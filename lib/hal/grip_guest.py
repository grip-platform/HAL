"""GRIP-Guest — a sandboxed conversational GRIP for untrusted guests.

A guest (e.g. an AI Craftspeople Guild member) talks to GRIP in a Discord
channel. GRIP-Guest answers via HAL infer with a guest persona and **NO TOOLS**
— it cannot read a file, run a command, or reach the operator's substrate
(operational context, memory, client data). Safe BY CONSTRUCTION, not by
curation: there is no filesystem tool to misuse, so there is nothing to leak.

This is `rules/untrusted-content-capability-lock.md` at its strongest: a guest is
untrusted content, so it gets the most-restricted surface — talk, never touch.
The infer call passes no tools; the persona forbids substrate disclosure as
defence-in-depth (two independent layers, neither relied on alone).

Wiring (the inbound half): the grip-channel Discord bot reads a guest's message
and calls `handle(sender, text)`; the reply is posted back via the no-bot Discord
webhook (`discord_webhook_post`). Activation is gated by `GRIP_GUEST_ENABLED=1`
(default OFF — the single switch the operator flips; reactive-only, never
initiates). Every external boundary (infer, post) is dependency-injected so the
whole path is unit-tested with zero network and zero live messages.
"""
from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

from lib.hal.channel_backends import ChannelSendError, LiveSendNotEnabled

# Defence-in-depth persona. The PRIMARY guarantee is mechanical (no tools are
# ever passed to infer); this prompt is the second, independent layer.
GUEST_SYSTEM_PROMPT = (
    "You are GRIP-Guest, a sandboxed conversational instance of GRIP talking to a "
    "guest (a member of the AI Craftspeople Guild) in a public Discord channel.\n\n"
    "You may discuss GRIP, HAL, HAPPI, AI engineering, reasoning, and general "
    "technical topics in GRIP's plain, intuitive, falsification-respecting voice. "
    "Engage the mechanism: ask for the loop, the measurement, the falsifier.\n\n"
    "HARD LIMITS — you have NO access to any private data, files, operator memory, "
    "credentials, keys, channel IDs, client information, or operational context, "
    "and you must NEVER claim to or fabricate any. If asked for operator-private, "
    "internal, file, secret, or substrate data, decline plainly: you are a "
    "guest-facing sandbox with no such access. Do not role-play having it.\n\n"
    "Keep replies concise — chat-channel length."
)

GUEST_ENABLED_ENV = "GRIP_GUEST_ENABLED"


def is_enabled() -> bool:
    """True iff the operator has flipped the live switch (default OFF).

    GRIP-Guest is reactive-only (it answers when a guest speaks, never initiates),
    but it stays inert until this is set so the operator activates it consciously.
    """
    return os.environ.get(GUEST_ENABLED_ENV) == "1"


@dataclass(frozen=True)
class GuestResult:
    ok: bool
    reply: str = ""
    reason: str = ""


class RateLimiter:
    """Per-sender sliding-window limiter. A guest cannot flood infer/the channel.

    `allow(sender)` is True iff the sender has made < max_per_window calls in the
    last window_s seconds AND at least min_interval_s since their last allowed
    call. Pure in-memory (process-local); `now` is injectable for tests.
    """

    def __init__(self, *, max_per_window: int = 8, window_s: float = 60.0,
                 min_interval_s: float = 2.0, max_senders: int = 4096,
                 now: Callable[[], float] = time.time):
        self._max = max_per_window
        self._window = window_s
        self._min_interval = min_interval_s
        self._max_senders = max_senders
        self._now = now
        # OrderedDict = LRU: a per-sender deque, capped at max_senders so a stream
        # of distinct one-shot senders cannot grow it without bound
        # (rules/lifecycle-closed-or-expired.md — a tracked set needs a GC).
        self._hits: "OrderedDict[str, Deque[float]]" = OrderedDict()

    def allow(self, sender: str) -> bool:
        t = self._now()
        hits = self._hits.get(sender)
        if hits is None:
            hits = deque()
        self._hits[sender] = hits
        self._hits.move_to_end(sender)                       # mark recently used
        while hits and t - hits[0] > self._window:           # drop expired
            hits.popleft()
        allowed = not (hits and t - hits[-1] < self._min_interval) and len(hits) < self._max
        if allowed:
            hits.append(t)
        while len(self._hits) > self._max_senders:           # bound: evict LRU
            self._hits.popitem(last=False)
        return allowed


def guest_respond(text: str, *, infer: Callable[[str], str],
                  persona: str = GUEST_SYSTEM_PROMPT) -> str:
    """Compose the guest persona + message and return the model's reply.

    grip_infer takes a single prompt (no system slot), so the persona is the
    prompt prefix. `infer` is injected (text -> text); no tools are ever passed.
    """
    prompt = f"{persona}\n\n--- Guest message ---\n{text.strip()}"
    return (infer(prompt) or "").strip()


def handle(sender: str, text: str, *,
           infer: Optional[Callable[[str], str]] = None,
           post: Optional[Callable[[str], dict]] = None,
           limiter: Optional[RateLimiter] = None,
           mention: bool = True) -> GuestResult:
    """End-to-end: rate-limit -> infer (no tools) -> post the reply via webhook.

    All boundaries are injected; defaults wire to the real HAL infer + Discord
    webhook lazily. Returns a GuestResult; never raises on a normal denial.
    """
    text = (text or "").strip()
    if not text:
        return GuestResult(ok=False, reason="empty_message")
    limiter = limiter if limiter is not None else _shared_limiter()
    if not limiter.allow(sender):
        # Rate-limited: do NOT infer or post (no spend, no flood).
        return GuestResult(ok=False, reason="rate_limited")
    infer = infer or _default_infer
    post = post or _default_post
    reply = guest_respond(text, infer=infer)
    if not reply:
        return GuestResult(ok=False, reason="empty_reply")
    # Mention only a numeric Discord snowflake — never emit <@non-numeric>.
    content = f"<@{sender}> {reply}" if (mention and sender.isdigit()) else reply
    try:
        post(content)
    except (LiveSendNotEnabled, ChannelSendError) as e:
        # Disabled outbound gate or a wire failure is a clean denial, not a crash —
        # handle() stays total (never raises on a normal not-sent path).
        return GuestResult(ok=False, reply=reply, reason=f"post_failed:{type(e).__name__}")
    return GuestResult(ok=True, reply=reply)


# --- default (real) boundaries — lazy to keep the import graph a clean DAG -----

_LIMITER_SINGLETON: Optional[RateLimiter] = None


def _shared_limiter() -> RateLimiter:
    global _LIMITER_SINGLETON
    if _LIMITER_SINGLETON is None:
        _LIMITER_SINGLETON = RateLimiter()
    return _LIMITER_SINGLETON


def _default_infer(prompt: str) -> str:
    """Wire to the canonical GRIP-via-HAL syscall (grip_infer), NO tools, NO media."""
    from lib.hal.grip_runtime import grip_infer
    result = grip_infer(prompt)
    return result.text if getattr(result, "ok", False) else ""


def _default_post(content: str) -> dict:
    """Post the reply to the Discord channel via the no-bot webhook (#430)."""
    from lib.hal.channel_backends import discord_webhook_post, _default_webhook_url
    return discord_webhook_post(_default_webhook_url(), content)


def main(argv=None) -> int:
    """CLI: read the guest message from STDIN (never argv — untrusted content),
    sender id from --sender. Inert unless GRIP_GUEST_ENABLED=1."""
    import argparse

    ap = argparse.ArgumentParser(prog="grip-guest",
                                 description="Sandboxed conversational GRIP for guests")
    ap.add_argument("--sender", required=True, help="Discord user id of the guest")
    ap.add_argument("--mention", action="store_true", default=True)
    args = ap.parse_args(argv)

    if not is_enabled():
        print(f"GRIP-Guest disabled (set {GUEST_ENABLED_ENV}=1 to enable)", file=sys.stderr)
        return 3
    text = sys.stdin.read()
    result = handle(args.sender, text, mention=args.mention)
    print(result.reply if result.ok else f"[no-send: {result.reason}]")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
