"""HAL GripChannelAdapter — the concrete wire to grip-channel.

One ChannelAdapter per platform (whatsapp / slack / discord). `send()` POSTs to
the grip-channel daemon. The network call is dependency-injected (`post_fn`) so:

  - tests run with a fake and zero network, and
  - autonomous loops never touch the wire (syscall doctrine; rules/dispatch-*).

The default `post_fn` (`grip_channel_post`) FAILS CLOSED unless the operator has
explicitly set HAL_CHANNEL_LIVE=1. This makes the "no autonomous broadcast" rail
mechanical, not merely a discipline: an accidental non-dry-run call in a loop
raises instead of messaging a real person.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Callable, Optional

from lib.hal.channels import ChannelRouter
from lib.hal.channel_adapter import (
    ChannelAdapter,
    CAPABILITY_TEXT,
    CAPABILITY_FILE,
    CAPABILITY_IMAGE,
    CAPABILITY_THREADS,
)
from lib.hal.human_channels import (
    PeerRegistry,
    HumanChannelGateway,
    JsonPresence,
    hal_home,
)

# Per-platform grip-channel endpoint (rules/operational-context.md W10 topology):
# WhatsApp stays on the Mac daemon (Hetzner IP is blocked by WhatsApp); Slack and
# Discord moved to the VPS in the W10 cutover.
_ENDPOINTS = {
    "whatsapp": "http://127.0.0.1:3101/mcp",
    "slack": "https://channel.grip-web.com/mcp",
    "discord": "https://channel.grip-web.com/mcp",
}

_CAPS = {
    "whatsapp": frozenset({CAPABILITY_TEXT, CAPABILITY_IMAGE, CAPABILITY_FILE}),
    "slack": frozenset({CAPABILITY_TEXT, CAPABILITY_FILE, CAPABILITY_THREADS, CAPABILITY_IMAGE}),
    "discord": frozenset({CAPABILITY_TEXT, CAPABILITY_FILE, CAPABILITY_THREADS, CAPABILITY_IMAGE}),
}


class UnknownPlatform(KeyError):
    """No grip-channel endpoint is known for the platform."""


class LiveSendNotEnabled(RuntimeError):
    """A real outbound was attempted without operator opt-in (HAL_CHANNEL_LIVE=1)."""


class ChannelSendError(RuntimeError):
    """The grip-channel wire returned but did NOT confirm delivery (e.g. the
    platform was not connected). A 200 response is not proof of delivery."""


def _check_send_response(resp_obj: dict) -> None:
    """Raise ChannelSendError unless the response confirms delivery.

    grip-channel returns a 200 result even on failure (e.g. 'Platform discord
    not connected' when the bridge is down), so a bare no-exception is NOT proof
    a message was sent — only a 'Sent ...' confirmation is. Fix silent failures
    loudly: an undelivered message must surface as an error, never ok.
    """
    if "error" in resp_obj:
        raise ChannelSendError(f"grip-channel error: {resp_obj['error']}")
    try:
        reply = resp_obj["result"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        reply = ""
    # grip-channel confirms delivery with "Sent to <platform> <recipient>" or
    # "Sent <n> blocks to ..." — always "Sent " + detail. Match the trailing space
    # so lookalikes ("Sentinel...") and any non-confirmation text are rejected.
    if not reply.startswith("Sent "):
        raise ChannelSendError(f"grip-channel did not confirm delivery: {reply or resp_obj!r}")


def _sse_json(raw: bytes) -> dict:
    """Extract the JSON object from a grip-channel SSE 'data:' line."""
    for line in raw.decode("utf-8", "replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("data:"):
            return json.loads(stripped[len("data:"):].strip())
    return {}


# Cloudflare fronts the VPS grip-channel endpoint and 403s the default
# Python-urllib User-Agent (verified 2026-06-06: default UA -> 403, browser UA ->
# 200). Every VPS-routed send (slack + discord) needs a browser-like UA; WhatsApp
# (local Mac daemon, no Cloudflare) is unaffected.
_USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
               "AppleWebKit/537.36 (KHTML, like Gecko) hal-channel/0.1")


def _base_headers() -> dict:
    """MCP request headers, incl. a browser-like User-Agent to pass Cloudflare."""
    return {"Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": _USER_AGENT}


def _resolve_recipient(address: str) -> str:
    """Resolve a gateway address to a wire recipient.

    Convention (platform-agnostic BY DESIGN, not a Discord-only special case): a
    'user:<id>' address denotes a direct message to that user id on ANY platform,
    and the wire wants the bare id; a bare address passes through unchanged. The
    convention lives here — named and tested — so it can never silently malform a
    recipient for a future prefixed platform.
    """
    prefix = "user:"
    return address[len(prefix):] if address.startswith(prefix) else address


def grip_channel_post(platform: str, address: str, text: str, *,
                      endpoints: Optional[dict] = None,
                      timeout: float = 15.0) -> dict:
    """Default post_fn — deliver via grip-channel MCP `channel_reply` (real wire).

    Fails closed unless HAL_CHANNEL_LIVE=1 (operator opt-in). Not exercised in
    tests (a fake post_fn is injected) and never called by autonomous loops.

    Recipient resolution is via _resolve_recipient (a 'user:<id>' address is a DM
    to that user on any platform). VERIFIED by code-read (no send needed):
    grip-channel index.ts channel_reply routes discord to DiscordBridge.sendText,
    which tries channels.fetch then falls back to users.fetch().createDM().send()
    (discord.ts:1150-1172) — so a bare user id is delivered as a Discord DM.
    """
    if os.environ.get("HAL_CHANNEL_LIVE") != "1":
        raise LiveSendNotEnabled(
            "live send disabled; set HAL_CHANNEL_LIVE=1 to enable real grip-channel "
            "delivery (operator-confirmed only). Use dry_run=True for a no-network check."
        )
    endpoints = endpoints or _ENDPOINTS
    url = endpoints.get(platform)
    if url is None:
        raise UnknownPlatform(f"no grip-channel endpoint for platform {platform!r}")
    recipient = _resolve_recipient(address)

    headers = _base_headers()

    def _rpc(body: dict, session: Optional[str] = None) -> tuple:
        # Context-managed: the response is always closed (no fd / connection leak).
        h = dict(headers)
        if session:
            h["Mcp-Session-Id"] = session
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.headers.get("Mcp-Session-Id"), resp.read()

    # 1) initialize -> capture Mcp-Session-Id
    session, _ = _rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                                  "clientInfo": {"name": "hal-channel", "version": "0.1.0"}}})
    # 2) acknowledge initialisation
    _rpc({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
         session=session)
    # 3) channel_reply
    _, raw = _rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                   "params": {"name": "channel_reply",
                              "arguments": {"platform": platform,
                                            "recipient": recipient,
                                            "message": text}}},
                  session=session)
    resp_obj = _sse_json(raw)
    _check_send_response(resp_obj)  # raise unless the wire confirms delivery
    return resp_obj


class GripChannelAdapter(ChannelAdapter):
    """Outbound-only ChannelAdapter that POSTs to grip-channel."""

    def __init__(self, router, platform: str, *, post_fn: Callable = None):
        self.transport = platform          # set before super().__init__ (it validates)
        super().__init__(router)
        self._post = post_fn or grip_channel_post

    def send(self, envelope) -> dict:
        return self._post(self.transport, envelope.thread_id, envelope.body)

    def capabilities(self) -> frozenset:
        return _CAPS.get(self.transport, frozenset({CAPABILITY_TEXT}))


# --- Discord via webhook (no bot) --------------------------------------------
# A Discord incoming webhook is a URL you POST JSON to; it drops a message into
# one channel. No bot, no bot token, no guild membership, no daemon, no
# mutual-guild wall (the bot path failed on "no mutual guilds"). To NOTIFY a
# person, mention them (<@id>) in a channel they can see — Discord's own rule,
# true of ANY reach. The webhook URL is a channel-post credential: kept in the
# keychain (grip-discord-webhook), never the repo; read in-process (never an arg).

_DISCORD_WEBHOOK_KEYCHAIN = "grip-discord-webhook"
_DISCORD_WEBHOOK_ENV = "HAL_DISCORD_WEBHOOK_URL"
_WEBHOOK_PLATFORM = "discord"


def _keychain(service: str) -> str:
    """Read a macOS keychain generic-password value in-process (never an arg).

    Returns '' on a non-macOS host (no `security` binary) so callers fall back to
    the env var — the VPS/Linux path (GRIP-Guest runs there, no keychain)."""
    if sys.platform != "darwin":
        return ""
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _default_webhook_url() -> str:
    # Env first (works on the VPS / Linux, no macOS keychain), then the keychain.
    url = os.environ.get(_DISCORD_WEBHOOK_ENV, "").strip() or _keychain(_DISCORD_WEBHOOK_KEYCHAIN)
    if not url:
        raise ChannelSendError(
            f"no Discord webhook configured (set ${_DISCORD_WEBHOOK_ENV} "
            f"or macOS keychain {_DISCORD_WEBHOOK_KEYCHAIN!r})"
        )
    return url


def _webhook_content(recipient: str, body: str) -> str:
    """Prefix a <@id> mention when the recipient is a 'user:<id>' endpoint, so the
    channel post pings that person; a bare recipient passes through unchanged."""
    if recipient.startswith("user:"):
        return f"<@{_resolve_recipient(recipient)}> {body}".strip()
    return body


def discord_webhook_post(url: str, content: str, *, timeout: float = 15.0) -> dict:
    """Deliver via a Discord incoming webhook (POST {content}). Fails closed unless
    HAL_CHANNEL_LIVE=1 (operator opt-in); a non-2xx raises (delivery confirmed)."""
    if os.environ.get("HAL_CHANNEL_LIVE") != "1":
        raise LiveSendNotEnabled(
            "live send disabled; set HAL_CHANNEL_LIVE=1 to enable real Discord "
            "webhook delivery (operator-confirmed only). Use dry_run=True for a check."
        )
    data = json.dumps({"content": content}).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise ChannelSendError(f"Discord webhook HTTP {e.code}: {e.read()[:200]!r}")
    except (urllib.error.URLError, OSError) as e:
        # DNS/TLS/timeout/connection — fail closed per the docstring contract.
        raise ChannelSendError(f"Discord webhook network error: {e}")
    if code not in (200, 204):
        raise ChannelSendError(f"Discord webhook HTTP {code}: {raw[:200]!r}")
    return {"ok": True, "status": code}


class DiscordWebhookAdapter(ChannelAdapter):
    """Outbound Discord adapter that posts via a channel webhook (no bot). The
    envelope's thread_id ('user:<id>') becomes a <@id> mention in the post."""

    def __init__(self, router, *, url_resolver=None, post_fn=None):
        self.transport = _WEBHOOK_PLATFORM
        super().__init__(router)
        self._url = url_resolver or _default_webhook_url
        self._post = post_fn or discord_webhook_post

    def send(self, envelope) -> dict:
        return self._post(self._url(), _webhook_content(envelope.thread_id, envelope.body))

    def capabilities(self) -> frozenset:
        return frozenset({CAPABILITY_TEXT})


def build_gateway(registry: PeerRegistry = None, *, post_fn: Callable = None,
                  presence=None, transcoder=None) -> HumanChannelGateway:
    """Wire a ready-to-use gateway: load peers, register an adapter per platform
    any peer uses (Discord via a no-bot channel webhook, others via grip-channel).
    Adding a peer/platform via config auto-registers its adapter — no code change
    (H-HCG-1). `post_fn` is injectable for tests (grip-channel platforms).
    """
    registry = registry or PeerRegistry.from_config()
    router = ChannelRouter()
    platforms = set()
    for p in registry.peers():
        platforms.update(p.channels())
    for plat in sorted(platforms):
        if plat == _WEBHOOK_PLATFORM:
            # Discord has its OWN post_fn contract (url, content) — distinct from
            # build_gateway's grip-channel `post_fn` (platform, address, text), so
            # it is NOT threaded here. The webhook adapter manages its own resolver;
            # tests inject DiscordWebhookAdapter(post_fn=...) directly.
            router.register(plat, DiscordWebhookAdapter(router))  # no bot — channel webhook
        else:
            router.register(plat, GripChannelAdapter(router, plat, post_fn=post_fn))
    if presence is None:
        presence = JsonPresence(hal_home() / "state" / "peer_presence.json")
    return HumanChannelGateway(registry, router, presence=presence, transcoder=transcoder)
