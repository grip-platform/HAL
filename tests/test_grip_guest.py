"""Tests for GRIP-Guest (lib/hal/grip_guest.py).

Mutation-resistant + zero network: infer and post are injected fakes. Each test
fails under a plausible wrong implementation (gutted persona, broken rate limit,
inferring/posting when rate-limited, missing mention).
"""
import pytest

from lib.hal.grip_guest import (
    GUEST_SYSTEM_PROMPT,
    GUEST_ENABLED_ENV,
    GuestResult,
    RateLimiter,
    guest_respond,
    handle,
    is_enabled,
)


class Clock:
    """Deterministic injectable clock."""

    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


# --- persona (the defence-in-depth layer) ------------------------------------

def test_persona_forbids_substrate_access():
    p = GUEST_SYSTEM_PROMPT.lower()
    # The sandbox prompt must state NO access to private/file/credential data.
    assert "no access" in p
    assert "never" in p
    for kw in ("file", "credential", "operational context", "decline"):
        assert kw in p, f"persona missing hard-limit keyword {kw!r}"


# --- guest_respond -----------------------------------------------------------

def test_guest_respond_composes_persona_and_returns_reply():
    seen = {}

    def fake_infer(prompt):
        seen["prompt"] = prompt
        return "  here is the answer  "

    out = guest_respond("what is HAPPI?", infer=fake_infer)
    assert out == "here is the answer"                 # stripped
    assert GUEST_SYSTEM_PROMPT in seen["prompt"]       # persona is the prefix
    assert "what is HAPPI?" in seen["prompt"]          # guest text included


# --- rate limiter ------------------------------------------------------------

def test_rate_limiter_blocks_within_min_interval():
    clk = Clock()
    rl = RateLimiter(max_per_window=10, window_s=60, min_interval_s=2, now=clk)
    assert rl.allow("u") is True
    clk.t += 1.0                                        # < 2s since last
    assert rl.allow("u") is False
    clk.t += 1.5                                        # now >= 2s
    assert rl.allow("u") is True


def test_rate_limiter_blocks_over_max_in_window():
    clk = Clock()
    rl = RateLimiter(max_per_window=3, window_s=60, min_interval_s=0, now=clk)
    for _ in range(3):
        clk.t += 0.1
        assert rl.allow("u") is True
    clk.t += 0.1
    assert rl.allow("u") is False                       # 4th within window blocked


def test_rate_limiter_window_expiry_frees_budget():
    clk = Clock()
    rl = RateLimiter(max_per_window=1, window_s=10, min_interval_s=0, now=clk)
    assert rl.allow("u") is True
    clk.t += 5
    assert rl.allow("u") is False                       # still in window
    clk.t += 6                                          # window elapsed (>10s total)
    assert rl.allow("u") is True


def test_rate_limiter_is_per_sender():
    clk = Clock()
    rl = RateLimiter(max_per_window=1, window_s=60, min_interval_s=0, now=clk)
    assert rl.allow("a") is True
    assert rl.allow("b") is True                        # b unaffected by a


# --- handle (end to end, injected) -------------------------------------------

def _ok_limiter():
    return RateLimiter(max_per_window=100, window_s=60, min_interval_s=0, now=Clock())


def test_handle_happy_path_infers_and_posts_with_mention():
    posts = []
    res = handle(
        "249557720464621568", "yo GRIP",
        infer=lambda prompt: "hello guest",
        post=lambda content: posts.append(content) or {"ok": True, "status": 204},
        limiter=_ok_limiter(),
    )
    assert res.ok is True and res.reply == "hello guest"
    assert posts == ["<@249557720464621568> hello guest"]   # mention prefix


def test_handle_rate_limited_does_not_infer_or_post():
    clk = Clock()
    rl = RateLimiter(max_per_window=1, window_s=60, min_interval_s=0, now=clk)
    rl.allow("u")  # exhaust the single slot
    infer_calls, posts = [], []
    res = handle(
        "u", "hi",
        infer=lambda p: infer_calls.append(p) or "x",
        post=lambda c: posts.append(c),
        limiter=rl,
    )
    assert res.ok is False and res.reason == "rate_limited"
    assert infer_calls == [] and posts == []            # no spend, no flood


def test_handle_empty_message_short_circuits():
    res = handle("u", "   ", infer=lambda p: "x", post=lambda c: None, limiter=_ok_limiter())
    assert res.ok is False and res.reason == "empty_message"


def test_handle_empty_reply_does_not_post():
    posts = []
    res = handle("u", "hi", infer=lambda p: "   ", post=lambda c: posts.append(c),
                 limiter=_ok_limiter())
    assert res.ok is False and res.reason == "empty_reply"
    assert posts == []


def test_handle_no_mention_when_disabled():
    posts = []
    handle("u", "hi", infer=lambda p: "yo", post=lambda c: posts.append(c),
           limiter=_ok_limiter(), mention=False)
    assert posts == ["yo"]


# --- activation switch -------------------------------------------------------

def test_is_enabled_default_off(monkeypatch):
    monkeypatch.delenv(GUEST_ENABLED_ENV, raising=False)
    assert is_enabled() is False


def test_is_enabled_true_when_set(monkeypatch):
    monkeypatch.setenv(GUEST_ENABLED_ENV, "1")
    assert is_enabled() is True


def test_handle_no_mention_for_non_numeric_sender():
    # A non-numeric id must NOT produce <@rawid>; post plain text instead.
    posts = []
    handle("not-a-snowflake", "hi", infer=lambda p: "yo",
           post=lambda c: posts.append(c), limiter=_ok_limiter())
    assert posts == ["yo"]


def test_handle_post_failure_returns_clean_result_not_raise():
    # If the outbound gate is off / wire fails, handle returns a clean denial,
    # never propagates the exception.
    from lib.hal.grip_guest import GuestResult
    from lib.hal.channel_backends import LiveSendNotEnabled

    def gated_post(content):
        raise LiveSendNotEnabled("HAL_CHANNEL_LIVE not set")

    res = handle("249557720464621568", "hi", infer=lambda p: "yo",
                 post=gated_post, limiter=_ok_limiter())
    assert res.ok is False and res.reason.startswith("post_failed")
    assert res.reply == "yo"   # the reply was produced, just not delivered


def test_rate_limiter_production_defaults_full_cycle():
    # Full coverage under the DOCUMENTED defaults (8/60s, 2s min interval), with
    # only the clock injected — proves allow -> min-interval-deny -> allow-after-2s
    # -> ... -> max-cap-deny, all at the real default thresholds.
    clk = Clock()
    rl = RateLimiter(now=clk)                 # defaults: max 8, window 60, min 2s
    allowed = 0
    for _ in range(8):
        if rl.allow("u"):
            allowed += 1
        clk.t += 2.0                          # respect the 2s min interval
    assert allowed == 8                        # exactly max_per_window in the window
    clk.t += 2.0
    assert rl.allow("u") is False             # 9th within 60s window -> capped


def test_rate_limiter_bounded_under_distinct_senders():
    # The unbounded-growth fix: a stream of one-shot senders cannot exceed the cap.
    clk = Clock()
    rl = RateLimiter(max_senders=10, min_interval_s=0, now=clk)
    for i in range(500):
        clk.t += 0.001
        rl.allow(f"sender-{i}")
    assert len(rl._hits) <= 10                 # LRU-bounded, not 500


def test_default_post_is_one_arg_adapting_to_two_arg_wire(monkeypatch):
    # Refutes the "post(content) vs discord_webhook_post(url, content) TypeError"
    # concern: grip_guest's _default_post is 1-arg and adapts to the 2-arg wire.
    import lib.hal.channel_backends as cb
    from lib.hal.grip_guest import _default_post
    seen = {}
    monkeypatch.setattr(cb, "_default_webhook_url", lambda: "https://wh.example/x")
    monkeypatch.setattr(cb, "discord_webhook_post",
                        lambda url, content: seen.update(url=url, content=content) or {"ok": True})
    out = _default_post("<@1> hi")             # ONE arg — no TypeError
    assert seen == {"url": "https://wh.example/x", "content": "<@1> hi"}
    assert out == {"ok": True}
