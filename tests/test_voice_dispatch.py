"""Two-sided Goodhart-proof tests for lib.hal.voice.dispatch.is_retryable.

Per Rule 14 every classification claim is tested BOTH as must-pass
(retryable subclasses return True, bare VoiceError with provider-side
text returns True) AND must-fail (caller-error hints return False,
non-VoiceError types return False). This prevents a regression where
the classifier silently retries caller errors (wasting API budget) or
silently skips transient failures (breaking fallover).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.voice.dispatch import NON_RETRYABLE_HINTS, is_retryable  # noqa: E402
from lib.hal.voice.providers._errors import (  # noqa: E402
    AuthFailure,
    NetworkError,
    QuotaExceeded,
    RateLimited,
)
from lib.hal.voice.types import VoiceError  # noqa: E402


# ---------- retryable subclasses (must-pass) ----------


def test_auth_failure_is_retryable() -> None:
    """Must-pass: next provider might have a working key."""
    assert is_retryable(AuthFailure("missing key")) is True


def test_network_error_is_retryable() -> None:
    """Must-pass: transport failure; sibling provider has different endpoint."""
    assert is_retryable(NetworkError("dns timeout")) is True


def test_quota_exceeded_is_retryable() -> None:
    """Must-pass: free-tier exhausted; local fallback still works."""
    assert is_retryable(QuotaExceeded("1M chars burned")) is True


def test_rate_limited_is_retryable() -> None:
    """Must-pass: this provider throttled; next rung may be fresh."""
    assert is_retryable(RateLimited("HTTP 429")) is True


# ---------- non-retryable caller errors (must-fail) ----------


def test_text_too_long_not_retryable() -> None:
    """Must-fail: every provider will reject the same oversize input."""
    assert is_retryable(VoiceError("text too long for TTS")) is False


def test_unsupported_voice_not_retryable() -> None:
    """Must-fail: voice-name mismatch is a caller bug; re-raise."""
    assert is_retryable(VoiceError("unsupported voice 'nonexistent'")) is False


def test_emotion_not_in_allowed_set_not_retryable() -> None:
    """Must-fail: emotion allowlist rejection; all providers similar check."""
    assert is_retryable(VoiceError("emotion 'aggressive' not in allowed set")) is False


def test_profile_tier_not_retryable() -> None:
    """Must-fail: profile-level caller policy; same result everywhere."""
    assert is_retryable(VoiceError("profile tier 2 cap exceeded")) is False


def test_non_retryable_hints_are_case_insensitive() -> None:
    """Must-fail: upper/lower/mixed case all blocked from retry."""
    assert is_retryable(VoiceError("TEXT TOO LONG")) is False
    assert is_retryable(VoiceError("Unsupported Voice")) is False


# ---------- bare VoiceError as retryable (must-pass) ----------


def test_bare_voice_error_is_retryable_by_default() -> None:
    """Must-pass: provider-side malfunction without a caller-error hint
    (e.g. schema drift, empty output) SHOULD fall over — sibling
    provider may succeed where this one doesn't."""
    assert is_retryable(VoiceError("malformed JSON response")) is True
    assert is_retryable(VoiceError("missing audioContent")) is True
    assert is_retryable(VoiceError("empty output file")) is True
    assert is_retryable(VoiceError("piper: subprocess exited 1")) is True


# ---------- non-VoiceError rejection (must-fail) ----------


def test_type_error_not_retryable() -> None:
    """Must-fail: programming errors recur on every provider, re-raise."""
    assert is_retryable(TypeError("bad arg")) is False


def test_key_error_not_retryable() -> None:
    assert is_retryable(KeyError("no such key")) is False


def test_value_error_not_retryable() -> None:
    assert is_retryable(ValueError("bad value")) is False


def test_arbitrary_exception_not_retryable() -> None:
    """Must-fail: a provider's internal bug (e.g. IndexError) must not
    trigger fallover — it indicates a real problem to surface."""
    assert is_retryable(IndexError("list out of range")) is False


# ---------- subclass discipline ----------


def test_subclasses_of_retryable_types_are_retryable() -> None:
    """Must-pass: a future subclass of RateLimited (e.g. an operator
    extension) still classifies as retryable via isinstance."""

    class MyRateLimit(RateLimited):
        pass

    assert is_retryable(MyRateLimit("custom")) is True


def test_non_retryable_hints_list_is_immutable_tuple() -> None:
    """Must-pass: NON_RETRYABLE_HINTS is a tuple so it cannot be
    mutated at runtime by an adversary patching the module."""
    assert isinstance(NON_RETRYABLE_HINTS, tuple)
    assert "text too long" in NON_RETRYABLE_HINTS
    assert "unsupported voice" in NON_RETRYABLE_HINTS
    assert "not in allowed set" in NON_RETRYABLE_HINTS
    assert "profile tier" in NON_RETRYABLE_HINTS
