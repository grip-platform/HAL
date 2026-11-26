"""Two-sided Goodhart-proof tests for lib/hal/voice/ladder.py.

Per Rule 14 each invariant has a must-pass and must-fail branch.

The must-fail branch on ``get_ladder`` is load-bearing: without it,
a typo like ``ladder: premuim`` in profiles.yaml would silently fall
through to the free ladder and ship without emotion support — the
worst kind of silent degradation because the operator sees zero
provider errors and only notices months later that the child voice
profile "sounds weird."
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.voice.ladder import (  # noqa: E402
    FREE_LADDER,
    LADDERS,
    PREMIUM_LADDER,
    ProviderLadder,
    get_ladder,
)


# ---------- ladder shape ----------


def test_free_ladder_has_tts_and_stt_lists() -> None:
    """Must-pass: FREE_LADDER is a ProviderLadder-shaped dict."""
    assert set(FREE_LADDER.keys()) == {"tts", "stt"}
    assert isinstance(FREE_LADDER["tts"], list)
    assert isinstance(FREE_LADDER["stt"], list)


def test_premium_ladder_has_tts_and_stt_lists() -> None:
    """Must-pass: PREMIUM_LADDER is a ProviderLadder-shaped dict."""
    assert set(PREMIUM_LADDER.keys()) == {"tts", "stt"}


def test_free_ladder_tts_uses_macos_say() -> None:
    """Must-pass: the free ladder TTS MUST remain macOS-built-in-only
    so profiles declaring 'ladder: free' never accidentally incur
    cloud costs."""
    assert FREE_LADDER["tts"] == ["macos_say"]


def test_free_ladder_stt_includes_whisper_first() -> None:
    """Whisper is the first STT rung in the free ladder (local, zero cost).
    macos_dictation is retained as a fallback for when whisper is unavailable."""
    assert FREE_LADDER["stt"][0] == "whisper"
    assert "macos_dictation" in FREE_LADDER["stt"]


def test_premium_ladder_has_macos_fallback() -> None:
    """Must-pass: every premium ladder ends with a macOS built-in
    so network/budget failures still produce audio. A premium
    ladder WITHOUT a free-tier fallback is a reliability regression."""
    assert "macos_say" in PREMIUM_LADDER["tts"]
    assert "macos_dictation" in PREMIUM_LADDER["stt"]


def test_premium_ladder_prefers_paid_provider_first() -> None:
    """Must-fail-if-misordered: the FIRST entry in premium tts must
    be the preferred paid provider. If someone reorders so macos_say
    comes first, every premium call degrades to free — defeating
    the purpose of the premium ladder."""
    assert PREMIUM_LADDER["tts"][0] != "macos_say"
    assert PREMIUM_LADDER["tts"][-1] == "macos_say"
    assert PREMIUM_LADDER["stt"][-1] == "macos_dictation"


def test_ladders_contain_named_ladders() -> None:
    """Must-pass: the top-level LADDERS dict exposes the named ladders
    so get_ladder() resolves them by name. Tier-3 W4 added
    ``fallover`` alongside the original ``free`` / ``premium``."""
    assert {"free", "premium", "fallover"}.issubset(LADDERS.keys())
    assert LADDERS["free"] is FREE_LADDER
    assert LADDERS["premium"] is PREMIUM_LADDER


# ---------- get_ladder resolver ----------


def test_get_ladder_returns_free_by_name() -> None:
    """Must-pass: known name round-trips to the ladder."""
    assert get_ladder("free") is FREE_LADDER


def test_get_ladder_returns_premium_by_name() -> None:
    """Must-pass: known name round-trips to the ladder."""
    assert get_ladder("premium") is PREMIUM_LADDER


def test_get_ladder_rejects_unknown_name_with_keyerror() -> None:
    """Must-fail: unknown name raises KeyError (not returns default)."""
    with pytest.raises(KeyError):
        get_ladder("does-not-exist")


def test_get_ladder_error_message_lists_valid_names() -> None:
    """Must-pass: the error string surfaces the valid ladder names
    so an operator seeing ``ladder: premuim`` gets ``free, premium``
    in the traceback and can correct the typo immediately."""
    try:
        get_ladder("premuim")  # deliberate typo
    except KeyError as exc:
        msg = str(exc)
        assert "free" in msg
        assert "premium" in msg
    else:
        pytest.fail("get_ladder should have raised KeyError")


def test_get_ladder_rejects_empty_string() -> None:
    """Must-fail: empty string is not a ladder name; early failure
    beats silent default."""
    with pytest.raises(KeyError):
        get_ladder("")


# ---------- order preservation ----------


def test_premium_ladder_order_is_stable() -> None:
    """Must-pass: list ordering is preserved across ladder accesses
    (the lists are not reshuffled by set operations under the hood).
    Order matters because dispatch tries index 0 first."""
    first_access = list(PREMIUM_LADDER["tts"])
    second_access = list(LADDERS["premium"]["tts"])
    assert first_access == second_access


def test_ladder_lists_are_non_empty() -> None:
    """Must-fail: an empty ladder list gives the dispatcher nothing
    to try. Every registered ladder must have at least one entry
    per slot."""
    for name, ladder in LADDERS.items():
        assert ladder["tts"], f"ladder '{name}' has empty tts list"
        assert ladder["stt"], f"ladder '{name}' has empty stt list"


# ---------- TypedDict shape documentation ----------


def test_provider_ladder_typeddict_keys() -> None:
    """Must-pass: TypedDict declares {tts, stt} — documented as the
    two-slot contract. New callers reading the type see these keys
    immediately."""
    annotations = ProviderLadder.__annotations__
    assert set(annotations.keys()) == {"tts", "stt"}
