"""End-to-end tests for lib.hal.voice.speak(ladder=...) fallover dispatch.

Per Rule 14 every fallover invariant is tested two-sided: happy path
must produce a SpeakResult AND failure paths must propagate / raise /
re-raise correctly. Audit events are captured and asserted against.

Strategy:
- Stub TTS providers are installed into ``PROVIDER_REGISTRY["tts"]``
  under unique test names, inside a setup/teardown fixture that
  restores the original registry on exit.
- Real providers stay registered during the test; we just add stubs
  for the ladder names being exercised.
- Fake player returns a fixed duration so no audio is produced.
- ``AuditLogger`` is pointed at a tmp file so the fallover audit
  events can be read back and asserted.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.voice import speak  # noqa: E402
from lib.hal.voice.audit import AuditLogger  # noqa: E402
from lib.hal.voice.ladder import FALLOVER_LADDER, LADDERS  # noqa: E402
from lib.hal.voice.providers import (  # noqa: E402
    PROVIDER_REGISTRY,
    GeminiTtsProvider,
    GoogleCloudTtsProvider,
    PiperProvider,
)
from lib.hal.voice.providers._errors import (  # noqa: E402
    AuthFailure,
    NetworkError,
    RateLimited,
)
from lib.hal.voice.providers.base import TtsSynthResult  # noqa: E402
from lib.hal.voice.types import VoiceError  # noqa: E402


# ---------- stub factory ----------


def _make_stub(name: str, behaviour: Any = None) -> type:
    """Return a TtsProvider-compatible class with configured behaviour.

    ``behaviour``:
    - ``BaseException`` instance → raised on every ``synthesise`` call
    - ``None`` → default happy-path synth result
    - callable → called with kwargs and returned (for custom results)
    """
    def synth(self, *, text, voice, emotion):
        b = behaviour
        if isinstance(b, BaseException):
            raise b
        if callable(b):
            return b(text=text, voice=voice, emotion=emotion)
        return TtsSynthResult(
            audio=b"fakewav",
            duration_s=0.05,
            voice=voice or name,
            provider_name=name,
            emotion_match=0.9,
        )

    return type(
        f"_Stub_{name}",
        (),
        {"name": name, "synthesise": synth},
    )


def _fake_player(path: Path) -> float:
    _ = path
    return 0.01


@pytest.fixture
def registry_guard():
    """Snapshot + restore PROVIDER_REGISTRY around each test."""
    saved = {
        "tts": dict(PROVIDER_REGISTRY["tts"]),
        "stt": dict(PROVIDER_REGISTRY["stt"]),
    }
    yield PROVIDER_REGISTRY
    PROVIDER_REGISTRY["tts"].clear()
    PROVIDER_REGISTRY["tts"].update(saved["tts"])
    PROVIDER_REGISTRY["stt"].clear()
    PROVIDER_REGISTRY["stt"].update(saved["stt"])


@pytest.fixture
def ladder_guard():
    """Snapshot + restore LADDERS around each test.

    Uses a shallow ``dict()`` copy so the stored VALUES stay identity-
    equal to the module-level ``FREE_LADDER`` / ``PREMIUM_LADDER`` /
    ``FALLOVER_LADDER`` constants. Deep-copying would break other
    tests that assert ``get_ladder('free') is FREE_LADDER`` after
    this fixture tears down."""
    saved = dict(LADDERS)
    yield LADDERS
    LADDERS.clear()
    LADDERS.update(saved)


@pytest.fixture
def audit_log(tmp_path: Path) -> AuditLogger:
    key_path = tmp_path / "audit.key"
    key_path.write_bytes(b"A" * 32)
    key_path.chmod(0o600)
    return AuditLogger(
        path=tmp_path / "audit.jsonl",
        key_path=key_path,
        set_append_only=False,
    )


# ---------- PROVIDER_REGISTRY wiring ----------


def test_piper_registered() -> None:
    """Must-pass: W4 wiring lands PiperProvider in the TTS sub-dict."""
    assert PROVIDER_REGISTRY["tts"]["piper"] is PiperProvider


def test_google_cloud_tts_registered() -> None:
    """Must-pass: GoogleCloudTtsProvider wired under free-tier name."""
    assert (
        PROVIDER_REGISTRY["tts"]["google_cloud_tts_free"]
        is GoogleCloudTtsProvider
    )


def test_gemini_tts_registered() -> None:
    """Must-pass: GeminiTtsProvider wired under free-tier name."""
    assert PROVIDER_REGISTRY["tts"]["gemini_tts_free"] is GeminiTtsProvider


def test_registry_stt_has_whisper() -> None:
    """Whisper STT provider ships as the first registered STT provider."""
    assert "whisper" in PROVIDER_REGISTRY["stt"]


def test_registry_tts_has_all_four_providers() -> None:
    """Must-pass: no auto-discovery surprise — the four TTS providers
    (three Tier-3 + macos_say) are the ONLY registrations at this
    ship point."""
    assert set(PROVIDER_REGISTRY["tts"].keys()) == {
        "piper", "google_cloud_tts_free", "gemini_tts_free", "macos_say"
    }


# ---------- FALLOVER_LADDER shape ----------


def test_fallover_ladder_tts_order() -> None:
    """Must-pass: cloud-first, local-last ordering."""
    assert FALLOVER_LADDER["tts"] == [
        "gemini_tts_free",
        "google_cloud_tts_free",
        "piper",
        "macos_say",
    ]


def test_fallover_registered_in_ladders_dict() -> None:
    """Must-pass: ``get_ladder('fallover')`` resolves to FALLOVER_LADDER."""
    assert LADDERS["fallover"] is FALLOVER_LADDER


# ---------- dispatch-mode validation ----------


def test_speak_rejects_both_provider_and_ladder() -> None:
    """Must-fail: caller supplied BOTH modes → VoiceError (mutual exclusion)."""
    with pytest.raises(VoiceError, match="mutually exclusive"):
        speak(
            text="hi",
            provider="piper",
            ladder="fallover",
            player=_fake_player,
        )


def test_speak_default_uses_macos_say(registry_guard) -> None:
    """Must-pass: no provider, no ladder → default falls to macos_say.
    With macos_say registered (post-Tier-3), the call reaches the
    registered class. We swap the real class for a stub here so the
    test doesn't require the actual ``say`` binary on CI and asserts
    the default-provider resolution path landed on the right key."""
    called: dict[str, bool] = {}

    def synth(self, *, text, voice, emotion):
        called["yes"] = True
        return TtsSynthResult(
            audio=b"fakewav",
            duration_s=0.01,
            voice=voice or "Jamie (Premium)",
            provider_name="macos_say",
            emotion_match=0.80,
        )

    stub = type("_MacosStub", (), {"name": "macos_say", "synthesise": synth})
    PROVIDER_REGISTRY["tts"]["macos_say"] = stub
    result = speak(text="hi", player=_fake_player)
    assert called.get("yes") is True
    assert result.provider == "macos_say"


def test_speak_missing_ladder_name_raises_voice_error() -> None:
    """Must-fail: misspelled ladder name → VoiceError with listing."""
    with pytest.raises(VoiceError, match="not found"):
        speak(
            text="hi",
            ladder="fallovr",  # typo
            player=_fake_player,
        )


# ---------- ladder happy path ----------


def test_ladder_first_provider_succeeds_others_untouched(
    registry_guard, ladder_guard
) -> None:
    """Must-pass: ladder[0] returns → rest of ladder not consulted."""
    called: list[str] = []

    def _track(name):
        def synth(self, *, text, voice, emotion):
            called.append(name)
            return TtsSynthResult(
                audio=b"a",
                duration_s=0.1,
                voice=voice,
                provider_name=name,
                emotion_match=0.9,
            )
        return type(f"_T_{name}", (), {"name": name, "synthesise": synth})

    PROVIDER_REGISTRY["tts"]["first"] = _track("first")
    PROVIDER_REGISTRY["tts"]["second"] = _track("second")
    LADDERS["mini"] = {"tts": ["first", "second"], "stt": []}

    result = speak(text="hi", ladder="mini", player=_fake_player)
    assert result.provider == "first"
    assert called == ["first"]  # "second" never invoked


def test_ladder_result_has_expected_speak_result_shape(
    registry_guard, ladder_guard
) -> None:
    """Must-pass: ladder dispatch returns a SpeakResult with the winning
    provider's identity, not a synthetic ladder-generic one."""
    PROVIDER_REGISTRY["tts"]["winner"] = _make_stub("winner")
    LADDERS["just_one"] = {"tts": ["winner"], "stt": []}
    r = speak(text="yo", ladder="just_one", player=_fake_player)
    assert r.provider == "winner"
    assert r.ok is True


# ---------- fallover behaviour ----------


def test_ladder_falls_over_on_network_error(
    registry_guard, ladder_guard
) -> None:
    """Must-fail at first rung, must-pass at second: NetworkError is
    retryable, ladder consults the next provider."""
    PROVIDER_REGISTRY["tts"]["flaky"] = _make_stub(
        "flaky", NetworkError("dns")
    )
    PROVIDER_REGISTRY["tts"]["steady"] = _make_stub("steady")
    LADDERS["pair"] = {"tts": ["flaky", "steady"], "stt": []}

    r = speak(text="hi", ladder="pair", player=_fake_player)
    assert r.provider == "steady"


def test_ladder_falls_over_on_rate_limited(
    registry_guard, ladder_guard
) -> None:
    PROVIDER_REGISTRY["tts"]["busy"] = _make_stub(
        "busy", RateLimited("429")
    )
    PROVIDER_REGISTRY["tts"]["ready"] = _make_stub("ready")
    LADDERS["pair"] = {"tts": ["busy", "ready"], "stt": []}
    r = speak(text="hi", ladder="pair", player=_fake_player)
    assert r.provider == "ready"


def test_ladder_falls_over_on_auth_failure(
    registry_guard, ladder_guard
) -> None:
    PROVIDER_REGISTRY["tts"]["bad_key"] = _make_stub(
        "bad_key", AuthFailure("missing key")
    )
    PROVIDER_REGISTRY["tts"]["good"] = _make_stub("good")
    LADDERS["pair"] = {"tts": ["bad_key", "good"], "stt": []}
    r = speak(text="hi", ladder="pair", player=_fake_player)
    assert r.provider == "good"


def test_ladder_does_not_fallover_on_non_retryable_voice_error(
    registry_guard, ladder_guard
) -> None:
    """Must-fail: caller-error (e.g. text too long) halts the ladder
    immediately; sibling providers are NOT tried because they'd
    produce the same error."""
    PROVIDER_REGISTRY["tts"]["strict"] = _make_stub(
        "strict", VoiceError("text too long for TTS")
    )
    PROVIDER_REGISTRY["tts"]["lenient"] = _make_stub("lenient")
    LADDERS["pair"] = {"tts": ["strict", "lenient"], "stt": []}

    with pytest.raises(VoiceError, match="text too long"):
        speak(text="hi", ladder="pair", player=_fake_player)


def test_ladder_does_not_fallover_on_type_error(
    registry_guard, ladder_guard
) -> None:
    """Must-fail: programming errors (TypeError) halt the ladder."""
    PROVIDER_REGISTRY["tts"]["buggy"] = _make_stub(
        "buggy", TypeError("internal")
    )
    PROVIDER_REGISTRY["tts"]["stable"] = _make_stub("stable")
    LADDERS["pair"] = {"tts": ["buggy", "stable"], "stt": []}
    with pytest.raises(TypeError):
        speak(text="hi", ladder="pair", player=_fake_player)


def test_ladder_all_providers_raise_retryable_reraises_last(
    registry_guard, ladder_guard
) -> None:
    """Must-fail: every rung fails with a retryable error → the last
    provider's exception is re-raised (caller sees the most recent
    failure, not the first)."""
    PROVIDER_REGISTRY["tts"]["a"] = _make_stub("a", NetworkError("a-net"))
    PROVIDER_REGISTRY["tts"]["b"] = _make_stub("b", RateLimited("b-rate"))
    PROVIDER_REGISTRY["tts"]["c"] = _make_stub("c", AuthFailure("c-auth"))
    LADDERS["triple"] = {"tts": ["a", "b", "c"], "stt": []}
    with pytest.raises(AuthFailure, match="c-auth"):
        speak(text="hi", ladder="triple", player=_fake_player)


def test_ladder_skips_unregistered_provider(
    registry_guard, ladder_guard
) -> None:
    """Must-pass: a name in the ladder that isn't registered is SKIPPED
    with an audit/warn — not a hard error. Lets ladder dispatch be ahead
    of registry as providers ship (e.g. macos_say will slot in later)."""
    PROVIDER_REGISTRY["tts"]["real"] = _make_stub("real")
    LADDERS["mixed"] = {"tts": ["ghost", "real"], "stt": []}
    r = speak(
        text="hi",
        ladder="mixed",
        player=_fake_player,
    )
    assert r.provider == "real"


def test_ladder_empty_tts_list_raises(
    registry_guard, ladder_guard
) -> None:
    LADDERS["empty"] = {"tts": [], "stt": []}
    with pytest.raises(VoiceError, match="no TTS providers configured"):
        speak(text="hi", ladder="empty", player=_fake_player)


def test_ladder_all_unregistered_raises(
    registry_guard, ladder_guard
) -> None:
    """Must-fail: a ladder of pure ghosts → VoiceError with name list."""
    LADDERS["ghosts"] = {"tts": ["a", "b"], "stt": []}
    with pytest.raises(VoiceError, match="no registered TTS providers"):
        speak(text="hi", ladder="ghosts", player=_fake_player)


# ---------- audit event emission ----------


def test_fallover_audit_event_recorded(
    registry_guard, ladder_guard, audit_log: AuditLogger
) -> None:
    """Must-pass: each skipped provider emits a voice.dispatch.fallover
    event with the correct payload shape."""
    PROVIDER_REGISTRY["tts"]["flaky"] = _make_stub(
        "flaky", NetworkError("dns")
    )
    PROVIDER_REGISTRY["tts"]["steady"] = _make_stub("steady")
    LADDERS["pair"] = {"tts": ["flaky", "steady"], "stt": []}

    speak(
        text="hi",
        ladder="pair",
        player=_fake_player,
        audit_logger=audit_log,
        profile="default",
    )

    import json
    lines = Path(audit_log._path).read_text().splitlines()  # type: ignore[attr-defined]
    events = [json.loads(line)["event"] for line in lines]
    assert "voice.dispatch.fallover" in events
    # Find the fallover payload
    fallover_record = next(
        json.loads(line)
        for line in lines
        if json.loads(line)["event"] == "voice.dispatch.fallover"
    )
    payload = fallover_record["payload"]
    assert payload["ladder"] == "pair"
    assert payload["skipped_provider"] == "flaky"
    assert "NetworkError" in payload["reason"]
    assert payload["profile"] == "default"


def test_unregistered_skip_audit_event(
    registry_guard, ladder_guard, audit_log: AuditLogger
) -> None:
    """Must-pass: unregistered-name skips emit an event with reason
    'unregistered' — operators can grep the audit for ghost providers."""
    PROVIDER_REGISTRY["tts"]["real"] = _make_stub("real")
    LADDERS["mixed"] = {"tts": ["ghost", "real"], "stt": []}
    speak(
        text="hi",
        ladder="mixed",
        player=_fake_player,
        audit_logger=audit_log,
    )
    import json
    lines = Path(audit_log._path).read_text().splitlines()  # type: ignore[attr-defined]
    fallover_records = [
        json.loads(line) for line in lines
        if json.loads(line)["event"] == "voice.dispatch.fallover"
    ]
    assert len(fallover_records) == 1
    assert fallover_records[0]["payload"]["skipped_provider"] == "ghost"
    assert fallover_records[0]["payload"]["reason"] == "unregistered"


# ---------- stderr suppression ----------


def test_hal_voice_quiet_suppresses_stderr(
    registry_guard, ladder_guard, capsys
) -> None:
    """Must-pass: HAL_VOICE_QUIET=1 → no stderr on fallover. Audit trail
    still emits; stderr is just the operator-facing UI signal."""
    PROVIDER_REGISTRY["tts"]["flaky"] = _make_stub(
        "flaky", NetworkError("x")
    )
    PROVIDER_REGISTRY["tts"]["ok"] = _make_stub("ok")
    LADDERS["pair"] = {"tts": ["flaky", "ok"], "stt": []}
    with mock.patch.dict(os.environ, {"HAL_VOICE_QUIET": "1"}):
        speak(text="hi", ladder="pair", player=_fake_player)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_stderr_emits_by_default(
    registry_guard, ladder_guard, capsys
) -> None:
    """Must-fail (silent-failure guard): without HAL_VOICE_QUIET, the
    fallover line DOES land on stderr so operators see the skip."""
    env = dict(os.environ)
    env.pop("HAL_VOICE_QUIET", None)
    PROVIDER_REGISTRY["tts"]["flaky"] = _make_stub(
        "flaky", NetworkError("xyz")
    )
    PROVIDER_REGISTRY["tts"]["ok"] = _make_stub("ok")
    LADDERS["pair"] = {"tts": ["flaky", "ok"], "stt": []}
    with mock.patch.dict(os.environ, env, clear=True):
        speak(text="hi", ladder="pair", player=_fake_player)
    captured = capsys.readouterr()
    assert "skipping 'flaky'" in captured.err
    assert "NetworkError" in captured.err
