"""Two-sided Goodhart-proof tests for lib/hal/voice/providers.

Per Rule 14 each Protocol invariant has a must-pass branch (a
valid-shape implementation is accepted) AND a must-fail branch (an
incomplete stub is rejected). Without the must-fail tests, a
refactor that removed @runtime_checkable would silently swap
Protocol compliance for "any object with any attribute."
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.voice.providers import (  # noqa: E402
    PROVIDER_REGISTRY,
    SttProvider,
    SttTranscribeResult,
    TtsProvider,
    TtsSynthResult,
)


# ---------- frozen result dataclasses ----------


def test_tts_synth_result_constructs_with_expected_fields() -> None:
    """Must-pass: all 5 fields round-trip into a TtsSynthResult."""
    r = TtsSynthResult(
        audio=b"WAV....",
        duration_s=1.5,
        voice="Iapetus",
        provider_name="gemini_tts",
        emotion_match=0.92,
    )
    assert r.audio == b"WAV...."
    assert r.emotion_match == 0.92


def test_tts_synth_result_is_frozen() -> None:
    """Must-fail: a provider cannot hand back a result then mutate
    emotion_match after handoff to game the cache threshold."""
    r = TtsSynthResult(
        audio=b"",
        duration_s=0.0,
        voice="v",
        provider_name="p",
        emotion_match=0.9,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.emotion_match = 0.5  # type: ignore[misc]


def test_stt_transcribe_result_constructs_with_expected_fields() -> None:
    """Must-pass: all 5 fields round-trip into an SttTranscribeResult."""
    r = SttTranscribeResult(
        text="hello there",
        confidence=0.88,
        language="en-US",
        duration_s=2.0,
        provider_name="macos_dictation",
    )
    assert r.text == "hello there"
    assert r.confidence == 0.88


def test_stt_transcribe_result_is_frozen() -> None:
    """Must-fail: the tier gate keys off confidence, so an immutable
    result stops a compromised provider from post-facto inflating
    the score after the gate's check."""
    r = SttTranscribeResult(
        text="t",
        confidence=0.5,
        language="en",
        duration_s=1.0,
        provider_name="p",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.confidence = 0.99  # type: ignore[misc]


# ---------- TtsProvider Protocol runtime check ----------


class _CompleteTtsStub:
    name = "complete_tts"

    def synthesise(
        self,
        *,
        text: str,
        voice: str,
        emotion: str | None,
    ) -> TtsSynthResult:
        return TtsSynthResult(
            audio=b"", duration_s=0.0, voice=voice,
            provider_name=self.name, emotion_match=1.0,
        )


class _IncompleteTtsStub:
    """Missing the synthesise method — should NOT satisfy the Protocol."""

    name = "incomplete_tts"


class _WrongAttrTtsStub:
    """Has synthesise but no `name` attribute."""

    def synthesise(
        self,
        *,
        text: str,
        voice: str,
        emotion: str | None,
    ) -> TtsSynthResult:
        raise NotImplementedError


def test_runtime_check_accepts_complete_tts() -> None:
    """Must-pass: a stub exposing name + synthesise satisfies TtsProvider."""
    assert isinstance(_CompleteTtsStub(), TtsProvider)


def test_runtime_check_rejects_incomplete_tts() -> None:
    """Must-fail: missing synthesise method → NOT a TtsProvider."""
    assert not isinstance(_IncompleteTtsStub(), TtsProvider)


def test_runtime_check_rejects_missing_name_attribute_tts() -> None:
    """Must-fail: Protocol requires `name` attribute; classes missing
    it are rejected even if synthesise is present."""
    assert not isinstance(_WrongAttrTtsStub(), TtsProvider)


# ---------- SttProvider Protocol runtime check ----------


class _CompleteSttStub:
    name = "complete_stt"

    def transcribe(
        self,
        *,
        audio_path: Path,
        language: str,
        max_duration_s: float,
    ) -> SttTranscribeResult:
        return SttTranscribeResult(
            text="", confidence=1.0, language=language,
            duration_s=0.0, provider_name=self.name,
        )


class _IncompleteSttStub:
    name = "incomplete_stt"


def test_runtime_check_accepts_complete_stt() -> None:
    """Must-pass: complete STT stub satisfies SttProvider."""
    assert isinstance(_CompleteSttStub(), SttProvider)


def test_runtime_check_rejects_incomplete_stt() -> None:
    """Must-fail: missing transcribe method → NOT a SttProvider."""
    assert not isinstance(_IncompleteSttStub(), SttProvider)


# ---------- cross-Protocol isolation ----------


def test_tts_provider_is_not_an_stt_provider() -> None:
    """Must-fail: a class implementing ONLY synthesise is NOT an
    SttProvider. Protocol isolation keeps the registry slot-safe —
    operators wiring a TTS provider into the STT registry slot
    should be rejected at validation time."""
    assert isinstance(_CompleteTtsStub(), TtsProvider)
    assert not isinstance(_CompleteTtsStub(), SttProvider)


def test_stt_provider_is_not_a_tts_provider() -> None:
    """Must-fail: a class implementing ONLY transcribe is NOT a
    TtsProvider. Same slot-safety rationale."""
    assert isinstance(_CompleteSttStub(), SttProvider)
    assert not isinstance(_CompleteSttStub(), TtsProvider)


# ---------- PROVIDER_REGISTRY ----------


def test_registry_has_tts_and_stt_keys() -> None:
    """Must-pass: top-level shape is {'tts': {...}, 'stt': {...}}."""
    assert set(PROVIDER_REGISTRY.keys()) == {"tts", "stt"}


def test_registry_populated_post_macos_say() -> None:
    """Must-pass: post-Tier-3 + macos_say the four TTS providers are
    registered; Whisper is the first STT provider.
    Supersedes both the alpha-2 W4 empty-dict assertion and
    the Tier-3 W4 three-provider variant."""
    assert set(PROVIDER_REGISTRY["tts"].keys()) == {
        "piper", "google_cloud_tts_free", "gemini_tts_free", "macos_say"
    }
    assert "whisper" in PROVIDER_REGISTRY["stt"]


def test_registry_accepts_protocol_conforming_class() -> None:
    """Must-pass: registering a class that satisfies TtsProvider
    via the explicit dict assignment is allowed. Tests simulate
    alpha-3 behaviour without importing provider modules."""
    original = dict(PROVIDER_REGISTRY["tts"])
    try:
        PROVIDER_REGISTRY["tts"]["test_complete"] = _CompleteTtsStub
        assert isinstance(
            PROVIDER_REGISTRY["tts"]["test_complete"](), TtsProvider
        )
    finally:
        PROVIDER_REGISTRY["tts"].clear()
        PROVIDER_REGISTRY["tts"].update(original)


def test_registry_lookup_returns_registered_class() -> None:
    """Must-pass: round-trip registration and lookup works as a plain
    dict — no hidden indirection layer."""
    original = dict(PROVIDER_REGISTRY["stt"])
    try:
        PROVIDER_REGISTRY["stt"]["rt"] = _CompleteSttStub
        got = PROVIDER_REGISTRY["stt"]["rt"]
        assert got is _CompleteSttStub
    finally:
        PROVIDER_REGISTRY["stt"].clear()
        PROVIDER_REGISTRY["stt"].update(original)


def test_registry_missing_key_raises_keyerror() -> None:
    """Must-fail: the registry is a plain dict; absent keys raise
    KeyError. No implicit default-provider fallback — the outer
    voice primitive is responsible for ladder fallback."""
    with pytest.raises(KeyError):
        _ = PROVIDER_REGISTRY["tts"]["does_not_exist_at_all"]
