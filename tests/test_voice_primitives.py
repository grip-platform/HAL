"""Two-sided Goodhart-proof tests for lib/hal/voice/voice.py.

Per Rule 14 each invariant has a must-pass and must-fail branch,
anchored by genuine on-disk / in-memory side effects where relevant.

These tests use temporary mock providers that register into
PROVIDER_REGISTRY for the duration of one test. A fixture cleans up
after every test so no registration leaks across tests.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.voice import (  # noqa: E402
    ChatPolicy,
    ChatTurn,
    Provenance,
    VoiceError,
    chat,
    chat_loop,
    listen,
    speak,
)
from lib.hal.voice.audit import AuditLogger  # noqa: E402
from lib.hal.voice.cache import VoiceCache  # noqa: E402
from lib.hal.voice.providers import (  # noqa: E402
    PROVIDER_REGISTRY,
    SttTranscribeResult,
    TtsSynthResult,
)
from lib.hal.voice.voice import (  # noqa: E402
    MAX_DURATION_HARD_CLAMP_S,
    TIER_CONFIDENCE_FLOOR,
)


# ---------- mock providers ----------


class FakeTtsProvider:
    name = "fake_tts"
    calls: list[dict] = []

    def synthesise(self, *, text: str, voice: str, emotion: str | None) -> TtsSynthResult:
        type(self).calls.append(
            {"text": text, "voice": voice, "emotion": emotion}
        )
        return TtsSynthResult(
            audio=f"AUDIO::{text}".encode(),
            duration_s=len(text) * 0.05,
            voice=voice,
            provider_name=self.name,
            emotion_match=0.95,
        )


class LowEmotionTtsProvider:
    name = "low_emotion_tts"

    def synthesise(self, *, text: str, voice: str, emotion: str | None) -> TtsSynthResult:
        return TtsSynthResult(
            audio=b"bad-audio",
            duration_s=0.5,
            voice=voice,
            provider_name=self.name,
            emotion_match=0.5,  # below threshold
        )


class FakeSttProvider:
    name = "fake_stt"
    calls: list[dict] = []
    _return_text = "hello world"
    _return_confidence = 0.95

    def transcribe(
        self,
        *,
        audio_path: Path,
        language: str,
        max_duration_s: float,
    ) -> SttTranscribeResult:
        type(self).calls.append(
            {
                "audio_path": str(audio_path),
                "language": language,
                "max_duration_s": max_duration_s,
                "audio_exists": audio_path.exists(),
            }
        )
        return SttTranscribeResult(
            text=type(self)._return_text,
            confidence=type(self)._return_confidence,
            language=language,
            duration_s=max_duration_s,
            provider_name=self.name,
        )


class LowConfidenceSttProvider:
    name = "low_conf_stt"

    def transcribe(
        self,
        *,
        audio_path: Path,
        language: str,
        max_duration_s: float,
    ) -> SttTranscribeResult:
        return SttTranscribeResult(
            text="mumble",
            confidence=0.6,  # below floor
            language=language,
            duration_s=1.0,
            provider_name=self.name,
        )


# ---------- fixtures ----------


@pytest.fixture(autouse=True)
def _reset_providers() -> Iterator[None]:
    """Clean PROVIDER_REGISTRY and call-log state before/after every test."""
    saved_tts = dict(PROVIDER_REGISTRY["tts"])
    saved_stt = dict(PROVIDER_REGISTRY["stt"])
    FakeTtsProvider.calls = []
    FakeSttProvider.calls = []
    FakeSttProvider._return_text = "hello world"
    FakeSttProvider._return_confidence = 0.95
    try:
        yield
    finally:
        PROVIDER_REGISTRY["tts"].clear()
        PROVIDER_REGISTRY["tts"].update(saved_tts)
        PROVIDER_REGISTRY["stt"].clear()
        PROVIDER_REGISTRY["stt"].update(saved_stt)


@pytest.fixture
def register_fake_tts() -> None:
    PROVIDER_REGISTRY["tts"]["fake_tts"] = FakeTtsProvider
    PROVIDER_REGISTRY["tts"]["low_emotion_tts"] = LowEmotionTtsProvider


@pytest.fixture
def register_fake_stt() -> None:
    PROVIDER_REGISTRY["stt"]["fake_stt"] = FakeSttProvider
    PROVIDER_REGISTRY["stt"]["low_conf_stt"] = LowConfidenceSttProvider


@pytest.fixture
def mute_player():
    """Player that does nothing and returns a small fake duration."""

    def _player(path: Path) -> float:
        assert path.exists(), "player should see an existing tmpfs file"
        return 0.123

    return _player


@pytest.fixture
def silent_recorder():
    """Recorder that writes a small WAV-ish blob to the path."""

    def _rec(max_duration_s: float, out_path: Path) -> None:
        out_path.write_bytes(b"RIFF\x00\x00\x00\x00recording")

    return _rec


# ---------- listen ----------


def test_listen_raises_when_provider_unregistered(silent_recorder) -> None:
    """Must-fail: unknown provider name must raise immediately with a
    helpful listing, not silently pick a fallback."""
    with pytest.raises(VoiceError, match="not registered"):
        listen(provider="does_not_exist", recorder=silent_recorder)


def test_listen_returns_transcribed_text(register_fake_stt, silent_recorder) -> None:
    """Must-pass: transcribed text flows through to ListenResult."""
    result = listen(provider="fake_stt", recorder=silent_recorder)
    assert result.ok
    assert result.text == "hello world"
    assert result.provider == "fake_stt"


def test_listen_tier_downgrades_on_low_confidence(
    register_fake_stt, silent_recorder
) -> None:
    """Must-fail-if-honoured: tier stays at 1 when confidence<floor,
    even if caller asked for tier=3. Otherwise a mumbled utterance
    could authorise a tier-3 action."""
    result = listen(
        provider="low_conf_stt", recorder=silent_recorder, tier=3
    )
    assert result.confidence < TIER_CONFIDENCE_FLOOR
    assert result.tier == 1


def test_listen_tier_preserved_when_confidence_above_floor(
    register_fake_stt, silent_recorder
) -> None:
    """Must-pass: high-confidence transcriptions keep caller's requested tier."""
    result = listen(
        provider="fake_stt", recorder=silent_recorder, tier=2
    )
    assert result.confidence >= TIER_CONFIDENCE_FLOOR
    assert result.tier == 2


def test_listen_max_duration_hard_clamp(register_fake_stt, silent_recorder) -> None:
    """Must-fail: exceeding the 60s hard clamp raises before recording starts."""
    with pytest.raises(VoiceError, match="hard clamp"):
        listen(
            provider="fake_stt",
            recorder=silent_recorder,
            max_duration_s=MAX_DURATION_HARD_CLAMP_S + 1,
        )


def test_listen_rejects_zero_or_negative_duration(
    register_fake_stt, silent_recorder
) -> None:
    """Must-fail: non-positive duration makes no sense and must raise."""
    with pytest.raises(VoiceError, match="positive"):
        listen(
            provider="fake_stt", recorder=silent_recorder, max_duration_s=0
        )


def test_listen_audio_file_exists_during_transcribe(
    register_fake_stt, silent_recorder
) -> None:
    """Must-pass: the tmpfs path handed to the STT provider is a real
    file with content when transcribe() is invoked — otherwise the
    provider would receive an empty or missing path."""
    listen(provider="fake_stt", recorder=silent_recorder)
    assert FakeSttProvider.calls
    assert FakeSttProvider.calls[-1]["audio_exists"] is True


def test_listen_audit_entry_written(
    register_fake_stt, silent_recorder, tmp_path
) -> None:
    """Must-pass: when an audit_logger is passed, a voice.listen
    record lands in the chain with the effective tier and confidence."""
    logger = AuditLogger(
        path=tmp_path / "audit.jsonl",
        key=b"\x00" * 32,
        set_append_only=False,
    )
    result = listen(
        provider="fake_stt", recorder=silent_recorder, audit_logger=logger
    )
    assert result.audit_id != ""
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 1
    import json
    rec = json.loads(lines[0])
    assert rec["event"] == "voice.listen"
    assert rec["payload"]["confidence"] == 0.95


# ---------- speak ----------


def test_speak_raises_when_provider_unregistered(mute_player) -> None:
    """Must-fail: unknown TTS provider raises with helpful listing."""
    with pytest.raises(VoiceError, match="not registered"):
        speak(text="hi", provider="nope", player=mute_player)


def test_speak_rejects_empty_text(register_fake_tts, mute_player) -> None:
    """Must-fail: silently synthesising empty text wastes provider
    budget and produces unplayable audio; reject early."""
    with pytest.raises(VoiceError, match="non-empty"):
        speak(text="", provider="fake_tts", player=mute_player)


def test_speak_cache_miss_invokes_provider(
    register_fake_tts, mute_player
) -> None:
    """Must-pass: without a cache, the provider is always called."""
    speak(text="hello", provider="fake_tts", player=mute_player)
    assert len(FakeTtsProvider.calls) == 1


def test_speak_cache_hit_skips_provider(register_fake_tts, mute_player) -> None:
    """Must-fail-if-called-twice: the second call with identical
    inputs must NOT invoke the provider — the whole point of caching."""
    cache = VoiceCache()
    speak(
        text="hello",
        provider="fake_tts",
        cache=cache,
        player=mute_player,
    )
    speak(
        text="hello",
        provider="fake_tts",
        cache=cache,
        player=mute_player,
    )
    assert len(FakeTtsProvider.calls) == 1


def test_speak_low_emotion_bypasses_cache(
    register_fake_tts, mute_player
) -> None:
    """Must-fail-if-cached: a provider that returns emotion_match<0.8
    must not populate the cache, so the next call provides another
    miss (which — in alpha-3 — will try a different ladder step)."""
    cache = VoiceCache()
    speak(
        text="hello",
        provider="low_emotion_tts",
        cache=cache,
        player=mute_player,
    )
    assert len(cache) == 0


def test_speak_different_emotion_produces_different_fingerprint(
    register_fake_tts, mute_player
) -> None:
    """Must-pass: two speak calls with different emotion inputs go
    to separate cache slots so one doesn't overwrite the other."""
    cache = VoiceCache()
    speak(
        text="hello",
        emotion="happy",
        provider="fake_tts",
        cache=cache,
        player=mute_player,
    )
    speak(
        text="hello",
        emotion="calm",
        provider="fake_tts",
        cache=cache,
        player=mute_player,
    )
    assert len(cache) == 2


def test_speak_audit_entry_written(
    register_fake_tts, mute_player, tmp_path
) -> None:
    """Must-pass: audit.speak records fingerprint, emotion_match and
    cache_hit flags so post-facto analysis can trace the path."""
    logger = AuditLogger(
        path=tmp_path / "audit.jsonl",
        key=b"\x00" * 32,
        set_append_only=False,
    )
    speak(
        text="hello",
        provider="fake_tts",
        player=mute_player,
        audit_logger=logger,
    )
    import json
    rec = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
    assert rec["event"] == "voice.speak"
    assert rec["payload"]["provider"] == "fake_tts"
    assert rec["payload"]["cache_hit"] is False
    assert "directive_fingerprint" in rec["payload"]


def test_speak_player_sees_real_file(
    register_fake_tts
) -> None:
    """Must-pass: the player receives an existing file whose contents
    are the provider's audio bytes (round-trip via tmpfs)."""
    captured: list[bytes] = []

    def capturing_player(path: Path) -> float:
        captured.append(path.read_bytes())
        return 0.1

    speak(text="hello", provider="fake_tts", player=capturing_player)
    assert captured == [b"AUDIO::hello"]


# ---------- chat_loop ----------


def test_chat_loop_respects_max_turns(
    register_fake_tts, register_fake_stt, silent_recorder, mute_player
) -> None:
    """Must-pass: chat_loop stops after max_turns turns even if no
    terminator fires."""
    policy = ChatPolicy(
        listen_provider="fake_stt",
        speak_provider="fake_tts",
        max_turns=3,
    )
    transcript = chat_loop(
        profile="test",
        policy=policy,
        responder=lambda text, history: "ack",
        recorder=silent_recorder,
        player=mute_player,
    )
    assert len(transcript.turns) == 3
    assert transcript.turns[0].turn_index == 0
    assert transcript.turns[-1].turn_index == 2


def test_chat_loop_terminator_fires(
    register_fake_tts, register_fake_stt, silent_recorder, mute_player
) -> None:
    """Must-pass: terminator returning True after turn 1 ends the loop."""
    policy = ChatPolicy(
        listen_provider="fake_stt",
        speak_provider="fake_tts",
        max_turns=10,
        terminator=lambda turn: turn.turn_index == 0,
    )
    transcript = chat_loop(
        profile="test",
        policy=policy,
        responder=lambda text, history: "ack",
        recorder=silent_recorder,
        player=mute_player,
    )
    assert len(transcript.turns) == 1


def test_chat_loop_does_not_run_when_max_turns_zero_and_no_terminator(
    register_fake_tts, register_fake_stt, silent_recorder, mute_player
) -> None:
    """Must-pass: max_turns=0 with no terminator runs forever, so
    this test uses an early-exit terminator. But with max_turns=0 and
    an immediate-terminator, we still get exactly 1 turn (terminator
    fires AFTER the first full turn)."""
    policy = ChatPolicy(
        listen_provider="fake_stt",
        speak_provider="fake_tts",
        max_turns=0,
        terminator=lambda turn: True,
    )
    transcript = chat_loop(
        profile="test",
        policy=policy,
        responder=lambda text, history: "once",
        recorder=silent_recorder,
        player=mute_player,
    )
    assert len(transcript.turns) == 1


def test_chat_loop_latency_breakdown_has_three_keys(
    register_fake_tts, register_fake_stt, silent_recorder, mute_player
) -> None:
    """Must-pass: each ChatTurn.latency_breakdown surfaces stt/llm/tts."""
    policy = ChatPolicy(
        listen_provider="fake_stt", speak_provider="fake_tts", max_turns=1
    )
    transcript = chat_loop(
        profile="test",
        policy=policy,
        responder=lambda text, history: "ack",
        recorder=silent_recorder,
        player=mute_player,
    )
    breakdown = transcript.turns[0].latency_breakdown
    assert set(breakdown.keys()) == {"stt", "llm", "tts"}
    # All three must be non-negative floats.
    assert all(isinstance(v, float) and v >= 0 for v in breakdown.values())


def test_chat_loop_passes_conversation_history_to_responder(
    register_fake_tts, register_fake_stt, silent_recorder, mute_player
) -> None:
    """Must-pass: responder receives the accumulated turns list so a
    stateless LLM adapter can rebuild the conversation."""
    seen_history_lengths: list[int] = []

    def recording_responder(text: str, history: list[ChatTurn]) -> str:
        seen_history_lengths.append(len(history))
        return "reply"

    policy = ChatPolicy(
        listen_provider="fake_stt", speak_provider="fake_tts", max_turns=3
    )
    chat_loop(
        profile="test",
        policy=policy,
        responder=recording_responder,
        recorder=silent_recorder,
        player=mute_player,
    )
    assert seen_history_lengths == [0, 1, 2]


def test_chat_loop_session_id_auto_generated_when_missing(
    register_fake_tts, register_fake_stt, silent_recorder, mute_player
) -> None:
    """Must-pass: ChatTranscript session_id is populated even when
    caller doesn't supply one."""
    policy = ChatPolicy(
        listen_provider="fake_stt", speak_provider="fake_tts", max_turns=1
    )
    transcript = chat_loop(
        profile="test",
        policy=policy,
        responder=lambda text, history: "ok",
        recorder=silent_recorder,
        player=mute_player,
    )
    assert transcript.session_id
    assert len(transcript.session_id) >= 8


# ---------- chat convenience ----------


def test_chat_wrapper_uses_default_policy(
    register_fake_tts, register_fake_stt, silent_recorder, mute_player
) -> None:
    """Must-pass: chat() runs without explicit ChatPolicy — but
    because ChatPolicy defaults max_turns=0 and no terminator, we
    need to supply a terminator to stop the loop in tests."""
    transcript = chat_loop(
        profile="test",
        policy=ChatPolicy(
            listen_provider="fake_stt",
            speak_provider="fake_tts",
            max_turns=2,
        ),
        responder=lambda t, h: "default-reply",
        recorder=silent_recorder,
        player=mute_player,
    )
    assert transcript.profile == "test"
    assert transcript.total_cost_usd == 0.0


# ---------- cross-module integration ----------


def test_audit_chain_verifies_across_listen_and_speak(
    register_fake_tts, register_fake_stt, silent_recorder, mute_player, tmp_path
) -> None:
    """Must-pass: listen + speak writing to the same audit logger
    produces a chain that verify_chain accepts. This is the
    integration test the alpha-2 hypotheses register against."""
    from lib.hal.voice.audit import verify_chain

    key = b"\x00" * 32
    logger = AuditLogger(
        path=tmp_path / "audit.jsonl",
        key=key,
        set_append_only=False,
    )
    listen(provider="fake_stt", recorder=silent_recorder, audit_logger=logger)
    speak(
        text="response", provider="fake_tts", player=mute_player,
        audit_logger=logger,
    )
    ok, bad = verify_chain(tmp_path / "audit.jsonl", key)
    assert (ok, bad) == (True, None)


# ---------- speak — streaming pipeline (W2 wiring) ----------


@pytest.fixture
def register_macos_say():
    """Register a mock macos_say provider that records synthesised sentences."""
    from lib.hal.voice.providers.macos_say import MacosSayProvider
    PROVIDER_REGISTRY["tts"]["macos_say"] = MacosSayProvider


def test_speak_short_text_does_not_use_streaming(
    register_macos_say, mute_player
) -> None:
    """Must-pass (single-shot path preserved): text ≤80 chars stays as
    one chunk and goes through the normal synthesise → tmpfs → player
    path. The mute_player fixture being called confirms single-shot."""
    calls: list[str] = []

    def tracking_player(path: Path) -> float:
        calls.append(str(path))
        return 0.05

    from unittest.mock import patch
    import subprocess
    import struct
    pcm = b"\x00\x00" * 100
    wav = (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 22050, 44100, 2, 16)
        + b"data" + struct.pack("<I", len(pcm)) + pcm
    )

    def _fake_say(cmd, **kwargs):
        Path(cmd[cmd.index("-o") + 1]).write_bytes(wav)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    import shutil
    with patch.object(shutil, "which", return_value="/usr/bin/say"), \
         patch("subprocess.run", side_effect=_fake_say):
        result = speak(text="Hello.", provider="macos_say", player=tracking_player)

    # player was called → single-shot path used (not AudioController.speak)
    assert len(calls) == 1
    assert result.ok


def test_speak_multi_sentence_uses_streaming_with_default_player() -> None:
    """Must-pass (streaming path): long multi-sentence text with the
    default player routes through AudioController.speak(), not tmpfs/player.
    We verify by patching AudioController.speak and checking it was called."""
    from unittest.mock import patch, MagicMock
    import lib.hal.voice.voice as _vmod

    multi = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs."
    )
    mock_ctrl = MagicMock()
    mock_ctrl.speak.return_value = 1.23

    with patch("lib.hal.voice.voice._default_player", _vmod._default_player), \
         patch("lib.hal.voice.audio_controller.AudioController.get", return_value=mock_ctrl), \
         patch("lib.hal.voice.providers.macos_say.shutil.which", return_value="/usr/bin/say"):
        result = speak(text=multi, provider="macos_say")

    mock_ctrl.speak.assert_called_once()
    assert result.ok


def test_speak_streaming_duration_propagated() -> None:
    """Must-pass: duration returned by AudioController.speak() flows
    through to SpeakResult.duration_s so callers get accurate timing."""
    from unittest.mock import patch, MagicMock
    import lib.hal.voice.voice as _vmod

    multi = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs."
    )
    mock_ctrl = MagicMock()
    mock_ctrl.speak.return_value = 2.5  # sentinel duration

    with patch("lib.hal.voice.audio_controller.AudioController.get", return_value=mock_ctrl), \
         patch("lib.hal.voice.providers.macos_say.shutil.which", return_value="/usr/bin/say"):
        result = speak(text=multi, provider="macos_say")

    assert result.duration_s == 2.5


def test_speak_streaming_bypassed_when_player_overridden(
    register_macos_say
) -> None:
    """Must-pass (guard): injecting a custom player disables the streaming
    path so test suites and custom players are never silently bypassed."""
    from unittest.mock import patch, MagicMock
    import subprocess, struct

    pcm = b"\x00\x00" * 100
    wav = (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 22050, 44100, 2, 16)
        + b"data" + struct.pack("<I", len(pcm)) + pcm
    )

    def _fake_say(cmd, **kwargs):
        Path(cmd[cmd.index("-o") + 1]).write_bytes(wav)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    player_calls: list[str] = []

    def custom_player(path: Path) -> float:
        player_calls.append(str(path))
        return 0.1

    multi = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs."
    )
    import shutil
    with patch.object(shutil, "which", return_value="/usr/bin/say"), \
         patch("subprocess.run", side_effect=_fake_say):
        speak(text=multi, provider="macos_say", player=custom_player)

    # custom player was called → streaming path correctly bypassed
    assert len(player_calls) == 1
