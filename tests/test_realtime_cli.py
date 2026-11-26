"""Tests for the `hal voice realtime` CLI: PCM→WAV, event accumulator, mic
seam, and the full turn driver.

No network, no audio hardware: ``run_realtime_turn``'s ``open_realtime_fn``
seam injects a fake session, the mic seam injects synthetic PCM, and
``_play_wav`` is monkeypatched. Goodhart-real — asserts exact WAV params and
exact event routing, so a dropped dispatch handler or a changed sample rate
fails here.
"""
from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lib.hal.voice.realtime_base import RealtimeConfig, RealtimeEvent  # noqa: E402
from lib.hal.voice import realtime_cli as rc  # noqa: E402


# ---------- PCM → WAV ----------

def test_pcm_chunks_to_wav_valid_params() -> None:
    wav = rc.pcm_chunks_to_wav([b"\x01\x02" * 100, b"\x03\x04" * 50])
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getframerate() == 24_000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getnframes() == 150  # 150 mono 16-bit samples in


def test_pcm_chunks_to_wav_empty_is_valid() -> None:
    wav = rc.pcm_chunks_to_wav([])
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getnframes() == 0
        assert wf.getframerate() == 24_000


def test_validate_wav_true_false() -> None:
    assert rc._validate_wav(rc.pcm_chunks_to_wav([b"\x00\x00" * 10])) is True
    assert rc._validate_wav(b"not a wav at all") is False


# ---------- _TurnAccumulator dispatch ----------

def test_accumulator_routes_audio_transcript_done() -> None:
    acc = rc._TurnAccumulator()
    acc.on_event(RealtimeEvent(kind="audio_delta", audio=b"\xaa\xbb"))
    acc.on_event(RealtimeEvent(kind="audio_delta", audio=b"\xcc\xdd"))
    acc.on_event(RealtimeEvent(kind="assistant_transcript", text="hello "))
    acc.on_event(RealtimeEvent(kind="assistant_transcript", text="world"))
    assert acc.done is False
    acc.on_event(RealtimeEvent(kind="turn_done"))
    assert acc.pcm_chunks == [b"\xaa\xbb", b"\xcc\xdd"]
    assert "".join(acc.transcript_parts) == "hello world"
    assert acc.done is True
    assert acc.error is None


def test_accumulator_error_sets_done_and_error() -> None:
    acc = rc._TurnAccumulator()
    acc.on_event(RealtimeEvent(kind="error", text="boom"))
    assert acc.error == "boom"
    assert acc.done is True


def test_accumulator_ignores_unknown_kind() -> None:
    acc = rc._TurnAccumulator()
    # partial_transcript is intentionally not in the dispatch table.
    acc.on_event(RealtimeEvent(kind="partial_transcript", text="ignored"))
    assert acc.transcript_parts == []
    assert acc.done is False


def test_accumulator_dedups_final_after_deltas() -> None:
    acc = rc._TurnAccumulator()
    acc.on_event(RealtimeEvent(kind="assistant_transcript", text="Hello, "))
    acc.on_event(RealtimeEvent(kind="assistant_transcript", text="friend!"))
    # GA's .done carries the FULL transcript (is_final=True) — must not double.
    acc.on_event(RealtimeEvent(kind="assistant_transcript", text="Hello, friend!", is_final=True))
    assert "".join(acc.transcript_parts) == "Hello, friend!"


def test_accumulator_uses_final_when_no_deltas() -> None:
    acc = rc._TurnAccumulator()
    # A backend that emits only a final transcript (no deltas) is still captured.
    acc.on_event(RealtimeEvent(kind="assistant_transcript", text="full text", is_final=True))
    assert "".join(acc.transcript_parts) == "full text"


# ---------- fake session (DIP seam) ----------

class _FakeSession:
    def __init__(self, script: list | None = None) -> None:
        self._cb = None
        self.sent_audio: list[bytes] = []
        self.sent_text: list[str] = []
        self.started = False
        self.closed = False
        self._script = script or []

    def on_transcript(self, cb) -> None:
        self._cb = cb

    def start(self) -> None:
        self.started = True

    def send_audio(self, pcm: bytes) -> None:
        self.sent_audio.append(pcm)

    def send_text(self, text: str) -> None:
        self.sent_text.append(text)

    def commit_turn(self, commit_audio: bool = True) -> None:
        self.commit_audio_flag = commit_audio
        for ev in self._script:
            self._cb(ev)

    def close(self) -> None:
        self.closed = True


# ---------- mic seam ----------

def test_drive_mic_turn_streams_synthetic_pcm() -> None:
    acc = rc._TurnAccumulator()
    session = _FakeSession()
    chunks = [b"\x01\x01", b"\x02\x02", b"\x03\x03"]
    rc._drive_mic_turn(session, lambda: iter(chunks), acc, timeout_s=0.05)
    assert session.sent_audio == chunks


# ---------- run_realtime_turn (full path, injected) ----------

def test_run_realtime_turn_text_path_returns_result(monkeypatch) -> None:
    monkeypatch.setattr(rc, "_play_wav", lambda *a, **k: None)  # no audio device
    script = [
        RealtimeEvent(kind="audio_delta", audio=b"\x10\x20" * 8),
        RealtimeEvent(kind="assistant_transcript", text="hi there"),
        RealtimeEvent(kind="turn_done"),
    ]
    session = _FakeSession(script=script)
    result = rc.run_realtime_turn(
        text="say hi",
        open_realtime_fn=lambda *, backend, config: session,
        realtime_config_cls=RealtimeConfig,
        timeout_s=0.1,
    )
    assert session.sent_text == ["say hi"]
    # text turn must NOT commit the (empty) audio buffer — GA rejects that.
    assert session.commit_audio_flag is False
    assert session.started and session.closed
    assert result["audio_bytes"] == 16
    assert result["wav_valid"] is True
    assert result["transcript"] == "hi there"
    assert result["error"] is None


def test_run_realtime_turn_surfaces_error(monkeypatch) -> None:
    monkeypatch.setattr(rc, "_play_wav", lambda *a, **k: None)
    session = _FakeSession(script=[RealtimeEvent(kind="error", text="rate limited")])
    result = rc.run_realtime_turn(
        text="x",
        open_realtime_fn=lambda *, backend, config: session,
        realtime_config_cls=RealtimeConfig,
        timeout_s=0.1,
    )
    assert result["error"] == "rate limited"
