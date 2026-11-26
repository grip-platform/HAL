"""Two-sided Goodhart-proof tests for lib.hal.voice.providers.macos_say.

Per Rule 14 every invariant has a must-pass branch (happy-path synth,
WAV shape, correct emotion_match constant) AND a must-fail branch
(missing binary, non-zero exit, timeout, empty output, malformed WAV).

Mocking:
- ``subprocess.run`` is patched with a fake writer that emits a
  canonical 44-byte-header WAV to the ``-o`` argument. No real ``say``
  binary invoked in CI (portable for Linux runners).
- ``shutil.which`` toggled per-test.
"""
from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.voice.providers import macos_say as ms  # noqa: E402
from lib.hal.voice.providers import (  # noqa: E402
    PROVIDER_REGISTRY,
    MacosSayProvider,
    SttProvider,
    TtsProvider,
)
from lib.hal.voice.providers.base import TtsSynthResult  # noqa: E402
from lib.hal.voice.providers.macos_say import (  # noqa: E402
    DEFAULT_VOICE,
    MACOS_SAY_EMOTION_MATCH,
    SAY_BIN,
    _parse_wav_duration_s,
    is_available,
    make_synthesise_fn,
    synthesise_bytes,
)
from lib.hal.voice.types import VoiceError  # noqa: E402


# ---------- helpers ----------


def _make_wav(
    pcm_bytes: bytes = b"\x00\x00" * 1000,
    sample_rate: int = 22050,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Canonical 44-byte-header RIFF/WAVE for mocked `say` output."""
    subchunk2 = len(pcm_bytes)
    chunk = 36 + subchunk2
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return (
        b"RIFF"
        + struct.pack("<I", chunk)
        + b"WAVE"
        + b"fmt "
        + struct.pack(
            "<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample
        )
        + b"data"
        + struct.pack("<I", subchunk2)
        + pcm_bytes
    )


def _writer_fake(wav: bytes):
    """Return a subprocess.run fake that writes ``wav`` to the -o path."""

    def _fake(cmd, **kwargs):
        idx = cmd.index("-o")
        Path(cmd[idx + 1]).write_bytes(wav)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    return _fake


# ---------- Protocol conformance ----------


def test_provider_satisfies_tts_protocol() -> None:
    assert isinstance(MacosSayProvider(), TtsProvider)


def test_provider_is_not_an_stt_provider() -> None:
    """Must-fail: slot-safety — macos_say is TTS-only."""
    assert not isinstance(MacosSayProvider(), SttProvider)


def test_provider_name_is_macos_say() -> None:
    """Must-pass: FALLOVER_LADDER final rung looks this key up."""
    assert MacosSayProvider().name == "macos_say"


def test_macos_say_registered_in_provider_registry() -> None:
    """Must-pass: W4 ladder finds macos_say via registry lookup rather
    than the 'unregistered' skip branch."""
    assert PROVIDER_REGISTRY["tts"]["macos_say"] is MacosSayProvider


# ---------- is_available ----------


def test_is_available_true_when_binary_present() -> None:
    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"):
        assert is_available() is True


def test_is_available_false_when_binary_missing() -> None:
    """Must-fail (ready-state): ladder skips cleanly on non-macOS hosts."""
    with mock.patch.object(ms.shutil, "which", return_value=None):
        assert is_available() is False


# ---------- happy path ----------


def test_synthesise_returns_tts_synth_result() -> None:
    wav = _make_wav()
    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_writer_fake(wav)):
        r = MacosSayProvider().synthesise(
            text="hello", voice=DEFAULT_VOICE, emotion=None
        )
    assert isinstance(r, TtsSynthResult)
    assert r.audio == wav
    assert r.voice == DEFAULT_VOICE
    assert r.provider_name == "macos_say"


def test_synthesise_emotion_match_is_the_macos_constant() -> None:
    """Must-pass: exact 0.80 — at the cache threshold so output is
    cached, but below Piper (0.85) and the cloud providers (0.9/0.95)
    for correct ladder-quality ordering."""
    wav = _make_wav()
    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_writer_fake(wav)):
        r = MacosSayProvider().synthesise(
            text="x", voice=DEFAULT_VOICE, emotion="calm"
        )
    assert r.emotion_match == MACOS_SAY_EMOTION_MATCH == 0.80


def test_synthesise_ignores_emotion_value() -> None:
    """Must-pass: `say` has no emotion channel; same emotion_match for
    every emotion including None. Crucially, the emotion string must
    NOT be prepended to the text (unlike Gemini) because `say` would
    speak the bracketed marker verbatim."""
    wav = _make_wav()
    captured: dict[str, str] = {}

    def _capture(cmd, **kwargs):
        # Last element of cmd is the spoken text
        captured["text"] = cmd[-1]
        Path(cmd[cmd.index("-o") + 1]).write_bytes(wav)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_capture):
        MacosSayProvider().synthesise(text="hello", voice=DEFAULT_VOICE, emotion="calm")
    assert captured["text"] == "hello"
    assert "[calm]" not in captured["text"]


def test_synthesise_empty_voice_falls_back_to_default() -> None:
    wav = _make_wav()
    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_writer_fake(wav)):
        r = MacosSayProvider().synthesise(text="x", voice="", emotion=None)
    assert r.voice == DEFAULT_VOICE


def test_synthesise_uses_wave_file_format_flags() -> None:
    """Must-pass: exact flag shape sent to `say` — file-format=WAVE +
    data-format=LEI16 so output matches the LINEAR16 PCM WAV shape
    the other providers produce."""
    wav = _make_wav()
    captured: dict[str, list[str]] = {}

    def _capture(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(wav)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_capture):
        MacosSayProvider().synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)
    cmd = captured["cmd"]
    assert cmd[0] == SAY_BIN
    assert "--file-format=WAVE" in cmd
    assert "--data-format=LEI16" in cmd
    assert "-v" in cmd and cmd[cmd.index("-v") + 1] == DEFAULT_VOICE


# ---------- error branches ----------


def test_synthesise_missing_binary_raises() -> None:
    """Must-fail: non-macOS host → VoiceError with actionable message."""
    with mock.patch.object(ms.shutil, "which", return_value=None):
        with pytest.raises(VoiceError, match="not found"):
            MacosSayProvider().synthesise(
                text="x", voice=DEFAULT_VOICE, emotion=None
            )


def test_synthesise_nonzero_exit_raises() -> None:
    """Must-fail: unknown voice or similar → VoiceError with stderr."""
    def _fail(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout="", stderr="Voice not found"
        )

    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_fail):
        with pytest.raises(VoiceError, match="exited 1"):
            MacosSayProvider().synthesise(
                text="x", voice="NonExistent", emotion=None
            )


def test_synthesise_timeout_raises() -> None:
    """Must-fail: hang → VoiceError so ladder can fallover — even though
    macos_say is the final rung, timeout at the final rung re-raises
    to the caller (no further fallover)."""
    def _timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_timeout):
        with pytest.raises(VoiceError, match="timed out"):
            MacosSayProvider(timeout_s=0.01).synthesise(
                text="x", voice=DEFAULT_VOICE, emotion=None
            )


def test_synthesise_empty_output_raises() -> None:
    """Must-fail: subprocess exit 0 but no bytes written → VoiceError."""
    def _empty(cmd, **kwargs):
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_empty):
        with pytest.raises(VoiceError, match="empty output"):
            MacosSayProvider().synthesise(
                text="x", voice=DEFAULT_VOICE, emotion=None
            )


def test_synthesise_malformed_wav_raises() -> None:
    """Must-fail: `say` emitted AIFF or MP3 by mistake → VoiceError."""
    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(
             ms.subprocess, "run", side_effect=_writer_fake(b"NOT-A-WAV" * 20)
         ):
        with pytest.raises(VoiceError, match="RIFF/WAVE"):
            MacosSayProvider().synthesise(
                text="x", voice=DEFAULT_VOICE, emotion=None
            )


# ---------- cleanup ----------


def test_synthesise_removes_tempfile_on_success() -> None:
    wav = _make_wav()
    captured: dict[str, str] = {}

    def _capture(cmd, **kwargs):
        captured["path"] = cmd[cmd.index("-o") + 1]
        Path(captured["path"]).write_bytes(wav)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_capture):
        MacosSayProvider().synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)
    assert not Path(captured["path"]).exists()


def test_synthesise_removes_tempfile_on_failure() -> None:
    captured: dict[str, str] = {}

    def _capture_fail(cmd, **kwargs):
        captured["path"] = cmd[cmd.index("-o") + 1]
        Path(captured["path"]).write_bytes(b"partial")
        return subprocess.CompletedProcess(cmd, returncode=2, stdout="", stderr="oops")

    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_capture_fail):
        with pytest.raises(VoiceError):
            MacosSayProvider().synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)
    assert not Path(captured["path"]).exists()


# ---------- duration helpers ----------


def test_parse_wav_duration_from_header() -> None:
    """Must-pass: 22050 samples mono 16-bit at 22050Hz → 1.0s."""
    pcm = b"\x00\x00" * 22050
    wav = _make_wav(pcm_bytes=pcm, sample_rate=22050)
    assert abs(_parse_wav_duration_s(wav) - 1.0) < 1e-6


def test_parse_wav_duration_rejects_non_riff() -> None:
    """Must-fail: non-RIFF bytes → VoiceError."""
    with pytest.raises(VoiceError, match="RIFF/WAVE"):
        _parse_wav_duration_s(b"not-a-wav-at-all" * 10)


# ---------- synthesise_bytes (W2 free function) ----------


def test_synthesise_bytes_returns_wav_bytes() -> None:
    """Must-pass: happy path — returns the same bytes the fake says wrote."""
    wav = _make_wav()
    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_writer_fake(wav)):
        result = synthesise_bytes("hello", voice=DEFAULT_VOICE)
    assert result == wav


def test_synthesise_bytes_empty_voice_falls_back_to_default() -> None:
    """Must-pass: empty voice resolved to DEFAULT_VOICE in the command."""
    wav = _make_wav()
    captured: dict[str, list[str]] = {}

    def _capture(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(wav)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_capture):
        synthesise_bytes("x", voice="")
    assert captured["cmd"][captured["cmd"].index("-v") + 1] == DEFAULT_VOICE


def test_synthesise_bytes_missing_binary_raises() -> None:
    """Must-fail: no `say` → VoiceError."""
    with mock.patch.object(ms.shutil, "which", return_value=None):
        with pytest.raises(VoiceError, match="not found"):
            synthesise_bytes("hello")


def test_synthesise_bytes_nonzero_exit_raises() -> None:
    """Must-fail: say exits 1 → VoiceError."""
    def _fail(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="oops")

    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_fail):
        with pytest.raises(VoiceError, match="exited 1"):
            synthesise_bytes("hello")


def test_synthesise_bytes_timeout_raises() -> None:
    """Must-fail: subprocess hangs → VoiceError."""
    def _timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_timeout):
        with pytest.raises(VoiceError, match="timed out"):
            synthesise_bytes("hello", timeout_s=0.001)


def test_synthesise_bytes_removes_tempfile() -> None:
    """Must-pass: temp file cleaned up after successful synthesis."""
    wav = _make_wav()
    captured: dict[str, str] = {}

    def _capture(cmd, **kwargs):
        captured["path"] = cmd[cmd.index("-o") + 1]
        Path(captured["path"]).write_bytes(wav)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_capture):
        synthesise_bytes("hello")
    assert not Path(captured["path"]).exists()


# ---------- make_synthesise_fn (W2 factory) ----------


def test_make_synthesise_fn_returns_callable() -> None:
    """Must-pass: factory produces a Callable[[str], bytes]."""
    fn = make_synthesise_fn()
    assert callable(fn)


def test_make_synthesise_fn_produces_bytes() -> None:
    """Must-pass: closure returns WAV bytes via synthesise_bytes."""
    wav = _make_wav()
    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_writer_fake(wav)):
        fn = make_synthesise_fn(voice=DEFAULT_VOICE)
        result = fn("hello")
    assert result == wav


def test_make_synthesise_fn_custom_voice_propagated() -> None:
    """Must-pass: voice kwarg passed through to subprocess command."""
    wav = _make_wav()
    captured: dict[str, list[str]] = {}

    def _capture(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(wav)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with mock.patch.object(ms.shutil, "which", return_value="/usr/bin/say"), \
         mock.patch.object(ms.subprocess, "run", side_effect=_capture):
        fn = make_synthesise_fn(voice="Moira")
        fn("hello")
    assert captured["cmd"][captured["cmd"].index("-v") + 1] == "Moira"
