"""Two-sided Goodhart-proof tests for lib.hal.voice.providers.piper.

Per Rule 14 each invariant has a must-pass branch (the correct shape is
accepted / the correct output is produced) AND a must-fail branch (a
deliberately broken precondition is rejected). Without the must-fail
tests a regression that removed the availability check or swallowed a
timeout would silently pass.

Mocking strategy: ``subprocess.run`` is patched by a fake that writes a
minimal valid WAV to the ``--output_file`` argument. No real piper binary
is invoked in CI. For the "binary missing" tests the patch makes
``shutil.which`` return None; for the "model missing" tests the model path
is simply absent from the temp dir.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.voice.providers import piper as piper_mod  # noqa: E402
from lib.hal.voice.providers.base import TtsSynthResult  # noqa: E402
from lib.hal.voice.providers.piper import (  # noqa: E402
    DEFAULT_VOICE,
    MODEL_FILE_SUFFIX,
    PIPER_EMOTION_MATCH,
    PiperProvider,
    _parse_wav_duration_s,
    is_available,
    model_dir,
    model_path,
)
from lib.hal.voice.providers import SttProvider, TtsProvider  # noqa: E402
from lib.hal.voice.types import VoiceError  # noqa: E402


# ---------- helpers ----------


def _make_wav(
    pcm_bytes: bytes = b"\x00\x00" * 1024,
    sample_rate: int = 22050,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Build a minimal canonical 44-byte-header RIFF/WAVE byte string."""
    subchunk2_size = len(pcm_bytes)
    chunk_size = 36 + subchunk2_size
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = (
        b"RIFF"
        + struct.pack("<I", chunk_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack(
            "<IHHIIHH",
            16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample,
        )
        + b"data"
        + struct.pack("<I", subchunk2_size)
    )
    return header + pcm_bytes


def _fake_run_writer(wav_bytes: bytes):
    """Return a fake subprocess.run that writes wav_bytes to --output_file."""

    def _fake(cmd, **kwargs):
        idx = cmd.index("--output_file")
        Path(cmd[idx + 1]).write_bytes(wav_bytes)
        return subprocess.CompletedProcess(
            cmd, returncode=0, stdout="", stderr=""
        )

    return _fake


@pytest.fixture
def model_root(tmp_path: Path) -> Path:
    """Isolated model directory with DEFAULT_VOICE present."""
    root = tmp_path / "models"
    root.mkdir()
    (root / f"{DEFAULT_VOICE}{MODEL_FILE_SUFFIX}").write_bytes(b"onnx-stub")
    return root


# ---------- Protocol conformance ----------


def test_piper_provider_satisfies_tts_protocol() -> None:
    """Must-pass: PiperProvider duck-types to TtsProvider (runtime_checkable)."""
    assert isinstance(PiperProvider(), TtsProvider)


def test_piper_provider_is_not_an_stt_provider() -> None:
    """Must-fail: PiperProvider has no transcribe() — slot-safety ensures it
    cannot be wired into the STT registry by accident."""
    assert not isinstance(PiperProvider(), SttProvider)


def test_piper_provider_name_is_piper() -> None:
    """Must-pass: registry key discipline — ``name`` must be the literal
    string ``piper`` because voices.yaml / the ladder look it up by that key."""
    assert PiperProvider().name == "piper"


# ---------- is_available ----------


def test_is_available_false_when_binary_missing(model_root: Path) -> None:
    """Must-fail (ready-state): ladder must skip cleanly when the piper
    binary is not installed."""
    with mock.patch.object(piper_mod.shutil, "which", return_value=None):
        assert is_available() is False


def test_is_available_false_when_model_missing(tmp_path: Path) -> None:
    """Must-fail (ready-state): binary alone is not sufficient; the named
    voice model must also be on disk."""
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ), mock.patch.dict(
        os.environ, {"HAL_PIPER_MODEL_DIR": str(empty_root)}
    ):
        assert is_available() is False


def test_is_available_true_when_both_present(model_root: Path) -> None:
    """Must-pass: with binary on PATH and model on disk, ladder keeps piper."""
    with mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ), mock.patch.dict(
        os.environ, {"HAL_PIPER_MODEL_DIR": str(model_root)}
    ):
        assert is_available() is True


def test_is_available_per_voice_is_scoped(model_root: Path) -> None:
    """Must-fail for the unknown voice, must-pass for the known one —
    availability is per-voice, not global."""
    with mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ), mock.patch.dict(
        os.environ, {"HAL_PIPER_MODEL_DIR": str(model_root)}
    ):
        assert is_available(DEFAULT_VOICE) is True
        assert is_available("not-a-real-voice") is False


# ---------- model_dir / model_path ----------


def test_model_dir_honours_env_override(tmp_path: Path) -> None:
    """Must-pass: HAL_PIPER_MODEL_DIR wins over the ~/.hal default."""
    override = tmp_path / "custom"
    with mock.patch.dict(os.environ, {"HAL_PIPER_MODEL_DIR": str(override)}):
        assert model_dir() == override


def test_model_dir_falls_back_to_default(tmp_path: Path) -> None:
    """Must-fail (env absence): no override → falls back to ~/.hal/..."""
    env = dict(os.environ)
    env.pop("HAL_PIPER_MODEL_DIR", None)
    with mock.patch.dict(os.environ, env, clear=True):
        assert model_dir() == piper_mod.DEFAULT_MODEL_DIR


def test_model_path_composes_voice_name(tmp_path: Path) -> None:
    """Must-pass: <dir>/<voice>.onnx layout — no lookup table needed."""
    with mock.patch.dict(os.environ, {"HAL_PIPER_MODEL_DIR": str(tmp_path)}):
        assert model_path("en_GB-alan-medium") == tmp_path / "en_GB-alan-medium.onnx"


# ---------- synthesise happy path ----------


def test_synthesise_returns_tts_synth_result(model_root: Path) -> None:
    """Must-pass: happy-path call produces the contract-required dataclass."""
    wav = _make_wav()
    provider = PiperProvider(model_dir_override=model_root)
    with mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ), mock.patch.object(
        piper_mod.subprocess, "run", side_effect=_fake_run_writer(wav)
    ):
        result = provider.synthesise(text="hello", voice=DEFAULT_VOICE, emotion=None)
    assert isinstance(result, TtsSynthResult)
    assert result.audio == wav
    assert result.voice == DEFAULT_VOICE
    assert result.provider_name == "piper"


def test_synthesise_emotion_match_is_the_piper_constant(model_root: Path) -> None:
    """Must-pass: exact 0.85 — above cache threshold (0.8) and below cloud
    (0.95). A drift here would break ladder ordering."""
    wav = _make_wav()
    provider = PiperProvider(model_dir_override=model_root)
    with mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ), mock.patch.object(
        piper_mod.subprocess, "run", side_effect=_fake_run_writer(wav)
    ):
        r = provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion="calm")
    assert r.emotion_match == PIPER_EMOTION_MATCH == 0.85


def test_synthesise_ignores_emotion_value(model_root: Path) -> None:
    """Must-pass: Piper has no emotion channel — the same emotion_match is
    reported regardless of the requested emotion."""
    wav = _make_wav()
    provider = PiperProvider(model_dir_override=model_root)
    with mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ), mock.patch.object(
        piper_mod.subprocess, "run", side_effect=_fake_run_writer(wav)
    ):
        calm = provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion="calm")
        angry = provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion="angry")
        none = provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)
    assert calm.emotion_match == angry.emotion_match == none.emotion_match


def test_synthesise_empty_voice_falls_back_to_default(model_root: Path) -> None:
    """Must-pass: a caller passing ``voice=""`` still lands on DEFAULT_VOICE
    instead of failing — ladder callers don't always supply explicit voices."""
    wav = _make_wav()
    provider = PiperProvider(model_dir_override=model_root)
    with mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ), mock.patch.object(
        piper_mod.subprocess, "run", side_effect=_fake_run_writer(wav)
    ):
        r = provider.synthesise(text="x", voice="", emotion=None)
    assert r.voice == DEFAULT_VOICE


# ---------- error branches ----------


def test_synthesise_missing_model_raises(tmp_path: Path) -> None:
    """Must-fail: no model on disk → VoiceError (provider cannot silently
    synthesise nothing)."""
    empty = tmp_path / "empty"
    empty.mkdir()
    provider = PiperProvider(model_dir_override=empty)
    with mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ):
        with pytest.raises(VoiceError, match="model file missing"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_synthesise_missing_binary_raises(model_root: Path) -> None:
    """Must-fail: no piper on PATH → VoiceError before any subprocess call."""
    provider = PiperProvider(model_dir_override=model_root)
    with mock.patch.object(piper_mod.shutil, "which", return_value=None):
        with pytest.raises(VoiceError, match="binary not found"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_synthesise_nonzero_exit_raises(model_root: Path) -> None:
    """Must-fail: piper exit code != 0 surfaces as VoiceError with stderr."""
    def _fail_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout="", stderr="piper: cannot load model"
        )

    provider = PiperProvider(model_dir_override=model_root)
    with mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ), mock.patch.object(piper_mod.subprocess, "run", side_effect=_fail_run):
        with pytest.raises(VoiceError, match="exited 1"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_synthesise_timeout_raises(model_root: Path) -> None:
    """Must-fail: subprocess timeout propagates as VoiceError — caller relies
    on this to trigger ladder fallover before a UI hang."""
    def _timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    provider = PiperProvider(model_dir_override=model_root, timeout_s=0.01)
    with mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ), mock.patch.object(piper_mod.subprocess, "run", side_effect=_timeout):
        with pytest.raises(VoiceError, match="timed out"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_synthesise_empty_output_raises(model_root: Path) -> None:
    """Must-fail: subprocess exit 0 but no bytes written → VoiceError.
    Prevents silent success returning empty audio to the speaker."""
    def _empty(cmd, **kwargs):
        idx = cmd.index("--output_file")
        Path(cmd[idx + 1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    provider = PiperProvider(model_dir_override=model_root)
    with mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ), mock.patch.object(piper_mod.subprocess, "run", side_effect=_empty):
        with pytest.raises(VoiceError, match="empty output"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_synthesise_malformed_wav_raises(model_root: Path) -> None:
    """Must-fail: non-RIFF output → VoiceError from the WAV parser.
    Catches a future provider bug where the wrong binary was invoked."""
    provider = PiperProvider(model_dir_override=model_root)
    with mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ), mock.patch.object(
        piper_mod.subprocess,
        "run",
        side_effect=_fake_run_writer(b"NOT-A-WAV" * 10),
    ):
        with pytest.raises(VoiceError, match="RIFF/WAVE"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


# ---------- duration parsing ----------


def test_duration_s_computed_from_wav_header() -> None:
    """Must-pass: 1 second of 22050 Hz mono 16-bit PCM → duration ≈ 1.0s."""
    pcm = b"\x00\x00" * 22050  # 22050 samples × 2 bytes = 1 second
    wav = _make_wav(pcm_bytes=pcm, sample_rate=22050, channels=1, bits_per_sample=16)
    assert abs(_parse_wav_duration_s(wav) - 1.0) < 0.001


def test_duration_s_scales_with_sample_rate() -> None:
    """Must-pass: same byte-count at half the sample rate → doubled duration."""
    pcm = b"\x00\x00" * 16000  # 1s at 16000Hz, but parse as 8000Hz → 2s
    wav = _make_wav(pcm_bytes=pcm, sample_rate=8000, channels=1, bits_per_sample=16)
    assert abs(_parse_wav_duration_s(wav) - 2.0) < 0.001


def test_duration_s_zero_for_header_only_wav() -> None:
    """Must-pass edge: 44-byte header with no PCM data → 0.0s playback."""
    empty_wav = _make_wav(pcm_bytes=b"")
    assert _parse_wav_duration_s(empty_wav) == 0.0


# ---------- cleanup ----------


def test_synthesise_removes_tempfile_on_success(model_root: Path) -> None:
    """Must-pass: tmpfile is unlinked after a successful synthesis so the
    tmpfs doesn't accumulate WAVs across many calls."""
    wav = _make_wav()
    captured: dict[str, str] = {}

    def _capture_and_write(cmd, **kwargs):
        idx = cmd.index("--output_file")
        captured["path"] = cmd[idx + 1]
        Path(captured["path"]).write_bytes(wav)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    provider = PiperProvider(model_dir_override=model_root)
    with mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ), mock.patch.object(piper_mod.subprocess, "run", side_effect=_capture_and_write):
        provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)
    assert captured["path"] != ""
    assert not Path(captured["path"]).exists()


def test_synthesise_removes_tempfile_on_failure(model_root: Path) -> None:
    """Must-pass: tmpfile is unlinked even when subprocess fails — no leak
    on the error path."""
    captured: dict[str, str] = {}

    def _capture_and_fail(cmd, **kwargs):
        idx = cmd.index("--output_file")
        captured["path"] = cmd[idx + 1]
        Path(captured["path"]).write_bytes(b"partial")
        return subprocess.CompletedProcess(cmd, returncode=2, stdout="", stderr="oops")

    provider = PiperProvider(model_dir_override=model_root)
    with mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ), mock.patch.object(piper_mod.subprocess, "run", side_effect=_capture_and_fail):
        with pytest.raises(VoiceError):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)
    assert captured["path"] != ""
    assert not Path(captured["path"]).exists()


# ---------- config override ----------


def test_model_dir_override_beats_env(tmp_path: Path, model_root: Path) -> None:
    """Must-pass: explicit constructor override wins over the env var —
    lets tests pin behaviour without touching os.environ."""
    env_dir = tmp_path / "env"
    env_dir.mkdir()  # deliberately missing the model file
    wav = _make_wav()
    provider = PiperProvider(model_dir_override=model_root)
    with mock.patch.dict(
        os.environ, {"HAL_PIPER_MODEL_DIR": str(env_dir)}
    ), mock.patch.object(
        piper_mod.shutil, "which", return_value="/usr/local/bin/piper"
    ), mock.patch.object(
        piper_mod.subprocess, "run", side_effect=_fake_run_writer(wav)
    ):
        r = provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)
    assert r.voice == DEFAULT_VOICE
