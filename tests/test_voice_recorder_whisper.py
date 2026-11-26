"""Tests for the ffmpeg recorder and Whisper STT provider.

Unit tests that never touch the real mic or whisper binary — all subprocess
calls are injected via the ``recorder`` / ``spawn`` seams the voice harness
explicitly provides for this purpose.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lib.hal.voice.providers.macos_recorder import is_available as ffmpeg_available
from lib.hal.voice.providers.macos_recorder import record as ffmpeg_record
from lib.hal.voice.providers.whisper_stt import WhisperSttProvider, is_available as whisper_available
from lib.hal.voice.providers import PROVIDER_REGISTRY
from lib.hal.voice.types import PlatformUnsupported, VoiceError


# ── ffmpeg recorder ──────────────────────────────────────────────────────────


class TestFfmpegRecorder:
    def test_is_available_delegates_to_shutil_which(self):
        with patch("lib.hal.voice.providers.macos_recorder.shutil.which", return_value="/usr/bin/ffmpeg"):
            assert ffmpeg_available() is True

    def test_is_available_false_when_missing(self):
        with patch("lib.hal.voice.providers.macos_recorder.shutil.which", return_value=None):
            assert ffmpeg_available() is False

    def test_record_raises_platform_unsupported_when_ffmpeg_missing(self, tmp_path):
        out = tmp_path / "out.wav"
        with patch("lib.hal.voice.providers.macos_recorder.shutil.which", return_value=None):
            with pytest.raises(PlatformUnsupported, match="ffmpeg not found"):
                ffmpeg_record(5.0, out)

    def test_record_calls_ffmpeg_with_correct_args(self, tmp_path):
        out = tmp_path / "rec.wav"
        with patch("lib.hal.voice.providers.macos_recorder.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("lib.hal.voice.providers.macos_recorder.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                ffmpeg_record(10.0, out)

        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "ffmpeg"
        assert "-f" in call_args and "avfoundation" in call_args
        assert "-i" in call_args and ":0" in call_args
        assert "-t" in call_args and "10.0" in call_args
        assert "-ar" in call_args and "16000" in call_args
        assert "-ac" in call_args and "1" in call_args
        assert str(out) in call_args

    def test_record_raises_voice_error_on_ffmpeg_failure(self, tmp_path):
        out = tmp_path / "rec.wav"
        with patch("lib.hal.voice.providers.macos_recorder.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch(
                "lib.hal.voice.providers.macos_recorder.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr=b"input error"),
            ):
                with pytest.raises(VoiceError, match="ffmpeg recording failed"):
                    ffmpeg_record(5.0, out)

    def test_record_raises_voice_error_on_timeout(self, tmp_path):
        out = tmp_path / "rec.wav"
        with patch("lib.hal.voice.providers.macos_recorder.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch(
                "lib.hal.voice.providers.macos_recorder.subprocess.run",
                side_effect=subprocess.TimeoutExpired("ffmpeg", 15.0),
            ):
                with pytest.raises(VoiceError, match="timed out"):
                    ffmpeg_record(5.0, out)


# ── Whisper STT provider ──────────────────────────────────────────────────────


class TestWhisperSttProvider:
    def _make_provider(self) -> WhisperSttProvider:
        return WhisperSttProvider()

    def test_name(self):
        assert self._make_provider().name == "whisper"

    def test_is_available_delegates_to_shutil_which(self):
        with patch("lib.hal.voice.providers.whisper_stt.shutil.which", return_value="/usr/bin/whisper"):
            assert whisper_available() is True

    def test_is_available_false_when_missing(self):
        with patch("lib.hal.voice.providers.whisper_stt.shutil.which", return_value=None):
            assert whisper_available() is False

    def test_transcribe_raises_when_whisper_missing(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")
        prov = self._make_provider()
        with patch("lib.hal.voice.providers.whisper_stt.shutil.which", return_value=None):
            with pytest.raises(PlatformUnsupported, match="whisper not found"):
                prov.transcribe(audio_path=audio, language="en", max_duration_s=5.0)

    def test_transcribe_returns_text_from_txt_file(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")
        expected_text = "Hello, this is a test transcription."
        prov = self._make_provider()

        def _fake_run(cmd, *, check, timeout, capture_output):
            # Simulate whisper writing audio.txt to --output_dir
            for i, arg in enumerate(cmd):
                if arg == "--output_dir":
                    out_dir = Path(cmd[i + 1])
                    (out_dir / "audio.txt").write_text(expected_text)
                    break
            return MagicMock(returncode=0)

        with patch("lib.hal.voice.providers.whisper_stt.shutil.which", return_value="/usr/bin/whisper"):
            with patch("lib.hal.voice.providers.whisper_stt.subprocess.run", side_effect=_fake_run):
                result = prov.transcribe(audio_path=audio, language="en", max_duration_s=5.0)

        assert result.text == expected_text
        assert result.provider_name == "whisper"
        assert result.confidence == 0.90
        assert result.language == "en"

    def test_transcribe_truncates_language_tag(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")
        prov = self._make_provider()
        captured = {}

        def _fake_run(cmd, *, check, timeout, capture_output):
            for i, arg in enumerate(cmd):
                if arg == "--language":
                    captured["lang"] = cmd[i + 1]
                if arg == "--output_dir":
                    (Path(cmd[i + 1]) / "audio.txt").write_text("ok")
            return MagicMock(returncode=0)

        with patch("lib.hal.voice.providers.whisper_stt.shutil.which", return_value="/usr/bin/whisper"):
            with patch("lib.hal.voice.providers.whisper_stt.subprocess.run", side_effect=_fake_run):
                prov.transcribe(audio_path=audio, language="en-US", max_duration_s=5.0)

        assert captured["lang"] == "en"  # "en-US" → "en"

    def test_transcribe_raises_voice_error_on_nonzero(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")
        prov = self._make_provider()
        with patch("lib.hal.voice.providers.whisper_stt.shutil.which", return_value="/usr/bin/whisper"):
            with patch(
                "lib.hal.voice.providers.whisper_stt.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "whisper", stderr=b"model error"),
            ):
                with pytest.raises(VoiceError, match="whisper transcription failed"):
                    prov.transcribe(audio_path=audio, language="en", max_duration_s=5.0)

    def test_transcribe_raises_voice_error_on_timeout(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")
        prov = self._make_provider()
        with patch("lib.hal.voice.providers.whisper_stt.shutil.which", return_value="/usr/bin/whisper"):
            with patch(
                "lib.hal.voice.providers.whisper_stt.subprocess.run",
                side_effect=subprocess.TimeoutExpired("whisper", 60.0),
            ):
                with pytest.raises(VoiceError, match="timed out"):
                    prov.transcribe(audio_path=audio, language="en", max_duration_s=5.0)

    def test_transcribe_raises_when_no_txt_output(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")
        prov = self._make_provider()
        with patch("lib.hal.voice.providers.whisper_stt.shutil.which", return_value="/usr/bin/whisper"):
            with patch(
                "lib.hal.voice.providers.whisper_stt.subprocess.run",
                return_value=MagicMock(returncode=0),
            ):
                with pytest.raises(VoiceError, match="no .txt output"):
                    prov.transcribe(audio_path=audio, language="en", max_duration_s=5.0)


# ── Registry integration ──────────────────────────────────────────────────────


class TestProviderRegistry:
    def test_whisper_registered_in_stt(self):
        assert "whisper" in PROVIDER_REGISTRY["stt"]
        assert PROVIDER_REGISTRY["stt"]["whisper"] is WhisperSttProvider

    def test_stt_registry_not_empty(self):
        assert len(PROVIDER_REGISTRY["stt"]) >= 1


# ── Ladder ────────────────────────────────────────────────────────────────────


class TestFreeLadderStt:
    def test_free_ladder_stt_includes_whisper(self):
        from lib.hal.voice.ladder import FREE_LADDER
        assert "whisper" in FREE_LADDER["stt"]

    def test_whisper_is_first_stt_rung(self):
        from lib.hal.voice.ladder import FREE_LADDER
        assert FREE_LADDER["stt"][0] == "whisper"
