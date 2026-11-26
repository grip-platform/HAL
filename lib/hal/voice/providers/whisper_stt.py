"""HAL Voice — Whisper CLI STT provider.

Transcribes a WAV file using the local ``whisper`` CLI binary (OpenAI Whisper,
installed via ``brew install openai-whisper``). Zero API cost, works offline,
high accuracy on clean microphone audio.

Shape:
- Shells to ``whisper <audio> --model base --language en --output_format txt
  --output_dir <tmpdir>`` and reads the resulting ``.txt`` file.
- ``is_available()`` returns True iff the ``whisper`` binary is on PATH.
- ``confidence = 0.90`` — Whisper base achieves ~95% accuracy on clean audio;
  0.90 keeps every result above ``TIER_CONFIDENCE_FLOOR`` (0.85) while
  signalling that it is not a perfect oracle.
- ``fp16 False`` ensures CPU-safe operation; GPU acceleration is automatic
  when CUDA/MPS is available and falls back gracefully without this flag, but
  the flag prevents a torch deprecation warning on CPU-only machines.

Registered into ``PROVIDER_REGISTRY["stt"]["whisper"]`` by
``providers/__init__.py``.

Per plans/voice-first-harness-triphase.md Rev 2.3 section 4.4.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import SttTranscribeResult
from ..types import PlatformUnsupported, VoiceError

WHISPER_BIN = "whisper"
DEFAULT_MODEL = "base"
WHISPER_CONFIDENCE = 0.90
TIMEOUT_MULTIPLIER = 3.0
TIMEOUT_GRACE_S = 30.0


def is_available() -> bool:
    """True iff the ``whisper`` binary is on PATH."""
    return shutil.which(WHISPER_BIN) is not None


class WhisperSttProvider:
    """Whisper CLI STT provider — local transcription, zero cost."""

    name = "whisper"

    def transcribe(
        self,
        *,
        audio_path: Path,
        language: str = "en",
        max_duration_s: float = 60.0,
    ) -> SttTranscribeResult:
        """Transcribe ``audio_path`` via the whisper CLI.

        Args:
            audio_path: Path to a WAV file (16 kHz mono recommended).
            language: BCP-47 language tag; only the 2-char prefix is used
                (``"en-US"`` → ``"en"``).
            max_duration_s: Upper bound on the audio length; used to set a
                generous subprocess timeout (``max_duration_s * 3 + 30s``).

        Raises:
            PlatformUnsupported: ``whisper`` binary not on PATH.
            VoiceError: whisper returned non-zero, timed out, or wrote no output.
        """
        if not is_available():
            raise PlatformUnsupported(
                "whisper not found — install with: brew install openai-whisper"
            )

        lang = language[:2]  # "en-US" → "en"
        timeout = max_duration_s * TIMEOUT_MULTIPLIER + TIMEOUT_GRACE_S

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                subprocess.run(
                    [
                        WHISPER_BIN,
                        str(audio_path),
                        "--model", DEFAULT_MODEL,
                        "--language", lang,
                        "--output_format", "txt",
                        "--output_dir", tmpdir,
                        "--fp16", "False",     # CPU-safe; GPU auto-detected by whisper
                    ],
                    check=True,
                    timeout=timeout,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode("utf-8", errors="replace").strip()
                raise VoiceError(
                    f"whisper transcription failed: {stderr[:300]}"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise VoiceError(
                    f"whisper timed out after {timeout:.0f}s"
                ) from exc

            txt_path = Path(tmpdir) / (Path(audio_path).stem + ".txt")
            if not txt_path.exists():
                raise VoiceError(
                    f"whisper produced no .txt output in {tmpdir}"
                )
            text = txt_path.read_text(encoding="utf-8").strip()

        return SttTranscribeResult(
            text=text,
            confidence=WHISPER_CONFIDENCE,
            language=lang,
            duration_s=max_duration_s,
            provider_name=self.name,
        )


__all__ = ["WhisperSttProvider", "is_available"]
