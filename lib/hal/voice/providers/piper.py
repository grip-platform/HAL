"""HAL Voice — Piper local neural TTS provider (Tier 3 W1).

Per plans/voice-tier3-cloud-fallover.md §4 W1.

Piper is a local ONNX-based neural TTS engine — unlimited use, $0 cost,
offline-capable. Lives on the FALLOVER_LADDER between cloud providers
and macos_say: when Gemini + Google Cloud TTS are unreachable or
quota-exhausted, Piper keeps speech working without network.

Shape:
- Shells to ``piper --model <path> --output_file <tmp>`` with text on stdin
- Voice models: ``~/.hal/models/piper/<voice>.onnx`` (configurable via env
  ``HAL_PIPER_MODEL_DIR``)
- Default voice: ``en_GB-alan-medium`` — fits GRIP tone; operator can swap
- ``is_available()`` returns False when the piper binary OR the named model
  is missing. The ladder dispatch layer uses this as a clean skip signal
  rather than an error — unavailability is a normal state on machines
  where Piper has not been installed yet.
- ``emotion_match = 0.85`` is above the cache's 0.8 floor (so Piper output
  is persisted) but below the 0.95 cloud-neural tier (so cloud wins when
  online and within quota).

``emotion`` kwarg is accepted for Protocol conformance but ignored: Piper
has no emotion-conditioning channel. The DirectiveResolver at α-3c is the
primary emotion gate; this provider is intentionally emotion-neutral.

Falsifiable hypothesis H-VO-P1: cold-start synthesis latency p50 ≤ 500ms
for 50-char input on M-series Macs. Deadline 2026-05-10.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..types import VoiceError
from ._wav_utils import parse_wav_duration_s as _parse_wav_duration_s
from .base import TtsSynthResult


PIPER_BIN = "piper"
DEFAULT_VOICE = "en_GB-alan-medium"
DEFAULT_MODEL_DIR = Path.home() / ".hal" / "models" / "piper"
MODEL_FILE_SUFFIX = ".onnx"
PIPER_EMOTION_MATCH = 0.85
SUBPROCESS_TIMEOUT_S = 30.0


def model_dir() -> Path:
    """Resolve the Piper model directory at call time.

    Honours ``HAL_PIPER_MODEL_DIR`` if set, else ``~/.hal/models/piper``.
    Lazy resolution lets tests monkeypatch the env var without reimporting
    the module.
    """
    override = os.environ.get("HAL_PIPER_MODEL_DIR")
    return Path(override) if override else DEFAULT_MODEL_DIR


def model_path(voice: str = DEFAULT_VOICE) -> Path:
    """Absolute path for a given voice's ONNX model file."""
    return model_dir() / f"{voice}{MODEL_FILE_SUFFIX}"


def is_available(voice: str = DEFAULT_VOICE) -> bool:
    """True iff the piper binary AND the named voice model are on disk.

    The ladder dispatch layer calls this at resolution time. When False,
    the next provider in the ladder is tried and no exception is raised —
    unavailability is a normal, non-error state.
    """
    if shutil.which(PIPER_BIN) is None:
        return False
    return model_path(voice).is_file()


# ``_parse_wav_duration_s`` is imported above from ``._wav_utils`` so
# piper-local tests that used the private name continue to resolve.
# The canonical implementation now lives in one place for all four
# WAV-producing providers.


class PiperProvider:
    """Local Piper ONNX neural TTS provider.

    Satisfies the ``TtsProvider`` Protocol. Does not inherit — the Protocol
    is ``@runtime_checkable`` so the registry's isinstance check validates
    shape at registration time, which catches incomplete stubs at boot
    rather than at first use.
    """

    name = "piper"

    def __init__(
        self,
        *,
        model_dir_override: Path | None = None,
        timeout_s: float = SUBPROCESS_TIMEOUT_S,
    ) -> None:
        self._model_dir_override = model_dir_override
        self._timeout_s = timeout_s

    def _resolve_model(self, voice: str) -> Path:
        root = self._model_dir_override or model_dir()
        return root / f"{voice}{MODEL_FILE_SUFFIX}"

    def synthesise(
        self,
        *,
        text: str,
        voice: str,
        emotion: str | None,
    ) -> TtsSynthResult:
        del emotion  # accepted for Protocol conformance; Piper has no channel
        resolved_voice = voice or DEFAULT_VOICE
        model = self._resolve_model(resolved_voice)
        if not model.is_file():
            raise VoiceError(
                f"piper: model file missing: {model} "
                "(set HAL_PIPER_MODEL_DIR or install the voice)"
            )
        if shutil.which(PIPER_BIN) is None:
            raise VoiceError(
                "piper: binary not found on PATH "
                "(install via `brew install piper-tts` or equivalent)"
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            out_path = Path(tf.name)
        try:
            try:
                result = subprocess.run(
                    [
                        PIPER_BIN,
                        "--model", str(model),
                        "--output_file", str(out_path),
                    ],
                    input=text,
                    text=True,
                    capture_output=True,
                    timeout=self._timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise VoiceError(
                    f"piper: synthesis timed out after {self._timeout_s}s"
                ) from exc
            if result.returncode != 0:
                raise VoiceError(
                    f"piper: subprocess exited {result.returncode}: "
                    f"{result.stderr.strip()}"
                )
            if not out_path.is_file() or out_path.stat().st_size == 0:
                raise VoiceError("piper: produced empty output file")
            audio = out_path.read_bytes()
        finally:
            out_path.unlink(missing_ok=True)

        duration_s = _parse_wav_duration_s(audio)
        return TtsSynthResult(
            audio=audio,
            duration_s=duration_s,
            voice=resolved_voice,
            provider_name=self.name,
            emotion_match=PIPER_EMOTION_MATCH,
        )


__all__ = [
    "DEFAULT_MODEL_DIR",
    "DEFAULT_VOICE",
    "MODEL_FILE_SUFFIX",
    "PIPER_BIN",
    "PIPER_EMOTION_MATCH",
    "PiperProvider",
    "SUBPROCESS_TIMEOUT_S",
    "is_available",
    "model_dir",
    "model_path",
]
