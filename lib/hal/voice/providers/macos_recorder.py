"""HAL Voice — macOS ffmpeg AVFoundation mic recorder.

Records from the default system microphone via ffmpeg's avfoundation driver.
Output is a 16 kHz mono WAV file — the format Whisper and all HAL STT
providers expect, and the same format macOS ``say`` and the TTS providers
produce so the cache and player see uniform bytes across the stack.

Requires: ffmpeg (brew install ffmpeg).
Permissions: Terminal must have Microphone access in
    System Preferences → Privacy & Security → Microphone.

Registered as the default recorder for macOS in
``lib.hal.voice.voice._default_recorder``. On non-Darwin platforms
``_default_recorder`` raises ``PlatformUnsupported`` before reaching
this module, so the import is guarded.

Per plans/voice-first-harness-triphase.md Rev 2.3 section 4.5 (alpha-3
recorder contract): caller injects the recorder into ``listen()`` and
``chat_loop()``, so tests can swap it for a fixture without touching the
real mic.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..types import PlatformUnsupported, VoiceError

FFMPEG_BIN = "ffmpeg"
SAMPLE_RATE = 16_000
CHANNELS = 1
TIMEOUT_GRACE_S = 10.0


def is_available() -> bool:
    """True iff ``ffmpeg`` is on PATH."""
    return shutil.which(FFMPEG_BIN) is not None


def record(max_duration_s: float, out_path: Path) -> None:
    """Record from the default mic to ``out_path`` via ffmpeg avfoundation.

    Blocks for exactly ``max_duration_s`` seconds (ffmpeg ``-t`` flag hard-
    clips the capture). The output file at ``out_path`` is a 16 kHz mono
    WAV on return; callers must not read it before this call returns.

    Raises:
        PlatformUnsupported: ``ffmpeg`` not on PATH.
        VoiceError: ffmpeg returned non-zero or timed out.
    """
    if not is_available():
        raise PlatformUnsupported(
            "ffmpeg not found — install with: brew install ffmpeg"
        )
    try:
        subprocess.run(
            [
                FFMPEG_BIN,
                "-y",                          # overwrite output without prompting
                "-f", "avfoundation",          # macOS AVFoundation capture backend
                "-i", ":0",                    # ":0" = default audio input, no video
                "-t", str(max_duration_s),     # hard clip at max_duration_s
                "-ar", str(SAMPLE_RATE),       # 16 kHz — Whisper / STT standard
                "-ac", str(CHANNELS),          # mono
                str(out_path),
            ],
            check=True,
            timeout=max_duration_s + TIMEOUT_GRACE_S,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise VoiceError(
            f"ffmpeg recording failed: {stderr[:300]}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise VoiceError(
            f"ffmpeg recording timed out after {max_duration_s + TIMEOUT_GRACE_S:.0f}s"
        ) from exc


__all__ = ["is_available", "record"]
