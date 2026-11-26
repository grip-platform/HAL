"""HAL Voice — shared RIFF/WAVE helpers.

Four providers (Piper, Google Cloud TTS, Gemini, macOS ``say``) each
need to parse or build canonical WAV bytes. Extracting the helpers
here preserves a single source of truth for byte-offset discipline
and for the 44-byte-header shape.

Scope:
- ``parse_wav_duration_s(wav_bytes)`` — playback duration from the
  fmt chunk of an existing RIFF/WAVE buffer (Piper + macos_say
  receive WAV from their subprocess; Gemini uses this for the
  ``audio/wav`` response mime branch).
- ``pcm16_duration_s(pcm_bytes, *, sample_rate, channels=1)`` —
  duration from raw LINEAR16 PCM byte count (Google Cloud TTS +
  Gemini compute this before wrapping).
- ``wrap_pcm16_as_wav(pcm_bytes, *, sample_rate, channels=1)`` —
  prepend a canonical 44-byte RIFF/WAVE header so cache + player
  see uniform WAV shape across every provider.

All helpers raise ``VoiceError`` on malformed input rather than
returning a sentinel so the caller can fallover (or halt, for
caller-error hints) without having to inspect return values.
"""
from __future__ import annotations

import struct

from ..types import VoiceError


WAV_HEADER_BYTES = 44


def parse_wav_duration_s(wav_bytes: bytes) -> float:
    """Compute playback duration from a RIFF/WAVE byte string.

    Reads ``num_channels`` / ``sample_rate`` / ``bits_per_sample``
    from the canonical fmt-chunk offsets. Assumes the 44-byte header
    layout that ``say``, Piper, and most modern WAV producers emit.
    Raises ``VoiceError`` on non-RIFF input.
    """
    if (
        len(wav_bytes) < WAV_HEADER_BYTES
        or wav_bytes[:4] != b"RIFF"
        or wav_bytes[8:12] != b"WAVE"
    ):
        raise VoiceError("output is not a RIFF/WAVE file")
    num_channels = struct.unpack_from("<H", wav_bytes, 22)[0]
    sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
    bits_per_sample = struct.unpack_from("<H", wav_bytes, 34)[0]
    bytes_per_sample = max(1, bits_per_sample // 8)
    divisor = max(1, sample_rate * num_channels * bytes_per_sample)
    pcm_bytes = len(wav_bytes) - WAV_HEADER_BYTES
    return max(0.0, pcm_bytes / divisor)


def pcm16_duration_s(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
) -> float:
    """Playback duration for raw LINEAR16 PCM.

    Used by providers that get raw PCM from the API and wrap it
    locally (Google Cloud TTS, Gemini audio in L16 mime). 16-bit =
    2 bytes per sample.
    """
    divisor = max(1, sample_rate * channels * 2)
    return len(pcm_bytes) / divisor


def wrap_pcm16_as_wav(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
) -> bytes:
    """Prefix a LINEAR16 PCM buffer with a canonical 44-byte RIFF/WAVE header.

    Matches the header shape Piper and ``say`` emit so cache + player
    treat all TTS outputs uniformly.
    """
    bits_per_sample = 16
    subchunk2_size = len(pcm_bytes)
    chunk_size = 36 + subchunk2_size
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return (
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
        + pcm_bytes
    )


__all__ = [
    "WAV_HEADER_BYTES",
    "parse_wav_duration_s",
    "pcm16_duration_s",
    "wrap_pcm16_as_wav",
]
