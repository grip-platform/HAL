"""HAL Voice — TTS / STT provider Protocol surface.

Per plans/voice-first-harness-triphase.md Rev 2.3 section 4.4.

The Protocols here define the narrowest contract a provider must
satisfy to plug into the voice harness. They're runtime_checkable so
the boot-time registry can ``isinstance``-validate each entry before
handing it to ``voice.speak`` / ``voice.listen`` — an adversary
dropping an incomplete stub in the import path gets rejected mechanically
instead of silently serving wrong audio.

Result types here are the RAW provider outputs: audio bytes or
transcribed text plus quality signals. The outer API (``ListenResult``,
``SpeakResult`` in ``hal.voice.types``) wraps these with tier, audit_id,
and Provenance — fields a provider cannot forge because they're
injected by the voice primitive after audit.py has signed them.

Shipped at PR alpha-2 W4. Concrete provider classes
(MacosDictationProvider, MacosSayProvider, GeminiTtsProvider,
ElevenLabsProvider) land in alpha-3 where tier_gate and capability
tokens land alongside.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


# ---------- raw provider results ----------


@dataclass(frozen=True)
class TtsSynthResult:
    """Raw TTS provider output.

    ``emotion_match`` is the provider's self-reported quality score
    for how well the synthesised audio matches the requested emotion.
    Below ``CACHE_EMOTION_THRESHOLD`` (0.8 per ``hal.voice.cache``)
    the cache silently rejects the entry so low-quality synth doesn't
    replay on subsequent hits.
    """

    audio: bytes
    duration_s: float
    voice: str
    provider_name: str
    emotion_match: float


@dataclass(frozen=True)
class SttTranscribeResult:
    """Raw STT provider output.

    ``confidence`` gates tier-2+ actions in the outer voice primitive:
    results below 0.85 are downgraded to tier-1 before the caller sees
    them. The provider must report a realistic score — the tier_gate
    layer does NOT second-guess a confidently-wrong transcription.
    """

    text: str
    confidence: float
    language: str
    duration_s: float
    provider_name: str


# ---------- provider Protocols ----------


@runtime_checkable
class TtsProvider(Protocol):
    """Text-to-speech provider contract.

    Implementations must expose a ``name`` attribute (used as the
    PROVIDER_REGISTRY key) and a ``synthesise`` method that produces
    raw audio bytes plus quality metadata. Providers are free to
    refuse a request (e.g., paid providers raising on missing keys);
    the voice primitive handles fallback via the provider ladder.
    """

    name: str

    def synthesise(
        self,
        *,
        text: str,
        voice: str,
        emotion: str | None,
    ) -> TtsSynthResult:
        ...


@runtime_checkable
class SttProvider(Protocol):
    """Speech-to-text provider contract.

    ``max_duration_s`` is a HARD clamp: the provider MUST return
    within the clamp or raise; callers rely on it as a fail-fast for
    hung microphones. The provider is also responsible for freeing
    any audio buffer / file it opened before returning.
    """

    name: str

    def transcribe(
        self,
        *,
        audio_path: Path,
        language: str,
        max_duration_s: float,
    ) -> SttTranscribeResult:
        ...


__all__ = [
    "SttProvider",
    "SttTranscribeResult",
    "TtsProvider",
    "TtsSynthResult",
]
