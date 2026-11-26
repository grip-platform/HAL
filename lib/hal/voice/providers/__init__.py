"""HAL Voice provider registry.

Per plans/voice-first-harness-triphase.md Rev 2.3 section 4.4 and
plans/voice-tier3-cloud-fallover.md §4 W4.

``PROVIDER_REGISTRY`` is an EXPLICIT dict-of-dicts populated by
concrete-provider modules at import time. It is NOT auto-populated by
filesystem scanning — that would let an adversary drop a malicious
provider into the import path and have it silently activate. Each
registration below is a deliberate, auditable assignment.

The structure is:

    PROVIDER_REGISTRY = {
        "tts": {"piper": PiperProvider, "gemini_tts_free": ..., ...},
        "stt": {"macos_dictation": ...},
    }

Operator audit is a one-liner:

    from lib.hal.voice.providers import PROVIDER_REGISTRY
    print(sorted(PROVIDER_REGISTRY["tts"].keys()))
    print(sorted(PROVIDER_REGISTRY["stt"].keys()))

At alpha-3e W4 the TTS sub-dict is populated with the three Tier-3
providers (``piper``, ``google_cloud_tts_free``, ``gemini_tts_free``).
``macos_say`` and ``macos_dictation`` remain unregistered until their
shipping commits land; ``FALLOVER_LADDER`` tolerates absent names by
skipping them with a warning.
"""
from __future__ import annotations

from typing import TypedDict

from .base import (
    SttProvider,
    SttTranscribeResult,
    TtsProvider,
    TtsSynthResult,
)


class ProviderRegistry(TypedDict):
    """Shape of ``PROVIDER_REGISTRY`` — TTS + STT sub-registries."""

    tts: dict[str, type[TtsProvider]]
    stt: dict[str, type[SttProvider]]


PROVIDER_REGISTRY: ProviderRegistry = {
    "tts": {},
    "stt": {},
}


# ---------- concrete provider registrations (W4) ----------
#
# Imports are deliberately AFTER the PROVIDER_REGISTRY definition so the
# modules can be imported without side-effects on circular paths. Each
# assignment is explicit — no decorator-based auto-register, no import
# magic. An adversary who drops a module into the import path but is NOT
# listed here stays invisible to dispatch.

from .gemini_tts import GeminiTtsProvider  # noqa: E402
from .google_cloud_tts import GoogleCloudTtsProvider  # noqa: E402
from .macos_say import MacosSayProvider  # noqa: E402
from .piper import PiperProvider  # noqa: E402
from .whisper_stt import WhisperSttProvider  # noqa: E402


PROVIDER_REGISTRY["tts"]["piper"] = PiperProvider
PROVIDER_REGISTRY["tts"]["google_cloud_tts_free"] = GoogleCloudTtsProvider
PROVIDER_REGISTRY["tts"]["gemini_tts_free"] = GeminiTtsProvider
PROVIDER_REGISTRY["tts"]["macos_say"] = MacosSayProvider

PROVIDER_REGISTRY["stt"]["whisper"] = WhisperSttProvider


__all__ = [
    "GeminiTtsProvider",
    "GoogleCloudTtsProvider",
    "MacosSayProvider",
    "PROVIDER_REGISTRY",
    "PiperProvider",
    "ProviderRegistry",
    "SttProvider",
    "SttTranscribeResult",
    "TtsProvider",
    "TtsSynthResult",
    "WhisperSttProvider",
]
