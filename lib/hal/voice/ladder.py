"""HAL Voice — provider fallback ladders.

Per plans/voice-first-harness-triphase.md Rev 2.3 section 4.6.

A ladder is an ordered list of provider names per (tts / stt) slot.
When a voice primitive invokes a ladder'd dispatch, it tries the
providers in order and falls through to the next only when the
current one raises. This lets a profile prefer a paid cloud
provider while still having an offline macOS fallback.

Two ladders ship here:

- ``FREE_LADDER`` — macOS-only built-ins. Zero API-cost for every
  call. The default for profiles without an explicit ladder.
- ``PREMIUM_LADDER`` — paid providers first, free fallback last.
  Used by profiles that opt into emotion-adaptive synthesis
  (emotion_adaptive=true implies a provider that actually supports
  emotion tags, which macos_say doesn't).

At alpha-3a the provider NAMES in the ladders are intentionally
ahead of PROVIDER_REGISTRY, which is still empty. The integration
commit (alpha-3 later) teaches the voice primitives to consult a
ladder, skip unregistered names with a warning, and raise only when
every name in the ladder is either unregistered or raises.

Ladders are pure data — no loading from YAML, no environment-based
overrides. Profiles select a ladder by name via profiles.yaml
(``ladder: free`` / ``ladder: premium``). A profile selecting a
ladder that doesn't exist in LADDERS raises at profile-load time
rather than silently falling back.
"""
from __future__ import annotations

from typing import TypedDict


class ProviderLadder(TypedDict):
    """Shape of a single ladder — TTS and STT ordered lists."""

    tts: list[str]
    stt: list[str]


FREE_LADDER: ProviderLadder = {
    "tts": ["macos_say"],
    "stt": ["whisper", "macos_dictation"],  # whisper ships first; macos_dictation planned for alpha-4
}


PREMIUM_LADDER: ProviderLadder = {
    "tts": ["gemini_tts", "elevenlabs", "macos_say"],
    "stt": ["gemini_stt", "macos_dictation"],
}


FALLOVER_LADDER: ProviderLadder = {
    "tts": [
        "gemini_tts_free",
        "google_cloud_tts_free",
        "piper",
        "macos_say",
    ],
    "stt": ["macos_dictation"],  # STT fallover lands in a later wave
}
"""Tier-3 free-tier fallover ladder.

Cloud-first for quality (Gemini 2.5 with emotion → Google Cloud TTS
Neural2), then local neural (Piper), with macOS ``say`` as the final
always-available rung. Tier-3 shipped providers target $0 total cost
within free-tier quotas; ``is_available`` on the local rungs lets
dispatch skip cleanly when binaries or model files are missing.
"""


LADDERS: dict[str, ProviderLadder] = {
    "free": FREE_LADDER,
    "premium": PREMIUM_LADDER,
    "fallover": FALLOVER_LADDER,
}


def get_ladder(name: str) -> ProviderLadder:
    """Return the named ladder or raise KeyError with an operator-
    helpful listing of valid names.

    Raising KeyError (not returning a default) is the SOLID choice:
    a profile mis-spelling 'premium' as 'premuim' must fail loudly
    at load time, not silently fall through to free and ship
    without emotion support.
    """
    if name not in LADDERS:
        valid = sorted(LADDERS.keys())
        raise KeyError(f"ladder '{name}' not found. Valid: {valid}")
    return LADDERS[name]


__all__ = [
    "FALLOVER_LADDER",
    "FREE_LADDER",
    "LADDERS",
    "PREMIUM_LADDER",
    "ProviderLadder",
    "get_ladder",
]
