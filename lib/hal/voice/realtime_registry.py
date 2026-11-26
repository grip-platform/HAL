"""HAL Voice — realtime (full-duplex) backend registry.

Sibling of ``providers/__init__.py``. Where that registry holds
single-shot ``TtsProvider`` / ``SttProvider`` classes, this one holds
full-duplex ``RealtimeBackend`` classes (bidirectional streaming audio +
transcript over a persistent websocket). A realtime backend is neither a
single-shot TTS nor an STT, so it lives in its own registry rather than
overloading ``PROVIDER_REGISTRY["tts"]`` / ``["stt"]`` — the narrow
Protocols in ``base.py`` stay narrow (design note R1 in the build spec).

``REALTIME_REGISTRY`` is an EXPLICIT dict populated by deliberate
assignment AFTER its definition — identical security stance to the TTS/STT
registry: an adversary who drops a backend module into the import path but
is NOT listed here stays invisible to dispatch. No decorator auto-register,
no filesystem scan.

Operator audit is a one-liner::

    from lib.hal.voice.realtime_registry import REALTIME_REGISTRY
    print(sorted(REALTIME_REGISTRY.keys()))

Adding the Gemini Live backend (build spec "RUNG 2 (planned)") is a single
explicit line here once that module ships::

    REALTIME_REGISTRY["gemini_live"] = GeminiLiveBackend

— same one-line extension as ``PROVIDER_REGISTRY["tts"][...] = ...``.
"""
from __future__ import annotations

from .realtime_base import RealtimeBackend

# The registry maps a backend name → its class (constructed lazily by
# ``open_realtime`` with a resolved RealtimeConfig). Type is the Protocol
# so any conforming class drops in; concrete imports follow the dict
# definition so the module imports cleanly with no circular side-effects.
REALTIME_REGISTRY: dict[str, type[RealtimeBackend]] = {}


# ---------- concrete backend registrations ----------
#
# Import AFTER the dict definition (mirrors providers/__init__.py). Each
# assignment is explicit and auditable.

from .providers.openai_realtime import OpenAiRealtimeBackend  # noqa: E402

REALTIME_REGISTRY["openai_realtime"] = OpenAiRealtimeBackend

# RUNG 2 (planned) — Gemini Live API backend. Same RealtimeBackend
# Protocol; only the wire grammar + keychain entry (grip-gemini /
# GEMINI_API_KEY, already wired in providers/gemini_tts.py) differ.
# Register with one line when lib/hal/voice/providers/gemini_live.py lands:
#   from .providers.gemini_live import GeminiLiveBackend
#   REALTIME_REGISTRY["gemini_live"] = GeminiLiveBackend


__all__ = [
    "OpenAiRealtimeBackend",
    "REALTIME_REGISTRY",
    "RealtimeBackend",
]
