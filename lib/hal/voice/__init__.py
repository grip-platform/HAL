"""HAL Voice — harness-level STT/TTS/chat primitives.

Scaffolded at PR alpha-2 per plans/voice-first-harness-triphase.md.
Rev 2.3 + alpha-1 audit placed this module at lib/hal/voice/ (matching
the lib/hal/happi/ precedent) rather than the plan's lib/voice/ draft —
this is an alpha-1-followup correction to the plan's path references.

Public surface:

    from lib.hal.voice import (
        # primitives
        listen, speak, chat_loop, chat,
        # result dataclasses
        ChatPolicy, ChatTranscript, ChatTurn,
        ListenResult, SpeakResult,
        # enums + exceptions
        Provenance, VoiceError,
        ProfileTierCapExceeded, PlatformUnsupported,
        DirectiveTamperDetected, FocusRaceDetected,
        UnauthorizedDiagnosticBypass,
    )

Submodules accessible via the namespace:
    - audit (AuditLogger, verify_chain)
    - cache (VoiceCache, fingerprint, tmpfs_audio_path)
    - providers (PROVIDER_REGISTRY, TtsProvider, SttProvider)

H-VO- hypotheses registered via hypothesis_engine.py as H129-H138 at
PR alpha-1 audit time.
"""
from .happi_responder import happi_responder
from .realtime_base import (
    RealtimeBackend,
    RealtimeConfig,
    RealtimeEvent,
)
from .realtime_session import (
    RealtimeSession,
    open_realtime,
    realtime_config_from_envelope,
    realtime_event_to_envelope_line,
)
from .types import (
    ChatPolicy,
    ChatTranscript,
    ChatTurn,
    DirectiveTamperDetected,
    FocusRaceDetected,
    ListenResult,
    PlatformUnsupported,
    ProfileTierCapExceeded,
    Provenance,
    SpeakResult,
    UnauthorizedDiagnosticBypass,
    VoiceError,
)
from .voice import chat, chat_loop, listen, speak

__all__ = [
    "ChatPolicy",
    "ChatTranscript",
    "ChatTurn",
    "DirectiveTamperDetected",
    "FocusRaceDetected",
    "ListenResult",
    "PlatformUnsupported",
    "ProfileTierCapExceeded",
    "Provenance",
    "RealtimeBackend",
    "RealtimeConfig",
    "RealtimeEvent",
    "RealtimeSession",
    "SpeakResult",
    "UnauthorizedDiagnosticBypass",
    "VoiceError",
    "chat",
    "chat_loop",
    "happi_responder",
    "listen",
    "open_realtime",
    "realtime_config_from_envelope",
    "realtime_event_to_envelope_line",
    "speak",
]
