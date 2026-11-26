"""HAL Voice — result dataclasses, Provenance enum, exception hierarchy.

Per plans/voice-first-harness-triphase.md Rev 2.3 section 4.1:
- All tier-3 paths take capability: bytes (HMAC-signed, single-use, TTL 180s
  with notification-tap clock). confirmed: bool / tier_override: int are
  explicitly forbidden on the public surface.
- Provenance is a typed enum shared with the osa plan (same enum, single
  source of truth) — tier_gate.py keys off the enum exclusively.
- audit_id on result types is generated internally via secrets.token_hex(16)
  in audit.py; the dataclasses accept it as a value but callers cannot
  construct results that bypass the audit writer.

Shipped at PR alpha-2 W1. Primitive bodies (voice.listen, voice.speak,
voice.chat_loop), audit chain, cache, tier gate, and providers land in
subsequent W1+ commits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Provenance(Enum):
    """Typed provenance enum. Shared single source of truth with the osa plan.

    tier_gate.py keys off this enum exclusively. Behavioural string flags
    are forbidden on tier-3 paths (see osa verdict invariant #1 and voice
    plan invariant V2).
    """

    human_typed = "human_typed"
    voice_human = "voice_human"
    tool_chain = "tool_chain"
    mcp_remote = "mcp_remote"
    socket_client = "socket_client"
    autonomous_loop = "autonomous_loop"
    yolo = "yolo"


class VoiceError(Exception):
    """Base voice exception. All voice-specific errors inherit this."""


class ProfileTierCapExceeded(VoiceError):
    """Raised when an operation exceeds the profile's declared tier_cap.

    Must fire on ALL entry points: voice.speak(directive=...) on tier_cap<3,
    voice.chat_loop with emotion_adaptive=True on a profile whose
    voices.yaml declares emotion_adaptive=False, etc. Silent success is
    prohibited — both paths are tested two-sided per Rule 14 Goodhart.
    """


class PlatformUnsupported(VoiceError):
    """Raised on the local-path voice primitives when running off darwin."""


class DirectiveTamperDetected(VoiceError):
    """Raised when directives.sig HMAC verification fails at load time.

    Also raised when voice-fixed-points.jsonl chain-verify detects a break.
    Corruption path: discard and rebuild from telemetry.
    """


class FocusRaceDetected(VoiceError):
    """Raised by macos_dictation.py when front-app PID changes mid-listen.

    Prevents transcribed speech from landing in the wrong window (Slack
    notification, iTerm alert) during the listen-to-buffer phase. Buffer
    is discarded on raise.
    """


class UnauthorizedDiagnosticBypass(VoiceError):
    """Raised when a call path other than chat_loop startup attempts to
    set the cafe-diagnostic thread-local bypass flag.

    Mechanical enforcement of the cafe-mode self-recovery UX — only one
    callable (_speak_diagnostic in macos_say.py) may bypass the headphone
    gate; all other attempts raise.
    """


@dataclass(frozen=True)
class ListenResult:
    """Result of a single voice.listen() call.

    audio_path is None by default (no disk persistence). When requested,
    the path is a tmpfs entry ($TMPDIR/voice-<uuid>.wav) that voice.listen
    unlinks immediately after the consumer reads it.

    confidence gates tier-2+ actions: drop to tier-1 if < 0.85.

    provenance is a typed enum (NOT a string) so the tier gate cannot be
    bypassed by a caller-supplied kwarg.

    audit_id is generated internally via secrets.token_hex(16) inside
    audit.py; the dataclass accepts the value as a non-settable field
    at construction time but callers cannot supply their own (audit.py
    rejects external audit_id kwargs at the boundary).
    """

    ok: bool
    text: str
    audio_path: str | None
    language: str
    confidence: float
    duration_s: float
    provider: str
    provenance: Provenance
    tier: int
    audit_id: str


@dataclass(frozen=True)
class SpeakResult:
    """Result of a single voice.speak() call.

    directive_fingerprint is sha256[:16] of the resolved directive text.
    Used for cache keying and audit correlation.
    """

    ok: bool
    audio_path: str | None
    duration_s: float
    provider: str
    voice: str
    profile: str
    emotion: str | None
    directive_fingerprint: str
    tier: int
    audit_id: str


@dataclass(frozen=True)
class ChatTurn:
    """One turn of a voice.chat_loop() transcript.

    latency_breakdown surfaces per-phase latency ({'stt': 0.8, 'llm': 0.9,
    'tts': 1.1}) — exposed by hal voice chat --verbose.
    """

    turn_index: int
    listen: ListenResult
    response_text: str
    speak: SpeakResult
    total_latency_s: float
    latency_breakdown: dict[str, float]


@dataclass(frozen=True)
class ChatTranscript:
    """Complete voice.chat_loop() session.

    total_cost_usd is 0.0 on the free-tier ladder; any premium-tier call
    appears here and is audit-logged as tier=premium (H-VO-2).
    """

    turns: list[ChatTurn]
    profile: str
    session_id: str
    started_at: str
    ended_at: str
    total_cost_usd: float


@dataclass(frozen=True)
class ChatPolicy:
    """Configuration for voice.chat_loop().

    Split from voice.chat() in MESH design round (invariant #3) — the
    original voice.chat(**many_kwargs) was the confirmed: bool antipattern
    in a different costume. ChatPolicy makes each dimension explicit,
    injectable, and testable.

    emotion_adaptive: None = use profile default (from voices.yaml);
    True/False = explicit override. Three-valued so the profile-level
    default is not silently overridden by a CLI invocation.

    capability is required when any tier-3 operation may fire. Single-use,
    nonce-tracked, 180s base TTL with notification-tap-starts-clock + 60s
    pre-invalidation if unacknowledged + renewal-on-tap (see plan section
    4.1). verify_capability() in tier_gate.py enforces.
    """

    listen_provider: str | None = None
    speak_provider: str | None = None
    emotion_adaptive: bool | None = None
    max_turns: int = 0
    terminator: Callable[[ChatTurn], bool] | None = None
    cafe_mode: bool = False
    capability: bytes | None = None
