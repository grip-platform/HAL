"""HAL Voice — RealtimeSession facade + factory + HAPPI envelope mapping.

The operator-facing entry point for full-duplex voice. ``RealtimeSession``
wraps a chosen ``RealtimeBackend`` (DIP: the backend is injected or
resolved by name from ``REALTIME_REGISTRY``) behind a stable,
provider-agnostic API — the syscall-doctrine shim for streaming voice:
one API, swappable backend (OpenAI Realtime today, Gemini Live next).

Public surface
--------------

    from lib.hal.voice import open_realtime, RealtimeConfig, RealtimeEvent

    session = open_realtime(
        backend="openai_realtime",
        config=RealtimeConfig(system="Be concise.", voice="alloy"),
    )
    session.on_transcript(lambda ev: print(ev.kind, ev.text))
    session.start()
    session.send_audio(mic_pcm_chunk)        # repeat for each chunk
    # ... events arrive on the callback (transcript + audio_delta) ...
    session.close()

The four spec-mandated methods — ``start`` / ``send_audio`` /
``on_transcript`` / ``close`` — are the core; ``send_text`` +
``commit_turn`` are additive for manual turn control (the nexus
confirm-turn path).

Two consumers, one shim
-----------------------
- **GRIP voice-chatting**: full-duplex sibling of ``voice.chat_loop`` —
  pipe mic PCM via ``send_audio``, play ``audio_delta`` events through
  ``audio_controller.AudioController`` (no new playback machinery).
- **nexus time_capture**: obtain a turn-final transcript + a REAL
  confidence (``RealtimeEvent.confidence``) to replace the browser Web
  Speech API, feeding the IDR confidence gate.

HAPPI envelope in/out
---------------------
``realtime_config_from_envelope`` reads a parsed ``HappiEnvelope``'s flags
into a ``RealtimeConfig``; ``realtime_event_to_envelope_line`` emits a JSON
line in the SAME event grammar the HAPPI runtime emits (``{"type":"delta",
"text":...}`` / ``{"type":"audio",...}`` / ``{"type":"error",...}``), so a
realtime stream is a drop-in for the existing ``happi_responder`` consumer
(``_collect_response_text``). Accepts ``happi/1.0`` and ``happi/1.1`` — the
envelope parser already does.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .realtime_base import (
    RealtimeBackend,
    RealtimeCallback,
    RealtimeConfig,
    RealtimeEvent,
)
from .realtime_registry import REALTIME_REGISTRY
from .types import VoiceError

DEFAULT_BACKEND = "openai_realtime"


class RealtimeSession:
    """Provider-agnostic full-duplex voice session.

    Thin facade over a ``RealtimeBackend``. Every method delegates to the
    backend; the value is the STABLE API (callers never import a concrete
    backend) + the registry/validation boundary in ``open_realtime``.
    """

    def __init__(self, backend: RealtimeBackend) -> None:
        self._backend = backend
        # Expose the resolved backend name as plain data (not a method) so
        # callers can log/audit which backend is live without growing the
        # behavioural surface.
        self.backend_name = backend.name

    def start(self) -> None:
        self._backend.start()

    def send_audio(self, pcm: bytes) -> None:
        self._backend.send_audio(pcm)

    def send_text(self, text: str) -> None:
        self._backend.send_text(text)

    def commit_turn(self, commit_audio: bool = True) -> None:
        # commit_audio=False creates a response WITHOUT committing the input
        # audio buffer (the text-turn path) — committing an empty buffer makes
        # GA reject the turn ("buffer too small"). Kept as ONE parameterised
        # method rather than a separate create_response() to hold the public
        # interface at 7 methods (ISP).
        self._backend.commit_turn(commit_audio=commit_audio)

    def on_transcript(self, callback: RealtimeCallback) -> None:
        """Register the per-event callback (spec-named entry point).

        Note: events are not only transcripts — the callback also receives
        ``audio_delta``, ``turn_done``, and ``error`` events. The method
        keeps the spec name ``on_transcript`` for the registration contract;
        callers dispatch on ``RealtimeEvent.kind`` inside the callback.
        """
        self._backend.on_transcript(callback)

    def close(self) -> None:
        self._backend.close()

    def is_available(self) -> bool:
        return self._backend.is_available()

    # Context-manager sugar so callers can guarantee close() on exit.
    def __enter__(self) -> "RealtimeSession":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_realtime(
    *,
    backend: str = DEFAULT_BACKEND,
    config: RealtimeConfig | None = None,
    registry: dict[str, type[RealtimeBackend]] | None = None,
    api_key: str | None = None,
    ws_factory: Callable[[str, dict[str, str]], Any] | None = None,
) -> RealtimeSession:
    """Resolve a backend by name and return a ``RealtimeSession``.

    Backend resolution is an EXPLICIT registry lookup — an unknown name
    raises ``VoiceError`` with the valid-name list, parity with how
    ``voice.speak`` rejects an unknown TTS provider. No filesystem scan,
    no import magic (security stance inherited from the registry).

    ``api_key`` / ``ws_factory`` are forwarded to the backend constructor
    so callers (and tests) can inject without touching env/keychain/network.
    The constructed backend is ``isinstance``-validated against the
    ``RealtimeBackend`` Protocol before it is wrapped — an incomplete stub
    is rejected here, not mid-stream.
    """
    reg = registry if registry is not None else REALTIME_REGISTRY
    backend_cls = reg.get(backend)
    if backend_cls is None:
        valid = sorted(reg.keys())
        raise VoiceError(
            f"open_realtime: unknown backend {backend!r}. Valid: {valid}"
        )
    instance = backend_cls(
        config=config or RealtimeConfig(),
        api_key=api_key,
        ws_factory=ws_factory,
    )
    if not isinstance(instance, RealtimeBackend):
        raise VoiceError(
            f"open_realtime: backend {backend!r} does not satisfy the "
            "RealtimeBackend Protocol (missing required methods)"
        )
    return RealtimeSession(instance)


# ---------- HAPPI envelope adapters ----------


def realtime_config_from_envelope(envelope: Any) -> RealtimeConfig:
    """Build a ``RealtimeConfig`` from a parsed ``HappiEnvelope``'s flags.

    Reads ``flags.{voice,model,system,transcription_model,turn_detection,
    input_audio_format,output_audio_format,api_key}`` with the dataclass
    defaults as fallbacks. Accepts any object exposing a ``flags`` mapping (so the
    test can pass a real ``HappiEnvelope`` or a light stub). The provider
    spec (``flags.provider`` / ``flags.backend``) is read by the caller to
    pick the registry key; this function only shapes the config.
    """
    flags = getattr(envelope, "flags", {}) or {}
    defaults = RealtimeConfig()

    def _flag(name: str, fallback: Any) -> Any:
        value = flags.get(name)
        return value if value is not None else fallback

    return RealtimeConfig(
        model=str(_flag("model", defaults.model)),
        voice=str(_flag("voice", defaults.voice)),
        input_audio_format=str(_flag("input_audio_format", defaults.input_audio_format)),
        output_audio_format=str(_flag("output_audio_format", defaults.output_audio_format)),
        system=_flag("system", defaults.system),
        transcription_model=_flag("transcription_model", defaults.transcription_model),
        turn_detection=str(_flag("turn_detection", defaults.turn_detection)),
        # api_key completes the DIP injection seam: an envelope-driven caller
        # can supply a key via flags.api_key without touching the constructor
        # (parity with open_realtime(api_key=...)); absent -> None -> resolve
        # from env/keychain at start(). Never str()-coerced (None must stay None).
        api_key=_flag("api_key", defaults.api_key),
    )


def realtime_event_to_envelope_line(event: RealtimeEvent) -> str:
    """Render a ``RealtimeEvent`` as a HAPPI-runtime-grammar JSON line.

    Maps each event kind to the event grammar that
    ``happi_responder._collect_response_text`` consumes, so a realtime
    stream drops into the existing HAPPI consumer unchanged:

        partial/final/assistant transcript → {"type":"delta","text":...}
        audio_delta                        → {"type":"audio","audio_b64":...}
        error                              → {"type":"error","message":...}
        turn_done                          → {"type":"done"}

    For ``final_transcript`` the real ``confidence`` (or ``null``) rides
    along under a ``confidence`` key so a HAPPI consumer that wants the
    gate signal can read it without breaking consumers that only read
    ``text``.
    """
    if event.kind in ("partial_transcript", "final_transcript", "assistant_transcript"):
        payload: dict[str, Any] = {"type": "delta", "text": event.text}
        if event.kind == "final_transcript":
            payload["confidence"] = event.confidence
            payload["is_final"] = True
        return json.dumps(payload, separators=(",", ":"))
    if event.kind == "audio_delta":
        import base64
        return json.dumps(
            {"type": "audio", "audio_b64": base64.b64encode(event.audio).decode("ascii")},
            separators=(",", ":"),
        )
    if event.kind == "turn_done":
        return json.dumps({"type": "done"}, separators=(",", ":"))
    if event.kind == "error":
        return json.dumps(
            {"type": "error", "code": "realtime_error", "message": event.text},
            separators=(",", ":"),
        )
    # Unknown kind — emit a typed error line rather than a silent drop.
    return json.dumps(
        {"type": "error", "code": "unknown_event_kind", "message": event.kind},
        separators=(",", ":"),
    )


__all__ = [
    "DEFAULT_BACKEND",
    "RealtimeConfig",
    "RealtimeEvent",
    "RealtimeSession",
    "open_realtime",
    "realtime_config_from_envelope",
    "realtime_event_to_envelope_line",
]
