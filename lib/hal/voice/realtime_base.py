"""HAL Voice — realtime (full-duplex) backend Protocol + data types.

Sibling of ``providers/base.py``. ``base.py`` defines the single-shot
``TtsProvider`` / ``SttProvider`` contracts; this module defines the
full-duplex ``RealtimeBackend`` contract — the union type over every
realtime streaming backend (OpenAI Realtime, Gemini Live, …). One shim,
one API, every caller goes through it (syscall doctrine).

``RealtimeBackend`` is ``@runtime_checkable`` for the same reason the
TTS/STT Protocols are: ``open_realtime`` ``isinstance``-validates the
resolved backend before handing it a websocket, so an incomplete stub is
rejected mechanically rather than half-streaming.

Result/config types are ``@dataclass(frozen=True)`` to match ``types.py``.
``RealtimeEvent.confidence`` is the load-bearing field for the nexus
time-capture seam: when a backend reports a transcript confidence, nexus
gates on it directly; when it reports ``None``, nexus falls back to its
parse-derived confidence. Either way the downstream IDR gate gets a real
number and never a fabricated ``1.0``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable


# ---------- event kinds (closed vocabulary) ----------
#
# Frozen set so a typo in a backend's mapping fails a membership check in
# tests rather than silently inventing a new kind nobody consumes.
EVENT_KINDS: frozenset[str] = frozenset({
    "partial_transcript",   # interim STT of the user's speech
    "final_transcript",     # turn-final STT of the user's speech (gate input)
    "audio_delta",          # a chunk of synthesised assistant audio (bytes)
    "assistant_transcript",  # text transcript of the assistant's spoken reply
    "turn_done",            # the assistant finished a response turn
    "error",                # backend signalled an error
})


@dataclass(frozen=True)
class RealtimeEvent:
    """One event emitted by a realtime backend's reader loop.

    ``kind`` is always a member of ``EVENT_KINDS``. The other fields are
    populated per-kind:

    - ``partial_transcript`` / ``final_transcript`` / ``assistant_transcript``
      → ``text`` (and ``confidence`` when the backend reports one)
    - ``audio_delta`` → ``audio`` (decoded bytes; format is backend-declared,
      typically PCM16 mono)
    - ``error`` → ``text`` carries the message, ``raw`` the native payload
    - ``turn_done`` → no payload

    ``confidence`` is ``None`` when the backend gives none — callers MUST
    treat ``None`` as "no signal" and fall back, never coerce it to a
    number. ``raw`` is the backend-native dict for audit/replay.
    """

    kind: str
    text: str = ""
    audio: bytes = b""
    confidence: float | None = None
    is_final: bool = False
    raw: dict | None = None


@dataclass(frozen=True)
class RealtimeConfig:
    """Configuration for a realtime session.

    Provider-agnostic. ``api_key`` is the DIP injection seam: ``None`` (the
    default) means resolve from env/keychain at ``start()`` — never at
    import or ``__init__`` — so importing the module has no live-key
    dependency. ``model`` empty means the backend resolves its own default
    (also at runtime, via env), so no model id is hardcoded as a secret.

    ``turn_detection``:
      - ``"server_vad"`` (default) — the server auto-detects end-of-utterance
        and auto-commits the input buffer + starts a response.
      - ``"none"`` — manual: the caller signals end-of-turn via
        ``commit_turn()``. Used by the nexus confirm-turn path where the
        operator controls when a dictation is "done".
    """

    model: str = ""
    voice: str = "alloy"
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    system: str | None = None
    transcription_model: str | None = None
    turn_detection: str = "server_vad"
    api_key: str | None = None


# Callback signature a consumer registers via ``on_transcript`` /
# ``RealtimeSession.on_event``. Receives every event the reader loop emits.
RealtimeCallback = Callable[[RealtimeEvent], None]


@runtime_checkable
class RealtimeBackend(Protocol):
    """Full-duplex realtime voice backend contract.

    The union type over OpenAI Realtime, Gemini Live, etc. Implementations
    expose a ``name`` attribute (the ``REALTIME_REGISTRY`` key) and the
    lifecycle below. They do NOT inherit this Protocol — duck-typing +
    ``runtime_checkable`` ``isinstance`` validation at ``open_realtime``.

    Lifecycle: ``start`` → (``send_audio`` | ``send_text``)* →
    [``commit_turn``] → … → ``close``. ``on_transcript`` registers the
    event callback (idempotent re-registration replaces). ``is_available``
    lets a future ladder skip the backend when no key is present —
    parity with ``whisper_stt.is_available``.
    """

    name: str

    def start(self) -> None:
        """Open the connection and send the session configuration."""
        ...

    def send_audio(self, pcm: bytes) -> None:
        """Append a chunk of input audio (PCM16 mono) to the buffer."""
        ...

    def send_text(self, text: str) -> None:
        """Inject a text turn (optional; not all flows use it)."""
        ...

    def commit_turn(self, commit_audio: bool = True) -> None:
        """End the turn + request a response (manual-VAD path).

        commit_audio=False creates a response WITHOUT committing the input
        audio buffer (text-turn path) — committing an empty buffer makes GA
        reject the turn with "buffer too small".
        """
        ...

    def on_transcript(self, callback: RealtimeCallback) -> None:
        """Register the callback invoked for every emitted event."""
        ...

    def close(self) -> None:
        """Close the connection gracefully and flush. Idempotent."""
        ...

    def is_available(self) -> bool:
        """True iff a usable credential is resolvable (no network call)."""
        ...


__all__ = [
    "EVENT_KINDS",
    "RealtimeBackend",
    "RealtimeCallback",
    "RealtimeConfig",
    "RealtimeEvent",
]
