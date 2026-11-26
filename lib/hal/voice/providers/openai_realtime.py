"""HAL Voice — OpenAI Realtime API backend (full-duplex websocket).

The first concrete ``RealtimeBackend`` (see ``../realtime_base.py``).
Streams microphone PCM up and receives transcript + synthesised audio
down over a single persistent websocket to the OpenAI Realtime API.

Design parity with ``gemini_tts.py`` (the canonical cloud-provider
reference in this package):

- Auth: module-level ``_resolve_api_key()`` = ``OPENAI_API_KEY`` env →
  macOS keychain entry ``hal-openai`` (already a live HAL provider key —
  see rules/operational-context.md HAL keychain table). Constructor
  ``api_key=`` override for tests/injection. The key is read at
  ``start()`` (NOT at import or ``__init__``) so importing this module has
  ZERO dependency on a live key — the load-bearing invariant for the test
  suite and CI.
- Model: resolved at call time via ``OPENAI_REALTIME_MODEL`` env →
  ``DEFAULT_MODEL`` constant, so a model rename is an env change, never a
  recompile, and the model id is never a hardcoded secret.
- Errors: raises the shared ``AuthFailure`` / ``NetworkError`` from
  ``_errors.py`` (all ``VoiceError`` subclasses) so a future ladder's
  ``is_retryable`` classifies them uniformly.

Concurrency parity with the rest of the voice stack (``audio_controller``,
``voice.py``): SYNCHRONOUS + a daemon reader THREAD, not asyncio — to
avoid an async island in an otherwise synchronous package (design note R3).

Websocket transport (design note R2): the actual ws client is injected via
``ws_factory`` (DIP seam). The default factory imports a ws library lazily
INSIDE ``start()`` so (a) tests never import it — they inject a fake — and
(b) the dependency is a runtime-only concern, exactly like the API key.
The backend talks to whatever ``ws_factory`` returns through a tiny duck
interface (``send(str)`` / ``recv() -> str`` / ``close()``), so it is not
coupled to any single ws library.

Native OpenAI Realtime event → ``RealtimeEvent.kind`` mapping (the part a
mutation test pins):

    conversation.item.input_audio_transcription.delta      → partial_transcript
    conversation.item.input_audio_transcription.completed  → final_transcript
    response.output_audio.delta                            → audio_delta
    response.output_audio_transcript.delta / .done         → assistant_transcript
    response.done                                          → turn_done
    error                                                  → error

GA migration (2026-06-03): the response-audio events gained the ``output_``
prefix vs Beta (``response.audio.delta`` → ``response.output_audio.delta``);
the input-transcription events kept their Beta names. session.update moved to
the nested ``session.audio.{input,output}`` shape (see _session_update_payload).
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import subprocess
import threading
from typing import Any, Callable
from urllib.parse import quote

from lib.hal.spend_guard import require_openai_destination
from ..realtime_base import RealtimeCallback, RealtimeConfig, RealtimeEvent
from ..types import VoiceError
from ._errors import AuthFailure, NetworkError


# ---------- constants ----------

KEYCHAIN_SERVICE = "hal-openai"
API_KEY_ENV = "OPENAI_API_KEY"
MODEL_ENV = "OPENAI_REALTIME_MODEL"
# GA Realtime model. The Beta ``gpt-4o-realtime-preview`` 404s now that the
# Beta API is sunset ("model does not exist or you do not have access").
DEFAULT_MODEL = "gpt-realtime"
ENDPOINT_TEMPLATE = "wss://api.openai.com/v1/realtime?model={model}"
DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"

# GA Realtime nests audio config under ``session.audio.{input,output}`` and
# requires each ``format`` to be an OBJECT (the Beta string ``"pcm16"`` is
# rejected with ``invalid_type``). Map the legacy config string → GA object.
# All confirmed against the live GA endpoint 2026-06-03.
_GA_AUDIO_FORMATS: dict[str, dict[str, Any]] = {
    "pcm16": {"type": "audio/pcm", "rate": 24000},
    "g711_ulaw": {"type": "audio/pcmu"},
    "g711_alaw": {"type": "audio/pcma"},
}


def _ga_audio_format(fmt: str) -> dict[str, Any]:
    """Map a Beta-style format string to the GA ``audio.*.format`` object.

    Unknown/empty values default to PCM16 @ 24 kHz rather than passing an
    unsupported shape (or a bare string) through to GA — fail-safe to the
    canonical realtime format instead of erroring at session.update time.
    """
    return _GA_AUDIO_FORMATS.get(fmt, {"type": "audio/pcm", "rate": 24000})


# ---------- auth / model resolution (mirrors gemini_tts) ----------


def _fetch_keychain_token(service: str = KEYCHAIN_SERVICE) -> str | None:
    """Read the OpenAI API key from the macOS keychain.

    Exact shape of ``gemini_tts._fetch_keychain_token`` — shells
    ``security find-generic-password`` and returns ``None`` on any failure
    (missing binary, non-zero exit, timeout) so resolution falls through
    cleanly rather than raising at lookup time.
    """
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-w", "-s", service],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _resolve_api_key() -> str | None:
    """Env first, keychain fallback. Read at call time, never at import."""
    return os.environ.get(API_KEY_ENV) or _fetch_keychain_token()


def _resolve_model() -> str:
    """Model id at call time so a preview rename is an env change."""
    return os.environ.get(MODEL_ENV) or DEFAULT_MODEL


def is_available() -> bool:
    """True iff an OpenAI key is resolvable. No network call.

    Module-level function, parity with ``whisper_stt.is_available`` — lets
    a future realtime ladder skip this backend cleanly when no key exists.
    """
    return _resolve_api_key() is not None


# ---------- native-event → RealtimeEvent mapping (pure, testable) ----------


def _decode_audio_b64(data_b64: Any) -> bytes:
    """Strict base64 → bytes, same discipline as gemini_tts._decode_response.

    Raises ``VoiceError`` on a non-str or invalid base64 payload so a
    malformed audio frame surfaces loudly instead of replaying garbage.
    """
    if not isinstance(data_b64, str):
        raise VoiceError("openai_realtime: audio delta 'delta' is not a string")
    try:
        return base64.b64decode(data_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise VoiceError("openai_realtime: audio delta is not valid base64") from exc


def map_event(frame: dict[str, Any]) -> RealtimeEvent | None:
    """Map one native OpenAI Realtime frame to a ``RealtimeEvent``.

    Returns ``None`` for frames we do not surface (session.created,
    input_audio_buffer.committed, rate_limits.updated, …) so the reader
    loop can ignore them without a callback. Pure function — the unit
    tests drive it directly with canned frames (no websocket, no network).
    """
    etype = frame.get("type")

    if etype == "conversation.item.input_audio_transcription.delta":
        return RealtimeEvent(
            kind="partial_transcript",
            text=str(frame.get("delta", "")),
            raw=frame,
        )
    if etype == "conversation.item.input_audio_transcription.completed":
        return RealtimeEvent(
            kind="final_transcript",
            text=str(frame.get("transcript", "")),
            confidence=_extract_confidence(frame),
            is_final=True,
            raw=frame,
        )
    if etype == "response.output_audio.delta":
        return RealtimeEvent(
            kind="audio_delta",
            audio=_decode_audio_b64(frame.get("delta")),
            raw=frame,
        )
    if etype in ("response.output_audio_transcript.delta", "response.output_audio_transcript.done"):
        key = "delta" if etype.endswith(".delta") else "transcript"
        return RealtimeEvent(
            kind="assistant_transcript",
            text=str(frame.get(key, "")),
            is_final=etype.endswith(".done"),
            raw=frame,
        )
    if etype == "response.done":
        return RealtimeEvent(kind="turn_done", is_final=True, raw=frame)
    if etype == "error":
        err = frame.get("error", {})
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return RealtimeEvent(kind="error", text=str(msg or "unknown error"), raw=frame)

    return None


def _extract_confidence(frame: dict[str, Any]) -> float | None:
    """Pull a transcription confidence out of a completed-transcript frame.

    The Realtime transcription payload MAY carry a confidence/logprob;
    when absent we return ``None`` (no signal) rather than inventing one —
    the downstream gate falls back to its own parse confidence.
    """
    conf = frame.get("confidence")
    if isinstance(conf, (int, float)):
        return float(conf)
    return None


# ---------- the backend ----------


class OpenAiRealtimeBackend:
    """OpenAI Realtime API backend. Satisfies ``RealtimeBackend``.

    Does not inherit the Protocol. Constructed by ``open_realtime`` with a
    resolved ``RealtimeConfig``; ``ws_factory`` is the DIP test seam.
    """

    name = "openai_realtime"

    def __init__(
        self,
        *,
        config: RealtimeConfig,
        api_key: str | None = None,
        ws_factory: Callable[[str, dict[str, str]], Any] | None = None,
    ) -> None:
        self._config = config
        # Precedence mirrors gemini_tts: explicit config.api_key, then the
        # constructor override, then env/keychain at start().
        self._api_key_override = api_key or config.api_key
        self._ws_factory = ws_factory
        self._ws: Any = None
        self._callback: RealtimeCallback | None = None
        self._reader: threading.Thread | None = None
        self._closed = threading.Event()
        self._started = False

    # -- auth / model (mirror gemini_tts._resolve_key / _resolve_model_name) --

    def _resolve_key(self) -> str:
        key = self._api_key_override or _resolve_api_key()
        if not key:
            raise AuthFailure(
                "openai_realtime: missing API key "
                f"(set {API_KEY_ENV} env or keychain entry {KEYCHAIN_SERVICE!r})"
            )
        return key

    def _resolve_model_name(self) -> str:
        return self._config.model or _resolve_model()

    def is_available(self) -> bool:
        if self._api_key_override:
            return True
        return is_available()

    # -- lifecycle --

    def _session_update_payload(self) -> dict[str, Any]:
        """Build the GA ``session.update`` config sent right after connect.

        GA Realtime requires ``session.type = "realtime"`` and nests audio
        config under ``session.audio.{input,output}`` with object-form
        ``format`` values — Beta's flat ``voice`` / ``input_audio_format`` /
        ``output_audio_format`` / ``turn_detection`` / ``input_audio_transcription``
        keys are rejected by GA (``unknown_parameter`` / ``invalid_type``).
        Confirmed live against the GA endpoint 2026-06-03.
        """
        input_audio: dict[str, Any] = {
            "format": _ga_audio_format(self._config.input_audio_format),
            "transcription": {
                "model": self._config.transcription_model or DEFAULT_TRANSCRIPTION_MODEL
            },
            # turn_detection="none" disables server VAD (manual commit_turn path).
            "turn_detection": (
                None if self._config.turn_detection == "none"
                else {"type": self._config.turn_detection}
            ),
        }
        session: dict[str, Any] = {
            "type": "realtime",
            "audio": {
                "input": input_audio,
                "output": {
                    "format": _ga_audio_format(self._config.output_audio_format),
                    "voice": self._config.voice,
                },
            },
        }
        if self._config.system:
            session["instructions"] = self._config.system
        return {"type": "session.update", "session": session}

    def _default_ws_factory(self, url: str, headers: dict[str, str]) -> Any:
        """Lazily import a ws client and connect. Only runs in real use.

        Imported INSIDE this method so tests that inject ``ws_factory``
        never import the ws dependency (design note R2). Raises
        ``NetworkError`` if the dependency is absent or the connect fails.
        """
        try:
            from websocket import create_connection  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only live
            raise NetworkError(
                "openai_realtime: websocket-client not installed "
                "(pip install websocket-client) — or inject ws_factory"
            ) from exc
        header_list = [f"{k}: {v}" for k, v in headers.items()]
        try:  # pragma: no cover - live network
            return create_connection(url, header=header_list, timeout=30)
        except Exception as exc:  # pragma: no cover - live network
            raise NetworkError(f"openai_realtime: connect failed — {exc}") from exc

    def start(self) -> None:
        if self._started:
            return
        require_openai_destination(ENDPOINT_TEMPLATE)
        api_key = self._resolve_key()
        # Percent-encode the model value: it flows into the wss URL query
        # string and OPENAI_REALTIME_MODEL is operator-set — encoding makes a
        # reserved character inert instead of malforming or injecting the URL.
        url = ENDPOINT_TEMPLATE.format(model=quote(self._resolve_model_name(), safe=""))
        # GA Realtime takes NO ``OpenAI-Beta`` header — sending the Beta marker
        # now returns "The Realtime Beta API is no longer supported."
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        factory = self._ws_factory or self._default_ws_factory
        ws = factory(url, headers)
        # Validate the transport BEFORE storing it, so a bad ws_factory result
        # never leaves a half-open self._ws that a later close() would touch
        # (raise a clean NetworkError at the boundary, not an opaque
        # AttributeError later inside _send_json / _read_loop).
        for _method in ("send", "recv", "close"):
            if not callable(getattr(ws, _method, None)):
                raise NetworkError(
                    f"openai_realtime: ws_factory returned an object without a "
                    f"callable {_method!r} (expected a websocket transport)"
                )
        self._ws = ws
        try:
            self._send_json(self._session_update_payload())
        except Exception:
            # Handshake failed: tear down + clear the ws so a subsequent
            # close() never operates on a half-open connection.
            try:
                ws.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._ws = None
            raise
        self._started = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise VoiceError("openai_realtime: not started — call start() first")
        try:
            self._ws.send(json.dumps(payload))
        except Exception as exc:  # noqa: BLE001 - normalise transport errors
            raise NetworkError(f"openai_realtime: send failed — {exc}") from exc

    def send_audio(self, pcm: bytes) -> None:
        b64 = base64.b64encode(pcm).decode("ascii")
        self._send_json({"type": "input_audio_buffer.append", "audio": b64})

    def send_text(self, text: str) -> None:
        self._send_json({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        })

    def commit_turn(self, commit_audio: bool = True) -> None:
        # Server-VAD auto-commits; manual mode commits the buffer + asks for a
        # response. commit_audio=False skips the buffer commit (text-turn path):
        # committing an empty buffer makes GA reject with "buffer too small".
        if commit_audio:
            self._send_json({"type": "input_audio_buffer.commit"})
        self._send_json({"type": "response.create"})

    def on_transcript(self, callback: RealtimeCallback) -> None:
        # Idempotent re-registration replaces the prior callback.
        self._callback = callback

    def _read_loop(self) -> None:
        """Daemon thread: recv frames, map, dispatch to the callback.

        On an UNEXPECTED termination — a transport error from ``recv()`` or a
        peer close (empty payload) while the session was NOT deliberately
        closed — emit a single terminal ``error`` event to the callback so
        callers (nexus time_capture, chat_loop) can distinguish a dropped
        connection from a graceful ``close()`` and never silently lose the
        reader. A deliberate ``close()`` (``self._closed`` set) exits SILENTLY.
        """
        reason: str | None = None
        while not self._closed.is_set():
            try:
                raw = self._ws.recv()
            except Exception as exc:  # noqa: BLE001 - ws closed / transport gone
                reason = f"transport error — {exc}"
                break
            if not raw:
                reason = "connection closed by peer"
                break
            self._dispatch_raw(raw)
        # Signal a terminal error ONLY for an unexpected drop — never when the
        # caller asked to close() (an expected, silent shutdown).
        if reason is not None and not self._closed.is_set():
            self._emit_terminal_error(reason)
        self._closed.set()

    def _emit_terminal_error(self, reason: str) -> None:
        """Fire a single terminal ``error`` event (best-effort).

        Guards the callback so a raising consumer cannot itself kill the
        reader thread during teardown.
        """
        if self._callback is None:
            return
        try:
            self._callback(
                RealtimeEvent(kind="error", text=f"openai_realtime: {reason}", is_final=True)
            )
        except Exception:  # noqa: BLE001 - a callback raising must not crash teardown
            pass

    def _dispatch_raw(self, raw: str | bytes) -> None:
        """Parse one raw frame and fire the callback if it maps to an event.

        ANY exception from ``map_event`` (not only ``VoiceError``) is
        converted to an ``error`` event rather than propagating out of the
        daemon reader thread and killing it silently. The callback is also
        guarded for the same reason — a raising consumer must not kill the
        reader (it would silently lose every subsequent frame).
        """
        # websocket-client may deliver a bytes frame; normalise to str before
        # parsing. A non-UTF-8 binary frame is not a JSON event we surface, so
        # skip it cleanly rather than risk a TypeError downstream.
        if isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                return
        try:
            frame = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(frame, dict):
            return
        try:
            event = map_event(frame)
        except VoiceError as exc:
            event = RealtimeEvent(kind="error", text=str(exc), raw=frame)
        except Exception as exc:  # noqa: BLE001 - a mapping bug must not kill the reader
            event = RealtimeEvent(
                kind="error",
                text=f"openai_realtime: frame mapping failed — {exc}",
                raw=frame,
            )
        if event is not None and self._callback is not None:
            try:
                self._callback(event)
            except Exception:  # noqa: BLE001 - a callback raising must not kill the reader
                pass

    def close(self) -> None:
        # Idempotent: safe to call repeatedly.
        self._closed.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001 - best-effort close
                pass
        if self._reader is not None and self._reader.is_alive():
            self._reader.join(timeout=2.0)


__all__ = [
    "API_KEY_ENV",
    "AuthFailure",
    "DEFAULT_MODEL",
    "DEFAULT_TRANSCRIPTION_MODEL",
    "ENDPOINT_TEMPLATE",
    "KEYCHAIN_SERVICE",
    "MODEL_ENV",
    "NetworkError",
    "OpenAiRealtimeBackend",
    "is_available",
    "map_event",
]
