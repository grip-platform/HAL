"""Tests for the RealtimeSession facade + factory + HAPPI envelope mapping.

Goodhart-resistant (Rule 14): every test asserts concrete VALUES (exact
strings, exact bytes, exact confidence numbers, exact method-presence),
never call counts. A mutation that flips a mapping, drops the confidence
field, or re-adds a 1.0-style fabrication fails here.

Zero live key, zero websocket, zero network: the backend's ws transport is
injected via ``ws_factory``, the keychain lookup is monkeypatched, and the
API key is supplied via the constructor/config injection seam.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lib.hal.voice import (  # noqa: E402
    RealtimeBackend,
    RealtimeConfig,
    RealtimeEvent,
    RealtimeSession,
    open_realtime,
    realtime_config_from_envelope,
    realtime_event_to_envelope_line,
)
from lib.hal.voice.providers import openai_realtime as oair  # noqa: E402
from lib.hal.voice.providers.openai_realtime import (  # noqa: E402
    OpenAiRealtimeBackend,
)
from lib.hal.voice.realtime_registry import REALTIME_REGISTRY  # noqa: E402
from lib.hal.voice.types import VoiceError  # noqa: E402
from lib.hal.voice.providers._errors import AuthFailure  # noqa: E402


# ---------- a fake websocket (the DIP seam) ----------


class FakeWebSocket:
    """In-memory ws stand-in. Records sent frames, replays a script of
    inbound frames on ``recv``. No network, no thread races in tests that
    drive ``_dispatch_raw`` directly."""

    def __init__(self, inbound: list[str] | None = None) -> None:
        self.sent: list[str] = []
        self._inbound = list(inbound or [])
        self.closed = False

    def send(self, payload: str) -> None:
        self.sent.append(payload)

    def recv(self) -> str:
        if self._inbound:
            return self._inbound.pop(0)
        return ""  # empty => reader loop exits cleanly

    def close(self) -> None:
        self.closed = True


def _backend_with_fake(
    inbound: list[str] | None = None,
    config: RealtimeConfig | None = None,
) -> tuple[OpenAiRealtimeBackend, FakeWebSocket]:
    ws = FakeWebSocket(inbound)
    backend = OpenAiRealtimeBackend(
        config=config or RealtimeConfig(),
        api_key="test-key-not-real",  # injection seam — no env/keychain
        ws_factory=lambda url, headers: ws,
    )
    return backend, ws


# ---------- Protocol conformance + factory ----------


def test_realtime_backend_protocol_conformance() -> None:
    backend = OpenAiRealtimeBackend(config=RealtimeConfig(api_key="x"))
    assert isinstance(backend, RealtimeBackend)


def test_open_realtime_unknown_backend_raises() -> None:
    with pytest.raises(VoiceError) as exc:
        open_realtime(backend="does-not-exist", registry={})
    # parity with speak(): the error lists the valid names
    assert "Valid" in str(exc.value)


def test_open_realtime_resolves_registered_backend() -> None:
    ws = FakeWebSocket()
    session = open_realtime(
        backend="openai_realtime",
        config=RealtimeConfig(api_key="x"),
        ws_factory=lambda url, headers: ws,
    )
    assert isinstance(session, RealtimeSession)
    assert session.backend_name == "openai_realtime"


def test_default_registry_has_openai_realtime() -> None:
    assert "openai_realtime" in REALTIME_REGISTRY
    assert REALTIME_REGISTRY["openai_realtime"] is OpenAiRealtimeBackend


# ---------- native-event mapping (the mutation-pinned core) ----------


def test_map_event_partial_transcript() -> None:
    ev = oair.map_event({
        "type": "conversation.item.input_audio_transcription.delta",
        "delta": "hel",
    })
    assert ev is not None
    assert ev.kind == "partial_transcript"
    assert ev.text == "hel"
    assert ev.is_final is False


def test_map_event_final_transcript_carries_confidence() -> None:
    ev = oair.map_event({
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "log forty five minutes on the Muller matter",
        "confidence": 0.92,
    })
    assert ev is not None
    assert ev.kind == "final_transcript"
    assert ev.text == "log forty five minutes on the Muller matter"
    assert ev.confidence == 0.92  # exact value, not a call count
    assert ev.is_final is True


def test_map_event_final_transcript_no_confidence_is_none() -> None:
    # Backend supplies no confidence => None (no signal), never coerced.
    ev = oair.map_event({
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "hello",
    })
    assert ev is not None
    assert ev.kind == "final_transcript"
    assert ev.confidence is None


def test_map_event_audio_delta_decodes_base64() -> None:
    import base64
    raw_audio = b"\x01\x02\x03\x04PCM"
    ev = oair.map_event({
        "type": "response.output_audio.delta",
        "delta": base64.b64encode(raw_audio).decode("ascii"),
    })
    assert ev is not None
    assert ev.kind == "audio_delta"
    assert ev.audio == raw_audio  # exact bytes round-trip


def test_map_event_audio_delta_bad_base64_raises() -> None:
    # Parity with gemini_tts._decode_response: malformed audio -> VoiceError.
    with pytest.raises(VoiceError):
        oair.map_event({"type": "response.output_audio.delta", "delta": "!!!not-base64!!!"})


def test_map_event_assistant_transcript() -> None:
    ev = oair.map_event({
        "type": "response.output_audio_transcript.done",
        "transcript": "Logged 45 minutes.",
    })
    assert ev is not None
    assert ev.kind == "assistant_transcript"
    assert ev.text == "Logged 45 minutes."
    assert ev.is_final is True


def test_map_event_turn_done() -> None:
    ev = oair.map_event({"type": "response.done"})
    assert ev is not None
    assert ev.kind == "turn_done"
    assert ev.is_final is True


def test_map_event_error() -> None:
    ev = oair.map_event({
        "type": "error",
        "error": {"message": "rate limited"},
    })
    assert ev is not None
    assert ev.kind == "error"
    assert ev.text == "rate limited"


def test_map_event_unknown_frame_returns_none() -> None:
    # Frames we don't surface (session.created, etc.) => None, not an event.
    assert oair.map_event({"type": "session.created"}) is None
    assert oair.map_event({"type": "rate_limits.updated"}) is None


# ---------- key resolution: at start(), never at import ----------


def test_key_resolved_at_start_not_import(monkeypatch) -> None:
    """Load-bearing invariant: constructing the backend with NO key
    available must NOT raise; only start() raises AuthFailure."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(oair, "_fetch_keychain_token", lambda service="x": None)

    backend = OpenAiRealtimeBackend(config=RealtimeConfig())  # no raise
    assert backend.is_available() is False
    with pytest.raises(AuthFailure) as exc:
        backend.start()
    assert "missing API key" in str(exc.value)


def test_api_key_injection_overrides_env(monkeypatch) -> None:
    """DIP seam: config.api_key bypasses env/keychain entirely (offline)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(oair, "_fetch_keychain_token", lambda service="x": None)

    ws = FakeWebSocket()
    backend = OpenAiRealtimeBackend(
        config=RealtimeConfig(api_key="injected"),
        ws_factory=lambda url, headers: ws,
    )
    assert backend.is_available() is True
    backend.start()  # must not raise — key came from config
    backend.close()


def test_constructor_api_key_beats_config(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(oair, "_fetch_keychain_token", lambda service="x": None)
    ws = FakeWebSocket()
    backend = OpenAiRealtimeBackend(
        config=RealtimeConfig(api_key="from-config"),
        api_key="from-ctor",
        ws_factory=lambda url, headers: ws,
    )
    assert backend.is_available() is True


def test_is_available_reflects_key_presence(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(oair, "_fetch_keychain_token", lambda service="x": None)
    assert oair.is_available() is False
    monkeypatch.setenv("OPENAI_API_KEY", "present")
    assert oair.is_available() is True


# ---------- lifecycle guards ----------


def test_send_audio_before_start_raises() -> None:
    backend = OpenAiRealtimeBackend(config=RealtimeConfig(api_key="x"))
    with pytest.raises(VoiceError):
        backend.send_audio(b"\x00\x01")


def test_close_idempotent() -> None:
    backend, ws = _backend_with_fake()
    backend.start()
    backend.close()
    backend.close()  # second close must not raise
    assert ws.closed is True


def test_start_sends_session_update_with_config(monkeypatch) -> None:
    """start() sends a session.update carrying the config fields."""
    config = RealtimeConfig(
        voice="verse",
        system="Be concise.",
        turn_detection="none",
        transcription_model="whisper-1",
    )
    backend, ws = _backend_with_fake(config=config)
    backend.start()
    backend.close()
    assert len(ws.sent) >= 1
    first = json.loads(ws.sent[0])
    assert first["type"] == "session.update"
    sess = first["session"]
    # GA nests audio config under session.audio.{input,output} and requires
    # session.type="realtime"; format is an object, not the Beta "pcm16" string.
    assert sess["type"] == "realtime"
    assert sess["audio"]["output"]["voice"] == "verse"
    assert sess["instructions"] == "Be concise."
    # turn_detection="none" => server-VAD disabled (manual commit_turn)
    assert sess["audio"]["input"]["turn_detection"] is None
    assert sess["audio"]["input"]["transcription"]["model"] == "whisper-1"
    assert sess["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}


def test_start_uses_ga_endpoint_no_beta_header() -> None:
    """GA regression anchor (2026-06-03 Beta sunset): start() targets the GA
    model and sends NO ``OpenAI-Beta`` header. The injected-ws_factory frame
    tests cannot see URL/header construction — this is the test that would have
    caught the Beta drift, and now prevents reverting it.
    """
    captured: dict = {}
    ws = FakeWebSocket()

    def capturing_factory(url, headers):
        captured["url"] = url
        captured["headers"] = headers
        return ws

    backend = OpenAiRealtimeBackend(
        config=RealtimeConfig(),
        api_key="test-key-not-real",
        ws_factory=capturing_factory,
    )
    backend.start()
    backend.close()
    assert captured["url"].endswith("model=gpt-realtime")
    assert "gpt-4o-realtime-preview" not in captured["url"]
    assert "OpenAI-Beta" not in captured["headers"]
    assert captured["headers"]["Authorization"] == "Bearer test-key-not-real"


def test_server_vad_config_sets_turn_detection_object() -> None:
    backend, ws = _backend_with_fake(config=RealtimeConfig(turn_detection="server_vad"))
    backend.start()
    backend.close()
    sess = json.loads(ws.sent[0])["session"]
    assert sess["audio"]["input"]["turn_detection"] == {"type": "server_vad"}


def test_send_audio_emits_input_audio_buffer_append() -> None:
    import base64
    backend, ws = _backend_with_fake()
    backend.start()
    backend.send_audio(b"\xAA\xBB\xCC")
    backend.close()
    appends = [json.loads(s) for s in ws.sent if json.loads(s).get("type") == "input_audio_buffer.append"]
    assert len(appends) == 1
    assert base64.b64decode(appends[0]["audio"]) == b"\xAA\xBB\xCC"


def test_commit_turn_emits_commit_and_response_create() -> None:
    backend, ws = _backend_with_fake()
    backend.start()
    backend.commit_turn()
    backend.close()
    types = [json.loads(s).get("type") for s in ws.sent]
    assert "input_audio_buffer.commit" in types
    assert "response.create" in types


# ---------- callback dispatch through _dispatch_raw (no thread) ----------


def test_dispatch_raw_fires_callback_with_mapped_event() -> None:
    backend, _ = _backend_with_fake()
    received: list[RealtimeEvent] = []
    backend.on_transcript(received.append)
    backend._dispatch_raw(json.dumps({
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "test transcript",
        "confidence": 0.77,
    }))
    assert len(received) == 1
    assert received[0].kind == "final_transcript"
    assert received[0].text == "test transcript"
    assert received[0].confidence == 0.77


def test_dispatch_raw_ignores_unmapped_frame() -> None:
    backend, _ = _backend_with_fake()
    received: list[RealtimeEvent] = []
    backend.on_transcript(received.append)
    backend._dispatch_raw(json.dumps({"type": "session.created"}))
    assert received == []  # no callback for ignored frames


def test_dispatch_raw_bad_audio_emits_error_event_not_crash() -> None:
    # A malformed audio frame must surface as an error EVENT, not propagate
    # out of the reader loop and kill the session.
    backend, _ = _backend_with_fake()
    received: list[RealtimeEvent] = []
    backend.on_transcript(received.append)
    backend._dispatch_raw(json.dumps({"type": "response.output_audio.delta", "delta": "@@@"}))
    assert len(received) == 1
    assert received[0].kind == "error"


def test_dispatch_raw_non_json_is_swallowed() -> None:
    backend, _ = _backend_with_fake()
    received: list[RealtimeEvent] = []
    backend.on_transcript(received.append)
    backend._dispatch_raw("not json at all")
    assert received == []


def test_reader_thread_delivers_scripted_frames() -> None:
    """End-to-end through the real daemon thread: scripted inbound frames
    arrive on the callback. Drains then exits on empty recv()."""
    inbound = [
        json.dumps({"type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "forty five minutes", "confidence": 0.9}),
        json.dumps({"type": "response.done"}),
    ]
    backend, _ = _backend_with_fake(inbound)
    received: list[RealtimeEvent] = []
    backend.on_transcript(received.append)
    backend.start()
    if backend._reader is not None:
        backend._reader.join(timeout=2.0)
    backend.close()
    kinds = [e.kind for e in received]
    assert "final_transcript" in kinds
    assert "turn_done" in kinds
    final = next(e for e in received if e.kind == "final_transcript")
    assert final.text == "forty five minutes"
    assert final.confidence == 0.9
    # After the scripted frames drain, the fake's empty recv() models a peer
    # close while we did NOT call close() → a terminal error event is the LAST
    # thing delivered, so callers see the stream ended (not a silent stop).
    assert received[-1].kind == "error"
    assert received[-1].is_final is True
    assert "connection closed" in received[-1].text


def test_dispatch_raw_non_voiceerror_maps_to_error_event(monkeypatch) -> None:
    # AGORA #418 blocker: a mapping bug that raises something OTHER than
    # VoiceError (e.g. a future KeyError on a malformed dict) must surface as
    # an error EVENT — never propagate out of the daemon reader thread and
    # kill it silently.
    backend, _ = _backend_with_fake()
    received: list[RealtimeEvent] = []
    backend.on_transcript(received.append)

    def _boom(_frame: dict) -> RealtimeEvent:
        raise KeyError("unexpected frame shape")

    monkeypatch.setattr(oair, "map_event", _boom)
    backend._dispatch_raw(json.dumps({"type": "anything"}))  # must NOT raise
    assert len(received) == 1
    assert received[0].kind == "error"
    assert "frame mapping failed" in received[0].text


def test_reader_signals_terminal_error_on_transport_drop() -> None:
    # AGORA #418 blocker: recv() raising (transport gone) while we did NOT
    # call close() must surface a terminal error event to the callback — never
    # a silent thread death. This is the contract nexus time_capture /
    # chat_loop rely on to distinguish a dropped connection from graceful close.
    class _DropWebSocket(FakeWebSocket):
        def recv(self) -> str:
            raise OSError("connection reset by peer")

    ws = _DropWebSocket()
    backend = OpenAiRealtimeBackend(
        config=RealtimeConfig(),
        api_key="test-key-not-real",
        ws_factory=lambda url, headers: ws,
    )
    received: list[RealtimeEvent] = []
    backend.on_transcript(received.append)
    backend.start()
    if backend._reader is not None:
        backend._reader.join(timeout=2.0)
    errors = [e for e in received if e.kind == "error"]
    assert errors, "reader must emit a terminal error on transport drop"
    assert errors[-1].is_final is True
    assert "transport error" in errors[-1].text
    backend.close()


def test_start_url_encodes_model_value() -> None:
    # AGORA #418 blocker: the model id flows into the wss URL query string; a
    # value with reserved characters must be percent-encoded so it cannot
    # inject into or malform the URL.
    captured: dict[str, str] = {}

    def _factory(url: str, headers: dict[str, str]) -> FakeWebSocket:
        captured["url"] = url
        return FakeWebSocket()

    backend = OpenAiRealtimeBackend(
        config=RealtimeConfig(model="evil model&x=1 /../inject"),
        api_key="test-key-not-real",
        ws_factory=_factory,
    )
    backend.start()
    backend.close()
    url = captured["url"]
    assert " " not in url           # space → %20
    assert "&x=1" not in url        # & → %26
    assert "/../" not in url        # / → %2F
    assert "evil%20model" in url    # the encoded form is what reaches the URL


# ---------- HAPPI envelope adapters ----------


class _StubEnvelope:
    def __init__(self, flags: dict) -> None:
        self.flags = flags


def test_config_from_envelope_reads_flags() -> None:
    env = _StubEnvelope({
        "voice": "verse",
        "model": "gpt-realtime-mini",
        "system": "Legal time-capture assistant.",
        "turn_detection": "none",
    })
    config = realtime_config_from_envelope(env)
    assert config.voice == "verse"
    assert config.model == "gpt-realtime-mini"
    assert config.system == "Legal time-capture assistant."
    assert config.turn_detection == "none"


def test_config_from_envelope_uses_defaults_for_missing_flags() -> None:
    config = realtime_config_from_envelope(_StubEnvelope({}))
    defaults = RealtimeConfig()
    assert config.voice == defaults.voice
    assert config.model == defaults.model
    assert config.turn_detection == defaults.turn_detection


def test_config_from_real_happi_envelope() -> None:
    """A real happi/1.1 envelope flows through unchanged (parser accepts it)."""
    from lib.hal.happi.envelope import HappiEnvelope
    env = HappiEnvelope.parse({
        "v": "happi/1.1",
        "id": "rt-test-1",
        "cmd": "voice_realtime",
        "args": [],
        "flags": {"voice": "alloy", "system": "Be brief."},
    })
    config = realtime_config_from_envelope(env)
    assert config.voice == "alloy"
    assert config.system == "Be brief."


def test_event_to_envelope_line_transcript_is_delta() -> None:
    line = realtime_event_to_envelope_line(
        RealtimeEvent(kind="final_transcript", text="hello", confidence=0.88, is_final=True)
    )
    payload = json.loads(line)
    assert payload["type"] == "delta"  # matches happi_responder consumer grammar
    assert payload["text"] == "hello"
    assert payload["confidence"] == 0.88
    assert payload["is_final"] is True


def test_event_to_envelope_line_audio() -> None:
    import base64
    line = realtime_event_to_envelope_line(
        RealtimeEvent(kind="audio_delta", audio=b"\x10\x20\x30")
    )
    payload = json.loads(line)
    assert payload["type"] == "audio"
    assert base64.b64decode(payload["audio_b64"]) == b"\x10\x20\x30"


def test_event_to_envelope_line_error() -> None:
    line = realtime_event_to_envelope_line(
        RealtimeEvent(kind="error", text="provider exploded")
    )
    payload = json.loads(line)
    assert payload["type"] == "error"
    assert payload["message"] == "provider exploded"


def test_event_line_consumed_by_happi_responder_collector() -> None:
    """Exogenous-parity anchor: the emitted delta line is consumable by the
    SAME collector happi_responder uses (_collect_response_text), proving the
    realtime stream is a drop-in for the existing HAPPI consumer."""
    # NB: ``lib.hal.voice.__init__`` re-exports the FUNCTION
    # ``happi_responder``, which shadows the submodule as a package
    # attribute — so both ``from ... import happi_responder`` AND
    # ``import lib.hal.voice.happi_responder`` bind the function. Reach the
    # real module object via ``sys.modules``, which is never shadowed.
    import lib.hal.voice  # noqa: F401 — ensures the submodule is imported
    hr_mod = sys.modules["lib.hal.voice.happi_responder"]
    from lib.hal.happi.envelope import HappiEnvelope

    line = realtime_event_to_envelope_line(
        RealtimeEvent(kind="final_transcript", text="drop-in works", confidence=0.9, is_final=True)
    )
    # Drive the collector with a fake _dispatch yielding our line.
    orig = hr_mod._dispatch
    try:
        hr_mod._dispatch = lambda env: iter([line])  # type: ignore
        env = HappiEnvelope.parse({"v": "happi/1.0", "id": "x", "cmd": "chat"})
        text = hr_mod._collect_response_text(env)
    finally:
        hr_mod._dispatch = orig  # type: ignore
    assert text == "drop-in works"  # delta.text was collected verbatim


def test_event_to_envelope_line_unknown_kind_is_error_not_silent() -> None:
    line = realtime_event_to_envelope_line(RealtimeEvent(kind="weird_new_kind"))
    payload = json.loads(line)
    assert payload["type"] == "error"
    assert payload["code"] == "unknown_event_kind"


# ---------- facade ISP guard (mutation: re-adding a method fails) ----------


def test_session_public_surface_is_seven_methods() -> None:
    methods = sorted(
        m for m in dir(RealtimeSession)
        if not m.startswith("_") and callable(getattr(RealtimeSession, m))
    )
    assert methods == [
        "close", "commit_turn", "is_available",
        "on_transcript", "send_audio", "send_text", "start",
    ]


def test_session_delegates_to_backend() -> None:
    ws = FakeWebSocket()
    session = open_realtime(
        config=RealtimeConfig(api_key="x"),
        ws_factory=lambda url, headers: ws,
    )
    session.start()
    session.send_audio(b"\x01")
    session.close()
    types = [json.loads(s).get("type") for s in ws.sent]
    assert "session.update" in types
    assert "input_audio_buffer.append" in types


def test_session_context_manager_starts_and_closes() -> None:
    ws = FakeWebSocket()
    with open_realtime(
        config=RealtimeConfig(api_key="x"),
        ws_factory=lambda url, headers: ws,
    ) as session:
        assert isinstance(session, RealtimeSession)
        session.send_audio(b"\x02")
    assert ws.closed is True  # __exit__ closed it


# ---------- AGORA #418 round-2 hardening ----------


def test_config_from_envelope_reads_api_key_flag() -> None:
    # DIP seam completeness: an envelope-driven caller injects the key via
    # flags.api_key without touching the constructor.
    env = _StubEnvelope({"api_key": "envelope-injected-key", "voice": "verse"})
    config = realtime_config_from_envelope(env)
    assert config.api_key == "envelope-injected-key"
    assert config.voice == "verse"


def test_config_from_envelope_api_key_absent_is_none() -> None:
    # No api_key flag => None (resolve from env/keychain at start), never "None".
    config = realtime_config_from_envelope(_StubEnvelope({}))
    assert config.api_key is None


def test_dispatch_raw_accepts_bytes_frame() -> None:
    # websocket-client may deliver bytes; a bytes JSON frame must map exactly
    # like its str form.
    backend, _ = _backend_with_fake()
    received: list[RealtimeEvent] = []
    backend.on_transcript(received.append)
    backend._dispatch_raw(json.dumps({
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "bytes path", "confidence": 0.5,
    }).encode("utf-8"))
    assert len(received) == 1
    assert received[0].kind == "final_transcript"
    assert received[0].text == "bytes path"


def test_start_rejects_unusable_ws_factory_result() -> None:
    # A ws_factory returning an object missing recv/close must fail at start()
    # with a clean NetworkError, not an opaque AttributeError later.
    from lib.hal.voice.providers.openai_realtime import NetworkError

    class _OnlySend:
        def send(self, payload: str) -> None: ...

    backend = OpenAiRealtimeBackend(
        config=RealtimeConfig(),
        api_key="test-key-not-real",
        ws_factory=lambda url, headers: _OnlySend(),
    )
    with pytest.raises(NetworkError):
        backend.start()
    # validated BEFORE assignment -> no half-open ws is left behind
    assert backend._ws is None


def test_start_clears_ws_on_handshake_failure() -> None:
    # If the session.update handshake send() fails AFTER the ws is built,
    # start() must tear down + clear self._ws so a later close() never touches a
    # half-open connection.
    from lib.hal.voice.providers.openai_realtime import NetworkError

    class _SendFailsWebSocket(FakeWebSocket):
        def send(self, payload: str) -> None:
            raise OSError("handshake send failed")

    ws = _SendFailsWebSocket()
    backend = OpenAiRealtimeBackend(
        config=RealtimeConfig(),
        api_key="test-key-not-real",
        ws_factory=lambda url, headers: ws,
    )
    with pytest.raises(NetworkError):
        backend.start()
    assert backend._ws is None   # cleared on failure
    assert ws.closed is True     # the orphan ws was torn down
    backend.close()              # safe even after a failed start
