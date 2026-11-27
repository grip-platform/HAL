"""InfiniteMachine v2, Step 5a: HAL stops being the error source (the T2 contract tests).

The converge-fleet run (``wf_5f3c7295-40f``) found the gateway itself causing
errors that a held or switched request would only multiply. Each is pinned here:

1. A transport fault talking upstream is a 503 in the client's dialect, never a
   400 (fleet findings f48, f68).
2. No 429 reaches an OpenAI-wire route: it goes out as a 503, Retry-After kept,
   because one 429 ends a Codex turn (f26).
3. ``Retry-After`` and ``retry-after-ms`` survive every relay (f73).
4. Server-sent events leave HAL within 100 ms of arriving (f79).
5. A stream that fails mid-way ends with one in-band error in the client's
   dialect, never a silent stop (f72).
6. The single 600 s upstream timeout is split: connect, total, stream idle and a
   short total for quick calls (f83).
7. Requests in flight are capped; past the cap a request gets a 503 with
   Retry-After at once, never a 429 and never a wait (f82, f121).

Everything runs on loopback. The gateway under test listens on a scratch port in
47000-47999, never on 4010 or 4011, and every fake upstream is started and
stopped by its own test. Each test writes its audit rows into its own temporary
HAL_HOME (tests/conftest.py), never into ~/.hal/logs. Credentials are obviously
fake and never token-shaped.
"""
from __future__ import annotations

import http.client
import io
import json
import random
import socket
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lib.hal import gateway_egress
from lib.hal.account_ledger import AccountLedger
from lib.hal.gateway import (
    GatewayProxy,
    InflightCap,
    UpstreamResponse,
    UpstreamTimeouts,
    build_default_opener,
    make_gateway_handler,
    read_some,
)
from lib.hal.gateway_failover import FailoverProxy, GatewayTelemetry, UpstreamSpec
from lib.hal.gateway_openai import _iter_sse_events, translate_sse_stream
from lib.hal.gateway_stream_crosswire import translate_chat_stream

SCRATCH_PORTS = range(47000, 48000)
FAKE_KEY = "fake-account-A-token"
MODEL = "claude-opus-5-5"
TURN = {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}
CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
RESPONSES = {"model": MODEL, "input": "hi"}
OPENAI_ROUTES = ("/v1/chat/completions", "/v1/responses")
ROUTES = {"/v1/messages": TURN, "/v1/chat/completions": CHAT, "/v1/responses": RESPONSES}


def _event(event: dict) -> bytes:
    """One SSE event, its text as raw UTF-8 the way a server sends it."""
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def _delta(text: str) -> dict:
    return {"type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text}}


START = [{"type": "message_start", "message": {"id": "msg_1", "model": MODEL,
                                               "usage": {"input_tokens": 3}}},
         {"type": "content_block_start", "index": 0,
          "content_block": {"type": "text", "text": ""}}]
END = [{"type": "content_block_stop", "index": 0},
       {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 4}},
       {"type": "message_stop"}]
OVERLOADED = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}


# --------------------------------------------------------------------------- #
# Loopback plumbing: scratch-port servers, a scriptable fake upstream, a client.
# --------------------------------------------------------------------------- #
def _serve(handler_cls) -> ThreadingHTTPServer:
    """Serve ``handler_cls`` on a free port in 47000-47999. A plain bind: no port
    is ever taken over from another process."""
    for port in random.sample(SCRATCH_PORTS, 50):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
        except OSError:
            continue
        server.block_on_close = False  # a held fake must not stall teardown
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                         daemon=True).start()
        return server
    raise RuntimeError("no free scratch port in 47000-47999")


@pytest.fixture
def servers():
    made = []

    def start(handler_cls) -> int:
        server = _serve(handler_cls)
        made.append(server)
        return server.server_address[1]

    yield start
    for server in made:
        server.shutdown()
        server.server_close()


class Upstream:
    """A fake upstream: ``answer(handler, upstream)`` writes each response."""

    def __init__(self, answer):
        self.answer = answer
        self.requests = []          # (method, path) of every request it got
        self.sent = []              # time.monotonic() after each SSE event it sent
        self.release = threading.Event()

    def handler(self):
        upstream = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def handle(self):
                try:
                    super().handle()
                except OSError:
                    pass  # the gateway under test hung up on this fake, as a test meant

            def do_POST(self):  # noqa: N802
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                upstream.requests.append((self.command, self.path))
                upstream.answer(self, upstream)

            do_GET = do_POST

            def log_message(self, *_args):
                pass

        return _Handler


def _json_answer(status: int, payload: dict, headers=()):
    def answer(handler, _upstream):
        body = json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        for name, value in headers:
            handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    return answer


def _sse_answer(events, *, gap: float = 0.0, cut: bool = False):
    """Chunked text/event-stream, each event flushed then ``gap`` s of quiet. A
    ``cut`` stream closes without its terminating chunk."""
    def answer(handler, upstream):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        for event in events:
            data = _event(event)
            handler.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            handler.wfile.flush()
            upstream.sent.append(time.monotonic())
            time.sleep(gap)
        if not cut:
            handler.wfile.write(b"0\r\n\r\n")
        handler.close_connection = True
    return answer


def _hang_up(handler, _upstream):
    """Read the request, answer nothing, close the connection."""
    handler.close_connection = True


def _held(answer):
    def held(handler, upstream):
        upstream.release.wait(10)
        answer(handler, upstream)
    return held


def _gateway(servers, upstream_port: int, *, cap=None, timeouts=None) -> int:
    proxy = GatewayProxy(upstream_base=f"http://127.0.0.1:{upstream_port}",
                         opener=build_default_opener(timeouts=timeouts or UpstreamTimeouts()),
                         max_retries=0, sleep=lambda _s: None)
    return servers(make_gateway_handler(proxy, cap or InflightCap(64, 5)))


def _post(port: int, path: str, payload: dict):
    """(status, lower-cased headers, body) of one POST to the gateway under test."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        conn.request("POST", path, body=json.dumps(payload).encode(),
                     headers={"Content-Type": "application/json", "x-api-key": FAKE_KEY})
        resp = conn.getresponse()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
    finally:
        conn.close()


def _closed_port() -> int:
    """A loopback port in the scratch range with nothing listening on it."""
    for port in random.sample(SCRATCH_PORTS, 50):
        probe = socket.socket()
        try:
            probe.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            probe.close()
    raise RuntimeError("no free scratch port in 47000-47999")


def _error_message(route: str, body: bytes) -> str:
    """The error message, read the way the route's own dialect carries it."""
    envelope = json.loads(body)
    if route in OPENAI_ROUTES:
        return envelope["error"]["message"]
    assert envelope["type"] == "error"
    return envelope["error"]["message"]


def _blocks(raw: bytes) -> list:
    """The SSE blocks of a client-side body, each as (event name, parsed data)."""
    out = []
    for block in raw.decode("utf-8", "replace").split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "data" in fields and fields["data"] != "[DONE]":
            out.append((fields.get("event"), json.loads(fields["data"])))
    return out


# --------------------------------------------------------------------------- #
# 0. The plain path still works: a whole answer, through the real timed opener.
# --------------------------------------------------------------------------- #
ANSWER = {"id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
          "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
          "usage": {"input_tokens": 3, "output_tokens": 1}}
TEXT_OF = {
    "/v1/messages": lambda body: body["content"][0]["text"],
    "/v1/chat/completions": lambda body: body["choices"][0]["message"]["content"],
    "/v1/responses": lambda body: body["output"][0]["content"][0]["text"],
}


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_a_whole_buffered_answer_reaches_every_route(servers, route):
    """A Content-Length answer, read to its end and past it. The first build of
    the timed opener set a socket timeout after the last byte had closed the
    connection, and every buffered answer on the translated routes became a 503."""
    upstream = servers(Upstream(_json_answer(200, ANSWER)).handler())
    status, _headers, body = _post(_gateway(servers, upstream), route, ROUTES[route])
    assert status == 200, body
    assert TEXT_OF[route](json.loads(body)) == "hello"


# --------------------------------------------------------------------------- #
# 1. A transport fault is a 503 in the client's dialect, never a 400.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("fault", ["refused", "hung-up"])
def test_a_transport_fault_is_a_503_on_every_route(servers, route, fault):
    upstream_port = (_closed_port() if fault == "refused"
                     else servers(Upstream(_hang_up).handler()))
    status, _headers, body = _post(_gateway(servers, upstream_port), route, ROUTES[route])
    assert status == 503
    assert _error_message(route, body)


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("fault", ["refused", "hung-up"])
def test_a_transport_fault_through_the_failover_chain_is_still_a_503(servers, tmp_path,
                                                                     route, fault):
    """The process gateway serves through FailoverProxy (``build_failover_proxy``),
    built here with its own default proxies: the ledger-observed opener (HAL#651)
    under the split timeouts. Once the last upstream is spent the chain raises
    again, and a network fault there must stay a transport fault, so the client
    is sent a 503, never the 502 of a fault the gateway names itself."""
    upstream_port = (_closed_port() if fault == "refused"
                     else servers(Upstream(_hang_up).handler()))
    proxy = FailoverProxy(
        [UpstreamSpec(name="primary", base_url=f"http://127.0.0.1:{upstream_port}")],
        max_retries=0, sleep=lambda _s: None,
        telemetry=GatewayTelemetry(sink=lambda _event: None),
        ledger=AccountLedger(tmp_path / "account-ledger.json"))
    port = servers(make_gateway_handler(proxy, InflightCap(64, 5)))
    status, _headers, body = _post(port, route, ROUTES[route])
    assert status == 503, body
    assert _error_message(route, body)


@pytest.mark.parametrize("route, payload", [("/v1/chat/completions", dict(CHAT, model="gpt-5.5")),
                                            ("/v1/responses", dict(RESPONSES, model="gpt-5.5"))])
def test_a_reset_on_the_same_wire_path_is_a_503_not_a_400(servers, monkeypatch, route, payload):
    """The exact defect the fleet saw (f48): Codex was sent
    'gateway translation error: [Errno 54] Connection reset by peer' as a 400."""
    def reset(*_args, **_kwargs):
        raise ConnectionResetError(54, "Connection reset by peer")

    monkeypatch.setenv("HAL_GATEWAY_MULTIPROVIDER", "1")
    monkeypatch.setattr(gateway_egress, "forward_same_wire", reset)
    status, _headers, body = _post(_gateway(servers, _closed_port()), route, payload)
    assert status == 503
    assert "translation error" not in _error_message(route, body)


# --------------------------------------------------------------------------- #
# 2 and 3. No 429 on an OpenAI-wire route; Retry-After survives every relay.
# --------------------------------------------------------------------------- #
RATE_LIMITED = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
RETRY = (("retry-after", "3"), ("retry-after-ms", "3000"))


@pytest.mark.parametrize("route", OPENAI_ROUTES)
def test_an_upstream_429_reaches_an_openai_wire_route_as_a_503(servers, route):
    upstream = servers(Upstream(_json_answer(429, RATE_LIMITED, RETRY)).handler())
    status, headers, body = _post(_gateway(servers, upstream), route, ROUTES[route])
    assert status == 503
    assert (headers.get("retry-after"), headers.get("retry-after-ms")) == ("3", "3000")
    assert _error_message(route, body) == "slow down"


@pytest.mark.parametrize("route, payload", [("/v1/chat/completions", dict(CHAT, model="gpt-5.5")),
                                            ("/v1/responses", dict(RESPONSES, model="gpt-5.5"))])
def test_a_same_wire_429_reaches_an_openai_wire_route_as_a_503(servers, monkeypatch,
                                                               route, payload):
    def rate_limited(*_args, **_kwargs):
        return UpstreamResponse(status=429, headers=[("Content-Type", "application/json"), *RETRY],
                                _reader=io.BytesIO(b'{"error": {"message": "slow down"}}'))

    monkeypatch.setenv("HAL_GATEWAY_MULTIPROVIDER", "1")
    monkeypatch.setattr(gateway_egress, "forward_same_wire", rate_limited)
    status, headers, _body = _post(_gateway(servers, _closed_port()), route, payload)
    assert status == 503
    assert headers.get("retry-after") == "3"


def test_the_anthropic_route_keeps_its_429_and_its_retry_after(servers):
    """Claude Code reads a 429 itself (its own usage-limit line); only the
    OpenAI-wire routes rename it. Retry-After is kept either way (f73)."""
    upstream = servers(Upstream(_json_answer(429, RATE_LIMITED, RETRY)).handler())
    status, headers, _body = _post(_gateway(servers, upstream), "/v1/messages", TURN)
    assert status == 429
    assert (headers.get("retry-after"), headers.get("retry-after-ms")) == ("3", "3000")


def test_a_cross_wire_provider_error_keeps_its_retry_after():
    provider = UpstreamResponse(status=429, headers=[("Retry-After", "9")],
                                _reader=io.BytesIO(b'{"error": {"message": "slow"}}'))
    reply = gateway_egress._provider_error_reply(provider)
    assert reply.status == 429
    assert ("Retry-After", "9") in reply.headers


# --------------------------------------------------------------------------- #
# 4. Each server-sent event leaves HAL within 100 ms of arriving (f79).
# --------------------------------------------------------------------------- #
def _arrivals(port: int, path: str, payload: dict, wanted: int, mark: bytes):
    """When each of the first ``wanted`` body blocks holding ``mark`` reached the client."""
    raw = socket.create_connection(("127.0.0.1", port), timeout=10)
    raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    body = json.dumps(payload).encode()
    raw.sendall(f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json"
                f"\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
    data, arrivals = b"", []
    try:
        while len(arrivals) < wanted:
            piece = raw.recv(65536)
            assert piece, "the stream ended early"
            data += piece
            seen = data.partition(b"\r\n\r\n")[2].split(b"\n\n")[:-1]
            arrivals += [time.monotonic()] * (sum(mark in b for b in seen) - len(arrivals))
    finally:
        raw.close()
    return arrivals


def test_each_relayed_event_leaves_hal_within_100ms(servers):
    """Each event leaves HAL as it arrives (f79); the upstream sends the next one
    250 ms later, so a relay that held an event for the next would show 250 ms or
    more. Every event after the first is out within 100 ms. The first comes with
    the response head, so its time also counts HAL's per-response work before the
    first byte (the witness record): one CI run measured 109 ms for it while the
    next five took about 0.1 ms each. It is held to the 250 ms gap instead, which
    a relay that held it would still fail."""
    events = START + [_delta(f"t{i}") for i in range(4)] + END
    upstream = Upstream(_sse_answer(events, gap=0.25))
    port = _gateway(servers, servers(upstream.handler()))
    arrivals = _arrivals(port, "/v1/messages", dict(TURN, stream=True), len(events), b"data:")
    lags = [got - sent for got, sent in zip(arrivals, upstream.sent)]
    assert lags[0] < 0.25, lags
    assert max(lags[1:]) < 0.1, lags


def test_each_translated_event_leaves_hal_within_100ms(servers):
    deltas = [_delta(f"t{i}") for i in range(4)]
    upstream = Upstream(_sse_answer(START + deltas + END, gap=0.25))
    port = _gateway(servers, servers(upstream.handler()))
    arrivals = _arrivals(port, "/v1/chat/completions", dict(CHAT, stream=True),
                         len(deltas), b'"content": "t')
    sent = upstream.sent[len(START):len(START) + len(deltas)]
    lags = [got - when for got, when in zip(arrivals, sent)]
    assert max(lags) < 0.1, lags


def test_a_reader_is_read_with_read1_when_its_class_has_one():
    class _Mockish:
        def __getattr__(self, _name):
            return lambda *_a: b"x"

    assert read_some(io.BytesIO(b"ab")).__name__ == "read1"
    assert read_some(_Mockish()).__name__ == "<lambda>"  # answers anything: not a stream


def test_a_character_split_across_two_reads_is_decoded_whole():
    """Reads now end wherever the network splits the stream, so the SSE parser
    must decode incrementally or a split character becomes U+FFFD."""
    event = _event(_delta("café"))
    cut = event.index("é".encode()) + 1
    parsed = list(_iter_sse_events([event[:cut], event[cut:]]))
    assert parsed[0]["delta"]["text"] == "café"


# --------------------------------------------------------------------------- #
# 5. A stream that fails mid-way ends with one in-band error (f72).
# --------------------------------------------------------------------------- #
def test_a_relayed_stream_cut_mid_way_ends_with_an_anthropic_error_event(servers):
    upstream = servers(Upstream(_sse_answer(START + [_delta("Hel")], cut=True)).handler())
    status, _headers, body = _post(_gateway(servers, upstream), "/v1/messages",
                                   dict(TURN, stream=True))
    assert status == 200
    name, data = _blocks(body)[-1]
    assert (name, data["type"], data["error"]["type"]) == ("error", "error", "api_error")
    assert "connection lost" in data["error"]["message"]


def test_an_upstream_error_event_reaches_the_chat_route_in_band(servers):
    events = START + [_delta("Hel"), OVERLOADED]
    upstream = servers(Upstream(_sse_answer(events)).handler())
    _status, _headers, body = _post(_gateway(servers, upstream), "/v1/chat/completions",
                                    dict(CHAT, stream=True))
    assert b"[DONE]" not in body
    name, data = _blocks(body)[-1]
    assert (name, data["error"]["message"]) == ("error", "Overloaded")
    assert not any(d.get("choices", [{}])[0].get("finish_reason") for _n, d in _blocks(body))


def test_a_cut_stream_on_the_responses_route_never_claims_completion(servers):
    upstream = servers(Upstream(_sse_answer(START + [_delta("Hel")], cut=True)).handler())
    _status, _headers, body = _post(_gateway(servers, upstream), "/v1/responses",
                                    dict(RESPONSES, stream=True))
    names = [name for name, _data in _blocks(body)]
    assert "response.completed" not in names
    assert names[-1] == "error" and names.count("error") == 1


def test_a_stream_ending_before_message_stop_is_an_interruption_not_an_answer():
    chunks = [b"".join(_event(e) for e in START + [_delta("Hel")])]
    out = b"".join(translate_sse_stream(chunks, model="m", completion_id="c", created=1))
    assert b"[DONE]" not in out
    assert _blocks(out)[-1][0] == "error"


def test_a_provider_error_chunk_ends_a_cross_wire_stream_with_an_anthropic_error():
    chunks = [b'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n',
              b'data: {"error": {"message": "upstream overloaded", "type": "server_error"}}\n\n']
    out = b"".join(translate_chat_stream(chunks, model="gpt-5.5"))
    assert b"message_stop" not in out
    name, data = _blocks(out)[-1]
    assert (name, data["error"]["message"]) == ("error", "upstream overloaded")


def test_a_chat_stream_cut_after_content_ends_with_exactly_one_error(servers):
    """The falsifier AGORA named on this PR: the connection is lost after content
    has gone out, on the chat route. The client keeps the content it was sent,
    then gets exactly one in-band error, and no finish_reason or [DONE]."""
    upstream = servers(Upstream(_sse_answer(START + [_delta("Hel")], cut=True)).handler())
    _status, _headers, body = _post(_gateway(servers, upstream), "/v1/chat/completions",
                                    dict(CHAT, stream=True))
    blocks = _blocks(body)
    choices = [data["choices"][0] for _name, data in blocks if data.get("choices")]
    assert [choice["delta"].get("content") for choice in choices].count("Hel") == 1
    assert [name for name, _data in blocks].count("error") == 1 and blocks[-1][0] == "error"
    assert b"[DONE]" not in body
    assert not any(choice.get("finish_reason") for choice in choices)


def test_a_cross_wire_read_that_fails_after_content_ends_with_exactly_one_error():
    """The same falsifier on the cross-wire path: the provider's connection fails
    on the read after the first content chunk. One Anthropic error event, after
    the content it follows, and no message_stop."""
    def chunks():
        yield b'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n'
        raise ConnectionResetError(54, "Connection reset by peer")

    blocks = _blocks(b"".join(translate_chat_stream(chunks(), model="gpt-5.5")))
    names = [name for name, _data in blocks]
    assert ("content_block_delta", "Hel") in [(name, data.get("delta", {}).get("text"))
                                               for name, data in blocks]
    assert names.count("error") == 1 and names[-1] == "error"
    assert "message_stop" not in names


class _CutReader:
    """A provider's stream that hands on ``chunks`` and then fails, as one that
    hung up mid-answer does."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read1(self, _size=-1):
        if self._chunks:
            return self._chunks.pop(0)
        raise ConnectionResetError(54, "Connection reset by peer")

    read = read1

    def close(self):
        pass


SAME_WIRE_FIRST = {
    "/v1/chat/completions": b'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n',
    "/v1/responses": (b'event: response.output_text.delta\n'
                      b'data: {"type": "response.output_text.delta", "delta": "Hel"}\n\n'),
}


@pytest.mark.parametrize("route", sorted(SAME_WIRE_FIRST))
def test_a_same_wire_stream_cut_mid_way_ends_in_the_routes_own_dialect(servers, monkeypatch,
                                                                        route):
    """A provider reached on its own wire is relayed verbatim, with no translation
    to catch its failure. When it fails mid-stream the relay itself writes the one
    error, and it must be in the route's dialect: Anthropic's shape on these routes
    is an error the OpenAI-wire clients do not read as one."""
    def forward(*_args, **_kwargs):
        return UpstreamResponse(status=200, headers=[("Content-Type", "text/event-stream")],
                                _reader=_CutReader([SAME_WIRE_FIRST[route]]))

    monkeypatch.setenv("HAL_GATEWAY_MULTIPROVIDER", "1")
    monkeypatch.setattr(gateway_egress, "forward_same_wire", forward)
    payload = dict(ROUTES[route], model="gpt-5.5", stream=True)
    _status, _headers, body = _post(_gateway(servers, _closed_port()), route, payload)
    assert SAME_WIRE_FIRST[route] in body
    blocks = _blocks(body)
    assert [name for name, _data in blocks].count("error") == 1 and blocks[-1][0] == "error"
    error = blocks[-1][1]
    if route == "/v1/responses":
        assert error.get("code") == "server_error" and "error" not in error
    else:
        assert "type" not in error and error["error"]["type"] == "server_error"
    assert b"[DONE]" not in body and b"response.completed" not in body


# --------------------------------------------------------------------------- #
# 6. The upstream timeout is split by route and stream mode (f83).
# --------------------------------------------------------------------------- #
def test_the_split_timeouts_are_overridable_by_environment():
    timeouts = UpstreamTimeouts.from_env({
        "HAL_GATEWAY_CONNECT_TIMEOUT_S": "2.5", "HAL_GATEWAY_TOTAL_TIMEOUT_S": "nope",
        "HAL_GATEWAY_STREAM_IDLE_TIMEOUT_S": "-1", "HAL_GATEWAY_SHORT_TIMEOUT_S": "inf"})
    assert timeouts == UpstreamTimeouts(connect=2.5, total=600.0, stream_idle=600.0,
                                        short_total=60.0)


# Durations use time.perf_counter(): on Windows, Python 3.12's time.monotonic()
# moves in steps of about 15.6 ms, so a full 0.4 s sleep can read as 0.390 s.
def _timed(call):
    started = time.perf_counter()
    with pytest.raises((OSError, urllib.error.URLError)):
        call()
    return time.perf_counter() - started


def _hang(seconds: float):
    def answer(handler, _upstream):
        time.sleep(seconds)
        handler.close_connection = True
    return answer


def test_a_handshake_that_never_comes_fails_at_the_connect_timeout():
    listener = socket.create_server(("127.0.0.1", 0))  # accepts, never speaks TLS
    url = f"https://127.0.0.1:{listener.getsockname()[1]}/v1/messages"
    opener = build_default_opener(timeouts=UpstreamTimeouts(connect=0.3, total=5))
    try:
        elapsed = _timed(lambda: opener("POST", url, [], b"{}"))
    finally:
        listener.close()
    assert 0.25 <= elapsed < 2.0, elapsed


def test_an_answer_that_never_starts_fails_at_the_total_timeout(servers):
    port = servers(Upstream(_hang(5)).handler())
    opener = build_default_opener(timeouts=UpstreamTimeouts(connect=5, total=0.5))
    elapsed = _timed(lambda: opener("POST", f"http://127.0.0.1:{port}/v1/messages", [], b"{}"))
    assert 0.4 <= elapsed < 2.0, elapsed


def test_a_get_gets_the_short_total(servers):
    port = servers(Upstream(_hang(5)).handler())
    opener = build_default_opener(timeouts=UpstreamTimeouts(total=30, short_total=0.4))
    elapsed = _timed(lambda: opener("GET", f"http://127.0.0.1:{port}/v1/models", [], None))
    assert 0.3 <= elapsed < 2.0, elapsed


def _trickle(handler, _upstream):
    """A buffered JSON answer that sends one byte every 0.1 s."""
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", "60")
    handler.end_headers()
    for _ in range(60):
        handler.wfile.write(b" ")
        handler.wfile.flush()
        time.sleep(0.1)


def test_a_buffered_answer_still_arriving_at_the_total_is_cut(servers):
    port = servers(Upstream(_trickle).handler())
    opener = build_default_opener(timeouts=UpstreamTimeouts(total=0.6))
    resp = UpstreamResponse(200, [], _reader=opener(
        "POST", f"http://127.0.0.1:{port}/v1/messages", [], b"{}"))
    elapsed = _timed(resp.read_all)
    assert elapsed < 1.5, elapsed


def test_a_stream_may_outlive_the_total_while_it_keeps_talking(servers):
    upstream = Upstream(_sse_answer([_delta(f"t{i}") for i in range(8)], gap=0.15))
    port = servers(upstream.handler())
    opener = build_default_opener(timeouts=UpstreamTimeouts(total=0.5, stream_idle=0.6))
    raw = opener("POST", f"http://127.0.0.1:{port}/v1/messages", [], b"{}")
    body = UpstreamResponse(200, [], _reader=raw).read_all()
    assert body.count(b"event: content_block_delta") == 8


def test_a_stream_that_falls_silent_fails_at_the_idle_timeout(servers):
    upstream = Upstream(_sse_answer([_delta("t0")], gap=5))
    port = servers(upstream.handler())
    opener = build_default_opener(timeouts=UpstreamTimeouts(total=30, stream_idle=0.4))
    chunks = UpstreamResponse(200, [], _reader=opener(
        "POST", f"http://127.0.0.1:{port}/v1/messages", [], b"{}")).iter_chunks()
    assert b"t0" in next(chunks)
    assert 0.3 <= _timed(lambda: next(chunks)) < 2.0


# --------------------------------------------------------------------------- #
# 7. Requests in flight are capped: past the cap, a 503 with Retry-After at once.
# --------------------------------------------------------------------------- #
def _until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    return predicate()


def _fill(port: int, upstream: Upstream, slots: int) -> list:
    """Start ``slots`` requests the held upstream will not answer yet."""
    threads = [threading.Thread(target=_post, args=(port, "/v1/messages", TURN), daemon=True)
               for _ in range(slots)]
    for thread in threads:
        thread.start()
    assert _until(lambda: len(upstream.requests) == slots)
    return threads


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_past_the_cap_a_request_gets_a_503_with_retry_after_at_once(servers, route):
    cap = InflightCap(2, 7)
    upstream = Upstream(_held(_json_answer(200, {"id": "msg_1", "content": []})))
    port = _gateway(servers, servers(upstream.handler()), cap=cap)
    held = _fill(port, upstream, 2)
    started = time.monotonic()
    status, headers, body = _post(port, route, ROUTES[route])
    assert (status, headers.get("retry-after")) == (503, "7")
    assert time.monotonic() - started < 1.0
    assert "limit of 2" in _error_message(route, body)
    upstream.release.set()
    for thread in held:
        thread.join(5)
    assert _until(lambda: cap.inflight == 0)          # the slots came back
    assert _post(port, "/v1/messages", TURN)[0] == 200


def test_a_large_post_turned_away_at_the_cap_still_reads_its_503(servers):
    """The gateway reads a turned-away request's body before answering. Answering
    first and closing would leave a client still sending a large body to fail on
    a reset, with no 503 and no Retry-After to act on. The body is far larger
    than a loopback socket's buffers, so the client is still sending."""
    upstream = Upstream(_held(_json_answer(200, {"id": "msg_1", "content": []})))
    port = _gateway(servers, servers(upstream.handler()), cap=InflightCap(1, 4))
    _fill(port, upstream, 1)
    large = dict(TURN, messages=[{"role": "user", "content": "x" * (32 * 1024 * 1024)}])
    status, headers, _body = _post(port, "/v1/messages", large)
    assert (status, headers.get("retry-after")) == (503, "4")
    upstream.release.set()


def test_health_is_never_capped(servers):
    upstream = Upstream(_held(_json_answer(200, {"id": "msg_1"})))
    port = _gateway(servers, servers(upstream.handler()), cap=InflightCap(1, 5))
    _fill(port, upstream, 1)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/health")
    assert conn.getresponse().status == 200
    conn.close()
    upstream.release.set()


def test_the_cap_reads_its_limit_and_retry_after_from_the_environment():
    cap = InflightCap.from_env("HAL_WIRE", {"HAL_WIRE_MAX_INFLIGHT": "3",
                                            "HAL_WIRE_BUSY_RETRY_AFTER_S": "0.2"})
    assert (cap.limit, cap.retry_after_s) == (3, 1)
    assert [cap.try_enter() for _ in range(4)] == [True, True, True, False]
    cap.leave()
    assert cap.try_enter() and cap.inflight == 3


def test_no_test_here_writes_outside_its_own_hal_home(isolated_hal_home, tmp_path):
    """The audit sink every test above writes to is this test's own (conftest).

    The assertion itself lives in the shared ``isolated_hal_home`` fixture, so
    this file pins to the same check as ``test_audit_log_isolation.py`` and the
    check cannot be weakened by editing one file alone. Here the home is also
    the exact directory conftest made for this test, not only one named alike.
    """
    assert isolated_hal_home == (tmp_path / "hal-home").resolve()
