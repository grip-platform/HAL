"""The HAL wire proxy (InfiniteMachine v2, Step 2), against local fake Anthropic hosts.

Everything runs on loopback; nothing reaches a real provider. Each fake serves TLS
with a certificate from a test-only CA, and the wire is told to trust that CA for
its upstream connections, so the wire's upstream verification is real. The test
client trusts the wire's own CA, as Claude Code does through NODE_EXTRA_CA_CERTS.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import plistlib
import random
import select
import socket
import ssl
import statistics
import struct
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._public_tree import require

pytest.importorskip("cryptography")

from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402

from lib.hal import wire_ca, wire_proxy  # noqa: E402
from lib.hal.gateway import InflightCap  # noqa: E402
from lib.hal.gateway_failover import GatewayTelemetry  # noqa: E402
from lib.hal.gateway_witness import build_chat_record, witness_chat  # noqa: E402


@pytest.fixture(autouse=True)
def _healthy_attempts(monkeypatch):
    from lib.hal.wire_cascade import Attempts
    monkeypatch.setattr(Attempts, '_unhealthy', False, raising=False)

REPO = Path(__file__).resolve().parents[1]
API = "api.anthropic.com"
BEARER = "Bearer wire-test-bearer-0000"
API_KEY = "wire-test-api-key-0000"
EXACT_REPLY = (b"HTTP/1.1 200 OK\r\nX-Reply:  kept as sent\r\ncontent-type: application/json"
               b"\r\ncontent-length: 17\r\n\r\n{\"ok\": \"exactly\"}")
WS_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
TURN = {"model": "claude-opus-5-5", "max_tokens": 64000,
        "messages": [{"role": "user", "content": "hello"}]}
QUOTA = {"model": "claude-fable-5-1", "max_tokens": 1,
         "messages": [{"role": "user", "content": "quota"}]}
SSE_EVENTS = [
    {"type": "message_start", "message": {
        "id": "msg_1", "model": "claude-opus-5-5",
        "usage": {"input_tokens": 21, "cache_read_input_tokens": 5,
                  "cache_creation_input_tokens": 3, "output_tokens": 1}}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 9}},
    {"type": "message_stop"},
]


# --------------------------------------------------------------------------- #
# Test-only PKI standing in for the real Anthropic certificate chain.
# --------------------------------------------------------------------------- #
def _cert(subject: str, issuer_key, issuer_name, key, ca: bool, hosts=()):
    now = wire_ca._utcnow()
    builder = (x509.CertificateBuilder()
               .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)]))
               .issuer_name(issuer_name).public_key(key.public_key())
               .serial_number(x509.random_serial_number())
               .not_valid_before(now - timedelta(minutes=5))
               .not_valid_after(now + timedelta(days=1))
               .add_extension(x509.BasicConstraints(ca=ca, path_length=None), True))
    if hosts:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(h) for h in hosts]), False)
    return builder.sign(issuer_key, hashes.SHA256())


class _TestPki:
    """A CA only these tests trust, and server contexts for the fake hosts."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "wire test upstream CA")])
        self.cert = _cert("wire test upstream CA", self.key, name, self.key, ca=True)

    def pem(self) -> str:
        return self.cert.public_bytes(serialization.Encoding.PEM).decode()

    def server_context(self, host: str) -> ssl.SSLContext:
        key = ec.generate_private_key(ec.SECP256R1())
        leaf = _cert(host, self.key, self.cert.subject, key, ca=False, hosts=[host])
        cert_path, key_path = self.folder / f"{host}.pem", self.folder / f"{host}.key"
        cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(wire_ca._key_pem(key))
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        return context

    def client_context(self) -> ssl.SSLContext:
        return _trusting(self.pem())


def _trusting(ca_pem: str) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.set_alpn_protocols(["http/1.1"])
    context.load_verify_locations(cadata=ca_pem)
    return context


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    return _TestPki(tmp_path_factory.mktemp("wire-pki"))


# --------------------------------------------------------------------------- #
# Byte-level helpers, independent of the code under test.
# --------------------------------------------------------------------------- #
class _Reader:
    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def _more(self) -> bool:
        data = self.sock.recv(65536)
        self.buf += data
        return bool(data)

    def until(self, marker: bytes) -> bytes:
        while marker not in self.buf:
            if not self._more():
                return b""
        head, self.buf = self.buf.split(marker, 1)
        return head + marker

    def exact(self, count: int) -> bytes:
        while len(self.buf) < count and self._more():
            pass
        out, self.buf = self.buf[:count], self.buf[count:]
        return out


def _header_int(head: bytes, name: bytes) -> int:
    lines = head.lower().split(b"\r\n")
    values = [line.split(b":", 1)[1].strip() for line in lines if line.startswith(name + b":")]
    return int(values[0]) if values else 0


def _read_response(reader: _Reader) -> bytes:
    head = reader.until(b"\r\n\r\n")
    if b"transfer-encoding: chunked" in head.lower():
        return head + reader.until(b"0\r\n\r\n")
    return head + reader.exact(_header_int(head, b"content-length"))


def _post(path: str, payload: dict, extra: str = "") -> bytes:
    body = json.dumps(payload).encode()
    return (f"POST {path} HTTP/1.1\r\nHost: {API}\r\nAuthorization: {BEARER}\r\n"
            f"x-api-key: {API_KEY}\r\nContent-Type: application/json\r\n{extra}"
            f"Content-Length: {len(body)}\r\n\r\n").encode() + body


def _get(path: str) -> bytes:
    return (f"GET {path} HTTP/1.1\r\nHost: {API}\r\nAuthorization: {BEARER}\r\n"
            f"x-api-key: {API_KEY}\r\n\r\n").encode()


def _loopback(port: int):
    """A client socket to 127.0.0.1:``port`` that sends each write at once."""
    raw = socket.create_connection(("127.0.0.1", port), timeout=10)
    raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return raw


def _quiet_close(sock) -> None:
    try:
        sock.close()
    except OSError:
        pass


def _wait_for(predicate, timeout: float = 3.0):
    """Poll ``predicate``; a log line caught mid-write (ValueError) counts as not yet."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = predicate()
        except ValueError:
            value = None
        if value:
            return value
        time.sleep(0.02)
    return predicate()


def _inside_an_exchange(frame) -> bool:
    """True when ``frame``'s stack is inside the wire's own ``_exchange`` (not this
    file's client helper of the same name)."""
    while frame is not None:
        code = frame.f_code
        if ((code.co_name == "_exchange" and Path(code.co_filename).name == "wire_proxy.py")
                or (code.co_name in ("prepare", "consume_fallback")
                    and Path(code.co_filename).name == "wire_hold.py")):
            return True
        frame = frame.f_back
    return False


def _exchanges_running() -> bool:
    import sys  # only this helper reads other threads' frames
    me = threading.get_ident()
    return any(_inside_an_exchange(frame) for ident, frame in sys._current_frames().items()
               if ident != me)


def wait_for_exchanges(timeout: float = 5.0) -> None:
    """Wait until no thread is still relaying or recording a wire exchange.

    A test can end while its wire still relays a stream or writes a witness row,
    because the wire's connection threads are daemons nobody joins. A row written
    after the test's HAL_HOME is restored lands in the next test's, or -- between
    two tests -- in the HAL_HOME the job started with. Seen 2026-10-05 in CI: the
    wire job failed its "no test wrote into the HAL_HOME" step with every test
    green, and the same leftover threads show on main (an SSE exchange still
    relaying, and a witness still being written, after their tests' teardown).
    """
    deadline = time.monotonic() + timeout
    while _exchanges_running() and time.monotonic() < deadline:
        time.sleep(0.02)


# --------------------------------------------------------------------------- #
# A fake Anthropic host: records every request's raw bytes, answers by path.
# --------------------------------------------------------------------------- #
def _bearer_of(head: bytes) -> str:
    for line in head.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"authorization" and value.strip()[:7].lower() == b"bearer ":
            return value.strip()[7:].decode("latin-1")
    return ""


class FakeUpstream:
    def __init__(self, context: ssl.SSLContext, accept_delay: float = 0.0,
                 event_gap: float = 0.15):
        self.context = context
        self.accept_delay = accept_delay
        self.event_gap = event_gap
        self.requests: list = []        # (connection number, raw request bytes)
        self.sent_events: list = []     # time.monotonic() after each SSE event was sent
        self.release = threading.Event()  # a "hold" turn is answered once this is set
        self.connections = 0
        self.reply_headers = b""        # extra header lines on each JSON reply
        self.orgs = {}                 # fake bearer -> organisation, for free models probes
        self.limit_bearers = set()      # bearers whose turns get a usage-limit 429
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self) -> None:
        _quiet_close(self._listener)

    def _accept(self) -> None:
        while True:
            try:
                raw, _ = self._listener.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._serve, args=(raw, self.connections),
                             daemon=True).start()

    def _serve(self, raw, number: int) -> None:
        time.sleep(self.accept_delay)          # stands in for a WAN handshake
        raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            tls = self.context.wrap_socket(raw, server_side=True)
            reader = _Reader(tls)
            while self._one(tls, reader, number):
                pass
        except (OSError, ssl.SSLError, ValueError):
            pass
        finally:
            _quiet_close(raw)

    def _one(self, tls, reader: _Reader, number: int) -> bool:
        head = reader.until(b"\r\n\r\n")
        if not head:
            return False
        body = reader.exact(_header_int(head, b"content-length"))
        self.requests.append((number, head + body))
        path = head.split(b" ", 2)[1].split(b"?")[0]
        if path == b"/api/ws/speech_to_text/voice_stream":
            return self._websocket(tls, reader, head)
        if path == b"/v1/models":
            org = self.orgs.get(_bearer_of(head), "").encode()
            tls.sendall(b"HTTP/1.1 200 OK\r\nanthropic-organization-id: " + org
                        + b"\r\ncontent-length: 2\r\n\r\n{}")
            return True
        if path == b"/v1/messages" and _bearer_of(head) in self.limit_bearers:
            return self._limit(tls)
        if path == b"/v1/messages":
            return self._messages(tls, json.loads(body))
        if path == b"/v1/messages/count_tokens":
            return self._json(tls, {"input_tokens": 42})
        tls.sendall(EXACT_REPLY)
        return True

    @staticmethod
    def _limit(tls) -> bool:
        body = b'{"type":"error","error":{"type":"rate_limit_error","message":"limit"}}'
        tls.sendall(b"HTTP/1.1 429 Too Many Requests\r\ncontent-type: application/json\r\n"
                    b"anthropic-ratelimit-unified-status: rejected\r\n"
                    b"anthropic-ratelimit-unified-overage-status: rejected\r\n"
                    + f"content-length: {len(body)}\r\n\r\n".encode() + body)
        return True

    def _json(self, tls, payload: dict) -> bool:
        body = json.dumps(payload).encode()
        tls.sendall(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n" + self.reply_headers
                    + f"content-length: {len(body)}\r\n\r\n".encode() + body)
        return True

    def _messages(self, tls, payload: dict) -> bool:
        if payload.get("model") == "hangup":   # read the request, answer nothing, close
            return False
        if payload.get("model") == "hold":
            self.release.wait(10)
        if payload.get("model") in ("disconnect", "reset"):
            tls.sendall(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                        b"transfer-encoding: chunked\r\n\r\n" + _chunk(SSE_EVENTS[0]))
            if payload["model"] == "reset":    # an RST, once the first event is read
                time.sleep(0.2)
                tls.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                tls.close()
            return False                       # close mid-stream, no terminating chunk
        if payload.get("stream"):
            return self._sse(tls)
        return self._json(tls, {"id": "msg_2", "model": "claude-opus-5-5",
                                "usage": {"input_tokens": 11, "output_tokens": 2,
                                          "cache_read_input_tokens": 0,
                                          "cache_creation_input_tokens": 0}})

    def _sse(self, tls) -> bool:
        tls.sendall(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                    b"transfer-encoding: chunked\r\n\r\n")
        for event in SSE_EVENTS:
            tls.sendall(_chunk(event))
            self.sent_events.append(time.monotonic())
            time.sleep(self.event_gap)
        tls.sendall(b"0\r\n\r\n")
        return True

    @staticmethod
    def _websocket(tls, reader: _Reader, head: bytes) -> bool:
        key = [line.split(b":", 1)[1].strip() for line in head.split(b"\r\n")
               if line.lower().startswith(b"sec-websocket-key:")][0]
        accept = base64.b64encode(hashlib.sha1(key + WS_GUID).digest())
        tls.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                    b"Connection: Upgrade\r\nSec-WebSocket-Accept: " + accept + b"\r\n\r\n")
        while True:
            payload = _ws_read_masked(reader)
            if payload is None:
                return False
            tls.sendall(bytes([0x81, len(payload)]) + payload)


def _chunk(event: dict) -> bytes:
    data = f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


def _ws_frame(payload: bytes) -> bytes:
    mask = os.urandom(4)
    masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    return bytes([0x81, 0x80 | len(payload)]) + mask + masked


def _ws_read_masked(reader: _Reader):
    header = reader.exact(2)
    if len(header) < 2:
        return None
    mask, data = reader.exact(4), reader.exact(header[1] & 0x7F)
    return bytes(byte ^ mask[i % 4] for i, byte in enumerate(data))


def _ws_read(reader: _Reader) -> bytes:
    header = reader.exact(2)
    return reader.exact(header[1] & 0x7F)


# --------------------------------------------------------------------------- #
# The wire under test, dialling the fakes.
# --------------------------------------------------------------------------- #
def _start_in_scratch_range(proxy):
    """``start_server`` on a free port in 47000-47999, where Step 5a's live tests
    run. ``reclaim=False``: a port in use is skipped, never taken over."""
    for port in random.sample(range(47000, 48000), 50):
        try:
            return wire_proxy.start_server(proxy, port, reclaim=False)
        except OSError:
            continue
    raise RuntimeError("no free scratch port in 47000-47999")


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / "hal-home"
    monkeypatch.setenv("HAL_HOME", str(path))
    monkeypatch.setenv(wire_ca.STORE_ENV, "file")
    monkeypatch.setenv("HAL_INFINITE_BRAKE_MARKER", str(tmp_path / "no-brake.json"))
    monkeypatch.setenv("HAL_WIRE_EGRESS_BRAKE_MARKER", str(tmp_path / "no-egress-brake.json"))
    return path


@pytest.fixture
def make_wire(home, pki):
    made = []

    def build(pool_size: int = 2, accept_delay: float = 0.0, upstream=None, cap=None,
              scratch: bool = False):
        fake = FakeUpstream(pki.server_context(API), accept_delay=accept_delay)
        blind = FakeUpstream(pki.server_context("claude.ai"))
        # Every host but the API resolves to the blind fake: a host the wire does
        # not know is tunnelled there too, so a test can prove bytes went through.
        proxy = wire_proxy.WireProxy(
            wire_ca.load_or_create(), pool_size=pool_size,
            resolve=lambda host, port: ("127.0.0.1", fake.port if host == API else blind.port),
            upstream_context=upstream or pki.client_context(), cap=cap)
        server = (_start_in_scratch_range(proxy) if scratch
                  else wire_proxy.start_server(proxy, 0, reclaim=False))
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                         daemon=True).start()
        made.append((server, proxy, fake, blind))
        client = _trusting(wire_ca.ca_cert_path().read_text())
        return SimpleNamespace(port=server.server_address[1], proxy=proxy, fake=fake,
                               blind=blind, client=client, home=home)

    yield build
    for server, proxy, fake, blind in made:
        server.shutdown()
        server.server_close()
        proxy.stop()
        fake.close()
        blind.close()
    wait_for_exchanges()  # while this test's HAL_HOME is still the one set


@pytest.fixture
def wire(make_wire):
    return make_wire()


def _connect_reply(port: int, target: str) -> bytes:
    raw = _loopback(port)
    raw.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
    reply = _Reader(raw).until(b"\r\n\r\n")
    _quiet_close(raw)
    return reply


def _tunnel(port: int, host: str, context: ssl.SSLContext):
    raw = _loopback(port)
    raw.sendall(f"CONNECT {host}:443 HTTP/1.1\r\nHost: {host}:443\r\n\r\n".encode())
    reply = _Reader(raw).until(b"\r\n\r\n")
    assert reply.startswith(b"HTTP/1.1 200"), reply
    return context.wrap_socket(raw, server_hostname=host)


def _exchange(env, request: bytes) -> bytes:
    tls = _tunnel(env.port, API, env.client)
    try:
        tls.sendall(request)
        return _read_response(_Reader(tls))
    finally:
        _quiet_close(tls)


def _health(port: int) -> dict:
    raw = _loopback(port)
    raw.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    response = _read_response(_Reader(raw))
    _quiet_close(raw)
    assert response.startswith(b"HTTP/1.1 200")
    return json.loads(response.split(b"\r\n\r\n", 1)[1])


def _rows(home: Path, name: str = "gateway-audit.jsonl") -> list:
    path = home / "logs" / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _witness_rows(home: Path) -> list:
    return [row for row in _rows(home) if row.get("event") == "gateway.chat.witness"]


# --------------------------------------------------------------------------- #
# Identity, routing, brakes.
# --------------------------------------------------------------------------- #
def test_health_names_the_wire_and_its_ca(wire, home):
    health = _health(wire.port)
    assert health["ok"] is True and health["service"] == "hal-wire"
    assert health["brake"]["engaged"] is False
    assert health["egress_brake"]["engaged"] is False
    assert health["ca_fingerprint_sha256"] == wire_ca.load().fingerprint
    # The certificate under the HAL_HOME this wire runs with, spelled out here
    # rather than asked of wire_ca, so the check does not grade a function by itself.
    expected = home / "state" / "wire-ca" / "hal-wire-ca.pem"
    assert health["ca_cert"] == str(expected) and expected.is_file()


def test_an_engaged_brake_is_reported_and_traffic_still_flows(wire, tmp_path, monkeypatch):
    marker = tmp_path / "brake.json"
    marker.write_text(json.dumps({"reason": "test"}))
    monkeypatch.setenv("HAL_INFINITE_BRAKE_MARKER", str(marker))
    brake = _health(wire.port)["brake"]
    assert (brake["engaged"], brake["marker"], brake["detail"]) == (
        True, str(marker), {"reason": "test"})   # engaged, and able to say why
    assert _exchange(wire, _post("/v1/messages", TURN)).startswith(b"HTTP/1.1 200")


def test_the_leaf_is_minted_again_only_in_its_last_hour(wire):
    """AGORA's falsifier from its fourth review of HAL#644: 61 minutes from expiry
    the same leaf serves on; 59 minutes from expiry a fresh 24-hour one is minted."""
    proxy = wire.proxy
    first = proxy._server_context()
    leaf, now = proxy._leaf, datetime.now(timezone.utc)
    proxy._leaf = dataclasses.replace(leaf, not_after=now + timedelta(minutes=61))
    assert proxy._server_context() is first
    proxy._leaf = dataclasses.replace(leaf, not_after=now + timedelta(minutes=59))
    assert proxy._server_context() is not first
    assert proxy._leaf.not_after - now > timedelta(hours=23)


def _blind_get(port: int, target: str, pki) -> bytes:
    """CONNECT to ``target``, then speak TLS to the blind fake behind it. The
    client trusts ONLY the fake's CA, so a handshake proves nothing was decrypted."""
    raw = _loopback(port)
    raw.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
    reply = _Reader(raw).until(b"\r\n\r\n")
    assert reply.startswith(b"HTTP/1.1 200"), reply
    tls = pki.client_context().wrap_socket(raw, server_hostname="claude.ai")
    try:
        tls.sendall(b"GET / HTTP/1.1\r\nHost: claude.ai\r\n\r\n")
        return _read_response(_Reader(tls))
    finally:
        _quiet_close(tls)


# Inverted, not deleted (GRIP#7460): this test pinned the old rule, under which
# every host off the allowlist was refused. That rule cut GitHub, the connectors
# and WebFetch out of every wired session; such hosts are now tunnelled blind.
@pytest.mark.parametrize("target", ["example.com:443", "anthropic.com:443",
                                    "evil-sentry.io:443", "example.com:8443", "claude.ai:80"])
def test_a_host_the_wire_does_not_know_is_tunnelled_blind_and_logged_by_host_only(
        wire, pki, target):
    assert _blind_get(wire.port, target, pki) == EXACT_REPLY
    assert wire.blind.requests and not wire.fake.requests
    rows = _rows(wire.home)
    assert [row["event"] for row in rows] == ["wire.tunnelled"]
    host, port = wire_proxy._split_target(target)
    assert set(rows[0]) == {"ts", "event", "host", "port"}
    assert (rows[0]["host"], rows[0]["port"]) == (host, port)


@pytest.mark.parametrize("target", ["api.anthropic.com:8443", "127.0.0.1:443", "localhost:443",
                                    "[::1]:443", "[::ffff:127.0.0.1]:443", "0.0.0.0:443",
                                    "169.254.169.254:80", "[::ffff:169.254.169.254]:80",
                                    "example.com:0"])
def test_the_api_off_443_and_local_targets_are_refused_and_logged_by_host_only(wire, target):
    assert _connect_reply(wire.port, target).startswith(b"HTTP/1.1 403")
    refused = [row for row in _rows(wire.home) if row["event"] == "wire.refused"]
    assert refused and set(refused[-1]) == {"ts", "event", "host"}
    assert refused[-1]["host"] == wire_proxy._split_target(target)[0]
    assert not wire.blind.requests and not wire.fake.requests


def test_the_routing_table():
    """Every row of _route's table, so a host added to BLIND_HOSTS cannot quietly
    move another row."""
    route = wire_proxy._route
    assert route("api.anthropic.com", 443) == "decrypt"
    assert route("api.anthropic.com", 8443) == "refuse"
    assert route("claude.ai", 443) == "known" and route("o1.ingest.sentry.io", 443) == "known"
    assert route("claude.ai", 80) == "unknown" and route("github.com", 443) == "unknown"
    for host in ("localhost", "a.localhost", "127.0.0.1", "::1", "::ffff:127.0.0.1", "0.0.0.0",
                 "169.254.169.254", "::ffff:169.254.169.254", "fe80::1"):
        assert route(host, 443) == "refuse", host
    assert route("", 443) == route("example.com", 0) == route("example.com", 65536) == "refuse"


def test_an_unreadable_egress_marker_still_brakes(wire, tmp_path, monkeypatch):
    marker = tmp_path / "wire-egress-brake.json"
    marker.write_text("{not json")
    monkeypatch.setenv("HAL_WIRE_EGRESS_BRAKE_MARKER", str(marker))
    assert _connect_reply(wire.port, "example.com:443").startswith(
        b"HTTP/1.1 403 Egress Brake Engaged")
    state = _health(wire.port)["egress_brake"]
    assert (state["engaged"], state["detail"]) == (True, None)


@pytest.mark.skipif(os.name == "nt",
                    reason="chmod(0) fault injection does not deny directory traversal on Windows; "
                           "it cannot produce the unstattable marker this test requires")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root can inspect any path, so nothing is uncheckable")
def test_an_egress_marker_that_cannot_be_checked_still_brakes(wire, tmp_path, monkeypatch):
    """Python 3.14's Path.exists() answers False here (3.12 and 3.13 raise), which
    would have released the brake; the marker check must not depend on that."""
    locked = tmp_path / "locked"
    locked.mkdir()
    monkeypatch.setenv("HAL_WIRE_EGRESS_BRAKE_MARKER", str(locked / "wire-egress-brake.json"))
    locked.chmod(0)
    try:
        assert _connect_reply(wire.port, "example.com:443").startswith(
            b"HTTP/1.1 403 Egress Brake Engaged")
    finally:
        locked.chmod(0o700)


def test_the_egress_brake_refuses_unknown_hosts_says_why_and_releases(wire, pki, tmp_path,
                                                                    monkeypatch):
    marker = tmp_path / "wire-egress-brake.json"
    marker.write_text(json.dumps({"reason": "test"}))
    monkeypatch.setenv("HAL_WIRE_EGRESS_BRAKE_MARKER", str(marker))
    reply = _connect_reply(wire.port, "example.com:443")
    assert reply.startswith(b"HTTP/1.1 403 Egress Brake Engaged")
    raw = _loopback(wire.port)
    raw.sendall(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n")
    assert str(marker).encode() in _read_response(_Reader(raw))  # names the marker
    _quiet_close(raw)
    refused = [row for row in _rows(wire.home) if row["event"] == "wire.refused"]
    assert refused[-1] == {**refused[-1], "host": "example.com", "reason": "egress-brake"}
    assert _health(wire.port)["egress_brake"]["detail"] == {"reason": "test"}
    # Claude Code's own hosts and the API are served while it is engaged.
    assert _blind_get(wire.port, "claude.ai:443", pki) == EXACT_REPLY
    assert _exchange(wire, _post("/v1/messages", TURN)).startswith(b"HTTP/1.1 200")
    # Released: the very next CONNECT is tunnelled.
    marker.unlink()
    assert _blind_get(wire.port, "example.com:443", pki) == EXACT_REPLY
    assert _health(wire.port)["egress_brake"]["engaged"] is False


def test_claude_codes_own_hosts_are_tunnelled_blind_with_no_row(wire, pki):
    """The client trusts ONLY the fake's CA: a handshake proves nothing was decrypted.
    A known host (BLIND_HOSTS) writes no row, unlike a host the wire does not know."""
    tls = _tunnel(wire.port, "claude.ai", pki.client_context())
    try:
        tls.sendall(b"GET / HTTP/1.1\r\nHost: claude.ai\r\n\r\n")
        assert _read_response(_Reader(tls)) == EXACT_REPLY
    finally:
        _quiet_close(tls)
    assert wire.blind.requests and not wire.fake.requests
    assert _rows(wire.home) == []


# --------------------------------------------------------------------------- #
# Byte-faithful relay.
# --------------------------------------------------------------------------- #
ODD_REQUEST = (b"POST /v1/code/sessions/cse_EXAMPLE/worker/events?seq=7 HTTP/1.1\r\n"
               b"hOsT: api.anthropic.com\r\nX-Odd-Header:   spaced   value  \r\n"
               b"authorization: Bearer wire-test-bearer-0000\r\n"
               b"content-type: application/json\r\ncontent-length: 25\r\n\r\n"
               b'{"events": [1, 2, 3]}    ')


def test_an_account_bound_request_and_its_answer_pass_byte_for_byte(wire):
    assert _exchange(wire, ODD_REQUEST) == EXACT_REPLY
    assert wire.fake.requests[-1][1] == ODD_REQUEST


def test_an_inference_request_passes_byte_for_byte_too(wire):
    request = _post("/v1/messages?beta=true", TURN, extra="X-Stainless-Retry-Count: 0\r\n")
    _exchange(wire, request)
    assert wire.fake.requests[-1][1] == request


def test_a_body_too_big_to_copy_is_relayed_whole_and_recorded_by_name_only(wire, monkeypatch):
    """AGORA's falsifier from its fifth review of HAL#644, scaled down by a small copy
    limit: a body past the limit still reaches Anthropic byte for byte, its copy is
    dropped rather than kept, and the call is recorded by method and path only."""
    class _SmallCopy(wire_proxy._Copy):
        def __init__(self, limit: int = 64):
            super().__init__(limit)

    monkeypatch.setattr(wire_proxy, "_Copy", _SmallCopy)
    request = _post("/v1/messages", TURN)
    assert _exchange(wire, request).startswith(b"HTTP/1.1 200")
    assert wire.fake.requests[-1][1] == request
    assert _wait_for(lambda: any(r["event"] == "wire.unknown" for r in _rows(wire.home)))
    assert not [r for r in _rows(wire.home) if r["event"] == "gateway.chat.witness"]


def test_an_account_bound_body_is_never_copied(wire, monkeypatch):
    made = []

    class _CountingCopy(wire_proxy._Copy):
        def __init__(self, *args, **kwargs):
            made.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(wire_proxy, "_Copy", _CountingCopy)
    _exchange(wire, ODD_REQUEST)
    assert made == []
    _exchange(wire, _post("/v1/messages", TURN))
    assert made


def test_each_sse_event_arrives_within_100ms_of_being_sent(wire):
    tls = _tunnel(wire.port, API, wire.client)
    try:
        tls.sendall(_post("/v1/messages?beta=true", {**TURN, "stream": True}))
        arrivals, data = [], b""
        while len(arrivals) < len(SSE_EVENTS):
            piece = tls.recv(65536)
            assert piece, "the stream ended early"
            data += piece
            arrivals += [time.monotonic()] * (data.count(b"\n\n") - len(arrivals))
    finally:
        _quiet_close(tls)
    lags = [arrived - sent for arrived, sent in zip(arrivals, wire.fake.sent_events)]
    assert max(lags) < 0.1, lags


def _websocket_round_trip(env) -> list:
    """Upgrade the voice route, echo two frames, close; returns what came back."""
    key = base64.b64encode(os.urandom(16))
    tls = _tunnel(env.port, API, env.client)
    try:
        tls.sendall(b"GET /api/ws/speech_to_text/voice_stream?language=en HTTP/1.1\r\n"
                    b"Host: api.anthropic.com\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                    b"Sec-WebSocket-Version: 13\r\nSec-WebSocket-Key: " + key + b"\r\n\r\n")
        reader = _Reader(tls)
        head = reader.until(b"\r\n\r\n")
        assert head.startswith(b"HTTP/1.1 101")
        assert base64.b64encode(hashlib.sha1(key + WS_GUID).digest()) in head
        echoes = []
        for word in (b"hello voice", b"second frame"):
            tls.sendall(_ws_frame(word))
            echoes.append(_ws_read(reader))
        return echoes
    finally:
        _quiet_close(tls)


def test_a_websocket_echoes_through_the_wire(wire):
    assert _websocket_round_trip(wire) == [b"hello voice", b"second frame"]


def _upgrade_then_connect(env) -> bytes:
    """One WebSocket round trip, then a new CONNECT; the reply, or what went wrong."""
    try:
        echoes = _websocket_round_trip(env)
        if echoes != [b"hello voice", b"second frame"]:
            return b"echoes lost: " + repr(echoes).encode()[:60]
        return _exchange(env, _post("/v1/messages", TURN))
    except (AssertionError, IndexError, OSError) as exc:
        return repr(exc).encode()[:80]


def test_a_finished_splice_never_touches_the_next_connection(wire):
    """Over the real TLS path: 200 WebSocket round trips, each followed at once by a
    new CONNECT that must get its 200. CI caught two failures here. On 41fd54e8 a
    splice closed a socket its other thread was still inside, and the next CONNECT
    got a stray TLS alert ("501 Unsupported method ('CO')"). On 87595782 an echo
    was lost mid-splice; that one was never reproduced locally. Races are caught
    only some of the time, so the test after this one checks the rule directly."""
    replies = [_upgrade_then_connect(wire) for _ in range(200)]
    broken = [(i, r[:40]) for i, r in enumerate(replies) if not r.startswith(b"HTTP/1.1 200")]
    assert not broken, broken


class _ThreadNotingSocket:
    """A real socket that notes which threads call it."""

    def __init__(self, sock):
        self._sock = sock
        self.threads = set()

    def __getattr__(self, name):
        attribute = getattr(self._sock, name)
        if not callable(attribute):
            return attribute

        def noted(*args, **kwargs):
            self.threads.add(threading.get_ident())
            return attribute(*args, **kwargs)

        return noted


def test_a_splice_does_every_read_and_write_on_one_thread():
    """Two threads on one TLS connection is what OpenSSL does not promise to survive,
    and what let one thread close a socket another was still inside. So the rule is
    checked directly: every call on either socket comes from the splice's thread."""
    near, near_peer = socket.socketpair()
    far, far_peer = socket.socketpair()
    near_peer.settimeout(5)
    far_peer.settimeout(5)
    left, right = _ThreadNotingSocket(near), _ThreadNotingSocket(far)
    splice = threading.Thread(target=wire_proxy._splice,
                              args=(wire_proxy._Stream(left), wire_proxy._Stream(right)),
                              daemon=True)
    splice.start()
    try:
        near_peer.sendall(b"up")
        assert far_peer.recv(16) == b"up"
        far_peer.sendall(b"down")
        assert near_peer.recv(16) == b"down"
        near_peer.close()          # the client hangs up
        splice.join(timeout=10)
        assert not splice.is_alive(), "the splice never returned"
        assert left.threads | right.threads == {splice.ident}
    finally:
        _quiet_close(near_peer)
        _quiet_close(far_peer)


def test_tls_sessions_are_kept_only_from_reusable_exchanges_and_do_resume(wire):
    """AGORA's falsifier on HAL#644: an upgraded connection leaves no session behind;
    a reusable exchange does, and the next upstream connection resumes it."""
    dialer = wire.proxy.pool.dialer
    _websocket_round_trip(wire)
    assert dialer._session is None
    assert _exchange(wire, _post("/v1/messages", TURN)).startswith(b"HTTP/1.1 200")
    assert _wait_for(lambda: dialer._session is not None)
    resumed = dialer.dial()
    try:
        assert resumed.session_reused
    finally:
        _quiet_close(resumed)


def test_a_mid_stream_disconnect_reaches_the_client_as_a_closed_stream(wire):
    tls = _tunnel(wire.port, API, wire.client)
    tls.settimeout(5)
    received, started = b"", time.monotonic()
    try:
        tls.sendall(_post("/v1/messages", {**TURN, "model": "disconnect", "stream": True}))
        piece = b"x"
        while piece:
            piece = _recv_or_closed(tls)
            received += piece
    finally:
        _quiet_close(tls)
    assert b"message_start" in received
    assert time.monotonic() - started < 3


_CUT_MODELS = {"upstream-closes": "disconnect", "upstream-resets": "reset",
               "client-hangs-up": "claude-opus-5-5"}


@pytest.mark.parametrize("ending", sorted(_CUT_MODELS))
def test_a_turn_cut_off_mid_stream_is_still_witnessed_as_not_served(wire, ending):
    """AGORA's falsifier from its eighth review of HAL#644: a streamed turn cut off
    after its first event is witnessed once, as not served, with the usage seen
    before the cut. When the client hangs up (an interrupted turn), the relay's next
    write fails, and that used to raise past the record: the turn left no row."""
    tls = _tunnel(wire.port, API, wire.client)
    tls.settimeout(5)
    try:
        tls.sendall(_post("/v1/messages", {**TURN, "model": _CUT_MODELS[ending],
                                           "stream": True}))
        if ending == "client-hangs-up":
            _Reader(tls).until(b"message_start")
        else:
            while _recv_or_closed(tls):
                pass
    finally:
        _quiet_close(tls)

    def witnessed() -> list:
        return [row for row in _rows(wire.home) if row["event"] == "gateway.chat.witness"]

    assert _wait_for(witnessed)
    rows = witnessed()
    assert len(rows) == 1
    assert (rows[0]["served"], rows[0]["status"], rows[0]["input_tokens"]) == (False, 200, 21)


def _recv_or_closed(tls) -> bytes:
    try:
        return tls.recv(65536)
    except (ssl.SSLError, ConnectionError):
        return b""


def test_one_client_connection_keeps_one_upstream_connection(wire):
    tls = _tunnel(wire.port, API, wire.client)
    reader = _Reader(tls)
    try:
        for _ in range(3):
            tls.sendall(_post("/v1/messages", TURN))
            assert _read_response(reader).startswith(b"HTTP/1.1 200")
    finally:
        _quiet_close(tls)
    numbers = {number for number, _ in wire.fake.requests}
    assert len(wire.fake.requests) == 3 and len(numbers) == 1


# --------------------------------------------------------------------------- #
# Witness and audit rows; no header in any of them.
# --------------------------------------------------------------------------- #
def _send_one_of_each(env) -> None:
    for request in (_post("/v1/messages?beta=true", TURN),
                    _post("/v1/messages?beta=true", {**TURN, "stream": True}),
                    _post("/v1/messages?beta=true", QUOTA),
                    _post("/v1/messages/count_tokens?beta=true", TURN),
                    ODD_REQUEST, _get("/v1/models")):
        _exchange(env, request)
    # The witness writes the unsigned row first and signs after, from the
    # connection's own thread, so wait for both logs, not only the first.
    _wait_for(lambda: len(_witness_rows(env.home)) >= 4
              and len(_rows(env.home, "gateway-chat-signed.jsonl")) >= 4
              and any(r["event"] == "wire.unknown" for r in _rows(env.home)))


def test_inference_and_aux_are_witnessed_and_account_bound_is_not(wire):
    _send_one_of_each(wire)
    rows = _witness_rows(wire.home)
    assert sorted(row["route_class"] for row in rows) == ["aux", "aux", "inference",
                                                          "inference"]
    assert {(row["rail"], row["provider"]) for row in rows} == {("wire", "anthropic")}
    assert "/v1/code" not in json.dumps(_rows(wire.home))
    assert len(_rows(wire.home, "gateway-chat-signed.jsonl")) == 4


def test_a_streamed_turn_is_priced_from_its_events(wire):
    _send_one_of_each(wire)
    streamed = [row for row in _witness_rows(wire.home) if row.get("input_tokens") == 21]
    assert len(streamed) == 1
    row = streamed[0]
    assert (row["output_tokens"], row["cache_read_tokens"], row["cache_write_tokens"],
            row["model"], row["served"], row["status"]) == (9, 5, 3, "claude-opus-5-5",
                                                            True, 200)


def test_an_unknown_route_leaves_one_row_with_method_and_path_only(wire):
    _send_one_of_each(wire)
    unknown = [row for row in _rows(wire.home) if row["event"] == "wire.unknown"]
    assert unknown == [{"ts": unknown[0]["ts"], "event": "wire.unknown",
                        "method": "GET", "path": "/v1/models"}]


def test_no_header_value_reaches_any_log_or_row(wire):
    _send_one_of_each(wire)
    _connect_reply(wire.port, "example.com:443")
    text = "".join(path.read_text() for path in (wire.home / "logs").glob("*.jsonl"))
    assert text
    for needle in ("Bearer ", "x-api-key", "wire-test-bearer-0000", API_KEY,
                   "X-Odd-Header", "spaced"):
        assert needle not in text, needle


# --------------------------------------------------------------------------- #
# The cache-loss ledger: one row per /v1/messages response, hashes only.
# --------------------------------------------------------------------------- #
def _cache_loss_rows(home: Path) -> list:
    from lib.hal import cache_loss
    path = cache_loss.log_path(home)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_cache_loss_ledger_gets_one_row_per_v1_messages_response_and_no_secret(wire):
    from lib.hal import cache_loss
    secret = "sk-ant-WIRE-TEST-SECRET-0000-DO-NOT-USE"
    turn_with_secret = {**TURN, "system": f"remember: {secret}"}
    _exchange(wire, _post("/v1/messages?beta=true", turn_with_secret))
    # count_tokens is the same path FAMILY but not MESSAGES_PATH itself -- no row.
    _exchange(wire, _post("/v1/messages/count_tokens?beta=true", TURN))
    _wait_for(lambda: len(_cache_loss_rows(wire.home)) >= 1)
    text = cache_loss.log_path(wire.home).read_text()
    assert secret not in text
    assert API_KEY not in text and "wire-test-bearer-0000" not in text
    rows = _cache_loss_rows(wire.home)
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "claude-opus-5-5"
    assert row["account_label"] == "unpooled"
    assert row["usage"] == {"input_tokens": 11, "output_tokens": 2,
                            "cache_read_tokens": 0, "cache_creation_tokens": 0}
    assert row["system_hash"] == cache_loss.hash_hex(f"remember: {secret}")
    assert set(row) == {"ts", "session_id", "account_label", "org_id", "model",
                        "system_hash", "tools_hash", "first_message_hash", "usage",
                        "five_hour_utilization", "seven_day_utilization",
                        "continues_thread", "account_switched", "prompt_id"}
    assert (row["continues_thread"], row["account_switched"]) == (False, False)
    assert row["prompt_id"] is None  # this request sent no prompt-id header


def test_cache_loss_ledger_captures_the_organisation_and_utilization_headers(wire):
    from lib.hal import cache_loss
    wire.fake.reply_headers = (b"anthropic-organization-id: org_test123\r\n"
                               b"anthropic-ratelimit-unified-5h-utilization: 12.34\r\n"
                               b"anthropic-ratelimit-unified-7d-utilization: 5.00\r\n")
    _exchange(wire, _post("/v1/messages?beta=true", TURN))
    _wait_for(lambda: len(_cache_loss_rows(wire.home)) >= 1)
    row = _cache_loss_rows(wire.home)[-1]
    assert row["org_id"] == "org_test123"
    assert row["five_hour_utilization"] == 12.34
    assert row["seven_day_utilization"] == 5.0


def test_a_failing_cache_loss_ledger_never_costs_the_witness_row(wire, monkeypatch):
    """The ledger runs before the witness inside one try in _record, so an
    exception escaping it would skip the witness row too. It must not."""
    from lib.hal import cache_loss

    def ledger_down(**_kwargs):
        raise RuntimeError("cache-loss ledger down")

    monkeypatch.setattr(cache_loss, "record_response", ledger_down)
    _exchange(wire, _post("/v1/messages?beta=true", TURN))
    _wait_for(lambda: len(_witness_rows(wire.home)) >= 1)
    assert [row["route_class"] for row in _witness_rows(wire.home)] == ["inference"]
    assert _cache_loss_rows(wire.home) == []


def test_witness_extra_fields_add_but_never_override():
    record = build_chat_record(provider="anthropic", model="m", status=200, served=True,
                               failed_over=False, latency_ms=3, path="/v1/messages",
                               extra={"event": "forged", "status": 999, "rail": "wire"})
    assert (record["event"], record["status"], record["rail"]) == (
        "gateway.chat.witness", 200, "wire")


def test_witness_extra_cannot_forge_the_audit_timestamp(home):
    """AGORA's falsifier on HAL#644: ``ts`` is stamped by the sink, never by a caller."""
    rows: list = []
    witness_chat(provider="anthropic", model="m", status=200, served=True,
                 started=time.monotonic(), path="/v1/messages",
                 telemetry=GatewayTelemetry(sink=rows.append),
                 extra={"ts": "1970-01-01T00:00:00+00:00", "event": "forged",
                        "rail": "wire"})
    assert rows[0]["ts"] != "1970-01-01T00:00:00+00:00"
    assert (rows[0]["event"], rows[0]["rail"]) == ("gateway.chat.witness", "wire")


def test_witness_extra_cannot_invent_a_measurement():
    """AGORA's third review of HAL#644: with no usage measured, extra may not add one."""
    record = build_chat_record(provider="anthropic", model="m", status=200, served=True,
                               failed_over=False, latency_ms=3, path="/v1/messages",
                               extra={"input_tokens": 5, "output_tokens": 7,
                                      "cost_usd": 9.9, "rail": "wire"})
    assert not {"input_tokens", "output_tokens", "cost_usd"} & set(record)
    assert record["rail"] == "wire"


SESSION_ID = "5f0c2b1e-0000-4000-8000-000000000001"


def test_witnessed_calls_become_thread_rows_of_the_gateways_classes(wire, monkeypatch):
    """Step 4's signed thread (HAL main since b2b9d6fb) takes one row per witnessed
    exchange, keyed by Claude Code's session header. The wire must class them as the
    gateway does: a turn is ``served``, a token count ``aux``. Before it said so, the
    defaults labelled every wire row a served inference turn."""
    rows = []
    monkeypatch.setattr("lib.hal.gateway_witness.append_thread_row",
                        lambda record, **kwargs: rows.append(kwargs))
    _exchange(wire, _post("/v1/messages", TURN,
                          extra=f"x-claude-code-session-id: {SESSION_ID}\r\n"))
    _exchange(wire, _post("/v1/messages/count_tokens", TURN))
    assert _wait_for(lambda: len(rows) == 2)
    # Step 4 part B (HAL#646): the row also takes Anthropic's own usage, in the
    # thread's shape (cache lifetimes kept), and the credential's billing kind.
    # This test's credentials are neither an OAuth token nor an API key.
    turn_usage = {"input_tokens": 11, "output_tokens": 2, "cache_read_tokens": 0,
                  "cache_write_tokens": 0, "cache_write_1h_tokens": 0}
    assert sorted(rows, key=lambda row: row["row_class"]) == [
        {"row_class": "aux", "route": "aux", "session_id": None,
         "usage": None, "billing": "unknown"},
        {"row_class": "served", "route": "inference", "session_id": SESSION_ID,
         "usage": turn_usage, "billing": "unknown"}]


PROMPT_ID = "9c2f0a2e-1111-4000-8000-000000000042"


def test_the_prompt_id_rides_the_thread_row_and_the_cache_loss_row(wire, monkeypatch):
    """Claude Code 2.1.283 sends x-claude-code-prompt-id so a gateway can group
    the requests that serve one user prompt. The wire copies it onto the signed
    thread row (through the witness record; the field is allowlisted) and onto
    the cache-loss row. A request without it leaves the thread row without the
    field and the ledger row null, never "".
    Mutation: drop "prompt_id" from PAYLOAD_FIELDS, or the extra in _witness -> RED.
    """
    from lib.hal import infinite_thread

    records = []
    monkeypatch.setattr("lib.hal.gateway_witness.append_thread_row",
                        lambda record, **kwargs: records.append(record))
    _exchange(wire, _post("/v1/messages", TURN,
                          extra=(f"x-claude-code-session-id: {SESSION_ID}\r\n"
                                 f"x-claude-code-prompt-id: {PROMPT_ID}\r\n")))
    _exchange(wire, _post("/v1/messages", TURN))
    assert _wait_for(lambda: len(records) == 2 and len(_cache_loss_rows(wire.home)) == 2)
    assert {record.get("prompt_id") for record in records} == {PROMPT_ID, None}
    assert sum("prompt_id" in record for record in records) == 1
    assert {row["prompt_id"] for row in _cache_loss_rows(wire.home)} == {PROMPT_ID, None}
    # The store accepts the field, so the real append_thread_row keeps it.
    stored = infinite_thread.build_row("served", {"prompt_id": PROMPT_ID, "route": "inference"},
                                       host="h-test", ts="2026-10-17T00:00:00+00:00",
                                       session_id=SESSION_ID)
    assert stored["prompt_id"] == PROMPT_ID


def test_the_front_end_error_hook_never_raises_outside_an_exception():
    wire_proxy._quiet_error(None, None)


# --------------------------------------------------------------------------- #
# Upstream verification and failure.
# --------------------------------------------------------------------------- #
def test_an_upstream_with_an_untrusted_certificate_is_never_used(make_wire, tmp_path):
    stranger = _TestPki(tmp_path)  # a CA the wire is not told to trust
    env = make_wire(pool_size=0, upstream=stranger.client_context())
    response = _exchange(env, _post("/v1/messages", TURN))
    assert response.startswith(b"HTTP/1.1 503")
    assert b"could not be reached" in response and not env.fake.requests


# --------------------------------------------------------------------------- #
# Latency: warm upstream connections keep short control calls fast.
# --------------------------------------------------------------------------- #
ARCHIVE = (b"POST /v1/code/sessions/cse_EXAMPLE/archive HTTP/1.1\r\nHost: api.anthropic.com\r\n"
           b"content-type: application/json\r\ncontent-length: 2\r\n\r\n{}")


def _timed(open_tls, ready=lambda: True) -> float:
    """One archive-shaped call from a new client connection; ``ready`` waits untimed."""
    assert _wait_for(ready, 5)
    started = time.monotonic()
    tls = open_tls()
    try:
        tls.sendall(ARCHIVE)
        assert _read_response(_Reader(tls)) == EXACT_REPLY
    finally:
        _quiet_close(tls)
    return time.monotonic() - started


def _median_call(env, via_wire: bool, pki, runs: int = 5) -> float:
    def direct():
        return pki.client_context().wrap_socket(_loopback(env.fake.port),
                                                server_hostname=API)

    def wired():
        return _tunnel(env.port, API, env.client)

    def warmed() -> bool:  # a pool that warms must have a connection ready first
        return env.proxy.pool.idle_count() >= 1 or not env.proxy.pool._size

    calls = [_timed(wired, warmed) if via_wire else _timed(direct) for _ in range(runs)]
    return statistics.median(calls)


def test_warm_connections_keep_a_short_call_well_under_the_archive_deadline(make_wire, pki):
    """0.15 s per new upstream connection stands in for a WAN handshake."""
    warm = make_wire(pool_size=2, accept_delay=0.15)
    cold = make_wire(pool_size=0, accept_delay=0.15)
    plain = make_wire(pool_size=2, accept_delay=0.0)
    timings = {"direct, new connection (0.15 s setup)": _median_call(warm, False, pki),
               "wire, warm pool (0.15 s setup)": _median_call(warm, True, pki),
               "wire, no pool (0.15 s setup)": _median_call(cold, True, pki),
               "direct, new connection (loopback)": _median_call(plain, False, pki),
               "wire, warm pool (loopback)": _median_call(plain, True, pki)}
    print("\nwire latency, median of 5:", json.dumps(
        {name: f"{seconds * 1000:.1f} ms" for name, seconds in timings.items()}, indent=2))
    # Wide margins for a loaded runner: the pool must save at least half of the
    # simulated 0.15 s handshake, and a warm call must stay far inside 0.6 s.
    assert timings["wire, warm pool (0.15 s setup)"] < 0.3
    assert (timings["wire, no pool (0.15 s setup)"]
            - timings["wire, warm pool (0.15 s setup)"]) > 0.075


def _tls_server_that_hangs_up(pki) -> int:
    """A one-connection TLS server that completes the handshake, then closes."""
    listener = socket.create_server(("127.0.0.1", 0))

    def serve() -> None:
        raw, _ = listener.accept()
        with pki.server_context(API).wrap_socket(raw, server_side=True):
            pass
        listener.close()

    threading.Thread(target=serve, daemon=True).start()
    return listener.getsockname()[1]


def test_a_warm_connection_outlives_its_session_tickets_but_not_a_hang_up(pki):
    """A TLS 1.3 server sends session tickets just after the handshake, so an idle
    warm connection turns readable with nothing in it for the reader. Found
    2026-10-05: the pool took that for a closed connection and threw away every
    connection it had dialled but not yet used. The latency test above could not
    see it, because its warm calls reuse connections whose tickets were already read."""
    fake = FakeUpstream(pki.server_context(API))
    warm = wire_proxy.TlsDialer(lambda: ("127.0.0.1", fake.port), pki.client_context()).dial()
    gone = wire_proxy.TlsDialer(lambda: ("127.0.0.1", _tls_server_that_hangs_up(pki)),
                                pki.client_context()).dial()
    try:
        time.sleep(0.3)                    # the tickets, and the hang-up, arrive
        assert warm.version() == "TLSv1.3"
        assert select.select([warm], [], [], 0)[0], "no tickets arrived: nothing tested"
        assert wire_proxy._Upstream(warm).is_fresh(45.0)
        assert not wire_proxy._Upstream(gone).is_fresh(45.0)
        warm.sendall(ARCHIVE)              # and the warm one still carries a request
        assert _read_response(_Reader(warm)) == EXACT_REPLY
    finally:
        _quiet_close(warm)
        _quiet_close(gone)
        fake.close()


def test_an_idle_warm_pool_keeps_its_connections_instead_of_redialling(make_wire):
    """With tickets read as a close, every maintenance pass retired both warm
    connections and dialled two more: about 24 TLS handshakes a minute to
    Anthropic from a wire nobody was using."""
    env = make_wire(pool_size=2)
    pool = env.proxy.pool
    assert _wait_for(lambda: pool.idle_count() == 2)
    time.sleep(0.3)                        # their session tickets arrive
    dialled = env.fake.connections
    pool._prune()
    pool._refill()
    assert pool.idle_count() == 2
    assert env.fake.connections == dialled


# --------------------------------------------------------------------------- #
# The spend guard on the wire (InfiniteMachine v2, Step 5h).
# --------------------------------------------------------------------------- #
OAUTH = "Bearer sk-ant-oat01-wire-fake-token"
OVERAGE_REPLY = (b"anthropic-ratelimit-unified-status: rejected\r\n"
                 b"anthropic-ratelimit-unified-overage-status: allowed\r\n")
BEFORE_THE_DATE = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
AFTER_THE_DATE = datetime(2026, 10, 11, 0, 1, tzinfo=timezone.utc)


@pytest.fixture
def strict_spend(tmp_path, monkeypatch):
    """Replace the suite's permissive policy with the committed one's shape, on a
    frozen clock. Returns a setter for the opt-in list and the moment."""
    from lib.hal import spend_guard
    path = tmp_path / "strict-spend-guard.json"

    def set_policy(opt_in=(), moment=BEFORE_THE_DATE) -> None:
        path.write_text(json.dumps({"paid_opt_in": list(opt_in)}))
        monkeypatch.setattr(spend_guard, "_clock", lambda: moment)

    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(path))
    set_policy()
    return set_policy


def _oauth_post(path: str, payload: dict) -> bytes:
    body = json.dumps(payload).encode()
    return (f"POST {path} HTTP/1.1\r\nHost: {API}\r\nAuthorization: {OAUTH}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
            ).encode() + body


def test_a_paid_key_is_refused_and_nothing_reaches_anthropic(wire, home, strict_spend):
    strict_spend(opt_in=["anthropic-overage"])  # overage opted in; a metered key is not
    response = _exchange(wire, _post("/v1/messages", TURN))
    assert response.startswith(b"HTTP/1.1 403")
    assert b"permission_error" in response and b"spend guard" in response
    assert wire.fake.requests == []
    rows = [row for row in _rows(home) if row.get("event") == "wire.spend_refused"]
    assert [(row["source"], row["status"]) for row in rows] == [("anthropic-api-key", 403)]
    assert set(rows[0]) == {"ts", "event", "source", "status"}   # never a header


SESSION_INGRESS = "sk-ant-si-fake-header.eyJmYWtlIjp0cnVlfQ.fake-signature"   # payload {"fake":true}


def test_remote_controls_session_token_is_relayed_while_paid_keys_are_refused(
        wire, home, strict_spend):
    """Remote Control's bridge authenticates with Claude Code's session-ingress
    token, ``sk-ant-si-`` then a JWT. Read as a paid key, every bridge request was
    refused here, so a session registered on claude.ai and then never synced
    (GRIP#7460 rollout, 2026-10-06). A malformed token of that prefix stays paid."""
    strict_spend()                         # paid keys refused, as committed
    poll = ("GET /v1/environments/fake-env/work/poll HTTP/1.1\r\nHost: {host}\r\n"
            "Authorization: Bearer {token}\r\n\r\n")
    relayed = _exchange(wire, poll.format(host=API, token=SESSION_INGRESS).encode())
    refused = _exchange(wire, poll.format(host=API, token="sk-ant-si-not-a-jwt").encode())
    assert relayed.startswith(b"HTTP/1.1 200"), relayed
    assert refused.startswith(b"HTTP/1.1 403"), refused
    assert len(wire.fake.requests) == 1    # only the session token reached Anthropic
    rows = [row for row in _rows(home) if row.get("event") == "wire.spend_refused"]
    assert [(row["source"], row["status"]) for row in rows] == [("anthropic-api-key", 403)]


def test_an_opted_in_key_flows_after_the_date(wire, strict_spend):
    strict_spend(opt_in=["anthropic-api-key"], moment=AFTER_THE_DATE)
    assert _exchange(wire, _post("/v1/messages", TURN)).startswith(b"HTTP/1.1 200")
    assert len(wire.fake.requests) == 1


def test_overage_on_a_2xx_spends_the_account_on_the_wire(wire, strict_spend):
    wire.fake.reply_headers = OVERAGE_REPLY
    first = _exchange(wire, _oauth_post("/v1/messages", TURN))
    second = _exchange(wire, _oauth_post("/v1/messages", TURN))
    assert first.startswith(b"HTTP/1.1 200")
    assert second.startswith(b"HTTP/1.1 429"), second
    assert b"retry-after: " in second.lower() and b"rate_limit_error" in second
    assert len(wire.fake.requests) == 1, "the refused turn reached Anthropic"
    # Nothing that does not bill is held back: a token count, an account-bound call.
    assert _exchange(wire, _oauth_post("/v1/messages/count_tokens", TURN)).startswith(
        b"HTTP/1.1 200")
    assert _exchange(wire, _oauth_post("/v1/code/sessions", {})).startswith(b"HTTP/1.1 200")
    assert len(wire.fake.requests) == 3


# --------------------------------------------------------------------------- #
# The account switch on the wire (InfiniteMachine v2, Step 5e, pre-send form).
# --------------------------------------------------------------------------- #
POOL_TOKEN = "sk-ant-oat01-wire-pool-teammate"


@pytest.fixture
def one_pool_account(monkeypatch):
    from lib.hal import account_switch, spend_guard
    account_switch.forget_exhausted()
    monkeypatch.setattr(spend_guard, "_SPENT", {})
    monkeypatch.setattr(account_switch, "_read_pool", lambda: [("teammate", POOL_TOKEN)])
    monkeypatch.setattr(account_switch, "_organisation",
                        lambda token, probe: spend_guard.account_key(token))
    yield account_switch
    account_switch.forget_exhausted()


def test_a_spent_accounts_next_turn_moves_to_a_pool_account(wire, home, strict_spend,
                                                             one_pool_account):
    wire.fake.reply_headers = OVERAGE_REPLY
    first = _exchange(wire, _oauth_post("/v1/messages", TURN))
    wire.fake.reply_headers = b""
    second = _exchange(wire, _oauth_post("/v1/messages", TURN))
    assert first.startswith(b"HTTP/1.1 200") and second.startswith(b"HTTP/1.1 200"), second
    assert len(wire.fake.requests) == 2
    sent = wire.fake.requests[1][1].lower()
    assert f"authorization: bearer {POOL_TOKEN}".encode() in sent
    assert b"wire-fake-token" not in sent, "the spent account's bearer went upstream"
    assert wire.fake.requests[0][1].split(b"\r\n\r\n", 1)[1] == sent.split(b"\r\n\r\n", 1)[1]
    rows = [r for r in _rows(home) if r.get("event") == "wire.account_switched"]
    assert [(r["to"], r["rank"]) for r in rows] == [("teammate", "subscription")]
    assert POOL_TOKEN not in json.dumps(_rows(home)), "a token reached the audit log"


def test_the_switch_brake_keeps_a_spent_account_on_its_refusal(wire, strict_spend,
                                                               one_pool_account):
    one_pool_account.set_switching(False)
    wire.fake.reply_headers = OVERAGE_REPLY
    _exchange(wire, _oauth_post("/v1/messages", TURN))
    second = _exchange(wire, _oauth_post("/v1/messages", TURN))
    assert second.startswith(b"HTTP/1.1 429"), second
    assert len(wire.fake.requests) == 1


def test_a_pool_account_that_answers_429_is_relayed_once_and_never_retried(
        wire, home, strict_spend, one_pool_account):
    """AGORA's falsifier on HAL#662: the account switched to is out too. Its 429
    reaches the client once (no replay), no paid key is ever presented, and the next
    turn is the guard's honest refusal -- or, once overage is opted in, the own
    account's overage."""
    wire.fake.reply_headers = OVERAGE_REPLY
    _exchange(wire, _oauth_post("/v1/messages", TURN))            # the own account: spent
    wire.fake.reply_headers = b""
    wire.fake.limit_bearers = {POOL_TOKEN}
    second = _exchange(wire, _oauth_post("/v1/messages", TURN))
    assert second.startswith(b"HTTP/1.1 429"), second               # the pool account's limit
    third = _exchange(wire, _oauth_post("/v1/messages", TURN))
    assert third.startswith(b"HTTP/1.1 429") and b"spend guard" in third, third
    assert len(wire.fake.requests) == 2, "the refused turn reached Anthropic"
    sent = b"".join(request for _number, request in wire.fake.requests).lower()
    assert b"x-api-key" not in sent and b"sk-ant-api" not in sent
    strict_spend(opt_in=["anthropic-overage"])
    fourth = _exchange(wire, _oauth_post("/v1/messages", TURN))
    assert fourth.startswith(b"HTTP/1.1 200"), fourth               # the own account's overage
    assert b"wire-fake-token" in wire.fake.requests[-1][1]


# --------------------------------------------------------------------------- #
# Running it.
# --------------------------------------------------------------------------- #
def test_the_cli_refuses_to_start_without_a_usable_ca(home, monkeypatch, capsys):
    def broken():
        raise wire_ca.WireCAError("no wire CA here")

    monkeypatch.setattr(wire_proxy.wire_ca, "load_or_create", broken)
    assert wire_proxy.main(["--port", "0", "--no-reclaim"]) == 2
    assert "not started: no wire CA here" in capsys.readouterr().err


@pytest.mark.parametrize("host,local", [
    ("::ffff:127.0.0.1", True),    # loopback, written as IPv4-mapped IPv6
    ("::ffff:169.254.1.1", True),  # link-local, IPv4-mapped
    ("::ffff:0.0.0.0", True),      # unspecified, IPv4-mapped
    ("::ffff:8.8.8.8", False),     # a public address stays public once unwrapped
    ("::1", True),
    ("8.8.8.8", False),
])
def test_an_ipv4_mapped_address_is_judged_by_the_address_it_carries(host, local):
    # HAL#876: the unwrap must refuse a mapped local target on every Python, and
    # must never turn a mapped public address into a local one.
    assert wire_proxy._is_local(host) is local


def test_the_launchd_template_runs_the_wire_on_4011():
    require("deploy")  # deploy/ is not in the public tree (HAL#859)
    from lib.hal import deploy_render

    raw = plistlib.loads((REPO / "deploy" / "com.hal.wire.plist").read_bytes())
    assert raw["Label"] == "com.hal.wire"
    # --no-reclaim: a managed wire never kills whatever holds its port.
    assert raw["ProgramArguments"][1:] == ["-m", "lib.hal.wire_proxy", "--port", "__PORT__",
                                           "--no-reclaim"]
    assert raw["EnvironmentVariables"]["HAL_HOME"] == "__HAL_HOME__"
    assert "/usr/bin" in raw["EnvironmentVariables"]["PATH"].split(":")
    # Step 6 fills the port; by default it is the wire's own port.
    rendered = plistlib.loads(deploy_render.render(
        "launchd", "wire", python="/opt/grip/venv/bin/python", hal_home="/opt/hal",
        log_dir="/opt/hal-logs", check_paths=False).encode("utf-8"))
    assert rendered["ProgramArguments"][-3:] == ["--port", str(wire_proxy.DEFAULT_PORT),
                                                 "--no-reclaim"]
    assert wire_proxy.DEFAULT_PORT == 4011


def test_the_hosts_claude_code_needs_beside_the_api_are_tunnelled_blind():
    """Measured 2026-10-05: Claude Code reached these through the wire and each
    was refused. A refusal of channel.grip-web.com cuts GRIP's own channel bridge
    out of every wired session."""
    from lib.hal import wire_proxy as wp
    for host in ("downloads.claude.ai", "http-intake.logs.us5.datadoghq.com",
                 "channel.grip-web.com"):
        assert wp._route(host, 443) == "known"   # served even under the egress brake
    assert wp._route("evil.example", 443) == "unknown"


# --------------------------------------------------------------------------- #
# InfiniteMachine v2, Step 5a: the wire is never the error source either.
# --------------------------------------------------------------------------- #
def _error_body(reply: bytes) -> dict:
    return json.loads(reply.split(b"\r\n\r\n", 1)[1])["error"]


def test_an_upstream_that_hangs_up_before_answering_gets_the_client_a_503(make_wire):
    """The connection used to close with no answer at all. Now the client hears,
    in Anthropic's dialect, that the request may have reached Anthropic."""
    env = make_wire(scratch=True)
    reply = _exchange(env, _post("/v1/messages", {**TURN, "model": "hangup"}))
    assert reply.startswith(b"HTTP/1.1 503")
    error = _error_body(reply)
    assert error["type"] == "api_error" and "may have reached it" in error["message"]


def test_a_hang_up_before_the_answer_ends_the_requests_flight_on_the_ledger(make_wire):
    """Step 5a's 503 and the account ledger (HAL#651) on one exchange. The head is
    read inside the ledger's watch, so when the upstream hangs up first the client
    still hears the 503, and the ledger has counted the request and ended its
    flight with no answer recorded. A request left counted in flight would make a
    later hold or switch wait on one that is gone."""
    from lib.hal.account_ledger import UNPOOLED, LedgerView

    env = make_wire(scratch=True)
    reply = _exchange(env, _post("/v1/messages", {**TURN, "model": "hangup"}))
    assert reply.startswith(b"HTTP/1.1 503")
    view = LedgerView()                      # this test's own $HAL_HOME
    entry = view.accounts().get(UNPOOLED)
    assert entry is not None, "the ledger never counted the request"
    assert view.in_flight(UNPOOLED) == 0
    assert entry["observations"] == 0 and entry["http_status"] is None


def test_past_the_cap_a_turn_gets_a_503_with_retry_after_at_once(make_wire):
    cap = InflightCap(1, 4)
    env = make_wire(cap=cap, scratch=True)
    held = threading.Thread(target=_exchange, daemon=True,
                            args=(env, _post("/v1/messages", {**TURN, "model": "hold"})))
    held.start()
    assert _wait_for(lambda: any(b'"hold"' in raw for _n, raw in env.fake.requests))
    started = time.monotonic()
    reply = _exchange(env, _post("/v1/messages", TURN))
    assert reply.startswith(b"HTTP/1.1 503") and b"\r\nRetry-After: 4\r\n" in reply
    assert time.monotonic() - started < 1.0
    assert "limit of 1" in _error_body(reply)["message"]
    assert _exchange(env, ODD_REQUEST) == EXACT_REPLY   # account-bound: never capped
    env.fake.release.set()
    held.join(5)
    # The held turn's client has its whole answer a moment before the wire has
    # written the turn's row and given the slot back.
    assert _wait_for(lambda: cap.inflight == 0)
    assert _exchange(env, _post("/v1/messages", TURN)).startswith(b"HTTP/1.1 200")


class _CascadeSocket:
    def __init__(self, inbound=b""):
        self.inbound, self.sent = inbound, []

    def recv(self, size):
        part, self.inbound = self.inbound[:size], self.inbound[size:]
        return part

    def sendall(self, data):
        self.sent.append(data)


@pytest.mark.parametrize('answered', [False, True])
def test_original_send_tracks_only_unknown_acceptance(monkeypatch, answered):
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_hold
    payload = json.dumps(TURN).encode()
    request = wire_proxy._parse_head(
        f'POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n'
        'x-claude-code-session-id: fake-session\r\n\r\n'.encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request.headers, time.monotonic(),
                              body=wire_proxy._Copy(), wire_body=wire_proxy._Copy())
    head = wire_proxy._parse_head(b'HTTP/1.1 429 Limit\r\nContent-Length: 0\r\n\r\n')
    def answer(client, conn, exchange):
        exchange.answer_head_seen = answered
        return head if answered else None
    proxy = SimpleNamespace(_answer=answer, _w4_attempts=wire_cascade.Attempts())
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=lambda *_: None))
    result = wire_proxy.WireProxy._send_for_answer(
        proxy, wire_proxy._Stream(_CascadeSocket(payload)), conn, request, ex)
    assert (result is head) is answered
    key = wire_hold.turn_key(ex.headers, json.loads(payload))
    assert key is not None and proxy._w4_attempts.reserve(key) is answered


@pytest.mark.parametrize('route', [wire_proxy.RouteClass.INFERENCE,
                                   wire_proxy.RouteClass.UNKNOWN])
def test_anthropic_answer_head_is_recorded_before_judgement(route):
    from contextlib import nullcontext
    from types import SimpleNamespace
    head = wire_proxy._parse_head(b'HTTP/1.1 429 Limit\r\nContent-Length: 0\r\n\r\n')
    ex = wire_proxy._Exchange('POST', '/v1/messages', (), time.monotonic(), route=route)
    def judge(*args):
        assert ex.answer_head_seen
        raise RuntimeError('judgement stopped')
    proxy = SimpleNamespace(_response_head=lambda *args, **kwargs: head,
        _ledger=SimpleNamespace(exchange=lambda *_: nullcontext(
            SimpleNamespace(answered=lambda *_: None))),
        _judge=judge)
    if route is wire_proxy.RouteClass.INFERENCE:
        with pytest.raises(RuntimeError, match='judgement stopped'):
            wire_proxy.WireProxy._answer(proxy, None, SimpleNamespace(stream=None), ex)
    else:
        assert wire_proxy.WireProxy._answer(proxy, None, SimpleNamespace(stream=None), ex) is head
    assert ex.answer_head_seen


def test_body_send_timeout_after_complete_delivery_closes_turn(monkeypatch):
    from lib.hal import wire_cascade, wire_hold
    payload = json.dumps(TURN).encode()
    request = wire_proxy._parse_head(
        f'POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n'
        'x-claude-code-session-id: fake-session\r\n\r\n'.encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request.headers, time.monotonic(),
                              body=wire_proxy._Copy(), wire_body=wire_proxy._Copy())
    sent = []
    def send(data):
        sent.append(data)
        if data == payload:
            raise TimeoutError('delivered then timed out')
    proxy = SimpleNamespace(_w4_attempts=wire_cascade.Attempts())
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=send))
    client = wire_proxy._Stream(_CascadeSocket(payload))
    assert wire_proxy.WireProxy._send_for_answer(proxy, client, conn, request, ex) is None
    assert sent[-1] == payload
    key = wire_hold.turn_key(request.headers, TURN)
    assert not proxy._w4_attempts.reserve(key)


def test_no_answer_records_the_transport_failure_detail():
    """HAL#871: a transport failure during the send left no trace of why -- only
    a Python debug line naming the exception's type, nothing in the audit log a
    human or a later session could read. One ``wire.no_answer`` row now names
    the stage that failed and the exception's own errno and message."""
    from lib.hal import wire_cascade
    payload = json.dumps(TURN).encode()
    request = wire_proxy._parse_head(
        f'POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n'
        'x-claude-code-session-id: fake-session\r\n\r\n'.encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request.headers, time.monotonic(),
                              body=wire_proxy._Copy(), wire_body=wire_proxy._Copy())
    def send(data):
        if data == payload:
            raise ConnectionResetError(104, 'Connection reset by peer')
    rows = []
    proxy = SimpleNamespace(_w4_attempts=wire_cascade.Attempts(),
                            _telemetry=GatewayTelemetry(sink=rows.append))
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=send))
    client = wire_proxy._Stream(_CascadeSocket(payload))
    assert wire_proxy.WireProxy._send_for_answer(proxy, client, conn, request, ex) is None
    no_answer = [row for row in rows if row['event'] == 'wire.no_answer']
    assert len(no_answer) == 1
    assert no_answer[0]['stage'] == 'request_body'
    assert no_answer[0]['exception'] == 'ConnectionResetError'
    assert no_answer[0]['errno'] == 104
    assert 'Connection reset' in no_answer[0]['reason']


def test_no_answer_records_a_clean_close_with_no_exception_to_name():
    """A connection that closes cleanly before any response byte raises nothing
    -- ``_answer`` just returns ``None`` -- so before this fix NOTHING was
    recorded, not even the debug line the exception path got. The row must
    still appear, naming the stage with no exception to report."""
    from lib.hal import wire_cascade
    payload = json.dumps(TURN).encode()
    request = wire_proxy._parse_head(
        f'POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n'
        'x-claude-code-session-id: fake-session\r\n\r\n'.encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request.headers, time.monotonic(),
                              body=wire_proxy._Copy(), wire_body=wire_proxy._Copy())
    rows = []
    proxy = SimpleNamespace(_answer=lambda *a, **k: None, _w4_attempts=wire_cascade.Attempts(),
                            _telemetry=GatewayTelemetry(sink=rows.append))
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=lambda *_: None))
    client = wire_proxy._Stream(_CascadeSocket(payload))
    assert wire_proxy.WireProxy._send_for_answer(proxy, client, conn, request, ex) is None
    no_answer = [row for row in rows if row['event'] == 'wire.no_answer']
    assert len(no_answer) == 1
    assert no_answer[0]['stage'] == 'response_head'
    assert 'exception' not in no_answer[0]
    # HAL PR 930 review (xAI Grok 4.3): with nothing to report, the row
    # carries only its timestamp and the two fields asserted above -- never
    # an incidentally-added key nobody meant to record.
    assert set(no_answer[0]) == {'ts', 'event', 'stage'}


def test_no_answer_records_a_request_head_transport_failure_with_no_errno():
    """HAL PR 930 review (Alibaba Qwen4-Max): every other no-answer test
    fails during the body relay, so none exercises the earliest stage -- a
    transport failure while sending the request head itself -- nor an
    ``OSError`` with no ``errno`` to report (a plain ``OSError(message)``,
    unlike a real socket error, never sets one)."""
    from lib.hal import wire_cascade
    payload = json.dumps(TURN).encode()
    request = wire_proxy._parse_head(
        f'POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n'
        'x-claude-code-session-id: fake-session\r\n\r\n'.encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request.headers, time.monotonic(),
                              body=wire_proxy._Copy(), wire_body=wire_proxy._Copy())
    def send(_data):
        raise OSError('the socket is not connected')
    rows = []
    proxy = SimpleNamespace(_w4_attempts=wire_cascade.Attempts(),
                            _telemetry=GatewayTelemetry(sink=rows.append))
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=send))
    client = wire_proxy._Stream(_CascadeSocket(payload))
    assert wire_proxy.WireProxy._send_for_answer(proxy, client, conn, request, ex) is None
    no_answer = [row for row in rows if row['event'] == 'wire.no_answer']
    assert len(no_answer) == 1
    assert no_answer[0]['stage'] == 'request_head'
    assert no_answer[0]['exception'] == 'OSError'
    assert 'the socket is not connected' in no_answer[0]['reason']
    assert 'errno' not in no_answer[0]


def test_no_answer_never_quotes_a_value_error_message():
    """HAL PR 930 review: ``_record_no_answer``'s docstring promises never the
    request or response body, but a ``ValueError`` from parsing a malformed
    chunk size or Content-Length can quote the offending bytes verbatim in
    its own message. Truncating that message to 200 chars caps the leak, it
    does not prevent it -- only the exception's type may be recorded for a
    ``ValueError``, never its message."""
    from lib.hal import wire_cascade
    payload = json.dumps(TURN).encode()
    request = wire_proxy._parse_head(
        f'POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n'
        'x-claude-code-session-id: fake-session\r\n\r\n'.encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request.headers, time.monotonic(),
                              body=wire_proxy._Copy(), wire_body=wire_proxy._Copy())
    def send(data):
        if data == payload:
            raise ValueError("invalid literal for int() with base 16: b'SECRET-BODY-FRAGMENT'")
    rows = []
    proxy = SimpleNamespace(_w4_attempts=wire_cascade.Attempts(),
                            _telemetry=GatewayTelemetry(sink=rows.append))
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=send))
    client = wire_proxy._Stream(_CascadeSocket(payload))
    assert wire_proxy.WireProxy._send_for_answer(proxy, client, conn, request, ex) is None
    no_answer = [row for row in rows if row['event'] == 'wire.no_answer']
    assert len(no_answer) == 1
    assert no_answer[0]['exception'] == 'ValueError'
    assert 'reason' not in no_answer[0]
    assert 'SECRET-BODY-FRAGMENT' not in json.dumps(no_answer[0])


def test_no_answer_never_quotes_a_wire_protocol_error_message():
    """Same guarantee for ``_WireProtocolError``, the proxy's own wrapper
    around a misframed stream: its message is never recorded, only its type
    name -- a future raise site could embed the offending bytes same as a
    raw ``ValueError`` can."""
    from lib.hal import wire_cascade
    payload = json.dumps(TURN).encode()
    request = wire_proxy._parse_head(
        f'POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n'
        'x-claude-code-session-id: fake-session\r\n\r\n'.encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request.headers, time.monotonic(),
                              body=wire_proxy._Copy(), wire_body=wire_proxy._Copy())
    def send(data):
        if data == payload:
            raise wire_proxy._WireProtocolError(
                "a malformed chunk size: b'SECRET-BODY-FRAGMENT'")
    rows = []
    proxy = SimpleNamespace(_w4_attempts=wire_cascade.Attempts(),
                            _telemetry=GatewayTelemetry(sink=rows.append))
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=send))
    client = wire_proxy._Stream(_CascadeSocket(payload))
    assert wire_proxy.WireProxy._send_for_answer(proxy, client, conn, request, ex) is None
    no_answer = [row for row in rows if row['event'] == 'wire.no_answer']
    assert len(no_answer) == 1
    assert no_answer[0]['exception'] == '_WireProtocolError'
    assert 'reason' not in no_answer[0]
    assert 'SECRET-BODY-FRAGMENT' not in json.dumps(no_answer[0])


def test_no_answer_recording_failure_never_costs_the_client_its_503():
    """HAL PR 930 review: a failing telemetry sink must never cost the client
    its 503. Here the ``_telemetry`` object's own ``.record()`` raises
    directly, so the call never reaches ``GatewayTelemetry``'s sink wrapper
    at all -- this proves ``_record_no_answer``'s own try/except is what
    holds, not merely a property of one telemetry implementation."""
    from lib.hal import wire_cascade

    class _RaisingTelemetry:
        def record(self, event: dict) -> None:
            raise RuntimeError('disk full')

    payload = json.dumps(TURN).encode()
    request = wire_proxy._parse_head(
        f'POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n'
        'x-claude-code-session-id: fake-session\r\n\r\n'.encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request.headers, time.monotonic(),
                              body=wire_proxy._Copy(), wire_body=wire_proxy._Copy())
    proxy = SimpleNamespace(_answer=lambda *a, **k: None, _w4_attempts=wire_cascade.Attempts(),
                            _telemetry=_RaisingTelemetry())
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=lambda *_: None))
    client = wire_proxy._Stream(_CascadeSocket(payload))
    assert wire_proxy.WireProxy._send_for_answer(proxy, client, conn, request, ex) is None
    assert client.sock.sent == [wire_proxy._NO_ANSWER]


def test_client_body_ends_early_without_closing_turn():
    """The client declares two more bytes than it sends, after a COMPLETE JSON value.
    A key can be built from what arrived, so only the early-end exemption keeps the turn
    open: Anthropic is still waiting for the rest and cannot have accepted it."""
    from lib.hal import wire_cascade, wire_hold
    payload = json.dumps(TURN).encode()
    request = wire_proxy._parse_head(
        f'POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload) + 2}\r\n'
        'x-claude-code-session-id: fake-session\r\n\r\n'.encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request.headers, time.monotonic(),
                              body=wire_proxy._Copy(), wire_body=wire_proxy._Copy())
    proxy = SimpleNamespace(_w4_attempts=wire_cascade.Attempts())
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=lambda *_: None))
    assert wire_proxy.WireProxy._send_for_answer(
        proxy, wire_proxy._Stream(_CascadeSocket(payload)), conn, request, ex) is None
    assert wire_hold.exchange_key(ex) == wire_hold.turn_key(request.headers, TURN)
    assert proxy._w4_attempts.reserve(wire_hold.turn_key(request.headers, TURN), 'fake-session')


def _streamed_send(answered, on_last_bytes=None):
    """Run one streamed send of the standard turn; return (registry, turn key).
    ``on_last_bytes`` runs while the final body bytes are being handed to Anthropic."""
    from lib.hal import wire_cascade, wire_hold
    payload = json.dumps(TURN).encode()
    request = wire_proxy._parse_head(
        f'POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n'
        'x-claude-code-session-id: fake-session\r\n\r\n'.encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request.headers, time.monotonic(),
                              body=wire_proxy._Copy(), wire_body=wire_proxy._Copy())
    attempts, key = wire_cascade.Attempts(), wire_hold.turn_key(request.headers, TURN)
    head = wire_proxy._parse_head(b'HTTP/1.1 429 Limit\r\nContent-Length: 0\r\n\r\n')

    def sendall(data):
        if on_last_bytes and data.endswith(payload[-8:]):
            on_last_bytes(attempts, key)

    def answer(client, conn, exchange):
        exchange.answer_head_seen = answered
        return head if answered else None

    proxy = SimpleNamespace(_answer=answer, _w4_attempts=attempts)
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=sendall))
    wire_proxy.WireProxy._send_for_answer(
        proxy, wire_proxy._Stream(_CascadeSocket(payload)), conn, request, ex)
    return attempts, key


def test_a_streamed_send_blocks_w4_before_its_last_byte_is_delivered():
    """Astra's fourth review: the turn was registered only after its whole body had
    been relayed, so an offer could be authorised while the last bytes were leaving."""
    during = []
    attempts, key = _streamed_send(
        answered=False,
        on_last_bytes=lambda registry, turn: during.append(registry.reserve(turn, 'fake-session')))
    assert during == [False]
    assert not attempts.reserve(key, 'fake-session')


def test_an_answered_streamed_send_leaves_no_mark_behind():
    """A mark that outlived its answer would close the session to W4 for the life of
    the process."""
    attempts, key = _streamed_send(answered=True)
    assert attempts._outstanding == {}
    assert attempts.reserve(key, 'fake-session')


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("served", [False, True])
@pytest.mark.parametrize("hold_enabled", [False, True])
def test_spend_refusal_offers_bounded_fresh_body_once(monkeypatch, chunked, served, hold_enabled):
    from lib.hal import spend_guard, wire_cascade, wire_hold
    payload = json.dumps(TURN).encode()
    body = f"{len(payload):x}\r\n".encode() + payload + b"\r\n0\r\n\r\n" if chunked else payload
    framing = b"Transfer-Encoding: chunked\r\n" if chunked else f"Content-Length: {len(body)}\r\n".encode()
    head = wire_proxy._parse_head(b"POST /v1/messages HTTP/1.1\r\nx-claude-code-session-id: fake-session\r\n"
                                  + framing + b"\r\n")
    sock = _CascadeSocket(body)
    rows, attempts = [], []
    proxy = SimpleNamespace(_telemetry=SimpleNamespace(record=lambda row: None), _record=rows.append,
                            _cap=SimpleNamespace(try_enter=lambda: True, leave=lambda: None),
                            _w4_attempts=wire_cascade.Attempts())
    proxy._offer_spend_refusal = lambda *args: wire_proxy.WireProxy._offer_spend_refusal(proxy, *args)
    refusal = spend_guard.Decision(False, spend_guard.OVERAGE, "spent", status=429)
    monkeypatch.setattr(spend_guard, "check_anthropic_request", lambda *args: refusal)
    monkeypatch.setattr(wire_hold, "enabled", lambda: hold_enabled)
    monkeypatch.setattr(wire_hold, "exhaustion", lambda *args: (True, None))
    monkeypatch.setattr(wire_cascade, "configured_targets", lambda: [object()])
    def fake_move(proxy, client, ex):
        wire_cascade.eligible(ex)
        attempts.append((ex.body.value(), ex.wire_body.value()))
        if served:
            client.sock.sendall(b"TARGET")
        return served
    monkeypatch.setattr(wire_cascade, "move", fake_move)
    assert wire_proxy.WireProxy._refused(proxy, wire_proxy._Stream(sock), head)
    assert attempts == [(payload, body)]
    assert sock.sent == ([b"TARGET"] if served else [wire_proxy._spend_refusal(refusal)])
    assert len(rows) == 1 and (served or (rows[0].status == 429 and rows[0].completed))


@pytest.mark.parametrize("case", ["assistant", "oversized", "chunked_oversized",
                                  "route", "no_target", "forbidden"])
def test_spend_refusal_closed_paths_keep_exact_bytes(monkeypatch, case):
    from lib.hal import spend_guard, wire_cascade, wire_hold
    payload = dict(TURN)
    if case == "assistant":
        payload["messages"] = [{"role": "assistant", "content": "old"}, *TURN["messages"]]
    body = json.dumps(payload).encode()
    chunked = case == "chunked_oversized"
    body = f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n" if chunked else body
    path = "/v1/messages/count_tokens" if case == "route" else "/v1/messages"
    framing = "Transfer-Encoding: chunked" if chunked else f"Content-Length: {len(body)}"
    head = wire_proxy._parse_head((f"POST {path} HTTP/1.1\r\n{framing}\r\n"
                                   "x-claude-code-session-id: fake-session\r\n\r\n").encode())
    sock, targets = _CascadeSocket(body), []
    proxy = SimpleNamespace(_telemetry=SimpleNamespace(record=lambda row: None),
                            _record=lambda ex: None,
                            _cap=SimpleNamespace(try_enter=lambda: True, leave=lambda: None))
    proxy._offer_spend_refusal = lambda *args: wire_proxy.WireProxy._offer_spend_refusal(proxy, *args)
    refusal = spend_guard.Decision(False, spend_guard.OVERAGE, "spent", status=403 if case == "forbidden" else 429)
    monkeypatch.setattr(spend_guard, "check_anthropic_request", lambda *args: refusal)
    monkeypatch.setattr(wire_hold, "exhaustion", lambda *args: (True, None))
    monkeypatch.setattr(wire_cascade, "configured_targets", lambda: [] if case == "no_target" else [object()])
    if case in ("oversized", "chunked_oversized"):
        monkeypatch.setattr(wire_proxy, "_MAX_COPY", 16)
    if chunked:
        original = wire_proxy._Copy
        monkeypatch.setattr(wire_proxy, "_Copy", lambda: original(limit=16))
    monkeypatch.setattr(wire_cascade, "move", lambda p, c, e: targets.append(e) or True)
    assert wire_proxy.WireProxy._refused(proxy, wire_proxy._Stream(sock), head)
    assert not targets and sock.sent == [wire_proxy._spend_refusal(refusal)]


@pytest.mark.parametrize("delivered", [False, True])
def test_a_spend_refusal_is_recorded_as_answered_only_once_it_is_sent(monkeypatch, delivered):
    """Codex on HAL#747: the row was marked completed, and written, before the refusal
    was sent. A client that had hung up left a signed row claiming an answer it never
    got. The row is read here at the moment it is written."""
    from lib.hal import spend_guard, wire_cascade, wire_hold
    payload = json.dumps(TURN).encode()
    head = wire_proxy._parse_head(
        f"POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n"
        "x-claude-code-session-id: fake-session\r\n\r\n".encode())

    class Client(_CascadeSocket):
        def sendall(self, data):
            if not delivered:
                raise BrokenPipeError("the client hung up")
            super().sendall(data)

    sock, rows = Client(payload), []
    proxy = SimpleNamespace(_telemetry=SimpleNamespace(record=lambda row: None),
                            _record=lambda ex: rows.append((ex.status, ex.completed)),
                            _cap=SimpleNamespace(try_enter=lambda: True, leave=lambda: None),
                            _w4_attempts=wire_cascade.Attempts())
    proxy._offer_spend_refusal = lambda *args: wire_proxy.WireProxy._offer_spend_refusal(proxy, *args)
    refusal = spend_guard.Decision(False, spend_guard.OVERAGE, "spent", status=429)
    monkeypatch.setattr(spend_guard, "check_anthropic_request", lambda *args: refusal)
    monkeypatch.setattr(wire_hold, "exhaustion", lambda *args: (True, None))
    monkeypatch.setattr(wire_cascade, "configured_targets", lambda: [object()])
    monkeypatch.setattr(wire_cascade, "move", lambda proxy, client, ex: False)
    if delivered:
        assert wire_proxy.WireProxy._refused(proxy, wire_proxy._Stream(sock), head)
        assert sock.sent == [wire_proxy._spend_refusal(refusal)]
    else:
        with pytest.raises(BrokenPipeError):
            wire_proxy.WireProxy._refused(proxy, wire_proxy._Stream(sock), head)
    assert rows == [(429, delivered)]


def test_an_unanswered_send_is_closed_before_its_marks_go_and_leaves_none_behind(monkeypatch):
    """Codex on HAL#747: a send that got no answer kept its turn's mark for the life of
    the process. Both marks must still stand while the closure is written, so that no
    offer can pass between them, and both must be gone afterwards. The closure is the
    session's own (AGORA's third falsifier): the same turn from another session can
    still be reserved."""
    from lib.hal import wire_cascade, wire_hold
    at_closure, real_close = [], wire_cascade.Attempts.close

    def watched_close(self, key):
        at_closure.append(dict(self._outstanding))
        real_close(self, key)

    monkeypatch.setattr(wire_cascade.Attempts, "close", watched_close)
    attempts, key = _streamed_send(answered=False)
    assert at_closure == [{"fake-session": 1, key: 1}]
    assert attempts._outstanding == {} and attempts.closed(key)
    assert not attempts.reserve(key, "fake-session")
    other = wire_hold.turn_key((("x-claude-code-session-id", "other-session"),), TURN)
    assert other != key and attempts.reserve(other, "other-session")


def test_a_streamed_send_that_fails_part_way_through_its_body_stays_open():
    """AGORA on HAL#747 read this known limit as out of date. It is not. When the
    upstream fails before the whole body has been handed to it, no key can be built
    from what was copied, so no closure is written: Anthropic cannot have accepted an
    incomplete body. Both marks are still removed."""
    from lib.hal import wire_cascade, wire_hold
    payload = json.dumps(TURN).encode()
    request = wire_proxy._parse_head(
        f"POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n"
        "x-claude-code-session-id: fake-session\r\n\r\n".encode())
    ex = wire_proxy._Exchange("POST", "/v1/messages", request.headers, time.monotonic(),
                              body=wire_proxy._Copy(), wire_body=wire_proxy._Copy())

    class Trickle(_CascadeSocket):
        def recv(self, size):
            return super().recv(min(size, 16))

    sent = []

    def send(data):
        sent.append(data)
        if len(sent) == 2:  # the head went out; the first sixteen body bytes did not
            raise ConnectionResetError("the upstream went away mid-body")

    proxy = SimpleNamespace(_w4_attempts=wire_cascade.Attempts())
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=send))
    assert wire_proxy.WireProxy._send_for_answer(
        proxy, wire_proxy._Stream(Trickle(payload)), conn, request, ex) is None
    key = wire_hold.turn_key(request.headers, TURN)
    assert len(sent) == 2 and wire_hold.exchange_key(ex) is None
    assert not proxy._w4_attempts.closed(key) and proxy._w4_attempts._outstanding == {}
    assert proxy._w4_attempts.reserve(key, "fake-session")
