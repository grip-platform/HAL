"""wire_proxy -- the HAL wire: a loopback intercepting HTTPS proxy on 127.0.0.1:4011.

InfiniteMachine v2, Step 2 (``docs/infinite-machine-contract.md``). Claude Code
keeps ``api.anthropic.com`` as its address -- which Remote Control requires -- and
sends its HTTPS through this proxy. The proxy decrypts ``api.anthropic.com`` only,
under the per-host CA in ``lib/hal/wire_ca.py``, so HAL sees every inference call.
Every other host is tunnelled blind: bytes are copied, nothing is decrypted.

How GRIP's launcher wires a Claude Code session (Step 3):

    HTTPS_PROXY=http://127.0.0.1:4011          (and https_proxy, the same)
    NO_PROXY=127.0.0.1,localhost,.ts.net
    NODE_EXTRA_CA_CERTS=$HAL_HOME/state/wire-ca/hal-wire-ca.pem
    ANTHROPIC_BASE_URL                         unset

It wires nothing unless ``GET http://127.0.0.1:4011/health`` answers ``{"ok": true,
"service": "hal-wire", "brake": {...}}``, so a dead wire means a direct session
with its "unwitnessed" badge, never a broken one.

What the proxy does with each CONNECT:

  api.anthropic.com:443  decrypt with a leaf minted in memory from the wire CA
                         (ALPN http/1.1); verify the REAL upstream certificate
                         against the system trust store; relay every request and
                         response byte for byte; sort each request with
                         ``lib/hal/wire_classify.py``:
                           inference, aux   one witness row through
                                            ``gateway_witness.witness_chat``:
                                            status, timing, model, and usage from
                                            the body or the stream; and the
                                            account ledger
                                            (``lib/hal/account_ledger.py``)
                                            counts it in flight until its
                                            answer begins, then reads that
                                            answer's rate-limit headers into
                                            numbers (observe only)
                           account-bound    nothing recorded; the body is not read
                           unknown          one audit row: method and path only
  BLIND_HOSTS, *.sentry.io (port 443)
                         the hosts Claude Code itself needs: a blind tunnel,
                         nothing recorded
  any other host         a blind tunnel and one ``wire.tunnelled`` row naming the
                         host and port only (GRIP#7460: commands, connectors and
                         WebFetch reach GitHub, registries and the open web).
                         While the egress brake is engaged: 403, naming the
                         brake's marker, and one ``wire.refused`` row
  api.anthropic.com on another port, loopback or link-local targets
                         403, and one audit row naming the host only

Relay rules. Apart from an eligible account switch's bearer, the request line,
headers, body and response go through exactly as they arrived.
Each read is forwarded the moment it returns, and every relayed socket sets
TCP_NODELAY so a small write is not held back either, so a server-sent event
reaches the client as soon as the upstream sends it (HAL's gateway waited for up
to 64 KiB, fleet finding f79). Bodies are framed (chunked, Content-Length, or until close)
so one connection can carry the next request. A 101 (voice's WebSocket) turns the
connection into a raw copy in both directions; interim 1xx answers are relayed as
they come. If the upstream fails after the response has started, the client
connection is closed, so the client sees a cut stream, never a hang. If it fails
before any of the answer arrives (a reset, a timeout, a close), the client is sent
a 503 in Anthropic's dialect, which Claude Code retries, rather than a closed
connection with no answer (InfiniteMachine v2, Step 5a). A limit-reached 429 on
an inference turn can be replayed once on another account, only before response
bytes reach the client. The original 429 stays available if the replacement
rejects a thinking signature or fails before answering. Holds with pings remain
outside this change (Step 5d). A 401 or 403 on a witnessed route has its body
read, then put back byte for byte, and judged (``lib/hal/auth_signal.py``): a
message that names a revocation or a suspension, from any account, stops all
account switching and tells the operator (``account_switch.stop_for_enforcement``).
A 403 on the presented token is never switched or replayed, and neither is a 401
that names an enforcement action; only another 401 is (the dead-token switch).
Headers, bearers and bodies are never written to any log; a failed connection is
logged by its exception type only.

In-flight cap (Step 5a; fleet findings f82, f121). At most ``HAL_WIRE_MAX_INFLIGHT``
(default 64) requests on the messages rail -- ``POST /v1/messages`` and its aux
paths -- are relayed at once; past that a request is answered at once with a 503
and ``Retry-After: HAL_WIRE_BUSY_RETRY_AFTER_S`` (default 5), never a 429 and never
a wait, so a fan-out cannot pile up behind the wire. Account-bound routes (Remote
Control, voice, OAuth) are never capped.

Warm upstream connections. On exit Claude Code gives its Remote Control archive
call under 0.6 s. The 2026-10-04 probe's interceptor missed that deadline
("timeout of 599ms exceeded") and orphaned a session. So a small pool keeps ready
TLS connections to api.anthropic.com. It refreshes them before they can go stale
and resumes TLS sessions where the server allows. Each client connection also
keeps its upstream connection across keep-alive requests.

Two brakes, both marker files (``lib/hal/gateway_brake.py``), both reported by
``/health``. The InfiniteMachine brake (``brake``) stops HAL's own outward actions,
never the user's traffic: every request is still relayed while it is engaged. The
egress brake (``egress_brake``) is the operator's control over the open tunnel: it
refuses hosts the wire does not know, from the next CONNECT, and leaves the API and
Claude Code's own hosts alone. Its marker is ``$HAL_WIRE_EGRESS_BRAKE_MARKER`` when
set, else ``${CLAUDE_DIR:-~/.claude}/state/session/wire-egress-brake.json``; like the
other brake, a marker that is present but unreadable, or a path that cannot be
checked, counts as engaged.

Run: ``python -m lib.hal.wire_proxy [--port 4011] [--no-reclaim]``. As a service:
``deploy/`` holds a launchd plist, a systemd user unit and a Task Scheduler task,
filled by ``lib/hal/deploy_render.py`` and installed by GRIP's
``bin/grip-hal-services`` (Step 6).
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import selectors
import socket
import ssl
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from typing import Callable, Optional

from lib.hal import account_switch, auth_signal, spend_guard, wire_ca
from lib.hal.gateway import InflightCap
from lib.hal.gateway_brake import MarkerBrake, brake_state
from lib.hal.gateway_failover import GatewayTelemetry
from lib.hal.gateway_witness import session_id_from_headers, witness_chat
from lib.hal.wire_classify import (AUX_PATHS, MESSAGES_PATH, RouteClass, StreamDecoder,
                                   classify, decoded_body, header_value, request_json,
                                   request_path)

SERVICE = "hal-wire"
DEFAULT_PORT = 4011
TLS_PORT = 443
DECRYPT_HOST = wire_ca.CONSTRAINED_HOST
# Hosts Claude Code 2.1.289 reached through the wire in the first live hour
# (2026-10-05) besides api.anthropic.com. Each is tunnelled blind: the wire
# never reads them. downloads.claude.ai carries auto-updates, the Datadog
# intake carries Claude Code's own telemetry, channel.grip-web.com is GRIP's
# channel bridge, which a wired session must still reach. Each was refused
# before this list carried it (37 wire.refused rows that hour).
BLIND_HOSTS = frozenset({
    "platform.claude.com", "claude.ai", "statsig.anthropic.com",
    "downloads.claude.ai", "http-intake.logs.us5.datadoghq.com",
    "channel.grip-web.com",
})
BLIND_SUFFIXES = (".sentry.io",)
#: The egress brake (GRIP#7460). Any other host is tunnelled blind, with one
#: audit row naming it; while this marker exists such hosts are refused again,
#: with a 403 that names the marker. Claude Code's own hosts above and the API
#: are served either way. Distinct from the InfiniteMachine brake, which never
#: touches the user's traffic.
EGRESS_BRAKE = MarkerBrake("HAL_WIRE_EGRESS_BRAKE_MARKER",
                           ("state", "session", "wire-egress-brake.json"))

_CHUNK = 65536                      # most bytes taken per read; reads return early
_MAX_HEAD = 65536                   # a request or response head larger than this is refused
_MAX_LINE = 8192                    # a chunk-size or trailer line
_MAX_COPY = 64 * 1024 * 1024        # body copies kept for classification and usage
_MAX_EVENT = 1024 * 1024            # one server-sent event held while it is scanned
_CONNECT_TIMEOUT_S = 10.0
_HANDSHAKE_TIMEOUT_S = 15.0
_IDLE_TIMEOUT_S = 900.0             # no healthy stream is silent this long
_POOL_SIZE = 2
_POOL_IDLE_TTL_S = 45.0
_POOL_TICK_S = 5.0
_LEAF_REFRESH_S = 3600.0
#: The routes the wire witnesses, and the signed thread row each earns (Step 4),
#: exactly as the gateway assigns them: a turn is ``served``; a quota check, a
#: token count or a title call is ``aux``.
_THREAD_ROW = {RouteClass.INFERENCE: "served", RouteClass.AUX: "aux"}
_WITNESSED = frozenset(_THREAD_ROW)
_BODY_PATHS = frozenset({MESSAGES_PATH}) | AUX_PATHS
#: The replay outcome recorded when a turn is answered with the honest exhaustion 429.
_PAUSED = "paused_until_reset"
#: The replay outcome recorded when that 429 names the present: a pooled
#: subscription has capacity again (HAL#748 item 18).
_RETRY_NOW = "retry_now"

_ESTABLISHED = b"HTTP/1.1 200 Connection Established\r\n\r\n"
_FORBIDDEN = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
_BAD_GATEWAY = b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"

_log = logging.getLogger("hal.wire")


class _WireProtocolError(Exception):
    """A message the wire cannot frame. Fixed text only, never request data."""


def _error_response(status: int, reason: str, message: str, *,
                    error_type: str = "api_error",
                    retry_after: Optional[int] = None) -> bytes:
    body = json.dumps({"type": "error",
                       "error": {"type": error_type, "message": message}}).encode()
    retry = f"Retry-After: {retry_after}\r\n" if retry_after else ""
    head = (f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n{retry}"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()
    return head + body


def _spend_refusal(decision: spend_guard.Decision) -> bytes:
    """The spend guard's refusal, Anthropic-shaped. Nothing was sent upstream:
    403 for a paid key; 429 with Retry-After for a spent subscription."""
    if decision.status == 429:
        return _error_response(429, "Too Many Requests", decision.reason,
                               error_type="rate_limit_error",
                               retry_after=decision.retry_after)
    return _error_response(decision.status, "Forbidden", decision.reason,
                           error_type="permission_error")


_UNREACHABLE = _error_response(
    503, "Service Unavailable",
    "HAL wire: api.anthropic.com could not be reached; nothing was sent to it")
# Sent when the upstream fails after the request went out but before any of the
# answer came back. Honest about the ambiguity: Anthropic may have the request.
_NO_ANSWER = _error_response(
    503, "Service Unavailable",
    "HAL wire: the connection to api.anthropic.com failed before an answer "
    "arrived; the request may have reached it; retry your request")


def _busy_response(cap: InflightCap) -> bytes:
    """The answer past the in-flight cap: 503, Retry-After, Anthropic's dialect.
    No ``Connection: close``, so the client may send its retry on this connection."""
    body = json.dumps({"type": "error", "error": {"type": "api_error",
                                                  "message": cap.busy_message()}}).encode()
    head = (f"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
            f"Retry-After: {cap.retry_after_s}\r\nContent-Length: {len(body)}\r\n\r\n")
    return head.encode() + body


class _Discard:
    """A sink for a request body nobody will read (one turned away at the cap)."""

    def sendall(self, _data: bytes) -> None:
        return None


def _send_quietly(sock, data: bytes) -> None:
    """Send ``data``; a client that has gone away is not an error."""
    try:
        sock.sendall(data)
    except (OSError, ValueError):
        pass


def _no_delay(sock) -> None:
    """Send every write at once. A relay must not hold small writes back (Nagle)."""
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except (OSError, AttributeError):
        pass


def _wait_readable(socks, timeout: float) -> list:
    """Those of ``socks`` with something to read (bytes or a close), waiting up to
    ``timeout`` seconds. ``selectors``, because ``select.select`` refuses any
    descriptor numbered above 1023."""
    with selectors.DefaultSelector() as selector:
        for sock in socks:
            selector.register(sock, selectors.EVENT_READ)
        return [key.fileobj for key, _ in selector.select(timeout)]


def _recv_now(sock, size: int) -> Optional[bytes]:
    """One read that never waits: bytes, b'' at a close, or None when nothing could
    be handed over. A TLS socket can be readable with nothing for the reader: a
    TLS 1.3 server sends session tickets just after the handshake, for one."""
    timeout = sock.gettimeout()
    sock.settimeout(0)
    try:
        return sock.recv(size)
    except (ssl.SSLWantReadError, ssl.SSLWantWriteError, BlockingIOError):
        return None
    finally:
        sock.settimeout(timeout)


def _close(*socks) -> None:
    """Shut down and close, quietly. Only for sockets no other thread is using."""
    for sock in socks:
        if sock is None:
            continue
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except (OSError, ValueError):
            pass
        try:
            sock.close()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Reading bytes exactly as they arrive.
# --------------------------------------------------------------------------- #
class _Stream:
    """Buffered reads over one socket that keep every byte exactly as it arrived."""

    def __init__(self, sock):
        self.sock = sock
        self.buf = bytearray()

    def _fill(self) -> bool:
        chunk = self.sock.recv(_CHUNK)
        if not chunk:
            return False
        self.buf += chunk
        return True

    def _take(self, count: int) -> bytes:
        data = bytes(self.buf[:count])
        del self.buf[:count]
        return data

    def read_some(self, limit: int) -> bytes:
        """Up to ``limit`` bytes: what is buffered, else what one read returns; b'' at EOF."""
        if not self.buf and not self._fill():
            return b""
        return self._take(min(limit, len(self.buf)))

    def read_head(self) -> Optional[bytes]:
        """One message head through its blank line; None at EOF before any byte."""
        return self._read_through(b"\r\n\r\n", _MAX_HEAD) or None

    def read_line(self) -> bytes:
        """Through the next CRLF; b'' at EOF before any byte."""
        return self._read_through(b"\r\n", _MAX_LINE)

    def _read_through(self, marker: bytes, limit: int) -> bytes:
        start = 0
        while True:
            end = self.buf.find(marker, start)
            if end >= 0:
                return self._take(end + len(marker))
            if len(self.buf) > limit:
                raise _WireProtocolError("a message head or line is too long")
            start = max(0, len(self.buf) - len(marker) + 1)
            if not self._fill():
                if self.buf:
                    raise _WireProtocolError("the connection closed inside a head or line")
                return b""


@dataclass(frozen=True)
class _Head:
    """A request or response head: the raw bytes plus a parsed view of them."""

    raw: bytes
    parts: tuple
    headers: tuple

    def header(self, name: str) -> str:
        return header_value(self.headers, name)


def _parse_head(raw: bytes) -> _Head:
    lines = raw.decode("latin-1").split("\r\n")
    parts = tuple(lines[0].split(" ", 2))
    if len(parts) < 2:
        raise _WireProtocolError("a malformed start line")
    headers = tuple(tuple(item.strip() for item in line.split(":", 1))
                    for line in lines[1:] if ":" in line)
    return _Head(raw=raw, parts=parts, headers=headers)


def _with_bearer(head: _Head, token: str) -> _Head:
    """``head`` with its Authorization replaced by ``Bearer token``; nothing else changes."""
    lines = head.raw.decode("latin-1").split("\r\n")
    for index, line in enumerate(lines[1:], 1):
        if line.split(":", 1)[0].strip().lower() == "authorization":
            lines[index] = f"Authorization: Bearer {token}"
    return _parse_head("\r\n".join(lines).encode("latin-1"))


def _status(head: _Head) -> int:
    try:
        return int(head.parts[1])
    except ValueError as exc:
        raise _WireProtocolError("a malformed status line") from exc


# --------------------------------------------------------------------------- #
# Framing and relaying bodies.
# --------------------------------------------------------------------------- #
def _content_length(head: _Head) -> Optional[int]:
    raw = head.header("content-length")
    if not raw:
        return None
    values = {value.strip() for value in raw.split(",")}
    value = values.pop() if len(values) == 1 else ""
    if not value.isdigit():
        raise _WireProtocolError("an unusable Content-Length")
    return int(value)


def _is_chunked(head: _Head) -> bool:
    codings = [c.strip().lower() for c in head.header("transfer-encoding").split(",")]
    return codings[-1] == "chunked"


def _request_framing(head: _Head) -> tuple:
    if _is_chunked(head):
        return ("chunked", 0)
    return ("length", _content_length(head) or 0)


def _response_framing(head: _Head, method: str) -> tuple:
    status = _status(head)
    if method == "HEAD" or status in (204, 304) or status < 200:
        return ("length", 0)
    if _is_chunked(head):
        return ("chunked", 0)
    length = _content_length(head)
    return ("close", 0) if length is None else ("length", length)


def _feed(tap, piece: bytes) -> None:
    if tap is not None:
        tap(piece)


def _relay_length(src: _Stream, dst, length: int, tap) -> bool:
    """Forward exactly ``length`` bytes, each read as it returns. False on early EOF."""
    remaining = length
    while remaining > 0:
        piece = src.read_some(min(remaining, _CHUNK))
        if not piece:
            return False
        dst.sendall(piece)
        _feed(tap, piece)
        remaining -= len(piece)
    return True


def _chunk_size(line: bytes) -> int:
    try:
        return int(line.split(b";", 1)[0].strip(), 16)
    except ValueError as exc:
        raise _WireProtocolError("a malformed chunk size") from exc


def _relay_chunked(src: _Stream, dst, _length: int, tap) -> bool:
    """Forward a chunked body as it arrives, framing included; the tap sees payload only."""
    while True:
        line = src.read_line()
        if not line:
            return False
        dst.sendall(line)
        size = _chunk_size(line)
        if size == 0:
            return _relay_trailers(src, dst)
        if not _relay_length(src, dst, size, tap):
            return False
        ending = src.read_line()
        if ending != b"\r\n":
            return False
        dst.sendall(ending)


def _relay_trailers(src: _Stream, dst) -> bool:
    while True:
        line = src.read_line()
        if not line:
            return False
        dst.sendall(line)
        if line == b"\r\n":
            return True


def _relay_until_close(src: _Stream, dst, _length: int, tap) -> bool:
    while True:
        piece = src.read_some(_CHUNK)
        if not piece:
            return True
        dst.sendall(piece)
        _feed(tap, piece)


_RELAYS = {"length": _relay_length, "chunked": _relay_chunked, "close": _relay_until_close}


def _relay(src: _Stream, dst, framing: tuple, tap=None) -> bool:
    """Forward one body from ``src`` to ``dst``. True when the body ended where framed."""
    kind, length = framing
    return _RELAYS[kind](src, dst, length, tap)


def _splice(left: _Stream, right: _Stream) -> None:
    """Copy bytes both ways until either side ends, then close both. One thread.

    Every read and write happens on the calling thread. A thread per direction
    would use each TLS connection from two threads at once, which OpenSSL does not
    promise to survive ("most objects are not safe for simultaneous use"), and
    CPython's ``SSLSocket.shutdown`` drops the TLS layer before it shuts the socket,
    so a second thread still writing could put plaintext on the wire. Two threads
    also let one close a socket the other was still inside: seen 2026-10-05, a
    stray TLS alert written into the next CONNECT. Buffered bytes go first. The
    splice ends when either side closes or fails, or both stay silent too long.
    """
    peer = {left: right, right: left}
    try:
        while _hand_over(_ready(left, right), peer):
            pass
    except (OSError, ValueError):
        return
    finally:
        _close(left.sock, right.sock)


def _hand_over(ready: list, peer: dict) -> bool:
    """Pass each ready stream's bytes to its peer. False once either side closed."""
    for src in ready:
        piece = _read_now(src)
        if piece == b"":
            return False
        if piece:
            peer[src].sock.sendall(piece)
    return True


def _ready(*streams: _Stream) -> list:
    """The streams with something to hand over, waiting up to the idle timeout."""
    ready = [stream for stream in streams if stream.buf or _pending(stream.sock)]
    if ready:
        return ready
    socks = set(_wait_readable([stream.sock for stream in streams], _IDLE_TIMEOUT_S))
    if not socks:
        raise TimeoutError("both directions of a splice stayed silent")
    return [stream for stream in streams if stream.sock in socks]


def _pending(sock) -> int:
    """Decrypted bytes a TLS socket already holds, which no readiness check sees."""
    pending = getattr(sock, "pending", None)
    return pending() if pending else 0


def _read_now(stream: _Stream) -> Optional[bytes]:
    """Buffered bytes first, else one read that never waits (see ``_recv_now``)."""
    if stream.buf:
        return stream.read_some(len(stream.buf))
    return _recv_now(stream.sock, _CHUNK)


def _keep_alive(request: _Head, response: _Head, framing: str) -> bool:
    if framing == "close" or len(request.parts) < 3 or request.parts[2] != "HTTP/1.1":
        return False
    tokens = f"{request.header('connection')},{response.header('connection')}".lower()
    return "close" not in {token.strip() for token in tokens.split(",")}


# --------------------------------------------------------------------------- #
# Copies kept for classification and usage. Headers are never kept.
# --------------------------------------------------------------------------- #
class _Copy:
    """A bounded copy of a body's payload (chunk framing removed)."""

    def __init__(self, limit: int = _MAX_COPY):
        self.data = bytearray()
        self.limit = limit
        self.overflow = False

    def __call__(self, piece: bytes) -> None:
        if self.overflow:
            return
        if len(self.data) + len(piece) > self.limit:
            self.overflow = True
            self.data.clear()
            return
        self.data += piece

    def value(self) -> bytes:
        return b"" if self.overflow else bytes(self.data)

    def sendall(self, data: bytes) -> None:
        self(data)
        if self.overflow:
            raise _WireProtocolError("a refused body exceeds the copy bound")


def _capture_refused_body(client: _Stream, request: _Head):
    """Keep decoded and exact wire bytes within the existing copy bound."""
    try:
        framing = _request_framing(request)
        if framing[0] == "length" and framing[1] > _MAX_COPY:
            return None
        body, wire = _Copy(), _Copy()
        if not _relay(client, wire, framing, body):
            return None
        return body, wire
    except (OSError, ValueError, _WireProtocolError):
        return None


class _RequestTee:
    """Forward body bytes while retaining their exact HTTP framing for one replay."""

    def __init__(self, sock, copy: _Copy):
        self.sock, self.copy = sock, copy

    def sendall(self, data: bytes) -> None:
        self.copy(data)
        self.sock.sendall(data)


#: The most of a 401's or 403's body read to judge it. It is never kept, logged or
#: changed: its bytes go back on the stream they came from, unread by the client.
_MAX_JUDGED_BODY = 65536


class _Peek:
    """A sink that keeps every byte it is handed, and stops past its limit."""

    def __init__(self, limit: int):
        self.raw = bytearray()
        self.limit = limit

    def sendall(self, data: bytes) -> None:
        self.raw += data
        if len(self.raw) > self.limit:
            raise _WireProtocolError("a body too large to judge")


def _peek_body(stream: "_Stream", head: "_Head", method: str) -> Optional[bytes]:
    """The body that follows ``head`` (chunk framing removed), read and then put
    back on ``stream`` byte for byte, so the next relay sends it exactly as it
    came. None when it is larger than ``_MAX_JUDGED_BODY`` or could not be read
    whole; what was read goes back either way. Never raises."""
    sink, payload, whole = _Peek(_MAX_JUDGED_BODY), _Copy(_MAX_JUDGED_BODY), False
    try:
        whole = _relay(stream, sink, _response_framing(head, method), payload)
    except Exception:  # noqa: BLE001 -- judging never fails the answer it reads
        whole = False
    finally:
        if sink.raw:
            stream.buf[:0] = bytes(sink.raw)
    return payload.value() if whole else None


def _json_object(data: bytes) -> Optional[dict]:
    try:
        payload = json.loads(data)
    except (ValueError, RecursionError):
        return None
    return payload if isinstance(payload, dict) else None


def _anthropic_usage(raw: dict) -> Optional[dict]:
    """Anthropic's usage block in witness form; None when nothing was measured."""
    if not raw:
        return None
    return {"input_tokens": raw.get("input_tokens", 0),
            "output_tokens": raw.get("output_tokens", 0),
            "cache_read_tokens": raw.get("cache_read_input_tokens", 0),
            "cache_write_tokens": raw.get("cache_creation_input_tokens", 0),
            "reasoning_tokens": 0}


class _UsageTap:
    """Reads the served model and token usage from a response, as it streams by."""

    def __init__(self, head: _Head):
        self.sse = "text/event-stream" in head.header("content-type").lower()
        self.coding = (("content-encoding", head.header("content-encoding")),)
        # A stream is decoded before it is scanned: Anthropic gzips a streamed reply
        # whenever the client offers gzip, as Claude Code does, and compressed
        # bytes hold no events to read. The client still gets the bytes as sent.
        self.decode = StreamDecoder(self.coding, _CHUNK)
        self.copy = _Copy()
        self.pending = bytearray()
        self.model: Optional[str] = None
        self.usage: dict = {}
        self.raw: dict = {}  # the whole usage block: cache lifetimes, speed, tier

    def __call__(self, piece: bytes) -> None:
        if not self.sse:
            self.copy(piece)
            return
        for block in self.decode(piece):
            self._scan(block)

    def _scan(self, block: bytes) -> None:
        self.pending += block
        end = self.pending.find(b"\n\n")
        while end >= 0:
            self._event(bytes(self.pending[:end]))
            del self.pending[:end + 2]
            end = self.pending.find(b"\n\n")
        if len(self.pending) > _MAX_EVENT:  # one huge delta: it carries no usage
            self.pending.clear()

    def _event(self, block: bytes) -> None:
        data = b"".join(line[5:].strip() for line in block.split(b"\n")
                        if line.startswith(b"data:"))
        if b"message_start" not in data and b"message_delta" not in data:
            return
        payload = _json_object(data) or {}
        message = payload.get("message")
        if isinstance(message, dict):
            self._note(message)
        self._merge(payload.get("usage"))

    def _note(self, message: dict) -> None:
        if isinstance(message.get("model"), str):
            self.model = message["model"]
        self._merge(message.get("usage"))

    def _merge(self, usage) -> None:
        if isinstance(usage, dict):
            self.raw.update(usage)
            self.usage.update({key: value for key, value in usage.items()
                               if type(value) is int})

    def thread_usage(self) -> Optional[dict]:
        """The usage a signed thread row takes (cache lifetimes kept), or None."""
        from lib.hal.infinite_cost import anthropic_usage  # lazy: wire imports stay light
        return anthropic_usage(self.raw)

    def result(self) -> tuple:
        """(model or None, normalised usage or None)."""
        if not self.sse:
            self._note(_json_object(decoded_body(self.coding, self.copy.value()) or b"")
                       or {})
        return self.model, _anthropic_usage(self.usage)


@dataclass
class _Exchange:
    """One request and its response, as far as the wire needs to know them."""

    method: str
    target: str
    headers: tuple
    started: float
    route: RouteClass = RouteClass.UNKNOWN
    status: Optional[int] = None
    completed: bool = False
    framing: str = "length"
    body: Optional[_Copy] = None
    tap: Optional[_UsageTap] = None
    wire_body: Optional[_Copy] = None
    client_started: bool = False
    w4_uncertain: bool = False
    w4_accepted: bool = False
    answer_head_seen: bool = False
    hold_deadline: Optional[float] = None
    auth: object = None  # a 401's or 403's judgement (auth_signal.AuthSignal), else None
    replay: Optional[dict] = None
    hold: Optional[dict] = None
    w4_target: object = None
    response_headers: tuple = ()
    # True when the bearer sent upstream was chosen by ``_switched`` (an account
    # already marked exhausted) rather than presented by the client. A replay's
    # switch is recorded in ``replay`` instead; the ledger reads both.
    switched: bool = False

    @property
    def path(self) -> str:
        return request_path(self.target)

    def request_model(self) -> str:
        payload = request_json(self.headers, self.body.value()) if self.body else None
        model = payload.get("model") if payload else None
        return model if isinstance(model, str) else ""


def _close_uncertain_turn(proxy, ex, answer_seen=None):
    """Close a bounded inference turn only when no Anthropic answer arrived."""
    from lib.hal.wire_hold import close_turn
    if answer_seen is None:
        answer_seen = ex.answer_head_seen
    if answer_seen or ex.method != 'POST' or ex.path != MESSAGES_PATH:
        return
    close_turn(proxy, ex)


def _mark_session_upload(proxy, ex, request):
    """Mark the session while a streamed turn is on its way to Anthropic.

    The turn's own key is unknown until its body has been read, and the last body
    byte can reach Anthropic before that. So from before the first byte until the
    send is settled, the SESSION is marked, and a W4 reservation for any turn of
    that session is refused. Memory only: a dictionary count, no file.
    """
    attempts = getattr(proxy, '_w4_attempts', None)
    if attempts is None or ex.method != 'POST' or ex.path != MESSAGES_PATH:
        return None
    from lib.hal.gateway_witness import session_id_from_headers
    session = session_id_from_headers(request.headers)
    if session:
        attempts.begin(session)
    return session


def _begin_anthropic_send(proxy, ex, source=None):
    from lib.hal.wire_hold import exchange_key
    attempts = getattr(proxy, '_w4_attempts', None)
    ex.send_key = exchange_key(source or ex) if attempts else None
    if ex.send_key:
        attempts.begin(ex.send_key)


def _end_anthropic_send(proxy, ex, status=None):
    """Settle a turn's send; ``status`` is its answer's, when one came."""
    key = getattr(ex, 'send_key', None)
    if key:
        proxy._w4_attempts.end(key, status)
        ex.send_key = None


# --------------------------------------------------------------------------- #
# Warm upstream connections.
# --------------------------------------------------------------------------- #
class _Upstream:
    """One TLS connection to the decrypted host, with its read buffer."""

    def __init__(self, sock):
        self.sock = sock
        self.stream = _Stream(sock)
        self.stamp = time.monotonic()

    def is_fresh(self, ttl: float) -> bool:
        """Idle less than ``ttl``, nothing unread, and not closed by the server.

        A readable socket is read once, without waiting. Session tickets alone,
        which every TLS 1.3 server sends just after the handshake, leave the
        connection fresh; data or a close retire it. (Treating "readable" as
        "closed" retired every warm connection within ~50 ms of its handshake,
        so the pool dialled a new one for each call. Found 2026-10-05.)
        """
        if self.stream.buf or time.monotonic() - self.stamp > ttl:
            return False
        try:
            if self.sock.pending():
                return False
            return not _wait_readable([self.sock], 0) or _recv_now(self.sock, 1) is None
        except (OSError, ValueError):
            return False


class TlsDialer:
    """Opens verified TLS connections to the decrypted host, resuming sessions."""

    def __init__(self, address: Callable[[], tuple], context: ssl.SSLContext):
        self._address = address
        self._context = context
        self._session = None

    def dial(self, timeout: float = _CONNECT_TIMEOUT_S):
        raw = socket.create_connection(self._address(), timeout=timeout)
        _no_delay(raw)
        try:
            sock = self._context.wrap_socket(raw, server_hostname=DECRYPT_HOST,
                                             session=self._session)
        except BaseException:
            raw.close()
            raise
        sock.settimeout(_IDLE_TIMEOUT_S)
        return sock

    def remember(self, sock) -> None:
        """Keep a resumable TLS session for the next connection."""
        session = getattr(sock, "session", None)
        if session is not None and getattr(session, "has_ticket", False):
            self._session = session


class UpstreamPool:
    """Ready TLS connections to api.anthropic.com, so no request waits on a handshake.

    ``size`` connections are kept idle and ready; a background thread drops any
    that went stale (older than ``idle_ttl``, or closed by the server) and opens
    replacements. ``size=0`` turns warming off: every request connects cold.
    """

    def __init__(self, dialer: TlsDialer, *, size: int = _POOL_SIZE,
                 idle_ttl: float = _POOL_IDLE_TTL_S):
        self.dialer = dialer
        self._size = size
        self._ttl = idle_ttl
        self._idle: deque = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopped = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._maintain, name="hal-wire-pool", daemon=True).start()

    def stop(self) -> None:
        self._stopped.set()
        self._wake.set()
        with self._lock:
            idle, self._idle = list(self._idle), deque()
        _close(*(conn.sock for conn in idle))

    def idle_count(self) -> int:
        with self._lock:
            return len(self._idle)

    def take(self, *, timeout: float = _CONNECT_TIMEOUT_S) -> _Upstream:
        """A ready connection, else a new one. Raises OSError or SSLError when unreachable."""
        self._wake.set()
        conn = self._pop_fresh()
        return conn if conn is not None else _Upstream(self.dialer.dial(timeout=timeout))

    def give_back(self, conn: _Upstream) -> None:
        """Keep a still-healthy connection for the next client, else close it."""
        self.dialer.remember(conn.sock)
        conn.stamp = time.monotonic()
        with self._lock:
            keep = (not self._stopped.is_set() and len(self._idle) < self._size
                    and self.fresh(conn))
            if keep:
                self._idle.append(conn)
        if not keep:
            _close(conn.sock)

    def fresh(self, conn: _Upstream) -> bool:
        """Whether ``conn`` may carry the next request (see ``_Upstream.is_fresh``)."""
        return conn.is_fresh(self._ttl)

    def _pop_fresh(self) -> Optional[_Upstream]:
        while True:
            with self._lock:
                conn = self._idle.popleft() if self._idle else None
            if conn is None or self.fresh(conn):
                return conn
            _close(conn.sock)

    def _maintain(self) -> None:
        while not self._stopped.is_set():
            self._prune()
            self._refill()
            self._wake.wait(_POOL_TICK_S)
            self._wake.clear()

    def _prune(self) -> None:
        keep, stale = [], []
        with self._lock:
            for conn in self._idle:
                (keep if self.fresh(conn) else stale).append(conn)
            self._idle = deque(keep)
        _close(*(conn.sock for conn in stale))

    def _refill(self) -> None:
        while self.idle_count() < self._size and not self._stopped.is_set():
            try:
                conn = _Upstream(self.dialer.dial())
            except (OSError, ssl.SSLError, ValueError):
                return  # offline or refused: the next tick tries again
            with self._lock:
                self._idle.append(conn)


def default_upstream_context() -> ssl.SSLContext:
    """Verifies api.anthropic.com's real certificate; TLS 1.2+, ALPN http/1.1 only."""
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.set_alpn_protocols(["http/1.1"])
    if not context.get_ca_certs():
        _load_certifi(context)
    return context


def _load_certifi(context: ssl.SSLContext) -> None:
    """Some Python builds ship an empty default store; certifi fills it when present."""
    try:
        import certifi
    except ImportError:
        return
    context.load_verify_locations(certifi.where())


# --------------------------------------------------------------------------- #
# The wire.
# --------------------------------------------------------------------------- #
def _split_target(target: str) -> tuple:
    host, _, port = (target or "").rpartition(":")
    host = host.strip("[]").rstrip(".").lower()
    return (host, int(port)) if host and port.isdigit() else ("", 0)


def _is_local(host: str) -> bool:
    """Loopback, link-local or unspecified, by name or by address literal. An
    IPv4-mapped IPv6 literal (``::ffff:127.0.0.1``) answers as its IPv4 address
    does. Newer standard libraries do that themselves; the Python 3.12 that
    Ubuntu 24.04 ships did not (measured 2026-11-25), so the address is unwrapped
    here.

    Only the literal CONNECT target is read. A name that resolves to loopback is
    tunnelled like any other host, which reaches nothing the local client could
    not reach directly."""
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    # Some Python releases report an IPv4-mapped IPv6 address such as
    # ::ffff:127.0.0.1 as neither loopback nor link-local, so check the IPv4
    # address it carries (HAL#859: on Ubuntu 24.04's Python 3.12 it got through).
    if address.version == 6 and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback or address.is_link_local or address.is_unspecified


def _route(host: str, port: int) -> str:
    """What the wire does with one CONNECT, before the egress brake is asked.

    ``decrypt``  api.anthropic.com on 443: witnessed.
    ``known``    a host Claude Code itself needs (BLIND_HOSTS, BLIND_SUFFIXES) on
                 443: tunnelled blind, with no row, and still served while the
                 egress brake is engaged.
    ``unknown``  any other host: tunnelled blind with one ``wire.tunnelled`` row,
                 or refused while the egress brake is engaged.
    ``refuse``   api.anthropic.com on any other port (a tunnel there would leave
                 the witness), a loopback or link-local target, or no target.
    """
    if not host or not 0 < port < 65536 or _is_local(host):
        return "refuse"
    if host == DECRYPT_HOST:
        return "decrypt" if port == TLS_PORT else "refuse"
    if port == TLS_PORT and (host in BLIND_HOSTS or host.endswith(BLIND_SUFFIXES)):
        return "known"
    return "unknown"


def _braked_response() -> bytes:
    """The 403 an unknown host gets while the egress brake is engaged; it says why."""
    message = (f"The HAL wire's egress brake is engaged, so hosts it does not know are "
               f"refused. Remove {EGRESS_BRAKE.path()} to release it "
               f"(GRIP: bin/grip-wire-brake off).\n").encode()
    head = ("HTTP/1.1 403 Egress Brake Engaged\r\nContent-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(message)}\r\nConnection: close\r\n\r\n")
    return head.encode() + message


class WireProxy:
    """The wire's behaviour behind the front end: one call per CONNECT.

    ``resolve(host, port) -> (host, port)`` picks where a host is dialled (tests
    point it at local fakes); ``upstream_context`` verifies the decrypted host's
    certificate (default: the system trust store); ``cap`` bounds the messages
    rail (default ``InflightCap.from_env("HAL_WIRE")``); ``ledger`` is the
    per-account ledger (``lib/hal/account_ledger.AccountLedger``; default: the
    one under ``$HAL_HOME``).
    """

    def __init__(self, ca: wire_ca.WireCA, *, resolve: Optional[Callable] = None,
                 upstream_context: Optional[ssl.SSLContext] = None,
                 pool_size: int = _POOL_SIZE, idle_ttl: float = _POOL_IDLE_TTL_S,
                 telemetry: Optional[GatewayTelemetry] = None,
                 cap: Optional[InflightCap] = None, ledger=None):
        from lib.hal.account_ledger import AccountLedger  # lazy, like the cost imports
        self._ca = ca
        self._resolve = resolve or (lambda host, port: (host, port))
        self._telemetry = telemetry or GatewayTelemetry()
        self._cap = cap if cap is not None else InflightCap.from_env("HAL_WIRE")
        self._busy = _busy_response(self._cap)
        self._busy_head = _parse_head(self._busy.split(b"\r\n\r\n", 1)[0] + b"\r\n\r\n")
        self._ledger = ledger or AccountLedger()
        from lib.hal.wire_hold import Fallbacks
        self._hold_fallbacks = Fallbacks()
        from lib.hal.wire_cascade import Attempts, Sessions
        self._w4_sessions = Sessions()
        self._w4_attempts = Attempts()
        dialer = TlsDialer(lambda: self._resolve(DECRYPT_HOST, TLS_PORT),
                           upstream_context or default_upstream_context())
        self.pool = UpstreamPool(dialer, size=pool_size, idle_ttl=idle_ttl)
        self._leaf_lock = threading.Lock()
        self._leaf: Optional[wire_ca.Leaf] = None
        self._context: Optional[ssl.SSLContext] = None

    def start(self) -> None:
        self.pool.start()

    def stop(self) -> None:
        self.pool.stop()

    def health(self) -> dict:
        return {"ok": True, "service": SERVICE, "brake": brake_state(),
                "account_switch": "on" if account_switch.switching_on() else "off",
                "account_switch_stop": account_switch.off_reason(),
                "egress_brake": EGRESS_BRAKE.state(),
                "ca_cert": str(wire_ca.ca_cert_path()),
                "ca_fingerprint_sha256": self._ca.fingerprint}

    def handle_connect(self, sock, target: str) -> None:
        """Serve one CONNECT. A failure ends this connection and nothing else."""
        host, port = _split_target(target)
        try:
            self._dispatch(sock, host, port)
        except Exception as exc:  # noqa: BLE001 -- one connection never stops the wire
            _log.debug("a %s connection ended: %s",
                       "decrypted" if host == DECRYPT_HOST else "tunnelled",
                       type(exc).__name__)
        finally:
            _close(sock)

    def _dispatch(self, sock, host: str, port: int) -> None:
        route = _route(host, port)
        if route == "unknown" and EGRESS_BRAKE.engaged():
            self._telemetry.record({"event": "wire.refused", "host": host[:255],
                                    "reason": "egress-brake"})
            sock.sendall(_braked_response())
            return
        if route == "refuse":
            self._telemetry.record({"event": "wire.refused", "host": host[:255]})
            sock.sendall(_FORBIDDEN)
            return
        _no_delay(sock)
        if route == "decrypt":
            sock.sendall(_ESTABLISHED)
            self._serve_decrypted(sock)
            return
        if route == "unknown":
            self._telemetry.record({"event": "wire.tunnelled", "host": host[:255],
                                    "port": port})
        self._blind_tunnel(sock, host, port)

    def _blind_tunnel(self, sock, host: str, port: int) -> None:
        try:
            upstream = socket.create_connection(self._resolve(host, port),
                                                timeout=_CONNECT_TIMEOUT_S)
        except OSError:
            sock.sendall(_BAD_GATEWAY)
            return
        try:
            sock.sendall(_ESTABLISHED)
            _no_delay(upstream)
            upstream.settimeout(_IDLE_TIMEOUT_S)
            sock.settimeout(_IDLE_TIMEOUT_S)
            _splice(_Stream(sock), _Stream(upstream))
        finally:
            _close(upstream)

    def _server_context(self) -> ssl.SSLContext:
        """The serving context; its leaf is re-minted an hour before it expires."""
        with self._leaf_lock:
            if self._leaf is None or self._leaf.expires_within(_LEAF_REFRESH_S):
                self._leaf = self._ca.mint_leaf(DECRYPT_HOST)
                self._context = wire_ca.server_context(self._leaf)
            return self._context

    def _serve_decrypted(self, sock) -> None:
        sock.settimeout(_HANDSHAKE_TIMEOUT_S)
        tls = self._server_context().wrap_socket(sock, server_side=True)
        tls.settimeout(_IDLE_TIMEOUT_S)
        try:
            conn = self._exchange_loop(_Stream(tls))
        finally:
            _close(tls)
        if conn is not None:
            self.pool.give_back(conn)

    def _exchange_loop(self, client: _Stream) -> Optional[_Upstream]:
        """Requests in order on one client connection; returns the upstream if reusable."""
        conn = None
        try:
            while True:
                raw = client.read_head()
                if raw is None:
                    return conn
                from lib.hal import wire_hold
                original = _parse_head(raw)
                from lib.hal.wire_cascade import serve_bound
                if serve_bound(self, client, original):
                    _close(conn.sock if conn else None)
                    return None
                if wire_hold.consume_fallback(self, client, original):
                    _close(conn.sock if conn else None)
                    return None
                request = self._switched(original)
                if self._refused(client, request):
                    return conn  # the client is closed; any upstream stays good
                if wire_hold.prepare(self, client, request, original.headers):
                    _close(conn.sock if conn else None)
                    return None
                conn = self._upstream_for(client, conn)
                # Identity is the whole test: _switched returns ``original`` itself
                # when no account is chosen and a new _Head only when the bearer
                # changed, and nothing between here and there rebuilds ``request``
                # (_refused and wire_hold.prepare only read it; _upstream_for
                # never sees it).
                if conn is None or not self._exchange(client, conn, request,
                                                      switched=request is not original):
                    _close(conn.sock if conn else None)
                    return None
                self.pool.dialer.remember(conn.sock)
                conn.stamp = time.monotonic()
        except BaseException:
            _close(conn.sock if conn else None)
            raise

    def _switched(self, request: _Head) -> _Head:
        """The request on another pool account when its own has no capacity left,
        else unchanged (``lib/hal/account_switch.py``). Only the bearer changes,
        and only the label and the rank are recorded, never a token."""
        choice = account_switch.choose(request.parts[0], request.parts[1], request.headers,
                                       probe_org=self._probe_org)
        if choice is None:
            return request
        label, token, rank = choice
        self._telemetry.record({"event": "wire.account_switched", "to": label[:64],
                                "rank": rank})
        return _with_bearer(request, token)

    def _refused(self, client: _Stream, request: _Head) -> bool:
        """Answer, and end, a request the spend guard refuses: a paid credential,
        or a spent account's billable request. Nothing is sent upstream, and
        no header is recorded (only the source and the status)."""
        refusal = spend_guard.check_anthropic_request(request.parts[0], request.parts[1],
                                                      request.headers)
        if refusal is None:
            return False
        self._telemetry.record({"event": "wire.spend_refused", "source": refusal.source,
                                "status": refusal.status})
        if refusal.status != 429 or not self._offer_spend_refusal(client, request, refusal):
            client.sock.sendall(_spend_refusal(refusal))
        return True

    def _offer_spend_refusal(self, client: _Stream, request: _Head, refusal) -> bool:
        """Offer a spent account's fresh turn to W4, and answer it here either way.

        True when this method answered the request: a target served it, or no target
        did and the refusal was sent from here. The refusal is sent before the row is
        written, so the row never says an answer was delivered that the client did
        not get. False when no offer could be made; the caller sends the refusal."""
        from lib.hal import wire_cascade, wire_hold
        if (request.parts[0].upper() != "POST" or request_path(request.parts[1]) != MESSAGES_PATH
                or not wire_cascade.configured_targets()
                or not wire_hold.exhaustion(self, request.headers)[0]):
            return False
        if not self._cap.try_enter():
            return False
        try:
            copies = _capture_refused_body(client, request)
            if copies is None:
                return False
            body, wire = copies
            ex = _Exchange("POST", request.parts[1], request.headers, time.monotonic(),
                           route=classify("POST", request.parts[1], request.headers, body.value()),
                           body=body, wire_body=wire, status=429)
            try:
                if not wire_hold.offer_cascade(self, client, ex, "unsent"):
                    client.sock.sendall(_spend_refusal(refusal))
                    ex.completed = True
                return True
            finally:
                self._record(ex)
        finally:
            self._cap.leave()

    def _probe_org(self, token: str) -> Optional[str]:
        """Free identity lookup over the wire's verified, direct TLS upstream.

        Raw sockets ignore HTTPS_PROXY: this never loops back through CONNECT.
        Close after the headers, since the models body is not consumed. A 401 or
        403 is the account's own answer, so it is judged like any other: one that
        names a revocation or a suspension stops all switching.
        """
        for attempt in range(2):
            conn = None
            try:
                conn = self.pool.take(timeout=3.0)
                conn.sock.settimeout(3.0)
                conn.sock.sendall((f"GET /v1/models?limit=1 HTTP/1.1\r\nHost: {DECRYPT_HOST}\r\n"
                                   f"Authorization: Bearer {token}\r\n"
                                   "anthropic-version: 2023-06-01\r\n"
                                   "anthropic-beta: oauth-2025-04-20\r\n"
                                   "Connection: close\r\n\r\n").encode("latin-1"))
                raw = conn.stream.read_head()
                if raw is None:
                    raise ConnectionError("organisation probe closed before headers")
                head = _parse_head(raw)
                if _status(head) != 200:
                    self._judge(conn.stream, head, (("Authorization", f"Bearer {token}"),), "GET")
                    return None
                return head.header("anthropic-organization-id").strip() or None
            except OSError:
                continue
            except (ValueError, _WireProtocolError):
                return None
            finally:
                if conn is not None:
                    _close(conn.sock)
        return None

    def _upstream_for(self, client: _Stream, conn: Optional[_Upstream]) -> Optional[_Upstream]:
        """The current connection if still fresh, else a ready one; 503 if none can open."""
        if conn is not None and self.pool.fresh(conn):
            return conn
        _close(conn.sock if conn else None)
        try:
            return self.pool.take()
        except (OSError, ssl.SSLError, ValueError):
            client.sock.sendall(_UNREACHABLE)
            return None

    def _exchange(self, client: _Stream, conn: _Upstream, request: _Head,
                  switched: bool = False) -> bool:
        """Relay one request and its response, then record it. True: keep both open.

        A request on the messages rail holds one slot of the in-flight cap while
        it runs; past the cap it is answered at once, 503 with Retry-After.
        ``switched`` says the bearer is a pool account's, chosen up front."""
        if not (request.parts[0].upper() == "POST"
                and request_path(request.parts[1]) in _BODY_PATHS):
            return self._relay_exchange(client, conn, request, switched=switched)
        if not self._cap.try_enter():
            return self._turn_away(client, request)
        try:
            return self._relay_exchange(client, conn, request, switched=switched)
        finally:
            self._cap.leave()

    def _turn_away(self, client: _Stream, request: _Head) -> bool:
        """Answer a request past the cap: its body read and dropped, then the busy
        503. True when the connection can carry the client's next request."""
        drained = _relay(client, _Discard(), _request_framing(request))
        client.sock.sendall(self._busy)
        return drained and _keep_alive(request, self._busy_head, "length")

    def _relay_exchange(self, client: _Stream, conn: _Upstream, request: _Head,
                        switched: bool = False) -> bool:
        ex = _Exchange(method=request.parts[0].upper(), target=request.parts[1],
                       headers=request.headers, started=time.monotonic(), switched=switched)
        if ex.method == "POST" and ex.path in _BODY_PATHS:
            ex.body = _Copy()
            ex.wire_body = _Copy()
        response = self._send_for_answer(client, conn, request, ex)
        if response is None:
            return False
        try:
            if self._replay_rejected(client, request, ex, response):
                return False  # the original upstream still has its unread 429 body
            upgraded = self._relay_response(client, conn, ex, response)
        finally:
            # Also when the response broke off. A client that hangs up mid-stream
            # (an interrupted turn) makes the relay's next write raise, and the
            # turn Anthropic was sent must still leave its row: served=False, with
            # whatever usage arrived before the cut.
            self._record(ex)
        return not upgraded and ex.completed and _keep_alive(request, response, ex.framing)

    def _send_for_answer(self, client: _Stream, conn: _Upstream, request: _Head,
                         ex: _Exchange) -> Optional[_Head]:
        """Send upstream and read its head; answer failed upstreams with 503."""
        sent = incomplete = False
        upload = _mark_session_upload(self, ex, request)
        try:
            sent = True
            conn.sock.sendall(request.raw)
            sink = _RequestTee(conn.sock, ex.wire_body) if ex.wire_body else conn.sock
            if not _relay(client, sink, _request_framing(request), ex.body):
                # An incomplete client body cannot have been accepted as this turn.
                incomplete = True
                return None
            ex.route = classify(ex.method, ex.target, request.headers,
                                ex.body.value() if ex.body else b"")
            # Before this point the complete body was not sent, so no turn was accepted.
            _begin_anthropic_send(self, ex)
            response = self._answer(client, conn, ex)
            if ex.answer_head_seen:
                _end_anthropic_send(self, ex)
        except (OSError, ValueError, _WireProtocolError) as exc:
            _log.debug("the upstream failed before answering: %s", type(exc).__name__)
            response = None
        finally:
            # The closure is written before either mark goes, so no offer for this
            # turn can pass between them. The turn's own mark goes here too when no
            # answer came: left behind, it would stay for the life of the process.
            if sent and not incomplete and not ex.answer_head_seen:
                _close_uncertain_turn(self, ex)
            _end_anthropic_send(self, ex)
            if upload:
                self._w4_attempts.end(upload)
        if response is None:
            _send_quietly(client.sock, _NO_ANSWER)
        return response

    def _answer(self, client: _Stream, conn: _Upstream, ex: _Exchange, *,
                relay_interim: bool = True) -> Optional[_Head]:
        """The response head. For inference and aux the account ledger counts the
        request in flight until it arrives, then reads its rate-limit headers into
        numbers; the ledger never fails or delays the exchange beyond its bounded write.
        A 401 or 403 there is judged from its body, which is put back unread
        (:meth:`_judge`); the judgement is kept on ``ex.auth``."""
        if ex.route not in _WITNESSED:
            head = self._response_head(client, conn, ex, relay_interim=relay_interim)
            ex.answer_head_seen = head is not None
            if head is not None:
                _end_anthropic_send(self, ex, _status(head))
            return head
        with self._ledger.exchange(ex.headers) as watch:
            head = self._response_head(client, conn, ex, relay_interim=relay_interim)
            if head is not None:
                ex.answer_head_seen = True
                _end_anthropic_send(self, ex, _status(head))
                watch.answered(_status(head), head.headers)
        if head is not None:
            ex.auth = self._judge(conn.stream, head, ex.headers, ex.method)
        return head

    def _judge(self, stream: _Stream, head: _Head, request_headers, method: str):
        """A 401 or 403 judged from its body (``lib/hal/auth_signal.py``), or None for
        any other status. The body goes back on ``stream`` byte for byte. An
        enforcement signal, from whichever account, stops all account switching
        here, before anything else decides whether to switch. Never raises."""
        status = _status(head)
        if status not in auth_signal.STATUSES:
            return None
        signal = auth_signal.classify(status, _peek_body(stream, head, method), head.headers)
        if signal.enforcement:
            self._enforcement_stop(request_headers, signal)
        return signal

    def _enforcement_stop(self, request_headers, signal) -> None:
        """Stop switching and tell the operator (``account_switch``), hold a 403's
        token dead (the ledger already holds a 401's), and leave one row: the
        account's label and the judgement, never the message, token or header."""
        label = self._ledger._label_for(request_headers)[0]
        stopped = account_switch.stop_for_enforcement(signal, label)
        if signal.status == 403:
            self._ledger.mark_dead(request_headers)
        self._telemetry.record({"event": "wire.enforcement_signal", "account": label[:64],
                                "stopped_switching": stopped, **signal.fields()})

    @staticmethod
    def _response_head(client: _Stream, conn: _Upstream, ex: _Exchange, *,
                       relay_interim: bool = True) -> Optional[_Head]:
        """The final (or 101) response head; interim 1xx heads are relayed as they come."""
        while True:
            raw = conn.stream.read_head()
            if raw is None:
                return None
            head = _parse_head(raw)
            if _status(head) >= 200 or _status(head) == 101:
                return head
            if relay_interim:
                ex.client_started = True
                client.sock.sendall(head.raw)

    def _replay_rejected(self, client: _Stream, request: _Head,
                         ex: _Exchange, response: _Head) -> bool:
        """One replacement attempt, never recursive. Keep the original 429 unread
        until a usable replacement answer exists. A transport failure restores it
        too, without retrying a request whose acceptance is now uncertain.

        Only a 401 on the presented token is a dead token, and only when it names
        no enforcement action (``ex.auth``). A 403 is never switched, replayed,
        cascaded or paused: it is a permission answer, which can be an
        enforcement action, so it reaches the client unchanged. So does a 401
        that names one; :meth:`_judge` has already stopped all switching."""
        status = _status(response)
        rejected = status in (401, 403) and ex.route is RouteClass.INFERENCE
        dead = rejected and status == 401 and not getattr(ex.auth, "enforcement", False)
        label = self._ledger._label_for(ex.headers)[0] if rejected else None
        served = False
        try:
            served = self._replay_rejected_once(client, request, ex, response, dead)
            return served
        finally:
            if rejected:
                paused = bool(ex.replay) and ex.replay.get("outcome") == _PAUSED
                outcome = _PAUSED if paused else ("served" if served else "original")
                self._telemetry.record({"event": "wire.dead_token", "account": label,
                    "switched": served and not paused, "outcome": outcome,
                    "status": status, "signal": getattr(ex.auth, "kind", None)})

    def _replay_rejected_once(self, client, request, ex, response, dead):
        """Orchestration only: whether anything replaces ``response`` lives in
        :meth:`_no_replacement` (no account to switch to) or
        :meth:`_replay_with` (one found); both are split out so this stays short."""
        exhausted = account_switch.observe(_status(response), response.headers, ex.headers)
        if ((exhausted is None and not dead) or ex.client_started or ex.route is not RouteClass.INFERENCE
                or ex.wire_body is None or ex.wire_body.overflow):
            return False
        choice = account_switch.choose(ex.method, ex.target, ex.headers,
                                       probe_org=self._probe_org, reason="dead_token" if dead else "limit")
        if choice is None:
            return self._no_replacement(client, ex, dead, response)
        return self._replay_with(client, request, ex, dead, choice)

    def _no_replacement(self, client: _Stream, ex: _Exchange, dead: bool,
                        response: _Head) -> bool:
        """No account can carry this turn now: try the W4 cascade, then -- for a
        dead presented token's 401 only -- the honest exhaustion pause
        (``wire_hold.dead_token_pause``, V>>'s 2026-10-08 split). A 403 is
        never paused: it can be a permission or enforcement answer the
        operator must see. False leaves ``response`` to be relayed unchanged,
        inventing no reset."""
        from lib.hal import wire_hold
        if _status(response) != 403 and wire_hold.offer_cascade(self, client, ex, "rejected"):
            return True
        return dead and _status(response) == 401 and self._pause_dead_token(client, ex)

    def _pause_dead_token(self, client: _Stream, ex: _Exchange) -> bool:
        """Answer a dead presented token's 401 with the honest exhaustion 429
        and its real earliest reset, when at least one other pooled account is
        merely spent and known to come back -- so Claude Code's auto-continue
        resumes then, instead of retrying a dead key for minutes before asking
        for a re-login. False (no known future reset) leaves the 401 standing."""
        from lib.hal import wire_hold
        paused = wire_hold.dead_token_pause(self, ex.headers)
        if paused is None:
            return False
        WireProxy._answer_paused(client, ex, paused)
        ex.replay = {"outcome": _PAUSED}
        return True

    @staticmethod
    def _answer_paused(client: _Stream, ex: _Exchange, paused: bytes) -> None:
        """Send an honest exhaustion 429; the exchange counts as answered only once
        it has gone."""
        client.sock.sendall(paused)
        ex.client_started = True
        ex.status, ex.completed = 429, True

    def _replay_with(self, client, request, ex, dead, choice):
        _, token, rank = choice
        replacement = _with_bearer(request, token)
        refusal = spend_guard.check_anthropic_request(ex.method, ex.target, replacement.headers)
        if refusal is not None:
            return WireProxy._refused_replacement(self, client, ex, refusal)
        replay = _Exchange(ex.method, ex.target, replacement.headers, ex.started, route=ex.route)
        ex.replay = {"from": spend_guard.anthropic_credential(ex.headers)[1],
                     "to": spend_guard.anthropic_credential(replacement.headers)[1],
                     "reason": "dead_token" if dead else "limit_reached", "rank": rank, "outcome": "no_answer"}
        return WireProxy._replay_sent(self, client, replacement, replay, ex, choice)

    def _replay_sent(self, client, replacement, replay, ex, choice):
        conn = None
        sent = False
        decision = None  # set in the try; every except path returns before it is read
        try:
            conn = self.pool.take()
            sent = True
            _begin_anthropic_send(self, replay, ex)
            conn.sock.sendall(replacement.raw + ex.wire_body.value())
            head = self._answer(client, conn, replay, relay_interim=False)
            if head is None:
                _close_uncertain_turn(self, ex, False)
                _close(conn.sock)
                return False
            replay.answer_head_seen = True
            _end_anthropic_send(self, replay)
            decision = self._replay_response_decision(conn, replay, ex, head, choice)
            if decision == "move":
                _close(conn.sock)
            if decision == "original":
                return False
        except (OSError, ValueError, _WireProtocolError):
            if sent and not replay.answer_head_seen:
                _close_uncertain_turn(self, ex, False)
            _close(conn.sock if conn else None)
            return False
        finally:
            # After any closure above, never before it: a replay that got no answer
            # must not leave its turn's mark behind for the life of the process.
            _end_anthropic_send(self, replay)
        if decision == "move":
            return (WireProxy._cascade_rejected(self, client, ex)
                    or WireProxy._pause_spent_pool(self, client, ex))
        return self._relay_replacement(client, conn, ex, head, replacement)

    def _cascade_rejected(self, client, ex):
        from lib.hal import wire_hold
        return wire_hold.offer_cascade(self, client, ex, "rejected")

    def _pause_spent_pool(self, client: _Stream, ex: _Exchange) -> bool:
        """Every pooled account is spent and no fallback took the turn.

        A replay that began with a dead presented token gets the dead-token pause:
        its 401 would send Claude Code to a re-login while the spent replacement
        comes back by itself. The replay's own record says which kind it was; a
        missing mark does not, since a slow fallback attempt can outlast a limit.

        Otherwise the presented account's own refusal names only its own reset,
        and Claude Code waits for that one even when another account comes back
        sooner (HAL#676, the weekly S8 run). So when a pooled subscription has
        capacity again (:meth:`_subscription_returned`), the honest exhaustion
        429 names the present, and Claude Code retries at once; and when another
        pooled account's reset is earlier, the 429 names that one. It is read as
        the dead-token pause reads it: a dead pool token is skipped, because its
        stale reset is no promise that it will come back. False relays the
        original unchanged: when it already names the earliest reset, when no
        other account is known to come back, and when the presented account's
        own limit has already passed."""
        from lib.hal import wire_hold
        if ex.replay.get("reason") == "dead_token":
            return WireProxy._pause_dead_token(self, client, ex)
        own = account_switch.exhausted_until(spend_guard.anthropic_credential(ex.headers)[1])
        if own is None:
            return False
        if WireProxy._subscription_returned(self, ex):
            return WireProxy._answer_returned(client, ex)
        other = wire_hold.pooled_reset(self, ex.headers)
        if other is None or other >= own:
            return False
        WireProxy._answer_paused(client, ex, wire_hold.exhaustion_response(other))
        ex.replay["outcome"] = _PAUSED
        return True

    def _subscription_returned(self, ex: _Exchange) -> bool:
        """True when a retry of this turn would now be switched to a pooled
        subscription with capacity. Asked only after a move that found every
        account spent, so it means one came back while the fallback was tried
        (HAL#748 item 18). The switcher's own choice is the test, so an account
        it would not pick never reads as returned: an unknown or shared
        organisation, a token held dead, switching turned off. The choice reads
        only organisations already known and never probes one, so this answer
        never waits on the network, even at the end of a held stream's budget
        (Codex on HAL#770); a token whose organisation was never looked up does
        not count. Its choice must also carry no spent mark
        (:func:`wire_hold.carries_spent_mark`), which reads the ledger as well as
        this process's marks. An account offered only for paid overage always
        carries one, so overage never reads as returned. Never after a fallback
        send that may have been accepted: a quick retry could repeat a turn the
        target is already answering."""
        from lib.hal import wire_hold
        if ex.w4_uncertain or ex.w4_accepted:
            return False
        choice = account_switch.choose(ex.method, ex.target, ex.headers,
                                       probe_org=None, reason="limit")
        return choice is not None and not wire_hold.carries_spent_mark(self, choice[1])

    @staticmethod
    def _answer_returned(client: _Stream, ex: _Exchange) -> bool:
        """Answer the honest exhaustion 429 with the present as its reset."""
        from lib.hal import wire_hold
        WireProxy._answer_paused(client, ex, wire_hold.exhaustion_response(
            time.time(), wire_hold.RETURNED_MESSAGE))
        ex.replay["outcome"] = _RETRY_NOW
        return True

    def _refused_replacement(self, client, ex, refusal):
        """The spend guard refused the chosen replacement before it was sent.

        After a refused move the presented account's own refusal is relayed, as
        before W4, even when the pool is not wholly spent (HAL#748 item 19, left
        as it is). The race is narrow: the switcher already passes over a spent
        account and over paid overage the policy forbids, so this needs a spend
        mark, or a change of policy, between the choice and the send."""
        if refusal.status != 429:
            return False
        return WireProxy._cascade_rejected(self, client, ex)

    def _relay_replacement(self, client, conn, ex, head, replacement):
        # A relay failure after commitment must propagate, never append the saved 429.
        try:
            ex.replay["outcome"] = "relayed"
            ex.headers = replacement.headers
            self._relay_response(client, conn, ex, head)
            return True
        finally:
            _close(conn.sock)

    def _replay_response_decision(self, conn, replay, ex, head, choice):
        status = _status(head)
        ex.replay["status"] = status
        account_switch.observe(status, head.headers, replay.headers)
        if account_switch.limit_reached(status, head.headers):
            from lib.hal import wire_cascade, wire_hold
            if wire_cascade.configured_targets() and wire_hold.exhaustion(self, replay.headers)[0]:
                return "move"
        if status in (401, 403):
            ex.replay["outcome"] = "dead_replacement"
            self._telemetry.record({"event": "wire.dead_token",
                "account": choice[0], "switched": False,
                "outcome": "original", "status": status})
            _close(conn.sock)
            return "original"
        if status == 400 and self._thinking_rejection(conn, replay, head):
            ex.replay["outcome"] = "thinking_signature_rejected"
            return "original"
        return "relay"

    @staticmethod
    def _thinking_rejection(conn: _Upstream, ex: _Exchange, head: _Head) -> bool:
        """Inspect a bounded 400 body, then put its wire bytes back for normal relay."""
        raw, payload = _Copy(), _Copy()

        class Capture:
            def sendall(self, data):
                raw(data)
                if raw.overflow:
                    raise _WireProtocolError("replay error body too large")

        if not _relay(conn.stream, Capture(), _response_framing(head, ex.method), payload):
            raise _WireProtocolError("incomplete replay error")
        conn.stream.buf[:0] = raw.value()
        data = decoded_body(head.headers, payload.value()) or b""
        error = (_json_object(data) or {}).get("error", {})
        message = str(error.get("message", "")).lower() if isinstance(error, dict) else ""
        rejected = "thinking" in message and "signature" in message
        if rejected:
            _close(conn.sock)
        return rejected

    @staticmethod
    def _relay_response(client: _Stream, conn: _Upstream, ex: _Exchange,
                        head: _Head) -> bool:
        """Forward the response. Returns True when it switched protocols (101).
        A 2xx that shows paid overage marks its account spent (the spend guard)."""
        ex.client_started = True
        client.sock.sendall(head.raw)
        ex.status = _status(head)
        ex.response_headers = head.headers
        spend_guard.observe_anthropic_response(ex.status, head.headers, ex.headers)
        account_switch.observe(ex.status, head.headers, ex.headers)
        if ex.status == 101:
            _splice(client, conn.stream)
            ex.completed = True
            return True
        ex.framing, length = _response_framing(head, ex.method)
        ex.tap = _UsageTap(head) if ex.route in _WITNESSED else None
        ex.completed = _relay(conn.stream, client.sock, (ex.framing, length), ex.tap)
        return False

    def _record(self, ex: _Exchange) -> None:
        """A witness row for inference and aux, an audit row for unknown, else nothing."""
        try:
            from lib.hal.wire_cascade import observed
            observed(self, ex)
            if ex.path == MESSAGES_PATH:
                self._cache_loss(ex)
            if ex.route in _WITNESSED:
                self._witness(ex)
            elif ex.route is RouteClass.UNKNOWN:
                self._telemetry.record({"event": "wire.unknown", "method": ex.method[:16],
                                        "path": ex.path[:512]})
        except Exception:  # noqa: BLE001 -- observability never subtracts availability
            _log.debug("a record could not be written")

    def _cache_loss(self, ex: _Exchange) -> None:
        """One cache-loss-ledger line (``lib.hal.cache_loss``) for this
        ``/v1/messages`` response -- hashes and numbers only, never a body.
        Never raises: it runs before :meth:`_witness` inside one ``try`` in
        :meth:`_record`, so an exception from this optional ledger -- even one
        raised while gathering its arguments, outside ``record_response``'s own
        guard -- would otherwise cost the witness row as well."""
        try:
            from lib.hal import cache_loss  # lazy, like the tap's and the witness's
            model, _ = ex.tap.result() if ex.tap else (None, None)
            body = ex.body.value() if ex.body else b""
            cache_loss.record_response(
                session_id=session_id_from_headers(ex.headers),
                account_label=self._ledger._label_for(ex.headers)[0],
                model=model or ex.request_model() or None,
                request_headers=ex.headers, request_body=body,
                response_headers=ex.response_headers,
                usage=ex.tap.thread_usage() if ex.tap else None,
                account_switched=ex.switched or (ex.replay or {}).get("outcome") == "relayed")
        except Exception:  # noqa: BLE001 -- observability never subtracts availability
            _log.debug("a cache-loss row could not be written")

    def _witness(self, ex: _Exchange) -> None:
        from lib.hal.gateway_witness import prompt_id_from_headers  # lazy, like the tap's
        from lib.hal.infinite_cost import billing_from_headers
        model, usage = ex.tap.result() if ex.tap else (None, None)
        # The prompt id joins this row to the other requests that served the
        # same user prompt (Claude Code 2.1.283's gateway hint). Absent means
        # the client sent none in the safe set: the row then has no field,
        # never an empty one.
        prompt_id = prompt_id_from_headers(ex.headers)
        witness_chat(provider=ex.w4_target.provider if ex.w4_target else "anthropic",
                     model=model or (ex.w4_target.model if ex.w4_target else ex.request_model()),
                     status=ex.status, served=ex.completed, started=ex.started,
                     path=ex.path, usage=usage, telemetry=self._telemetry,
                     extra={"rail": "wire", "route_class": ex.route.value,
                            **({"prompt_id": prompt_id} if prompt_id else {}),
                            **({"wire_hold": ex.hold} if ex.hold else {}),
                            **({"account_replay": ex.replay} if ex.replay else {})},
                     session_id=session_id_from_headers(ex.headers),
                     row_class=_THREAD_ROW[ex.route], route=ex.route.value,
                     thread_usage=ex.tap.thread_usage() if ex.tap else None,
                     # A moved exchange is billed as its target declares, never
                     # as the inbound Anthropic credential suggests.
                     billing=(ex.w4_target.billing if ex.w4_target
                              else billing_from_headers(ex.headers)))


# --------------------------------------------------------------------------- #
# The front end: CONNECT and /health on 127.0.0.1.
# --------------------------------------------------------------------------- #
def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(wire: WireProxy) -> type:
    """The front-end request handler bound to ``wire``."""

    class _FrontHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        rbufsize = 0  # read the CONNECT head byte by byte, so nothing after it is taken

        def do_CONNECT(self):  # noqa: N802 (stdlib naming)
            self.close_connection = True
            wire.handle_connect(self.connection, self.path)

        def do_GET(self):  # noqa: N802
            self.close_connection = True
            if self.path.split("?", 1)[0] == "/health":
                _send_json(self, 200, wire.health())
            elif self.path.split("?", 1)[0] == "/w4/status":
                from urllib.parse import parse_qs, urlsplit
                from lib.hal.wire_cascade import status
                session = parse_qs(urlsplit(self.path).query).get("session", [""])[0]
                _send_json(self, 200, status(wire, session))
            else:
                _send_json(self, 404, {"ok": False,
                                       "error": "the HAL wire answers CONNECT and /health"})

        def log_message(self, fmt, *args):
            pass  # custody: request lines are never logged

    return _FrontHandler


def _quiet_error(_request, _client_address) -> None:
    """The server's error hook: the exception's type only, never its text.

    socketserver calls it from inside an ``except`` block; outside one there is
    no exception to name, and the hook must still not raise.
    """
    kind = sys.exc_info()[0]
    _log.warning("a front-end connection failed: %s",
                 kind.__name__ if kind is not None else "no exception")


def start_server(wire: WireProxy, port: int = DEFAULT_PORT, *, reclaim: bool = True):
    """Bind 127.0.0.1:``port`` and start the pool; the caller runs ``serve_forever``."""
    from lib.hal.gateway import _bind_gateway  # one loopback-only, reclaiming bind
    server = _bind_gateway(port, make_handler(wire), reclaim=reclaim)
    server.handle_error = _quiet_error
    wire.start()
    return server


def serve(port: int = DEFAULT_PORT, *, reclaim: bool = True) -> None:
    """Run the wire on 127.0.0.1:``port`` until interrupted."""
    wire = WireProxy(wire_ca.load_or_create())
    server = start_server(wire, port, reclaim=reclaim)
    _log.info("listening on http://127.0.0.1:%d (CONNECT; decrypts %s only); CA %s",
              port, DECRYPT_HOST, wire_ca.ca_cert_path())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        wire.stop()
        server.server_close()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lib.hal.wire_proxy",
        description="The HAL wire: a loopback HTTPS proxy that decrypts api.anthropic.com only.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"loopback port (default {DEFAULT_PORT})")
    parser.add_argument("--no-reclaim", action="store_true",
                        help="fail when the port is held instead of taking it over")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="hal-wire: %(message)s")
    try:
        serve(args.port, reclaim=not args.no_reclaim)
    except wire_ca.WireCAError as exc:
        print(f"hal-wire: not started: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
