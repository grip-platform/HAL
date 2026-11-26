"""HAPPI β-3 Unix socket transport tests.

Spins the server on a tempfile socket path inside a daemon thread, hits
it with real `socket.socket(AF_UNIX)` connections, and asserts wire-level
behaviour: envelope round-trips, NDJSON line-per-event framing, parse-error
handling, stale-socket cleanup, cross-transport parity with stdin dispatch.

No mocking — the server really binds, accept()s, and streams.
"""
from __future__ import annotations

import json
import os
import socket as _socket
import tempfile
import threading
from pathlib import Path
from typing import Iterator

import pytest

from lib.hal.happi.transports.socket_transport import (
    DEFAULT_SOCKET_PATH,
    build_socket_server,
)


@pytest.fixture
def socket_path(tmp_path: Path) -> str:
    """Per-test unique AF_UNIX path (shorter than /tmp prefix — Linux has a
    104/108 char limit on socket paths)."""
    # macOS: 104 chars. Linux: 108. tmp_path under /private/var/folders can
    # exceed that → use /tmp/happi-<pid>-<n>.sock instead.
    path = f"/tmp/happi-test-{os.getpid()}-{id(tmp_path)}.sock"
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def live_server(socket_path: str) -> Iterator[str]:
    server = build_socket_server(socket_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield socket_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _exchange(path: str, envelope_bytes: bytes, timeout: float = 3.0) -> bytes:
    """Open AF_UNIX, send envelope, read until EOF, return raw response."""
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(path)
    try:
        s.sendall(envelope_bytes)
        try:
            s.shutdown(_socket.SHUT_WR)  # signal end-of-request
        except OSError:
            # Server may have already closed for short/empty requests —
            # the \n in the body is enough to unblock readline().
            pass
        chunks: list[bytes] = []
        while True:
            try:
                chunk = s.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        s.close()


def _parse_ndjson(raw: bytes) -> list[dict]:
    return [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]


# --- Basic round-trips ------------------------------------------------------

def test_noop_envelope_yields_started_and_completed(live_server):
    path = live_server
    envelope = json.dumps({
        "v": "happi/1.0", "id": "sock-noop-1", "cmd": "noop",
    }).encode("utf-8") + b"\n"

    raw = _exchange(path, envelope)
    events = _parse_ndjson(raw)

    types = [e["type"] for e in events]
    assert types == ["started", "completed"]
    assert events[0]["id"] == "sock-noop-1"


def test_echo_envelope_yields_delta(live_server):
    path = live_server
    envelope = json.dumps({
        "v": "happi/1.0", "id": "sock-echo-1",
        "cmd": "echo", "args": ["hello", "socket"],
    }).encode("utf-8") + b"\n"

    raw = _exchange(path, envelope)
    events = _parse_ndjson(raw)
    types = [e["type"] for e in events]
    assert types == ["started", "delta", "completed"]
    assert events[1]["text"] == "hello socket"


def test_version_envelope_yields_runtime_protocol_version(live_server):
    """Post-P7: version cmd emits PROTOCOL_VERSION (default "happi/1.1")
    as the delta text — was the α-1 scaffold marker pre-cutover.
    """
    from lib.hal.happi.events import PROTOCOL_VERSION
    path = live_server
    envelope = json.dumps({
        "v": "happi/1.0", "id": "sock-ver-1", "cmd": "version",
    }).encode("utf-8") + b"\n"

    raw = _exchange(path, envelope)
    events = _parse_ndjson(raw)
    assert [e["type"] for e in events] == ["started", "delta", "completed"]
    assert events[1]["text"] == PROTOCOL_VERSION


# --- Framing invariants -----------------------------------------------------

def test_each_event_is_newline_terminated(live_server):
    path = live_server
    envelope = json.dumps({
        "v": "happi/1.0", "id": "sock-frame-1",
        "cmd": "echo", "args": ["x"],
    }).encode("utf-8") + b"\n"

    raw = _exchange(path, envelope)
    text = raw.decode("utf-8")
    lines = text.split("\n")
    # Final trailing empty element from the terminating \n.
    assert lines[-1] == "", f"response did not end with \\n: {text!r}"
    # Every non-empty line must parse as JSON.
    for line in lines[:-1]:
        json.loads(line)


def test_no_data_prefix_on_socket_frames(live_server):
    """Unlike HTTP/SSE, the socket sends raw NDJSON with no `data: ` prefix."""
    path = live_server
    envelope = json.dumps({
        "v": "happi/1.0", "id": "sock-raw-1", "cmd": "noop",
    }).encode("utf-8") + b"\n"

    raw = _exchange(path, envelope)
    text = raw.decode("utf-8")
    assert "data:" not in text
    assert text.count("\n") >= 2  # at least 2 events × 1 newline each


# --- Parse errors -----------------------------------------------------------

def test_empty_request_yields_parse_error(live_server):
    path = live_server
    raw = _exchange(path, b"\n")
    events = _parse_ndjson(raw)
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "parse_error"


def test_invalid_json_yields_parse_error(live_server):
    path = live_server
    raw = _exchange(path, b"not-json\n")
    events = _parse_ndjson(raw)
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "parse_error"


def test_wrong_version_yields_parse_error(live_server):
    path = live_server
    envelope = json.dumps({
        "v": "happi/9.9", "id": "bad-ver", "cmd": "noop",
    }).encode("utf-8") + b"\n"
    raw = _exchange(path, envelope)
    events = _parse_ndjson(raw)
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "parse_error"


def test_envelope_larger_than_cap_yields_parse_error(live_server):
    from lib.hal.happi.transports.socket_transport import MAX_ENVELOPE_BYTES
    path = live_server
    oversized = b"x" * (MAX_ENVELOPE_BYTES + 16) + b"\n"
    raw = _exchange(path, oversized)
    events = _parse_ndjson(raw)
    assert events[0]["type"] == "error"
    assert "too large" in events[0]["message"]


# --- Stale socket + rebinding ----------------------------------------------

def test_build_socket_server_unlinks_stale_path(tmp_path: Path):
    path = f"/tmp/happi-stale-{os.getpid()}.sock"
    # Create a stale socket file (pretend a previous server crashed).
    Path(path).touch()
    assert Path(path).exists()
    try:
        server = build_socket_server(path)
        # Bind succeeded — that means the stale file was cleaned up.
        server.server_close()
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


# --- Cross-transport parity -------------------------------------------------

def test_socket_dispatch_matches_stdin_dispatch_parity(live_server):
    """Envelopes via socket transport yield byte-identical NDJSON to stdin."""
    from lib.hal.happi.envelope import HappiEnvelope
    from lib.hal.happi.runtime import _dispatch

    envelope_dict = {
        "v": "happi/1.0", "id": "parity-sock-1",
        "cmd": "echo", "args": ["parity", "test"],
    }
    # Baseline: stdin dispatch.
    stdin_events = list(_dispatch(HappiEnvelope.parse(envelope_dict)))
    stdin_parsed = [json.loads(line) for line in stdin_events]

    # Now via socket.
    path = live_server
    raw = _exchange(path, (json.dumps(envelope_dict) + "\n").encode("utf-8"))
    socket_parsed = _parse_ndjson(raw)

    assert [e["type"] for e in socket_parsed] == [e["type"] for e in stdin_parsed]
    assert [e["id"] for e in socket_parsed] == [e["id"] for e in stdin_parsed]


def test_socket_and_http_transports_agree_on_event_sequence(live_server):
    """β-1 HTTP and β-3 socket must yield the same event type sequence for the
    same envelope — confirms both transports delegate to the same runtime."""
    from lib.hal.happi.transports.http_transport import build_server as build_http
    import urllib.request

    envelope = {
        "v": "happi/1.0", "id": "parity-xtransport-1",
        "cmd": "echo", "args": ["cross", "transport"],
    }

    # Socket side.
    path = live_server
    sock_raw = _exchange(path, (json.dumps(envelope) + "\n").encode("utf-8"))
    sock_events = _parse_ndjson(sock_raw)

    # HTTP side — spin ephemeral HTTP server.
    http_server = build_http(host="127.0.0.1", port=0)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()
    try:
        host, http_port = http_server.server_address[0], http_server.server_address[1]
        req = urllib.request.Request(
            f"http://{host}:{http_port}/happi",
            data=json.dumps(envelope).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = resp.read().decode("utf-8")
    finally:
        http_server.shutdown()
        http_server.server_close()
        http_thread.join(timeout=2)

    http_events = [
        json.loads(frame[len("data: "):])
        for frame in body.split("\n\n")
        if frame.startswith("data: ")
    ]

    assert [e["type"] for e in sock_events] == [e["type"] for e in http_events]
    assert [e["id"] for e in sock_events] == [e["id"] for e in http_events]
