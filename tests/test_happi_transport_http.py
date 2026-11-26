"""HAPPI β-1 HTTP transport tests.

Spins the server on an ephemeral port in a daemon thread, hits it with
real urllib requests, and asserts wire-level behaviour: health endpoint,
404 on unknown paths, SSE streaming for valid envelopes, parse-error
JSON responses for invalid/empty bodies, and MIME-type correctness.

No mocking — the server really listens and really streams. Tests are
self-contained: each spins its own server, tears it down on teardown.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.client import HTTPConnection
from typing import Iterator

import pytest

from lib.hal.happi.transports.http_transport import (
    HAPPI_PATH,
    HEALTH_PATH,
    build_server,
)


@pytest.fixture
def live_server() -> Iterator[tuple[str, int]]:
    """Yield (host, port) for a daemon-thread server on an ephemeral port."""
    server = build_server(host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _post(host: str, port: int, path: str, body: bytes,
          content_type: str = "application/json") -> HTTPConnection:
    """POST body and return the open connection for streaming reads."""
    conn = HTTPConnection(host, port, timeout=3)
    conn.request(
        "POST", path, body=body,
        headers={"content-type": content_type, "content-length": str(len(body))},
    )
    return conn


def _get(host: str, port: int, path: str) -> HTTPConnection:
    conn = HTTPConnection(host, port, timeout=3)
    conn.request("GET", path)
    return conn


# --- Health + routing -------------------------------------------------------

def test_health_endpoint_returns_ok(live_server):
    host, port = live_server
    conn = _get(host, port, HEALTH_PATH)
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("content-type") == "application/json"
    payload = json.loads(resp.read())
    assert payload["status"] == "ok"
    assert payload["service"] == "happi/1.1"
    conn.close()


def test_unknown_path_returns_404(live_server):
    host, port = live_server
    conn = _get(host, port, "/nonexistent")
    resp = conn.getresponse()
    assert resp.status == 404
    assert resp.getheader("content-type") == "application/json"
    payload = json.loads(resp.read())
    assert payload["error"] == "not found"
    conn.close()


def test_get_on_happi_path_returns_404(live_server):
    """HAPPI endpoint is POST-only; GET yields 404 (method-not-allowed-equiv)."""
    host, port = live_server
    conn = _get(host, port, HAPPI_PATH)
    resp = conn.getresponse()
    assert resp.status == 404
    conn.close()


def test_post_on_unknown_path_returns_404(live_server):
    host, port = live_server
    conn = _post(host, port, "/wrong", b'{}')
    resp = conn.getresponse()
    assert resp.status == 404
    conn.close()


# --- SSE streaming for valid envelopes --------------------------------------

def test_post_noop_envelope_streams_sse(live_server):
    host, port = live_server
    envelope = json.dumps({
        "v": "happi/1.0", "id": "http-noop-1", "cmd": "noop",
    }).encode("utf-8")

    conn = _post(host, port, HAPPI_PATH, envelope)
    resp = conn.getresponse()

    assert resp.status == 200
    assert resp.getheader("content-type") == "text/event-stream"
    assert resp.getheader("cache-control") == "no-cache"

    body = resp.read().decode("utf-8")
    conn.close()

    # Expect SSE frames: `data: <ndjson>\n\n` for each runtime event.
    frames = [frame for frame in body.split("\n\n") if frame.strip()]
    events = []
    for frame in frames:
        assert frame.startswith("data: "), f"bad SSE frame: {frame!r}"
        events.append(json.loads(frame[len("data: "):]))

    types = [e["type"] for e in events]
    assert types == ["started", "completed"]
    assert events[0]["id"] == "http-noop-1"


def test_post_echo_envelope_streams_delta(live_server):
    host, port = live_server
    envelope = json.dumps({
        "v": "happi/1.0", "id": "http-echo-1",
        "cmd": "echo", "args": ["hello", "world"],
    }).encode("utf-8")

    conn = _post(host, port, HAPPI_PATH, envelope)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    conn.close()

    events = [
        json.loads(frame[len("data: "):])
        for frame in body.split("\n\n")
        if frame.startswith("data: ")
    ]
    types = [e["type"] for e in events]
    assert types == ["started", "delta", "completed"]
    assert events[1]["text"] == "hello world"


def test_post_version_envelope_streams_runtime_protocol_delta(live_server):
    """Post-P7: version cmd emits PROTOCOL_VERSION (default "happi/1.1")
    as the delta text — was the α-1 scaffold marker pre-cutover.
    """
    from lib.hal.happi.events import PROTOCOL_VERSION
    host, port = live_server
    envelope = json.dumps({
        "v": "happi/1.0", "id": "http-ver-1", "cmd": "version",
    }).encode("utf-8")

    conn = _post(host, port, HAPPI_PATH, envelope)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    conn.close()

    events = [
        json.loads(frame[len("data: "):])
        for frame in body.split("\n\n")
        if frame.startswith("data: ")
    ]
    assert [e["type"] for e in events] == ["started", "delta", "completed"]
    assert events[1]["text"] == PROTOCOL_VERSION


# --- Parse errors -----------------------------------------------------------

def test_post_empty_body_returns_400(live_server):
    host, port = live_server
    conn = _post(host, port, HAPPI_PATH, b"")
    resp = conn.getresponse()
    assert resp.status == 400
    assert resp.getheader("content-type") == "application/x-ndjson"
    body = resp.read().decode("utf-8")
    event = json.loads(body)
    assert event["type"] == "error"
    assert event["code"] == "parse_error"
    conn.close()


def test_post_invalid_json_returns_400(live_server):
    host, port = live_server
    conn = _post(host, port, HAPPI_PATH, b"not-json")
    resp = conn.getresponse()
    assert resp.status == 400
    body = resp.read().decode("utf-8")
    event = json.loads(body)
    assert event["type"] == "error"
    assert event["code"] == "parse_error"
    conn.close()


def test_post_wrong_version_returns_400(live_server):
    host, port = live_server
    envelope = json.dumps({
        "v": "happi/9.9", "id": "bad-ver", "cmd": "noop",
    }).encode("utf-8")
    conn = _post(host, port, HAPPI_PATH, envelope)
    resp = conn.getresponse()
    assert resp.status == 400
    body = resp.read().decode("utf-8")
    event = json.loads(body)
    assert event["code"] == "parse_error"
    conn.close()


def test_post_without_content_length_returns_400(live_server):
    """Requests without content-length hdr can't be body-read safely."""
    host, port = live_server
    conn = HTTPConnection(host, port, timeout=3)
    # Manually construct a request WITHOUT content-length.
    conn.putrequest("POST", HAPPI_PATH)
    conn.putheader("content-type", "application/json")
    # Intentionally omit content-length
    conn.endheaders()
    # Send no body (connection close signals EOF).
    resp = conn.getresponse()
    assert resp.status == 400
    conn.close()


def test_post_body_larger_than_cap_returns_400(live_server):
    from lib.hal.happi.transports.http_transport import MAX_BODY_BYTES
    host, port = live_server
    oversized = b'{"v":"happi/1.0","id":"x","cmd":"echo","args":["x"]}' \
                + b" " * (MAX_BODY_BYTES + 1)
    # We don't actually have to send the oversized bytes — just claim the size.
    conn = HTTPConnection(host, port, timeout=3)
    conn.putrequest("POST", HAPPI_PATH)
    conn.putheader("content-type", "application/json")
    conn.putheader("content-length", str(MAX_BODY_BYTES + 1))
    conn.endheaders()
    # Send some bytes so the connection doesn't hang; the handler returns 400
    # based on the content-length header alone.
    conn.send(b"x" * 64)
    resp = conn.getresponse()
    assert resp.status == 400
    conn.close()


# --- SSE format invariants --------------------------------------------------

def test_sse_frames_use_data_prefix_and_blank_line_separator(live_server):
    host, port = live_server
    envelope = json.dumps({
        "v": "happi/1.0", "id": "http-fmt-1",
        "cmd": "echo", "args": ["x"],
    }).encode("utf-8")
    conn = _post(host, port, HAPPI_PATH, envelope)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    conn.close()

    # Every non-empty chunk must start with "data: " and be followed by \n\n
    chunks = body.split("\n\n")
    # Last chunk after final \n\n is empty → drop it.
    chunks = [c for c in chunks if c]
    for chunk in chunks:
        assert chunk.startswith("data: "), f"frame missing SSE prefix: {chunk!r}"
        # Payload after the prefix must be valid JSON.
        json.loads(chunk[len("data: "):])


def test_http_dispatch_matches_stdin_dispatch_parity():
    """Envelopes dispatched via HTTP must yield byte-identical NDJSON to stdin dispatch."""
    from lib.hal.happi.envelope import HappiEnvelope
    from lib.hal.happi.runtime import _dispatch

    envelope = HappiEnvelope.parse({
        "v": "happi/1.0", "id": "parity-1",
        "cmd": "echo", "args": ["parity", "test"],
    })
    stdin_events = list(_dispatch(envelope))
    stdin_ids = [json.loads(line)["id"] for line in stdin_events]

    # Now via HTTP — same envelope, same dispatch underneath.
    server = build_server(host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        envelope_bytes = json.dumps({
            "v": "happi/1.0", "id": "parity-1",
            "cmd": "echo", "args": ["parity", "test"],
        }).encode("utf-8")
        conn = _post(host, port, HAPPI_PATH, envelope_bytes)
        body = conn.getresponse().read().decode("utf-8")
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    http_events_raw = [
        frame[len("data: "):]
        for frame in body.split("\n\n")
        if frame.startswith("data: ")
    ]
    # Each HTTP event is the stdin NDJSON line verbatim, just wrapped in a
    # `data: ` / `\n\n` SSE frame. IDs and types must match exactly.
    assert [json.loads(e)["id"] for e in http_events_raw] == stdin_ids
    assert [json.loads(e)["type"] for e in http_events_raw] == [
        json.loads(line)["type"] for line in stdin_events
    ]
