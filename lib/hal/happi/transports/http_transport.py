"""HAPPI/1.1 β-1 HTTP transport shim.

Wraps the stdin/stdout HAPPI runtime behind an HTTP endpoint. Clients
POST a JSON envelope body to `/happi`; the response is a
`text/event-stream` SSE stream carrying the runtime's NDJSON events as
`data: <ndjson>\\n\\n` frames.

Routes:
    GET  /health      → {"status":"ok",...}
    POST /happi       → SSE stream of HAPPI events
    (anything else)   → 404 application/json

Stdlib-only (http.server + ThreadingHTTPServer). One thread per request.
No external deps, no framework.

TLS / cert-pinning lands at β-2 together with the PRE-CHECK gate.

Usage:
    from lib.hal.happi.transports.http_transport import serve_http
    serve_http(host="127.0.0.1", port=8787)

    # Non-blocking form for tests / embedders:
    server = build_server(host="127.0.0.1", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # ...
    server.shutdown(); server.server_close()
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Type

from lib.hal.happi.envelope import EnvelopeError, HappiEnvelope
from lib.hal.happi.events import emit_error
from lib.hal.happi.runtime import _dispatch


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
HAPPI_PATH = "/happi"
HEALTH_PATH = "/health"
MAX_BODY_BYTES = 1024 * 1024  # 1 MiB envelope ceiling — HAPPI envelopes are small


class HappiHTTPHandler(BaseHTTPRequestHandler):
    """Per-request handler. Stateless — `_dispatch` owns the work."""

    # Protocol identifier sent in the Server: header.
    server_version = "happi-http/1.0"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        """Suppress default stderr logging; hosts wire their own if wanted."""
        return

    def do_GET(self) -> None:  # noqa: N802 — stdlib signature
        if self.path == HEALTH_PATH:
            self._write_health()
            return
        self._write_not_found()

    def do_POST(self) -> None:  # noqa: N802 — stdlib signature
        if self.path != HAPPI_PATH:
            self._write_not_found()
            return
        body = self._read_body()
        if body is None:
            return  # error already written
        self._stream_envelope(body)

    def _read_body(self) -> bytes | None:
        length_hdr = self.headers.get("content-length")
        if length_hdr is None:
            self._write_parse_error("missing content-length header")
            return None
        try:
            length = int(length_hdr)
        except ValueError:
            self._write_parse_error(f"invalid content-length: {length_hdr!r}")
            return None
        if length < 0:
            self._write_parse_error(f"negative content-length: {length}")
            return None
        if length > MAX_BODY_BYTES:
            self._write_parse_error(
                f"body too large: {length} bytes (max {MAX_BODY_BYTES})"
            )
            return None
        return self.rfile.read(length) if length else b""

    def _write_health(self) -> None:
        payload = json.dumps(
            {"status": "ok", "service": "happi/1.1", "transport": "http-beta-1"}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _write_not_found(self) -> None:
        payload = json.dumps({"error": "not found", "path": self.path}).encode("utf-8")
        self.send_response(404)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _write_parse_error(self, message: str) -> None:
        ndjson = emit_error("req-invalid", message, code="parse_error")
        payload = ndjson.encode("utf-8")
        self.send_response(400)
        self.send_header("content-type", "application/x-ndjson")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _stream_envelope(self, body: bytes) -> None:
        raw = body.decode("utf-8", errors="replace").strip()
        if not raw:
            self._write_parse_error("empty body; expected a JSON envelope")
            return
        try:
            envelope = HappiEnvelope.parse(raw)
        except EnvelopeError as exc:
            self._write_parse_error(str(exc))
            return

        envelope_raw = raw.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        # Transfer-Encoding: chunked is implicit when content-length is absent
        self.end_headers()

        for ndjson_line in _dispatch(envelope, envelope_raw):
            frame = f"data: {ndjson_line}\n\n".encode("utf-8")
            try:
                self.wfile.write(frame)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # Client disconnected mid-stream — abort cleanly.
                return


def build_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    server_class: Type[ThreadingHTTPServer] = ThreadingHTTPServer,
) -> ThreadingHTTPServer:
    """Construct a server without blocking. Caller runs `serve_forever`."""
    return server_class((host, port), HappiHTTPHandler)


def serve_http(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    server_class: Type[ThreadingHTTPServer] = ThreadingHTTPServer,
) -> None:
    """Block and serve HAPPI requests until interrupted."""
    server = build_server(host, port, server_class)
    try:
        server.serve_forever()
    finally:
        server.server_close()
