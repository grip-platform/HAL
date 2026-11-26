"""HAPPI/1.0 β-3 Unix socket transport.

Listens on an AF_UNIX stream socket and exchanges HAPPI envelopes +
events as newline-delimited JSON (NDJSON in, NDJSON out).

Wire contract (one envelope per connection):

    Client -> Server: one JSON envelope terminated by "\\n"
    Server -> Client: zero or more NDJSON event lines (each "\\n"-terminated),
                      followed by orderly socket close

No headers, no length-prefix, no checksum. Dispatch semantics match the
stdin/stdout runtime and the β-1 HTTP transport byte-for-byte — the
socket transport just swaps the outer framing.

Use-case target: γ-1 voice-first loopback (replaces a duplicate local
IPC primitive with this canonical HAPPI shim) and any non-HTTP local
client that wants cheap per-request isolation without spawning a
subprocess.

Usage:
    from lib.hal.happi.transports.socket_transport import serve_socket
    serve_socket("/tmp/happi.sock")

    # Non-blocking form for tests / embedders:
    server = build_socket_server("/tmp/happi.sock")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # ...
    server.shutdown(); server.server_close(); os.unlink("/tmp/happi.sock")
"""
from __future__ import annotations

import os
import socketserver
import sys
from typing import Any, Type, cast

from lib.hal.happi.envelope import EnvelopeError, HappiEnvelope
from lib.hal.happi.events import emit_error
from lib.hal.happi.runtime import _dispatch


DEFAULT_SOCKET_PATH = "/tmp/happi.sock"
MAX_ENVELOPE_BYTES = 1024 * 1024  # 1 MiB ceiling — HAPPI envelopes are small


class HappiSocketHandler(socketserver.StreamRequestHandler):
    """Per-connection handler. Reads one envelope, writes N event lines, closes."""

    # StreamRequestHandler defaults: rfile/wfile are buffered io wrappers over
    # the connection. Disable timeout on the per-call read since the handler
    # itself is short-lived and we want to accept slow clients without a
    # per-byte deadline.
    timeout = 30  # seconds — bounds any single connection to 30s total

    def handle(self) -> None:
        raw_bytes = self.rfile.readline(MAX_ENVELOPE_BYTES + 1)
        if len(raw_bytes) > MAX_ENVELOPE_BYTES:
            self._write_parse_error(
                f"envelope too large: > {MAX_ENVELOPE_BYTES} bytes"
            )
            return
        raw = raw_bytes.decode("utf-8", errors="replace").strip()
        if not raw:
            self._write_parse_error("empty request; expected a JSON envelope")
            return
        try:
            envelope = HappiEnvelope.parse(raw)
        except EnvelopeError as exc:
            self._write_parse_error(str(exc))
            return

        envelope_raw = raw.encode("utf-8")
        for line in _dispatch(envelope, envelope_raw):
            try:
                self.wfile.write((line + "\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

    def _write_parse_error(self, message: str) -> None:
        ndjson = emit_error("req-invalid", message, code="parse_error") + "\n"
        try:
            self.wfile.write(ndjson.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


if hasattr(socketserver, "UnixStreamServer"):
    class ThreadingUnixStreamServer(
        socketserver.ThreadingMixIn, socketserver.UnixStreamServer
    ):
        """Per-connection threading over AF_UNIX stream."""
        daemon_threads = True
        allow_reuse_address = True
else:
    ThreadingUnixStreamServer = None


def _require_unix_sockets() -> None:
    if ThreadingUnixStreamServer is None:
        raise RuntimeError(
            f"AF_UNIX socket transport is unavailable on {sys.platform}; "
            "use the stdio transport or the HTTP transport in http_transport.py"
        )


def _resolve_server_class(server_class: Type[Any] | None) -> Type[Any]:
    resolved = server_class if server_class is not None else ThreadingUnixStreamServer
    if resolved is None:
        _require_unix_sockets()
    return cast(Type[Any], resolved)


def _unlink_if_exists(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def build_socket_server(
    socket_path: str = DEFAULT_SOCKET_PATH,
    server_class: Type[Any] | None = None,
) -> Any:
    """Construct (and bind) a socket server without starting its loop.

    Any stale socket file at `socket_path` is unlinked first — required
    for AF_UNIX bind to succeed on restart.
    """
    server_class = _resolve_server_class(server_class)
    _unlink_if_exists(socket_path)
    return server_class(socket_path, HappiSocketHandler)


def serve_socket(
    socket_path: str = DEFAULT_SOCKET_PATH,
    server_class: Type[Any] | None = None,
) -> None:
    """Block and serve HAPPI requests on `socket_path` until interrupted."""
    server_class = _resolve_server_class(server_class)
    server = build_socket_server(socket_path, server_class)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        _unlink_if_exists(socket_path)
