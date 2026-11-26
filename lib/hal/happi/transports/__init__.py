"""HAPPI/1.0 transport shims.

The HAPPI/1.0 wire contract is argv + stdin JSON envelope -> stdout NDJSON
event stream. Transports are optional wrappers that expose the same
contract over a different on-the-wire format:

    β-1  http_transport   — POST /happi with JSON body, stream SSE response
    β-3  socket_transport  — AF_UNIX stream socket, NDJSON in / NDJSON out

Every transport delegates to `lib.hal.happi.runtime._dispatch` so behaviour
is identical across transports — the only thing each transport does is
translate its wire format to and from the runtime's envelope + NDJSON.
"""
from lib.hal.happi.transports.http_transport import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    HAPPI_PATH,
    HEALTH_PATH,
    HappiHTTPHandler,
    build_server,
    serve_http,
)
from lib.hal.happi.transports.socket_transport import (
    DEFAULT_SOCKET_PATH,
    HappiSocketHandler,
    ThreadingUnixStreamServer,
    build_socket_server,
    serve_socket,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_SOCKET_PATH",
    "HAPPI_PATH",
    "HEALTH_PATH",
    "HappiHTTPHandler",
    "HappiSocketHandler",
    "ThreadingUnixStreamServer",
    "build_server",
    "build_socket_server",
    "serve_http",
    "serve_socket",
]
