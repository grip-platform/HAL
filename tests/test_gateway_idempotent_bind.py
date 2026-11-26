"""Tests for lib.hal.gateway._bind_gateway — idempotent (port-reclaiming) bind.

Real-socket anchors: the strict path must actually raise ``EADDRINUSE`` against
a genuinely-held port (proves the test can fail and that ``reclaim=False`` stays
strict), and the reclaim path must bind successfully once the holder is freed.
"""
from __future__ import annotations

import errno
import socket
from http.server import BaseHTTPRequestHandler

import pytest

from lib.hal import gateway


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _hold(port: int) -> socket.socket:
    """Bind+listen on 127.0.0.1:port and return the holding socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(1)
    return sock


class _DummyHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence test output
        pass


def test_bind_gateway_strict_raises_eaddrinuse_on_held_port():
    port = _free_port()
    holder = _hold(port)
    try:
        with pytest.raises(OSError) as excinfo:
            gateway._bind_gateway(port, _DummyHandler, reclaim=False)
        assert excinfo.value.errno == errno.EADDRINUSE
    finally:
        holder.close()


def test_bind_gateway_reclaims_then_binds(monkeypatch):
    port = _free_port()
    holder = _hold(port)
    reclaimed_for = []
    closed = {"v": False}

    def _fake_reclaim(p, **kwargs):
        reclaimed_for.append(p)
        holder.close()  # simulate killing the listener -> frees the port
        closed["v"] = True
        return [4242]

    monkeypatch.setattr(gateway, "reclaim_port", _fake_reclaim)

    server = gateway._bind_gateway(port, _DummyHandler, reclaim=True)
    try:
        assert reclaimed_for == [port]  # reclaim was invoked for the right port
        assert server.server_address[1] == port  # and we bound it afterwards
    finally:
        server.server_close()
        if not closed["v"]:  # holder already closed by the fake reclaim
            holder.close()


def test_bind_gateway_eaddrinuse_path_executes_without_unbound_names(monkeypatch, capsys):
    # Forward-anchor for the AGORA review's two (false) blocker claims —
    # "NameError: sys/time not imported" and "UnboundLocalError: server" on the
    # EADDRINUSE path. This drives that exact path (real held socket -> bind
    # raises EADDRINUSE -> except: server.server_close() + sys.stderr.write() +
    # time.sleep() -> rebind) and asserts it completes. Gutting any of: the
    # `import sys`/`import time` module imports, or the `server = ...` assignment
    # before the try, makes this test raise instead of pass.
    port = _free_port()
    holder = _hold(port)
    closed = {"v": False}

    def _fake_reclaim(p, **kwargs):
        holder.close()
        closed["v"] = True
        return [4242]

    monkeypatch.setattr(gateway, "reclaim_port", _fake_reclaim)

    server = gateway._bind_gateway(port, _DummyHandler, reclaim=True)
    try:
        # The except branch's sys.stderr.write ran -> `sys` resolved and `server`
        # was bound (close + write both executed) -> no NameError/UnboundLocalError.
        assert "reclaimed, rebinding" in capsys.readouterr().err
        assert server.server_address[1] == port
    finally:
        server.server_close()
        if not closed["v"]:
            holder.close()


def test_bind_gateway_binds_free_port_without_reclaim(monkeypatch):
    # No holder: bind succeeds on the first attempt and reclaim is never called.
    port = _free_port()

    def _must_not_reclaim(*a, **k):  # pragma: no cover - asserts non-invocation
        raise AssertionError("reclaim_port called for a free port")

    monkeypatch.setattr(gateway, "reclaim_port", _must_not_reclaim)
    server = gateway._bind_gateway(port, _DummyHandler, reclaim=True)
    try:
        assert server.server_address[1] == port
    finally:
        server.server_close()
