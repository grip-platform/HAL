"""Startup safety: no live services, real ephemeral sockets, mocked reclaim."""
import errno
import json
import runpy
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import pytest

from lib.hal import gateway


def forbidden(*args, **kwargs):
    raise AssertionError("startup must not bind or reclaim")


@pytest.mark.parametrize("argv,code,text", [
    (["--help"], 0, "usage:"),
    (["--unknown"], 2, "unrecognized arguments"),
    (["--por", "0"], 2, "unrecognized arguments"),
    (["--port"], 2, "expected one argument"),
    (["--port", "no"], 2, "invalid int value"),
    (["--port", "65536"], 2, "port must be"),
    (["--port", "0", "0"], 2, "use either"),
])
def test_cli_exits_before_startup(monkeypatch, capsys, argv, code, text):
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    monkeypatch.setattr("lib.hal.port_reclaim.reclaim_port", forbidden)
    monkeypatch.setattr(sys, "argv", ["lib.hal.gateway", *argv])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(gateway.__file__, run_name="__main__")
    assert exc.value.code == code
    captured = capsys.readouterr()
    assert text in captured.out + captured.err


@pytest.mark.parametrize("argv,env,expected", [
    ([], {}, (gateway.DEFAULT_GATEWAY_PORT, gateway.DEFAULT_UPSTREAM, True)),
    (["--port", "0"], {}, (0, gateway.DEFAULT_UPSTREAM, True)),
    (["--port=0"], {}, (0, gateway.DEFAULT_UPSTREAM, True)),
    (["0"], {}, (0, gateway.DEFAULT_UPSTREAM, True)),
    (["--upstream", "http://upstream.invalid"], {},
     (gateway.DEFAULT_GATEWAY_PORT, "http://upstream.invalid", True)),
    ([], {"HAL_GATEWAY_UPSTREAM": "http://env.invalid"},
     (gateway.DEFAULT_GATEWAY_PORT, "http://env.invalid", True)),
    (["--upstream=http://cli.invalid"], {"HAL_GATEWAY_UPSTREAM": "http://env.invalid"},
     (gateway.DEFAULT_GATEWAY_PORT, "http://cli.invalid", True)),
    (["--no-reclaim"], {}, (gateway.DEFAULT_GATEWAY_PORT, gateway.DEFAULT_UPSTREAM, False)),
    ([], {"HAL_GATEWAY_NO_RECLAIM": "1"},
     (gateway.DEFAULT_GATEWAY_PORT, gateway.DEFAULT_UPSTREAM, False)),
])
def test_cli_forms(monkeypatch, argv, env, expected):
    monkeypatch.delenv("HAL_GATEWAY_UPSTREAM", raising=False)
    monkeypatch.delenv("HAL_GATEWAY_NO_RECLAIM", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    serve = Mock()
    monkeypatch.setattr(gateway, "serve", serve)
    assert gateway.main(argv) == 0
    port, upstream, reclaim = expected
    serve.assert_called_once_with(port=port, upstream_base=upstream, reclaim=reclaim)


def test_upstream_reaches_proxy(monkeypatch):
    monkeypatch.setenv("HAL_GATEWAY_UPSTREAM", "http://env.invalid")
    proxies = []
    def handler(proxy, cap):
        proxies.append(proxy)
        return BaseHTTPRequestHandler
    monkeypatch.setattr(gateway, "make_gateway_handler", handler)
    server = Mock()
    monkeypatch.setattr(gateway, "_bind_gateway", Mock(return_value=server))
    gateway.serve(port=0, upstream_base="http://cli.invalid")
    assert proxies[0]._proxies[0].upstream_base == "http://cli.invalid"
    server.serve_forever.assert_called_once()


@pytest.mark.parametrize("status,body,label", [
    (200, {"service": "hal-gateway"}, "healthy hal-gateway"),
    (200, {"service": "other-service"}, "responding HTTP service"),
    (404, {"error": "missing"}, "responding HTTP service"),
    (200, "not-json", "responding HTTP service"),
])
def test_http_holder_is_never_reclaimed(monkeypatch, capsys, status, body, label):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/health"
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *args):
            pass
    holder = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = holder.server_port
    assert port not in {4010, 4011, 4012, 7416}
    thread = threading.Thread(target=holder.serve_forever, daemon=True)
    thread.start()
    reclaim = Mock(side_effect=forbidden)
    monkeypatch.setattr(gateway, "reclaim_port", reclaim)
    monkeypatch.delenv("HAL_GATEWAY_NO_RECLAIM", raising=False)
    # An invalid proxy setting proves the health check stays on loopback.
    monkeypatch.setenv("http_proxy", "http://proxy.invalid")
    try:
        assert gateway.main(["--port", str(port)]) == 1
        reclaim.assert_not_called()
        assert capsys.readouterr().err == (
            f"HAL Gateway: port {port} held by {label}; refusing reclaim\n")
        assert gateway._http_port_holder(port) == label
    finally:
        holder.shutdown()
        holder.server_close()
        thread.join(timeout=2)


def test_non_http_listener_reaches_reclaim(monkeypatch):
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        assert port not in {4010, 4011, 4012, 7416}
        reclaim = Mock(return_value=[])
        monkeypatch.setattr(gateway, "reclaim_port", reclaim)
        with pytest.raises(OSError) as exc:
            gateway._bind_gateway(port, BaseHTTPRequestHandler, attempts=1)
        assert exc.value.errno == errno.EADDRINUSE
        reclaim.assert_called_once_with(port)


def test_listen_receives_backlog_before_activation(monkeypatch):
    calls = []
    original = socket.socket.listen
    def listen(sock, backlog):
        calls.append(backlog)
        return original(sock, backlog)
    monkeypatch.setattr(socket.socket, "listen", listen)
    server = gateway._bind_gateway(0, BaseHTTPRequestHandler, reclaim=False)
    try:
        assert server.server_port not in {4010, 4011, 4012, 7416}
        assert calls == [128]
    finally:
        server.server_close()


@pytest.mark.parametrize("value", ["ftp://upstream.invalid", "https:///missing", "http://[broken"])
@pytest.mark.parametrize("source", ["flag", "env"])
def test_invalid_upstream_exits_before_startup(monkeypatch, capsys, value, source):
    monkeypatch.setattr(gateway, "serve", forbidden)
    monkeypatch.setenv("HAL_GATEWAY_UPSTREAM", value if source == "env" else gateway.DEFAULT_UPSTREAM)
    with pytest.raises(SystemExit) as exc:
        gateway.main(["--upstream", value] if source == "flag" else [])
    assert exc.value.code == 2
    assert "upstream must be an http(s) URL with a host" in capsys.readouterr().err
