"""Portable body-consumption checks for each gateway early POST refusal."""
import http.client
import io
from unittest.mock import Mock

import pytest

from lib.hal import gateway


@pytest.mark.parametrize("case,status", [("route", 404), ("access", 403)])
def test_gateway_refusals_drain_body(monkeypatch, case, status):
    cap = gateway.InflightCap(1, 1)
    cls = gateway.make_gateway_handler(Mock(), cap)
    h = cls.__new__(cls)
    h.path = "/missing" if case == "route" else "/v1/messages"
    h.command, h.request_version = "POST", "HTTP/1.0"
    h.requestline = f"POST {h.path} HTTP/1.0"
    raw = b'{"test":true}'
    h.headers = http.client.parse_headers(io.BytesIO(f"Content-Length: {len(raw)}\r\n\r\n".encode()))
    h.rfile, h.wfile = io.BytesIO(raw), io.BytesIO()
    h.close_connection = True
    h.connection = Mock()
    def access(handler):
        if case == "access":
            handler.send_error(403)
            return False
        return True
    monkeypatch.setattr(gateway, "enforce_gateway_access", access)
    h.do_POST()
    assert int(h.wfile.getvalue().split(b" ", 2)[1]) == status
    assert h.rfile.tell() == len(raw)
    h.connection.settimeout.assert_called_once_with(2.0)
