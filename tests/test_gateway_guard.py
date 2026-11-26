"""Goodhart-proof tests for the HAL gateway access guard.

Each test fails under the obvious mutation of the code it covers:
  * default-deny tests fail if remote-default flips to allow;
  * the fail-closed test fails if the except-clause is removed (fail-open);
  * the constant-time test fails if ``hmac.compare_digest`` is replaced by
    ``==``;
  * the 403 test fails if the deny path stops sending the error body.

Assertions are on returned values + captured I/O, never on call counts alone
(Rule 14 — a test that always passes is worse than no test).
"""
from __future__ import annotations

import hmac
import io
from unittest.mock import Mock

import pytest

from lib.hal import gateway_guard
from lib.hal.gateway_guard import (
    enforce_gateway_access,
    gateway_access_allowed,
    is_loopback_caller,
    is_remote_request,
)

# --- header fixtures --------------------------------------------------------

_LOCAL = [("Content-Type", "application/json"), ("Content-Length", "12")]
_REMOTE_CF = [("Content-Type", "application/json"), ("CF-Ray", "abc123-LHR")]
_TOKEN_HDR = "X-HAL-Gateway-Token"


def _remote_with_token(value: str):
    """Remote request (cf-ray) carrying a gateway token header."""
    return [("CF-Ray", "abc123-LHR"), (_TOKEN_HDR, value)]


# --- is_remote_request ------------------------------------------------------


def test_is_remote_request_cf_ray_true():
    assert is_remote_request([("CF-Ray", "x")]) is True


def test_is_remote_request_x_forwarded_for_true():
    assert is_remote_request([("X-Forwarded-For", "1.2.3.4")]) is True


def test_is_remote_request_true_client_ip_true():
    assert is_remote_request([("True-Client-IP", "1.2.3.4")]) is True


def test_is_remote_request_no_proxy_headers_false():
    assert is_remote_request(_LOCAL) is False


def test_is_remote_request_case_insensitive():
    # Mixed-case header name must still be detected.
    assert is_remote_request([("cF-rAy", "x")]) is True


def test_is_remote_request_accepts_mapping():
    # The (name, value) iterable contract must also accept a dict mapping.
    assert is_remote_request({"cf-connecting-ip": "9.9.9.9"}) is True
    assert is_remote_request({"content-type": "application/json"}) is False


# --- gateway_access_allowed: local ------------------------------------------


def _loopback_handler():
    """A handler whose connecting address is a loopback address."""
    handler = Mock()
    handler.client_address = ("127.0.0.1", 54321)
    return handler


def _remote_handler(ip="100.64.0.5"):
    """A handler whose connecting address is a non-loopback (tailnet) address."""
    handler = Mock()
    handler.client_address = (ip, 54321)
    return handler


def test_local_allowed_without_token():
    # Proven loopback (via the connecting address) is allowed with no token.
    assert gateway_access_allowed(_LOCAL, token="", handler=_loopback_handler()) is True


def test_local_allowed_even_with_token_set():
    # Local is always allowed regardless of token config.
    assert gateway_access_allowed(_LOCAL, token="s3cr3t", handler=_loopback_handler()) is True


# --- gateway_access_allowed: remote default-deny (CORE SECURITY) ------------


def test_remote_denied_when_token_unset():
    # CORE assertion: remote + no configured token => DENY by default.
    # A mutation flipping the default to allow MUST fail here.
    assert gateway_access_allowed(_REMOTE_CF, token="") is False


def test_remote_denied_when_token_unset_via_env(monkeypatch):
    # token=None resolves from the env var; absent => "" => deny.
    monkeypatch.delenv("HAL_GATEWAY_AUTH_TOKEN", raising=False)
    assert gateway_access_allowed(_REMOTE_CF, token=None) is False


# --- gateway_access_allowed: remote with token ------------------------------


def test_remote_denied_when_token_header_missing():
    assert gateway_access_allowed(_REMOTE_CF, token="s3cr3t") is False


def test_remote_denied_with_wrong_token():
    assert gateway_access_allowed(_remote_with_token("wrong"), token="s3cr3t") is False


def _authenticated_remote_whois(ip):
    """A ``tailscale whois --json`` result for a node carrying the ACL tag."""
    return {"Node": {"Tags": ["tag:grip-brain"]}}


def test_remote_allowed_with_correct_token(monkeypatch):
    # A remote peer is admitted only after Tailscale tag authentication AND,
    # when a static token is configured, a matching token header.
    monkeypatch.setenv("HAL_GATEWAY_TS_ACL_TAG", "tag:grip-brain")
    handler = _remote_handler()
    assert (
        gateway_access_allowed(
            _remote_with_token("s3cr3t"),
            token="s3cr3t",
            handler=handler,
            whois_func=_authenticated_remote_whois,
        )
        is True
    )


def test_remote_token_resolves_from_env(monkeypatch):
    monkeypatch.setenv("HAL_GATEWAY_AUTH_TOKEN", "env-secret")
    monkeypatch.setenv("HAL_GATEWAY_TS_ACL_TAG", "tag:grip-brain")
    handler = _remote_handler()
    assert (
        gateway_access_allowed(
            _remote_with_token("env-secret"),
            token=None,
            handler=handler,
            whois_func=_authenticated_remote_whois,
        )
        is True
    )
    assert (
        gateway_access_allowed(
            _remote_with_token("nope"),
            token=None,
            handler=handler,
            whois_func=_authenticated_remote_whois,
        )
        is False
    )


# --- fail-closed ------------------------------------------------------------


class _ExplodingHeaders:
    """A headers object whose iteration AND .get raise — the malformed case."""

    def items(self):
        raise RuntimeError("malformed header store")

    def __iter__(self):
        raise RuntimeError("malformed header store")

    def get(self, *_a, **_k):
        raise RuntimeError("malformed header store")


def test_fail_closed_on_exploding_headers():
    # A fail-OPEN implementation (no try/except, or `return True` on error)
    # fails this test. Must DENY, must not raise. The handler's non-loopback
    # address routes the request into the remote path so the exploding headers
    # actually reach the guard's header iteration.
    result = gateway_access_allowed(
        _ExplodingHeaders(), token="s3cr3t", handler=_remote_handler()
    )
    assert result is False


# --- constant-time compare (Goodhart anchor for hmac.compare_digest) --------


def test_uses_constant_time_compare(monkeypatch):
    # Prove the token comparison routes through hmac.compare_digest. A naive
    # `==` mutation never calls it, so `seen` stays empty and this fails.
    monkeypatch.setenv("HAL_GATEWAY_TS_ACL_TAG", "tag:grip-brain")
    seen = {}
    real = hmac.compare_digest

    def _spy(a, b):
        seen["called"] = (a, b)
        return real(a, b)

    monkeypatch.setattr(gateway_guard.hmac, "compare_digest", _spy)
    assert (
        gateway_access_allowed(
            _remote_with_token("s3cr3t"),
            token="s3cr3t",
            handler=_remote_handler(),
            whois_func=_authenticated_remote_whois,
        )
        is True
    )
    assert seen.get("called") == ("s3cr3t", "s3cr3t")


# --- enforce_gateway_access (BaseHTTPRequestHandler integration) ------------


class _FakeWFile:
    def __init__(self, *, fail: bool = False):
        self._buf = io.BytesIO()
        self._fail = fail

    def write(self, data):
        if self._fail:
            raise BrokenPipeError("client hung up")
        return self._buf.write(data)

    def getvalue(self):
        return self._buf.getvalue()


class _FakeHandler:
    """Captures the BaseHTTPRequestHandler surface the guard touches."""

    def __init__(self, header_pairs, *, wfile_fail: bool = False, client_ip=None):
        self.headers = dict(header_pairs)
        self.status = None
        self.sent_headers = []
        self.ended = False
        self.wfile = _FakeWFile(fail=wfile_fail)
        self.client_address = (client_ip, 12345) if client_ip is not None else None

    def send_response(self, code):
        self.status = code

    def send_header(self, name, value):
        self.sent_headers.append((name, value))

    def end_headers(self):
        self.ended = True


def test_enforce_denies_remote_with_403_and_body():
    handler = _FakeHandler(_REMOTE_CF)  # remote, no token configured => deny
    allowed = enforce_gateway_access(handler)

    assert allowed is False
    assert handler.status == 403
    assert handler.ended is True
    # Leak-free error envelope written verbatim.
    assert handler.wfile.getvalue() == (
        b'{"error":{"type":"forbidden","message":"gateway access denied"}}'
    )
    header_map = {n.lower(): v for n, v in handler.sent_headers}
    assert header_map["content-type"] == "application/json"
    assert header_map["connection"] == "close"


def test_enforce_allows_local_and_sends_nothing():
    handler = _FakeHandler(_LOCAL, client_ip="127.0.0.1")  # proven loopback => allow
    allowed = enforce_gateway_access(handler)

    assert allowed is True
    assert handler.status is None
    assert handler.ended is False
    assert handler.wfile.getvalue() == b""


def test_guard_wired_into_server_routing():
    # Goodhart anchor: the guard must be imported AND called in both verbs
    # (do_GET + do_POST). Removing the wire fails this even if the module tests pass.
    import os
    src = open(os.path.join(os.path.dirname(__file__), "..", "lib", "hal", "server.py"),
               encoding="utf-8").read()
    assert "from lib.hal.gateway_guard import enforce_gateway_access" in src
    assert src.count("enforce_gateway_access(self)") >= 2


def test_guard_wired_into_standalone_gateway():
    import os
    src = open(os.path.join(os.path.dirname(__file__), "..", "lib", "hal", "gateway.py"),
               encoding="utf-8").read()
    assert "from lib.hal.gateway_guard import enforce_gateway_access" in src
    assert src.count("enforce_gateway_access(self)") >= 2


def test_enforce_does_not_raise_on_broken_pipe():
    # Deny path with a wfile that raises BrokenPipe — must be swallowed.
    handler = _FakeHandler(_REMOTE_CF, wfile_fail=True)
    allowed = enforce_gateway_access(handler)
    assert allowed is False
    assert handler.status == 403


# --- is_loopback_caller ------------------------------------------------------
# Used by the public /api/health route (see tests/test_health.py and
# tests/test_server_api_access_gate.py) to decide whether the response may
# reveal the absolute HAL_HOME path. Composes the two existing primitives, so
# these tests exercise the public seam rather than re-deriving the logic.


def test_loopback_caller_ipv4():
    handler = _FakeHandler(_LOCAL, client_ip="127.0.0.1")
    assert is_loopback_caller(handler) is True


def test_loopback_caller_ipv6():
    handler = _FakeHandler(_LOCAL, client_ip="::1")
    assert is_loopback_caller(handler) is True


def test_remote_caller_is_not_loopback():
    handler = _FakeHandler(_REMOTE_CF, client_ip="203.0.113.5")
    assert is_loopback_caller(handler) is False


def test_missing_client_address_is_not_loopback():
    # Fail-safe: an unrecognised/malformed handler must default to the MORE
    # restrictive branch (redacted), never to revealing the path.
    handler = _FakeHandler(_LOCAL)  # client_ip=None -> client_address=None
    assert is_loopback_caller(handler) is False


def test_loopback_with_no_headers_attribute_fails_closed():
    # Ambiguous input (can't confirm "no forwarding header") must resolve to
    # NOT genuinely local -- the module's own fail-CLOSED invariant, applied
    # to this function too. A SimpleNamespace with only client_address set
    # has no .headers at all, unlike every _FakeHandler-based case above.
    from types import SimpleNamespace as _SNS
    handler = _SNS(client_address=("127.0.0.1", 1))
    assert is_loopback_caller(handler) is False


def test_loopback_with_real_stdlib_header_message_shape():
    # is_remote_request (via _header_pairs) is built for a plain mapping or
    # pair-iterable; the REAL handler.headers in production is an
    # email.message.Message (BaseHTTPRequestHandler's actual type), whose
    # .items() shape differs from the dict _FakeHandler uses everywhere
    # else in this file. Exercise that real shape directly rather than
    # arguing it is compatible.
    import email.message

    local_headers = email.message.Message()
    local_headers["Content-Type"] = "application/json"
    handler = _FakeHandler(_LOCAL, client_ip="127.0.0.1")
    handler.headers = local_headers
    assert is_loopback_caller(handler) is True

    tunnelled_headers = email.message.Message()
    tunnelled_headers["CF-Ray"] = "abc123-LHR"
    handler.headers = tunnelled_headers
    assert is_loopback_caller(handler) is False


def test_tunnelled_request_with_loopback_address_is_not_genuinely_local():
    # The production case this function exists to get right: hal-server sits
    # behind a Cloudflare Tunnel, whose local cloudflared agent connects to
    # the origin over loopback for EVERY request it forwards -- so a real
    # internet caller still arrives with client_address=127.0.0.1. Only the
    # preserved cf-ray header still says "this came through the tunnel".
    # Mirrors gateway_access_allowed's own loopback-plus-proxy-header case
    # (test_remote_with_correct_token_allowed above exercises the same
    # header shape on that function).
    handler = _FakeHandler(_REMOTE_CF, client_ip="127.0.0.1")
    assert is_loopback_caller(handler) is False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
