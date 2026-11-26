"""The ledger's observe hooks on the wire and on the gateway.

InfiniteMachine v2, Step 5b. The wire (inference and aux routes) and the gateway
read every Anthropic answer's rate-limit headers into the per-account ledger, and
decide nothing: the answer is relayed unchanged.

Everything runs on loopback against local fakes; nothing reaches a real provider.
The wire under test listens on a scratch port in 47000-47999 and never reclaims a
port. Every credential is an obvious fake.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("cryptography")

from lib.hal import account_ledger as al  # noqa: E402
from lib.hal import wire_ca, wire_proxy  # noqa: E402
from lib.hal.account_ledger import (UNPOOLED, AccountLedger, AccountResolver,  # noqa: E402
                                    PoolAccount)
from lib.hal.gateway_failover import FailoverProxy, GatewayTelemetry, UpstreamSpec  # noqa: E402
from tests._loopback_http import HttpUpstream, clear_proxies  # noqa: E402
from tests.test_wire_proxy import (API, _quiet_close, _read_response, _Reader,  # noqa: E402
                                   _TestPki, _trusting, wait_for_exchanges)

P = al.PREFIX
TOKEN_A = "fake-account-A-token"
TOKEN_B = "fake-account-B-token"
SCRATCH_PORTS = range(47000, 48000)


def _limits(reset: int) -> dict:
    """Header sets a fake answers with, the reset relative to the real clock."""
    return {
        "spent": [(al.STATUS, "rejected"), (al.RESET, str(reset)), (al.CLAIM, "five_hour"),
                  (f"{P}5h-utilization", "1"), (f"{P}5h-reset", str(reset)),
                  ("retry-after", "60"), ("request-id", "SEEDED-REQUEST-ID")],
        "fine": [(al.STATUS, "allowed"), (f"{P}5h-utilization", "0.25"),
                 (f"{P}5h-reset", str(reset)), (f"{P}7d-utilization", "0.5"),
                 (f"{P}7d-reset", str(reset + 86400))],
    }


def _answer(status: int, headers, body: dict) -> bytes:
    payload = json.dumps(body).encode()
    lines = [f"HTTP/1.1 {status} {'OK' if status < 400 else 'Error'}"]
    lines += [f"{name}: {value}" for name, value in headers]
    lines += ["content-type: application/json", f"content-length: {len(payload)}"]
    return ("\r\n".join(lines) + "\r\n\r\n").encode() + payload


# --------------------------------------------------------------------------- #
# The wire: a fake api.anthropic.com that answers with rate-limit headers.
# --------------------------------------------------------------------------- #
class LimitsUpstream:
    """Answers ``POST /v1/messages`` by the request's model; any other path gets the
    rate-limit headers too, which the ledger must not read (it is account-bound)."""

    def __init__(self, context, limits: dict):
        self.context = context
        self.limits = limits
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self) -> None:
        _quiet_close(self._listener)

    def _accept(self) -> None:
        while True:
            try:
                raw, _ = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(raw,), daemon=True).start()

    def _serve(self, raw) -> None:
        try:
            tls = self.context.wrap_socket(raw, server_side=True)
            reader = _Reader(tls)
            while self._one(tls, reader):
                pass
        except (OSError, ValueError):
            pass
        finally:
            _quiet_close(raw)

    def _one(self, tls, reader: _Reader) -> bool:
        head = reader.until(b"\r\n\r\n")
        if not head:
            return False
        length = [int(line.split(b":", 1)[1]) for line in head.lower().split(b"\r\n")
                  if line.startswith(b"content-length:")]
        body = reader.exact(length[0]) if length else b""
        path = head.split(b" ", 2)[1].split(b"?")[0]
        spent = path != b"/v1/messages" or json.loads(body).get("model") == "spent"
        tls.sendall(_answer(429 if spent else 200, self.limits["spent" if spent else "fine"],
                            {"type": "message", "model": "m",
                             "usage": {"input_tokens": 1, "output_tokens": 1}}))
        return True


def _start_on_scratch_port(proxy):
    for port in SCRATCH_PORTS:
        try:
            return wire_proxy.start_server(proxy, port, reclaim=False)
        except OSError:
            continue
    raise RuntimeError("no free scratch port in 47000-47999")


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    return _TestPki(tmp_path_factory.mktemp("ledger-wire-pki"))


@pytest.fixture
def wire(tmp_path, monkeypatch, pki):
    monkeypatch.setenv(wire_ca.STORE_ENV, "file")
    monkeypatch.setenv("HAL_INFINITE_BRAKE_MARKER", str(tmp_path / "no-brake.json"))
    reset = int(time.time()) + 3600
    fake = LimitsUpstream(pki.server_context(API), _limits(reset))
    ledger = AccountLedger(tmp_path / "ledger" / "account-ledger.json",
                           resolver=AccountResolver(lambda: [PoolAccount("acct-a", TOKEN_A)]))
    proxy = wire_proxy.WireProxy(wire_ca.load_or_create(), pool_size=0,
                                 resolve=lambda host, port: ("127.0.0.1", fake.port),
                                 upstream_context=pki.client_context(), ledger=ledger)
    server = _start_on_scratch_port(proxy)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                     daemon=True).start()
    assert server.server_address[1] in SCRATCH_PORTS
    yield SimpleNamespace(port=server.server_address[1], ledger=ledger, reset=reset,
                          client=_trusting(wire_ca.ca_cert_path().read_text()))
    server.shutdown()
    server.server_close()
    proxy.stop()
    fake.close()
    wait_for_exchanges()  # every witness row lands in this test's own HAL_HOME


def _post(path: str, model: str, token: str) -> bytes:
    body = json.dumps({"model": model, "max_tokens": 64000,
                       "messages": [{"role": "user", "content": "hello"}]}).encode()
    return (f"POST {path} HTTP/1.1\r\nHost: {API}\r\nAuthorization: Bearer {token}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
            ).encode() + body


def _through_wire(env, request: bytes) -> bytes:
    raw = socket.create_connection(("127.0.0.1", env.port), timeout=10)
    raw.sendall(f"CONNECT {API}:443 HTTP/1.1\r\nHost: {API}:443\r\n\r\n".encode())
    assert _Reader(raw).until(b"\r\n\r\n").startswith(b"HTTP/1.1 200")
    tls = env.client.wrap_socket(raw, server_hostname=API)
    try:
        tls.sendall(request)
        return _read_response(_Reader(tls))
    finally:
        _quiet_close(tls)


def test_the_wire_records_each_answer_on_its_account(wire):
    view = wire.ledger.view()
    assert _through_wire(wire, _post("/v1/messages", "fine", TOKEN_A)).startswith(
        b"HTTP/1.1 200")
    entry = view.accounts()["acct-a"]
    windows = entry["quota"]["windows"]
    assert (entry["quota"]["status"], entry["token"]["health"]) == ("allowed", "ok")
    assert (windows["five_hour"]["utilization"], windows["seven_day"]["utilization"]) == (
        0.25, 0.5)
    assert view.cooling("acct-a") is None and view.in_flight("acct-a") == 0
    reply = _through_wire(wire, _post("/v1/messages?beta=true", "spent", TOKEN_A))
    assert reply.startswith(b"HTTP/1.1 429")
    assert f"{al.RESET}: {wire.reset}".encode() in reply  # relayed unchanged: 5b only reads
    cooling = view.cooling("acct-a")
    assert (cooling.until, cooling.reason, cooling.claim) == (wire.reset, "spent", "five_hour")
    assert view.accounts()["acct-a"]["observations"] == 2


def test_the_wire_puts_another_credential_on_unpooled_and_never_reads_account_bound(wire):
    _through_wire(wire, _post("/v1/messages", "spent", "fake-claude-code-login"))
    _through_wire(wire, _post("/v1/code/sessions", "spent", TOKEN_B))
    accounts = wire.ledger.view().accounts()
    assert set(accounts) == {UNPOOLED}
    assert accounts[UNPOOLED]["observations"] == 1


def test_no_bearer_or_header_value_reaches_the_wires_sinks(wire, tmp_path):
    _through_wire(wire, _post("/v1/messages", "spent", TOKEN_A))
    _through_wire(wire, _post("/v1/messages", "fine", "fake-claude-code-login"))
    sinks = "".join(path.read_text(errors="replace") for path in tmp_path.rglob("*")
                    if path.is_file() and path.suffix in (".json", ".jsonl", ".log", ".lock"))
    assert "acct-a" in sinks
    for secret in (TOKEN_A, "fake-claude-code-login", "Bearer ", "SEEDED-REQUEST-ID"):
        assert secret.lower() not in sinks.lower(), secret  # case-blind, as the parser lower-cases


# --------------------------------------------------------------------------- #
# The gateway: every answer, each transport retry included.
# --------------------------------------------------------------------------- #
def _quiet() -> GatewayTelemetry:
    return GatewayTelemetry(sink=lambda event: None)


def test_the_gateway_records_every_answer_each_retry_included(tmp_path, monkeypatch):
    clear_proxies(monkeypatch)
    limits = _limits(int(time.time()) + 3600)
    # Retried answers are overloads: a usage-limit 429 is no longer retried on
    # the same account (HAL#678), which the next test pins.
    upstream = HttpUpstream([(529, []), (529, []), (200, limits["fine"])])
    ledger = AccountLedger(tmp_path / "account-ledger.json",
                           resolver=AccountResolver(lambda: [PoolAccount("acct-a", TOKEN_A)]))
    proxy = FailoverProxy([UpstreamSpec("primary", base_url=upstream.url)],
                          sleep=lambda seconds: None, ledger=ledger, telemetry=_quiet())
    try:
        resp = proxy.forward("/v1/messages", b'{"model": "m"}',
                             [("authorization", f"Bearer {TOKEN_A}"),
                              ("content-type", "application/json")])
        assert resp.status == 200
        resp.close()
    finally:
        upstream.close()
    entry = ledger.view().accounts()["acct-a"]
    assert entry["observations"] == 3  # both retried overloads, then the 200
    assert (entry["http_status"], entry["quota"]["status"], entry["cooling"]) == (
        200, "allowed", None)
    assert ledger.view().in_flight("acct-a") == 0


def test_a_usage_limit_429_is_recorded_once_and_never_retried(tmp_path, monkeypatch):
    # The inverse of what the test above used to require (HAL#678): the spent
    # account is not asked again; its answer is recorded and the 429 returned.
    clear_proxies(monkeypatch)
    limits = _limits(int(time.time()) + 3600)
    # No retry-after: its 60 s would exceed the gateway's 10 s throttle budget
    # on its own, and then this test would pass without the usage-limit rule.
    spent = [h for h in limits["spent"] if h[0] != "retry-after"]
    upstream = HttpUpstream([(429, spent), (200, limits["fine"])])
    ledger = AccountLedger(tmp_path / "account-ledger.json",
                           resolver=AccountResolver(lambda: [PoolAccount("acct-a", TOKEN_A)]))
    proxy = FailoverProxy([UpstreamSpec("primary", base_url=upstream.url)],
                          sleep=lambda seconds: None, ledger=ledger, telemetry=_quiet())
    try:
        resp = proxy.forward("/v1/messages", b'{"model": "m"}',
                             [("authorization", f"Bearer {TOKEN_A}"),
                              ("content-type", "application/json")])
        assert resp.status == 429
        resp.close()
    finally:
        upstream.close()
    entry = ledger.view().accounts()["acct-a"]
    assert entry["observations"] == 1
    assert entry["quota"]["status"] == "rejected" and entry["cooling"] is not None
    assert ledger.view().in_flight("acct-a") == 0


def test_the_gateway_marks_a_token_dead_on_a_401(tmp_path, monkeypatch):
    clear_proxies(monkeypatch)
    upstream = HttpUpstream([(401, [("www-authenticate", "SEEDED-AUTH-VALUE")])])
    ledger = AccountLedger(tmp_path / "account-ledger.json",
                           resolver=AccountResolver(lambda: [PoolAccount("acct-b", TOKEN_B)]))
    proxy = FailoverProxy([UpstreamSpec("primary", base_url=upstream.url)],
                          sleep=lambda seconds: None, ledger=ledger, telemetry=_quiet())
    try:
        assert proxy.forward("/v1/messages", b"{}", [("x-api-key", TOKEN_B)]).status == 401
    finally:
        upstream.close()
    assert ledger.view().token_health("acct-b") == "dead"
    assert ledger.view().token_health("acct-b", "fake-relogged-token") == "unknown"
    assert "SEEDED-AUTH-VALUE" not in (tmp_path / "account-ledger.json").read_text()
