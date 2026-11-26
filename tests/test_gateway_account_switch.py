"""Real loopback upstreams, fake credentials, production single-primary gateway."""
import http.client
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lib.hal import account_switch as switch, gateway, gateway_accounts, spend_guard
from lib.hal import gateway_openai, gateway_responses
from lib.hal.gateway_failover import build_failover_proxy

A, B = "fake-account-A-token", "fake-account-B-token"
LIMIT = "anthropic-ratelimit-unified-"
BODY = b'{"model":"claude-test","messages":[{"role":"user","content":"hi"}]}'
ANSWER = b'{"id":"msg_fake","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"B answer"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}}'


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    switch.forget_exhausted()
    switch._ORGS.clear()
    switch._ORG_RETRY.clear()
    monkeypatch.setattr(spend_guard, "_OAUTH_PREFIX", "fake-account-")
    monkeypatch.setattr(gateway, "_normalise_anthropic_auth", lambda h: h)
    monkeypatch.setattr(switch, "_read_pool", lambda: [("B", B)])
    monkeypatch.setenv("HAL_GATEWAY_MULTIPROVIDER", "0")
    policy = tmp_path / "policy.json"
    policy.write_text('{"paid_opt_in":[]}')
    monkeypatch.setenv("HAL_SPEND_GUARD_CONFIG", str(policy))
    def headers(pairs):
        return [(k, v) for k, v in pairs if k.lower() not in ("host", "content-length")]
    monkeypatch.setattr(gateway_openai, "anthropic_upstream_headers", headers)
    monkeypatch.setattr(gateway_responses, "anthropic_upstream_headers", headers)
    yield
    switch.forget_exhausted()
    switch._ORGS.clear()
    switch._ORG_RETRY.clear()


@contextmanager
def server(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    assert srv.server_port not in (3850, 4010, 4011, 4012, 7416)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join()


@contextmanager
def upstream(script):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.send_response(200)
            self.send_header("anthropic-organization-id", "org-" + self.headers.get("Authorization", ""))
            self.send_header("Content-Length", "0")
            self.end_headers()
        def do_POST(self):
            token = self.headers.get("Authorization", "").removeprefix("Bearer ")
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls.append((token, self.path, body, list(self.headers.items())))
            status, headers, answer = script(token, len(calls))
            self.send_response(status)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)
    with server(Handler) as srv:
        yield f"http://127.0.0.1:{srv.server_port}", calls


def limited(seconds=600):
    return 429, [(LIMIT + "status", "rejected"),
                 (LIMIT + "reset", str(time.time() + seconds))], b'original limit'


def send(proxy, path="/v1/messages", body=BODY, extra=()):
    return proxy.forward(path, body, [("Authorization", "Bearer " + A), *extra])


def test_limit_once(monkeypatch):
    monkeypatch.setattr(switch, "_read_pool", lambda: [])
    with upstream(lambda *_: limited()) as (url, calls):
        with pytest.raises(spend_guard.SpendRefused):
            send(build_failover_proxy(upstream_base=url))
        assert len(calls) == 1


@pytest.mark.parametrize("signal,value", [("representative-claim", "five_hour"), ("overage-status", "rejected")])
def test_shared_limit_predicate(signal, value, monkeypatch):
    monkeypatch.setattr(switch, "_read_pool", lambda: [])
    with upstream(lambda *_: (429, [(LIMIT + signal, value)], b'limit')) as (url, calls):
        with pytest.raises(spend_guard.SpendRefused):
            send(build_failover_proxy(upstream_base=url))
        assert len(calls) == 1


def test_plain_throttle_waits_and_stops_at_budget():
    sleeps = []
    with upstream(lambda *_: (429, [("Retry-After", "5")], b'throttle')) as (url, calls):
        response = send(gateway.GatewayProxy(upstream_base=url, max_retries=20, sleep=sleeps.append))
        assert response.status == 429
        response.close()
        assert sleeps == [5, 5]
        assert len(calls) == 3


def test_retry_after_too_long_is_not_shortened():
    with upstream(lambda *_: (429, [("Retry-After", "11")], b'throttle')) as (url, calls):
        response = send(gateway.GatewayProxy(upstream_base=url, sleep=lambda _: pytest.fail("early retry")))
        response.close()
        assert len(calls) == 1


def test_replay_and_cooldown(tmp_path):
    with upstream(lambda token, _: limited() if token == A else (200, [], ANSWER)) as (url, calls):
        proxy = build_failover_proxy(upstream_base=url)
        assert send(proxy).read_all() == ANSWER
        assert send(proxy).read_all() == ANSWER
        assert [c[0] for c in calls] == [A, B, B]
        assert all(c[2] == BODY for c in calls)
        without_auth = lambda c: [(k, v) for k, v in c[3] if k.lower() != "authorization"]
        assert without_auth(calls[0]) == without_auth(calls[1])
    import os
    from pathlib import Path
    rows = [json.loads(line) for line in (Path(os.environ["HAL_HOME"]) / "logs/gateway-audit.jsonl").read_text().splitlines()]
    switches = [r for r in rows if r["event"] == "gateway.account_switched"]
    assert [(r["to"], r["rank"]) for r in switches] == [("B", "subscription")] * 2
    assert A not in json.dumps(rows) and B not in json.dumps(rows)


def test_paid_replacement_refused(monkeypatch):
    monkeypatch.setattr(switch, "choose", lambda *a, **k: ("paid", "fake-paid", "overage"))
    real = spend_guard.anthropic_credential
    monkeypatch.setattr(spend_guard, "anthropic_credential", lambda h: (spend_guard.PAID, "")
        if any(v == "Bearer fake-paid" for _, v in h) else real(h))
    with upstream(lambda *_: (200, [], ANSWER)) as (url, calls):
        with pytest.raises(spend_guard.SpendRefused):
            send(build_failover_proxy(upstream_base=url))
        assert calls == []


@pytest.mark.parametrize("path", ["/v1/messages/batches", "/v1/messages/count_tokens", "/v1/code/test", "/api/oauth/test"])
def test_account_bound_unchanged(path):
    with upstream(lambda *_: limited()) as (url, calls):
        response = send(build_failover_proxy(upstream_base=url), path)
        assert response.status == 429 and response.read_all() == b'original limit'
        assert [c[0] for c in calls] == [A]
        assert switch.exhausted_until(spend_guard.account_key(A)) is None


def test_unbuffered_no_replay():
    with upstream(lambda *_: limited()) as (url, calls):
        response = send(build_failover_proxy(upstream_base=url), body=None)
        assert response.status == 429
        response.close()
        assert [c[0] for c in calls] == [A]


def test_off_switch():
    switch.set_switching(False)
    with upstream(lambda *_: limited()) as (url, calls):
        with pytest.raises(spend_guard.SpendRefused):
            send(build_failover_proxy(upstream_base=url))
        assert [c[0] for c in calls] == [A]


def test_no_switch_after_client_bytes():
    error = b'event: error\ndata: {"error":{"type":"rate_limit_error"}}\n\n'
    with upstream(lambda *_: (200, [("Content-Type", "text/event-stream")], b'data: first\n\n' + error)) as (url, calls):
        response = send(build_failover_proxy(upstream_base=url))
        chunks = response.iter_chunks(size=4)
        assert next(chunks) == b'data'
        assert b'first' in b''.join(chunks)
        assert [c[0] for c in calls] == [A]


@pytest.mark.parametrize("route", ["/v1/responses", "/v1/chat/completions"])
def test_real_exhaustion_last_rung(route):
    with upstream(lambda token, _: limited(600 if token == A else 120)) as (url, calls):
        proxy = build_failover_proxy(upstream_base=url)
        with server(gateway.make_gateway_handler(proxy, gateway.InflightCap.from_env())) as srv:
            conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
            payload = {"model": "claude-test", "input": "hi", "messages": [{"role": "user", "content": "hi"}], "stream": True}
            conn.request("POST", route, json.dumps(payload), {"Authorization": "Bearer " + A, "Content-Type": "application/json"})
            result = conn.getresponse()
            answer = result.read().decode()
            if route == "/v1/responses":
                assert result.status == 200
                assert answer.count('event: response.completed') == 1
                assert 'Earliest reset:' in answer
                earliest = datetime_stamp(time.time() + 120)
                assert earliest in answer
            else:
                assert result.status == 429
                assert 118 <= int(result.getheader("Retry-After")) <= 121
            conn.close()
        assert [c[0] for c in calls] == [A, B]


def datetime_stamp(epoch):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def test_partial_request_body_is_not_forwardable():
    import io
    from types import SimpleNamespace
    handler = SimpleNamespace(headers={"Content-Length": str(len(BODY) + 1)},
                              rfile=io.BytesIO(BODY))
    with pytest.raises(ValueError, match="incomplete request body"):
        gateway.read_request_body(handler)


def test_one_replay_does_not_claim_a_third_account_is_spent(monkeypatch):
    third = "fake-account-C-token"
    monkeypatch.setattr(switch, "_read_pool", lambda: [("B", B), ("C", third)])
    with upstream(lambda token, _: (200, [], ANSWER) if token == third else limited()) as (url, calls):
        proxy = build_failover_proxy(upstream_base=url)
        response = send(proxy)
        assert response.status == 429
        response.close()
        assert [c[0] for c in calls] == [A, B]
        assert send(proxy).read_all() == ANSWER
        assert [c[0] for c in calls] == [A, B, third]


def test_dead_pool_token_is_skipped():
    from lib.hal.account_ledger import AccountLedger, AccountResolver
    ledger = AccountLedger(resolver=AccountResolver(pool=lambda: [('B', B)]))
    with ledger.exchange([('Authorization', 'Bearer ' + B)]) as watch:
        watch.answered(401, [])
    with upstream(lambda token, _: limited() if token == A else (200, [], ANSWER)) as (url, calls):
        with pytest.raises(spend_guard.SpendRefused):
            send(build_failover_proxy(upstream_base=url))
        assert [c[0] for c in calls] == [A]
