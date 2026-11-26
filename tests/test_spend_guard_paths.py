"""T3: no billing path presents a paid credential the operator has not opted in (Step 5h).

InfiniteMachine v2, Step 5h. The oracle is the upstream, not the guard: a fake
Anthropic on loopback records every request it receives, and each test asserts
on what reached it. The real gateway handler sits in front, on all three rails
(Anthropic ``/v1/messages``, OpenAI ``/v1/chat/completions`` and
``/v1/responses``); the native provider is driven with ``urlopen`` replaced; the
removed rungs are driven as they used to fire. Strict policy, frozen clock:
tests/conftest.py's permissive policy is replaced in every test here.

Credentials are obvious fakes, shaped only by their public prefixes.
"""
from __future__ import annotations

import http.client
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lib.hal import spend_guard
from lib.hal.gateway import GatewayProxy, InflightCap, make_gateway_handler
from lib.hal.gateway_failover import build_failover_proxy
from lib.hal.providers import anthropic_native
from lib.hal.providers.anthropic_native import AnthropicError, AnthropicProvider

BEFORE = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
LAST_SECOND = datetime(2026, 10, 10, 23, 59, 59, tzinfo=timezone.utc)
AFTER = datetime(2026, 10, 11, 0, 0, 1, tzinfo=timezone.utc)
FAKE_KEY = "sk-ant-api03-fake-test-key"
FAKE_OAUTH = "sk-ant-oat01-fake-test-token"
FAKE_FALLBACK_KEY = "sk-ant-api03-fake-fallback-key"
MODEL = "claude-opus-5-5"
REPLY = {"id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
         "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
         "usage": {"input_tokens": 3, "output_tokens": 1}}
OVERAGE = {"anthropic-ratelimit-unified-status": "rejected",
           "anthropic-ratelimit-unified-overage-status": "allowed",
           "anthropic-ratelimit-unified-reset": str(int(BEFORE.timestamp()) + 600)}


@pytest.fixture
def policy(tmp_path, monkeypatch):
    """Strict policy (the committed default's shape) and a frozen clock. ``caps``
    is ``daily_caps_usd``: HAL's own API-key path needs one with room left."""
    path = tmp_path / "spend_guard.json"

    def write(opt_in=(), moment=BEFORE, caps=None) -> None:
        policy = {"paid_opt_in": list(opt_in)}
        if caps is not None:
            policy["daily_caps_usd"] = caps
        path.write_text(json.dumps(policy), encoding="utf-8")
        monkeypatch.setattr(spend_guard, "_clock", lambda: moment)

    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(path))
    write()
    return write


# --------------------------------------------------------------------------- #
# A fake Anthropic that records what reaches it.
# --------------------------------------------------------------------------- #
class FakeAnthropic:
    def __init__(self, status: int = 200, extra=None):
        self.requests = []           # (method, path, {lower-case name: value})
        self.status, self.extra = status, dict(extra or {})
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _serve(self):
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                owner.requests.append((self.command, self.path,
                                       {k.lower(): v for k, v in self.headers.items()}))
                body = json.dumps(REPLY).encode()
                self.send_response(owner.status)
                self.send_header("Content-Type", "application/json")
                for name, value in owner.extra.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_POST = do_GET = _serve

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        _serve_in_background(self.server)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def credentials(self) -> list:
        """Every credential header value that reached this upstream."""
        return [value for _m, _p, headers in self.requests
                for name, value in headers.items() if name in ("x-api-key", "authorization")]


def _serve_in_background(server) -> None:
    """Serve on a daemon thread, polling fast so ``shutdown`` returns at once."""
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                     daemon=True).start()


@pytest.fixture
def upstream(monkeypatch):
    # Loopback only, whatever proxy the shell running the suite exports.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    made = []

    def build(status: int = 200, extra=None) -> FakeAnthropic:
        fake = FakeAnthropic(status, extra)
        made.append(fake)
        return fake

    yield build
    for fake in made:
        fake.close()


@pytest.fixture
def gateway(upstream):
    """The real gateway handler on loopback, in front of a fresh fake Anthropic."""
    servers = []

    def build(status: int = 200, extra=None):
        fake = upstream(status, extra)
        proxy = GatewayProxy(upstream_base=fake.base, max_retries=0)
        # Step 5a: the handler now takes the per-host in-flight cap. This
        # suite drives one request at a time, so any permissive cap holds.
        server = ThreadingHTTPServer(("127.0.0.1", 0),
                                      make_gateway_handler(proxy, InflightCap(64, 5)))
        _serve_in_background(server)
        servers.append(server)
        return fake, server.server_address[1]

    yield build
    for server in servers:
        server.shutdown()
        server.server_close()


def _call(port: int, path: str, payload: dict, headers: dict):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("POST", path, body=json.dumps(payload).encode(),
                     headers={"Content-Type": "application/json", **headers})
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


MESSAGES = ("/v1/messages?beta=true",
            {"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})
CHAT = ("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
RESPONSES = ("/v1/responses", {"model": MODEL, "input": "hi"})
RAILS = {"messages": MESSAGES, "chat": CHAT, "responses": RESPONSES}
PAID = {"x-api-key": {"x-api-key": FAKE_KEY},
        "api-key-bearer": {"Authorization": f"Bearer {FAKE_KEY}"}}


# --------------------------------------------------------------------------- #
# The gateway, all three rails.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("moment", [BEFORE, LAST_SECOND])
@pytest.mark.parametrize("opt_in", [(), (spend_guard.OVERAGE,)], ids=["nothing", "overage-only"])
@pytest.mark.parametrize("credential", PAID.values(), ids=PAID.keys())
@pytest.mark.parametrize("rail", RAILS.values(), ids=RAILS.keys())
def test_no_paid_credential_reaches_anthropic_without_opt_in(
        policy, gateway, rail, credential, opt_in, moment):
    policy(opt_in=opt_in, moment=moment)
    fake, port = gateway()
    status, _headers, body = _call(port, rail[0], rail[1], credential)
    assert status == 403, body
    assert b"spend guard" in body
    assert fake.requests == [], "a paid credential reached Anthropic without opt-in"


@pytest.mark.parametrize("credential", PAID.values(), ids=PAID.keys())
@pytest.mark.parametrize("rail", RAILS.values(), ids=RAILS.keys())
def test_an_opted_in_key_is_presented_at_any_date(policy, gateway, rail, credential):
    policy(opt_in=[spend_guard.API_KEY], moment=BEFORE)
    fake, port = gateway()
    status, _headers, _body = _call(port, rail[0], rail[1], credential)
    assert status == 200
    assert fake.credentials() == [FAKE_KEY]


@pytest.mark.parametrize("credential", PAID.values(), ids=PAID.keys())
def test_a_key_not_opted_in_is_refused(policy, gateway, credential):
    policy(opt_in=[spend_guard.OVERAGE], moment=AFTER)
    fake, port = gateway()
    status, _headers, _body = _call(port, *MESSAGES, credential)
    assert status == 403
    assert fake.requests == []


@pytest.mark.parametrize("rail", RAILS.values(), ids=RAILS.keys())
def test_the_subscription_flows_with_nothing_opted_in(policy, gateway, rail):
    """The control: the guard refuses paid credentials, never the subscription."""
    fake, port = gateway()
    status, _headers, _body = _call(port, rail[0], rail[1],
                                    {"Authorization": f"Bearer {FAKE_OAUTH}"})
    assert status == 200
    assert fake.credentials() == [f"Bearer {FAKE_OAUTH}"]


def test_overage_on_a_2xx_spends_the_account_and_the_next_turn_is_refused(policy, gateway):
    fake, port = gateway(extra=OVERAGE)
    oauth = {"Authorization": f"Bearer {FAKE_OAUTH}"}
    first = _call(port, *MESSAGES, oauth)
    second = _call(port, *MESSAGES, oauth)
    assert first[0] == 200
    assert second[0] == 429, second[2]
    assert second[1].get("Retry-After") == "600"
    assert b"overage" in second[2]
    assert len(fake.requests) == 1, "the refused turn reached Anthropic"
    # A token count does not bill, so a spent account still reaches it.
    assert _call(port, "/v1/messages/count_tokens", MESSAGES[1], oauth)[0] == 200
    assert len(fake.requests) == 2


@pytest.mark.parametrize("rail", [CHAT], ids=["chat"])
def test_a_spent_subscription_gets_the_guards_429_on_an_openai_wire_rail(policy, gateway, rail):
    """Where Step 5a meets the guard. 5a sends an UPSTREAM 429 on these rails out
    as a 503, because one 429 ends a Codex turn. The guard's own refusal of a spent
    subscription is not renamed: it stays a 429 with Retry-After until the reset
    (HAL#656), since nothing went upstream and, with paid overage not opted in,
    the account cannot serve before then. Absorbing it is the account failover
    the plan gives these rails (5g), not a rename here."""
    fake, port = gateway(extra=OVERAGE)
    oauth = {"Authorization": f"Bearer {FAKE_OAUTH}"}
    assert _call(port, *MESSAGES, oauth)[0] == 200      # overage on a 2xx: spent
    status, headers, body = _call(port, rail[0], rail[1], oauth)
    assert status == 429, body
    assert headers.get("Retry-After") == "600"
    assert json.loads(body)["error"]["code"] == "spend_refused"
    assert len(fake.requests) == 1, "the refused turn reached Anthropic"



def _responses_notice(body, stream):
    if stream:
        events = [json.loads(line[6:]) for line in body.decode().splitlines()
                  if line.startswith("data: ")]
        assert sum(e.get("type") == "response.completed" for e in events) == 1
        reply = events[-1]["response"]
    else:
        reply = json.loads(body)
    assert reply["status"] == "completed"
    assert len(reply["output"]) == 1
    message = reply["output"][0]
    assert message["role"] == "assistant"
    return message["content"][0]["text"]


@pytest.mark.parametrize("stream", [False, True])
def test_a_spent_subscription_gets_a_final_notice_on_responses(policy, gateway, stream):
    """Invert only HAL#654's Responses case: Codex must never receive HTTP 429."""
    fake, port = gateway(extra=OVERAGE)
    oauth = {"Authorization": f"Bearer {FAKE_OAUTH}"}
    assert _call(port, *MESSAGES, oauth)[0] == 200
    status, headers, body = _call(port, RESPONSES[0], dict(RESPONSES[1], stream=stream), oauth)
    assert status == 200, body
    assert headers.get("Retry-After") is None
    text = _responses_notice(body, stream)
    assert "2026-10-05 12:10 UTC" in text
    assert "spend guard" in text
    assert len(fake.requests) == 1, "the refused turn reached Anthropic"


@pytest.mark.parametrize("stream", [False, True])
def test_spent_subscription_route_split(policy, gateway, stream):
    """Step 5g: the identical refusal is retryable for chat and final for Codex."""
    fake, port = gateway(extra=OVERAGE)
    oauth = {"Authorization": f"Bearer {FAKE_OAUTH}"}
    assert _call(port, *MESSAGES, oauth)[0] == 200
    chat_status, chat_headers, chat_body = _call(port, CHAT[0], dict(CHAT[1], stream=stream), oauth)
    response_status, response_headers, response_body = _call(
        port, RESPONSES[0], dict(RESPONSES[1], stream=stream), oauth)
    assert chat_status == 429, chat_body
    assert chat_headers["Retry-After"] == "600"
    chat_error = json.loads(chat_body)["error"]
    assert chat_error["code"] == "spend_refused"
    assert response_status == 200, response_body
    assert "Retry-After" not in response_headers
    notice = _responses_notice(response_body, stream)
    assert notice == chat_error["message"]
    assert "2026-10-05 12:10 UTC" in notice
    assert len(fake.requests) == 1


# --------------------------------------------------------------------------- #
# The native provider (hal-server's Anthropic provider).
# --------------------------------------------------------------------------- #
class _FakeHTTPResponse:
    headers = {"content-type": "application/json"}

    def read(self):
        return json.dumps(REPLY).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@pytest.fixture
def urlopen_calls(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(dict(request.header_items()))
        return _FakeHTTPResponse()

    monkeypatch.setattr(anthropic_native.urllib.request, "urlopen", fake_urlopen)
    return calls


class _UnhealthySubscription:
    def health_check(self) -> bool:
        return False


@pytest.mark.parametrize("make", [
    lambda: AnthropicProvider(api_key=FAKE_KEY, prefer="api_key"),
    lambda: AnthropicProvider(api_key=FAKE_KEY, subscription=_UnhealthySubscription()),
], ids=["forced-api-key", "auto-fallback"])
def test_the_native_provider_never_sends_a_key_not_opted_in(policy, urlopen_calls, make):
    policy(opt_in=[spend_guard.OVERAGE], moment=LAST_SECOND)
    with pytest.raises(AnthropicError, match="spend guard"):
        make().chat("hi", model=MODEL)
    assert urlopen_calls == [], "the paid key was sent"


def test_the_native_provider_sends_an_opted_in_key_at_any_date(policy, urlopen_calls):
    policy(opt_in=[spend_guard.API_KEY], moment=BEFORE, caps={spend_guard.API_KEY: 5})
    AnthropicProvider(api_key=FAKE_KEY, prefer="api_key").chat("hi", model=MODEL)
    assert len(urlopen_calls) == 1
    assert {k.lower(): v for k, v in urlopen_calls[0].items()}["x-api-key"] == FAKE_KEY


@pytest.mark.parametrize("caps", [None, {spend_guard.API_KEY: 0}], ids=["no-cap", "cap-0"])
def test_an_opted_in_key_without_room_under_a_cap_is_never_sent(policy, urlopen_calls, caps):
    """Since the daily cap, opt-in alone no longer lets HAL send its own key: the
    committed policy ships the cap at 0 so nothing is spent until it is set."""
    policy(opt_in=[spend_guard.API_KEY], moment=BEFORE, caps=caps)
    with pytest.raises(AnthropicError, match="spend guard"):
        AnthropicProvider(api_key=FAKE_KEY, prefer="api_key").chat("hi", model=MODEL)
    assert urlopen_calls == [], "the paid key was sent"


def test_a_paid_key_never_counts_as_capacity(policy):
    provider = AnthropicProvider(api_key=FAKE_KEY, subscription=_UnhealthySubscription())
    assert provider.health_check() is False
    policy(opt_in=[spend_guard.API_KEY], moment=AFTER, caps={spend_guard.API_KEY: 0})
    assert provider.health_check() is False, "a key with a $0 cap is not usable"
    policy(opt_in=[spend_guard.API_KEY], moment=AFTER, caps={spend_guard.API_KEY: 5})
    assert provider.health_check() is True


# --------------------------------------------------------------------------- #
# The removed env fallback chain, driven as it used to fire.
# --------------------------------------------------------------------------- #
def test_the_env_fallback_key_never_reaches_any_upstream(policy, upstream, monkeypatch):
    """HAL_GATEWAY_FALLBACK_KEY used to append a second upstream that sent this
    key as x-api-key whenever the primary answered 429. The primary answers 429
    here; the fallback upstream must receive nothing, and the key must reach no
    one."""
    primary, mirror = upstream(status=429), upstream()
    monkeypatch.setenv("HAL_GATEWAY_UPSTREAM", primary.base)
    monkeypatch.setenv("HAL_GATEWAY_FALLBACK_UPSTREAM", mirror.base)
    monkeypatch.setenv("HAL_GATEWAY_FALLBACK_KEY", FAKE_FALLBACK_KEY)
    resp = build_failover_proxy().forward(
        "/v1/messages", json.dumps(MESSAGES[1]).encode(),
        [("authorization", f"Bearer {FAKE_OAUTH}")])
    assert resp.status == 429
    resp.close()
    assert mirror.requests == []
    assert FAKE_FALLBACK_KEY not in primary.credentials()


@pytest.mark.parametrize("resets", [(600, 120), (120, 600), (600, 0)])
def test_spent_candidates_use_capacity_or_name_the_earliest_reset(resets):
    from types import SimpleNamespace
    from lib.hal.gateway_failover import FailoverProxy, GatewayTelemetry, UpstreamSpec

    calls = []
    def factory(spec):
        index = int(spec.name)
        def forward(*_args, **_kwargs):
            calls.append(index)
            seconds = resets[index]
            if seconds:
                raise spend_guard.SpendRefused(spend_guard.Decision(
                    False, "anthropic-overage", f"resets in {seconds}", 429, seconds))
            return SimpleNamespace(status=200)
        return SimpleNamespace(forward=forward)

    proxy = FailoverProxy([UpstreamSpec("0"), UpstreamSpec("1")], proxy_factory=factory,
                          telemetry=GatewayTelemetry(sink=lambda _: None))
    if 0 in resets:
        assert proxy.forward("/v1/messages", b"{}", []).status == 200
    else:
        with pytest.raises(spend_guard.SpendRefused) as caught:
            proxy.forward("/v1/messages", b"{}", [])
        assert caught.value.decision.retry_after == min(resets)
        assert caught.value.decision.reason == f"resets in {min(resets)}"
    assert calls == [0, 1]


@pytest.mark.parametrize("resets", [(-1, 600), (600, -1)])
def test_a_transport_fault_is_not_reported_as_all_accounts_spent(resets):
    from types import SimpleNamespace
    from lib.hal.gateway import GatewayTransportError
    from lib.hal.gateway_failover import FailoverProxy, GatewayTelemetry, UpstreamSpec

    def factory(spec):
        def forward(*_args, **_kwargs):
            seconds = resets[int(spec.name)]
            if seconds < 0:
                raise GatewayTransportError("fake network fault")
            raise spend_guard.SpendRefused(spend_guard.Decision(
                False, "anthropic-overage", "fake reset notice", 429, seconds))
        return SimpleNamespace(forward=forward)
    proxy = FailoverProxy([UpstreamSpec("0"), UpstreamSpec("1")], proxy_factory=factory,
                          telemetry=GatewayTelemetry(sink=lambda _: None))
    with pytest.raises(GatewayTransportError):
        proxy.forward("/v1/messages", b"{}", [])
