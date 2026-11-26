"""HAL decide's surfaces: the catalogue, ``decide()``, ``hal decide``, POST /api/decide, the spend policy.

Nothing here reaches the network: every provider is a stand-in. Each test names
the mutation that turns it red.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lib.hal import spend_guard
from lib.hal.decide import DecisionResult, catalog, decide, noul
from lib.hal.decide import http as decide_http
import lib.hal.decide.__main__ as cli

REPO = Path(__file__).resolve().parents[1]
QUESTIONS = {"urgent": noul("Is it urgent?")}


class FakeProvider:
    name = "fake"

    def __init__(self, result=None, raises=None):
        self.calls, self.result, self.raises = [], result, raises

    def decide(self, model, state, questions):
        self.calls.append((model, state, dict(questions)))
        if self.raises is not None:
            raise self.raises
        return self.result or DecisionResult(ok=True, answers={"urgent": {"type": "noul", "noul": 0.9}})


# --- the catalogue ---------------------------------------------------------------

def test_the_committed_catalogue_serves_jev_through_typesafe():
    """Mutation: a wrong endpoint, spend source, price or pin in config/decision_models.yaml."""
    model, spec = catalog.resolve("jev-latest")
    assert (model.provider, model.pinned) == ("typesafe", "jev-1.13.0")
    assert spec.endpoint == "https://api.typesafe.ai/v1/systemone"
    assert spec.spend_source == spend_guard.TYPESAFE_API_KEY and spec.billing == "prepaid"
    assert spec.usd_per_million_input == Decimal("0.042") and spec.usd_per_million_output == 0
    assert spec.key_source == "keychain:grip-typesafe"


def test_decision_models_stay_out_of_the_chat_catalogue():
    """Mutation: list a decision model in config/model_catalog.yaml, where it could be routed as chat."""
    assert "jev-" not in (REPO / "config" / "model_catalog.yaml").read_text(encoding="utf-8")


@pytest.mark.parametrize("text", [
    "not: [valid",
    "- a list",
    "providers: {p: {endpoint: 'http://plain.example.org', key_source: k, spend_source: s,"
    " billing: prepaid, usd_per_million_input_tokens: 1, usd_per_million_output_tokens: 0,"
    " max_request_bytes: 10}}",
    "providers: {}\nmodels: {m: {provider: missing}}",
    "providers: {p: {endpoint: 'https://p.example.org', key_source: k, spend_source: s,"
    " billing: prepaid, usd_per_million_input_tokens: 1, max_request_bytes: 10}}",
], ids=["unparseable", "not-a-map", "plain-http-endpoint", "unknown-provider", "missing-output-price"])
def test_an_unusable_catalogue_refuses(tmp_path, text):
    """Mutation: accept a catalogue that sends over plain http, names no provider, or
    leaves out a price (reading the missing price as zero)."""
    path = tmp_path / "decision_models.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(catalog.CatalogError):
        catalog.load(path)


# --- decide() --------------------------------------------------------------------

def test_decide_hands_the_resolved_model_to_the_provider():
    """Mutation: ``decide`` passes the alias string instead of the catalogue's model."""
    provider = FakeProvider()
    assert decide("x", QUESTIONS, provider=provider).ok
    model, state, questions = provider.calls[0]
    assert (model.id, model.pinned, state, questions) == ("jev-latest", "jev-1.13.0", "x", QUESTIONS)


@pytest.mark.parametrize("call, reason", [
    (lambda p: decide(None, QUESTIONS, provider=p), "invalid_request"),
    (lambda p: decide("x", {}, provider=p), "invalid_request"),
    (lambda p: decide("x", QUESTIONS, model="gpt-6", provider=p), "unknown_model"),
], ids=["no-state", "no-questions", "unknown-model"])
def test_decide_refuses_without_calling_the_provider(call, reason):
    """Mutation: validate after the provider is asked, or route an unknown model anyway."""
    provider = FakeProvider()
    result = call(provider)
    assert result.reason == reason and provider.calls == []


def test_an_unreadable_catalogue_is_a_reason_not_a_raise(tmp_path, monkeypatch):
    monkeypatch.setenv(catalog.CONFIG_ENV, str(tmp_path / "absent.yaml"))
    assert decide("x", QUESTIONS, provider=FakeProvider()).reason == "catalog_unavailable"


def test_a_provider_that_raises_becomes_a_reason():
    """Mutation: let the provider's exception escape ``decide`` (the contract is: never raises)."""
    result = decide("x", QUESTIONS, provider=FakeProvider(raises=RuntimeError("boom")))
    assert result.reason == "provider_error" and result.detail == "RuntimeError"


# --- hal decide ------------------------------------------------------------------

def _run(monkeypatch, argv, result=None):
    seen = []

    def fake_decide(state, questions, model):
        seen.append((state, questions, model))
        return result or DecisionResult(ok=True, answers={})

    monkeypatch.setattr(cli, "decide", fake_decide)
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli.main(argv)
    return code, seen, out.getvalue()


def test_the_cli_batches_every_noul_into_one_decision(monkeypatch):
    """Mutation: one call per --noul, or a flag that never reaches decide()."""
    code, seen, out = _run(monkeypatch, ["--state", "x", "--noul", "a", "A?", "--noul", "b", "B?",
                                         "--model", "jev-preview"])
    assert code == 0 and json.loads(out)["status"] == "answered"
    assert seen == [("x", {"a": noul("A?"), "b": noul("B?")}, "jev-preview")]


def test_the_cli_reads_a_relative_request_where_the_caller_typed_it(monkeypatch, tmp_path):
    """Mutation: resolve --request against HAL_HOME (bin/hal changes directory first)."""
    (tmp_path / "body.json").write_text(json.dumps({"state": "s", "questions": QUESTIONS}))
    monkeypatch.setenv("HAL_CALLER_CWD", str(tmp_path))
    code, seen, _ = _run(monkeypatch, ["--request", "body.json"])
    assert code == 0 and seen == [("s", QUESTIONS, "jev-latest")]


def test_the_cli_exit_codes(monkeypatch):
    """Mutation: exit 0 when no model answered, or run with no question at all."""
    unavailable = DecisionResult.unavailable("no_key")
    assert _run(monkeypatch, ["--state", "x", "--noul", "a", "A?"], unavailable)[0] == 3
    assert _run(monkeypatch, ["--state", "x"])[0] == 2


# --- POST /api/decide --------------------------------------------------------------

class Base:
    def do_POST(self):
        self.reached_base = True


def _handler(path, body=None):
    handler = decide_http.with_decide_route(Base)()
    handler.path, handler.sent, handler.reached_base = path, [], False
    handler._read_body = lambda: body if body is not None else {}
    handler._json = lambda data, code=200: handler.sent.append((data, code))
    return handler


def test_the_route_answers_on_loopback_and_leaves_other_paths_alone(monkeypatch):
    """Mutation: intercept every path, or skip the access check."""
    monkeypatch.setattr(decide_http, "enforce_gateway_access", lambda handler: True)
    monkeypatch.setattr(decide_http, "decide_response", lambda body: {"echo": body})
    handler = _handler("/api/decide?x=1", {"state": "s"})
    handler.do_POST()
    assert handler.sent == [({"echo": {"state": "s"}}, 200)] and not handler.reached_base
    other = _handler("/api/infer")
    other.do_POST()
    assert other.reached_base and other.sent == []


def test_the_route_refuses_a_caller_the_gateway_guard_refuses(monkeypatch):
    """Mutation: answer before the gateway guard has said yes."""
    discarded = []
    monkeypatch.setattr(decide_http, "enforce_gateway_access", lambda handler: False)
    monkeypatch.setattr(decide_http, "_discard_unread_body", discarded.append)
    handler = _handler("/api/decide", {"state": "s"})
    handler.do_POST()
    assert handler.sent == [] and discarded == [handler]


def test_the_route_is_fail_safe(monkeypatch):
    """Mutation: let an import or runtime failure become a 5xx, or choke on a body that is not a map."""
    import lib.hal.decide as package
    assert decide_http.decide_response(["not", "a", "map"])["reason"] == "invalid_request"
    monkeypatch.setattr(package, "decide", MagicMock(side_effect=RuntimeError("boom")))
    reply = decide_http.decide_response({"state": "s", "questions": QUESTIONS})
    assert reply["ok"] is False and reply["reason"] == "provider_error" and reply["status"] == "unavailable"


def test_hal_server_serves_the_route(monkeypatch):
    """Mutation: start() builds the handler without with_decide_route."""
    import lib.hal.server as server
    captured = {}

    class FakeServer:
        allow_reuse_address = False

        def __init__(self, address, handler):
            captured["handler"] = handler

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            captured["closed"] = True

    monkeypatch.setattr(server, "ThreadingHTTPServer", FakeServer)
    hal = server.HALServer.__new__(server.HALServer)
    for name in ("_registry", "_conversation", "_council", "_sessions", "_streamer",
                 "_agent_registry", "_fleet", "_executor", "_env_registry"):
        setattr(hal, name, MagicMock())
    hal._port = 0
    hal.start()
    assert captured["handler"].do_POST.__module__ == "lib.hal.decide.http" and captured["closed"]


# --- the committed spend policy ------------------------------------------------------

def test_the_committed_policy_meters_typesafe_under_a_ten_cent_cap(monkeypatch):
    """Mutation: TypeSafe opted in without a cap HAL enforces, or a cap that is not $0.10
    (D-1 set $1 on 2026-10-10; the operator lowered it to $0.10 on 2026-10-17)."""
    monkeypatch.delenv(spend_guard.CONFIG_ENV, raising=False)
    policy = spend_guard.load_policy()
    assert policy.opted_in(spend_guard.TYPESAFE_API_KEY)
    assert policy.cap(spend_guard.TYPESAFE_API_KEY) == Decimal("0.1")
    assert spend_guard.TYPESAFE_API_KEY in spend_guard.METERED
    assert not policy.opted_in(spend_guard.API_KEY), "the metered Anthropic key stays refused"
