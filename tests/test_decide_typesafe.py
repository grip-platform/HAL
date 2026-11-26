"""TypeSafe's decision provider: what it sends, what it records, when it refuses.

Every collaborator is injected (lib/hal/decide/typesafe.py), so nothing here can
reach the network: the transport is a scripted stand-in that records what it was
given. The spend guard is HAL's real one, under a policy each test writes, with
the ledger in this test's own HAL_HOME (tests/conftest.py). Each test names the
mutation that turns it red. The key is an obvious fake in TypeSafe's shape.
"""
from __future__ import annotations

import json
import logging
from decimal import Decimal

import pytest

from lib.hal import spend_guard
from lib.hal.decide import shield as decision_shield
from lib.hal.decide.contract import DecisionModel, ProviderSpec, choice, noul
from lib.hal.decide.typesafe import TypeSafeDecisionProvider, _Breaker, _no_proxy_transport

SOURCE = "typesafe-api-key"
SPEC = ProviderSpec(name="typesafe", endpoint="https://api.typesafe.ai/v1/systemone",
                    key_source="keychain:grip-typesafe", spend_source=SOURCE, billing="prepaid",
                    usd_per_million_input=Decimal("0.042"), usd_per_million_output=Decimal("0"),
                    max_request_bytes=524288)
MODEL = DecisionModel(id="jev-latest", provider="typesafe", pinned="jev-1.13.0")
CANARY = "apikey_FAKECANARYFAKECANARY_notarealkey_000000000000"
QUESTIONS = {"urgent": noul("Is it urgent?"),
             "team": choice("Which team?", {"billing": None, "technical": None})}
OK_BODY = {"model": "jev-1.13.0", "usage": {"input_tokens": 411, "output_tokens": 78},
           "answers": {"urgent": {"type": "noul", "noul": 0.97},
                       "team": {"type": "choice", "choice": "billing", "confidence": 0.8,
                                "probabilities": {"billing": 0.9, "technical": 0.1}}}}


def reply(status=200, body=None, headers=None):
    return status, dict(headers or {}), json.dumps(OK_BODY if body is None else body).encode()


class Wire:
    """The network's stand-in: replays scripted outcomes and records every send."""

    def __init__(self, *outcomes):
        self.outcomes, self.sent = list(outcomes), []

    def __call__(self, url, data, headers, timeout):
        self.sent.append({"url": url, "body": json.loads(data), "headers": dict(headers),
                          "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class Shield:
    """Records what it was asked to scrub, and marks the state so the send can be traced."""

    def __init__(self, raises=None):
        self.calls, self.raises = [], raises

    def __call__(self, body, *, to):
        self.calls.append(to)
        if self.raises is not None:
            raise self.raises
        return {**body, "state": "[scrubbed]"}


@pytest.fixture
def policy(tmp_path, monkeypatch):
    """Write a spend policy for this test: ``policy(opt_in=[...], caps={...})``."""
    def write(opt_in=(SOURCE,), caps=None):
        path = tmp_path / "spend-policy.json"
        path.write_text(json.dumps({"paid_opt_in": list(opt_in),
                                    "daily_caps_usd": {SOURCE: 1} if caps is None else caps}))
        monkeypatch.setenv(spend_guard.CONFIG_ENV, str(path))
    write()
    return write


def make(wire, *, shield=None, key=CANARY, breaker=None, clock=None, sleeps=None, spec=SPEC):
    return TypeSafeDecisionProvider(
        spec, transport=wire, shield=shield or Shield(), key_resolver=lambda source: key,
        sleep=(sleeps.append if sleeps is not None else (lambda seconds: None)),
        clock=clock or (lambda: 100.0), jitter=lambda: 0.0, breaker=breaker or _Breaker())


def today(name=SOURCE):
    return spend_guard._today_entry(name, spend_guard.today_utc().isoformat())


# --- an answer -------------------------------------------------------------------

def test_an_answer_is_checked_priced_and_recorded(policy):
    """Mutation: skip ``record_spend`` (the cap would then bound nothing)."""
    result = make(Wire(reply())).decide(MODEL, "A synthetic ticket.", QUESTIONS)
    assert result.ok and result.answers["urgent"]["noul"] == 0.97
    assert result.model == "jev-1.13.0" and not result.drifted and result.attempts == 1
    assert result.usage == {"input_tokens": 411, "output_tokens": 78}
    assert result.cost_usd == "0.00001726"
    entry = today()
    assert entry["spent_usd"] == "0.00001726" and entry["calls"] == 1 and entry["input_tokens"] == 411


def test_a_new_version_answering_is_flagged_as_drift(policy):
    """Mutation: ``drifted`` ignores the pinned version."""
    result = make(Wire(reply(body={**OK_BODY, "model": "jev-1.14.0"}))).decide(MODEL, "x", QUESTIONS)
    assert result.ok and result.drifted


def test_what_is_sent_is_the_shielded_body_and_the_key_is_only_in_the_header(policy):
    """Mutation: send the body as given instead of the shield's rewrite."""
    wire, shield = Wire(reply()), Shield()
    make(wire, shield=shield).decide(MODEL, {"from": "someone"}, QUESTIONS)
    sent = wire.sent[0]
    assert shield.calls == [SPEC.endpoint], "the shield is told where the body is going"
    assert sent["url"] == SPEC.endpoint and sent["body"]["state"] == "[scrubbed]"
    assert sent["body"]["questions"] == QUESTIONS and sent["body"]["model"] == "jev-latest"
    assert sent["headers"]["Authorization"] == f"Bearer {CANARY}"
    assert CANARY not in json.dumps(sent["body"])


def test_the_default_transport_disables_every_proxy(monkeypatch):
    """Mutation: build the opener without ``ProxyHandler({})`` (a TLS-intercepting proxy would see the key)."""
    import urllib.request
    seen = []

    class Response:
        status, headers = 200, {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"{}"

    class Opener:
        def open(self, request, timeout):
            seen.append((request.full_url, timeout))
            return Response()

    def build_opener(*handlers):
        seen.extend(handlers)
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    assert _no_proxy_transport(SPEC.endpoint, b"{}", {}, 10.0)[0] == 200
    proxies = [h for h in seen if isinstance(h, urllib.request.ProxyHandler)]
    assert len(proxies) == 1 and proxies[0].proxies == {}


# --- refusals before anything is sent --------------------------------------------

def test_no_send_without_the_spend_guards_permission(policy):
    """Mutation: skip ``decide_within_cap`` before the send."""
    policy(opt_in=())
    wire = Wire(reply())
    result = make(wire).decide(MODEL, "x", QUESTIONS)
    assert result.reason == "spend_refused" and "paid_opt_in" in result.detail and wire.sent == []


def test_no_send_once_the_daily_cap_is_reached(policy):
    """Mutation: check the opt-in but not today's spend against the cap."""
    policy(caps={SOURCE: "0.00001"})
    spend_guard.record_spend(SOURCE, "0.00001")
    wire = Wire(reply())
    result = make(wire).decide(MODEL, "x", QUESTIONS)
    assert result.reason == "spend_refused" and "cap" in result.detail and wire.sent == []


def test_no_send_without_a_key(policy):
    """Mutation: send with an empty Authorization header when no key resolves."""
    wire = Wire(reply())
    result = make(wire, key=None).decide(MODEL, "x", QUESTIONS)
    assert result.reason == "no_key" and wire.sent == []


@pytest.mark.parametrize("raised, reason", [
    (decision_shield.ShieldUnavailable("no shield"), "shield_unavailable"),
    (OSError("a key in the decision body holds text the shield would change"), "shield_refused"),
])
def test_no_send_when_the_shield_cannot_scrub(policy, raised, reason):
    """Mutation: fall back to sending the unscrubbed body when the shield fails."""
    wire = Wire(reply())
    result = make(wire, shield=Shield(raises=raised)).decide(MODEL, "x", QUESTIONS)
    assert result.reason == reason and wire.sent == []


def test_no_send_over_the_byte_ceiling(policy):
    """Mutation: drop the size check."""
    wire = Wire(reply())
    tight = ProviderSpec(**{**SPEC.__dict__, "max_request_bytes": 100})
    result = make(wire, spec=tight).decide(MODEL, "x" * 200, QUESTIONS)
    assert result.reason == "over_budget" and wire.sent == []


# --- retries, the way TypeSafe's SDK retries ---------------------------------------

@pytest.mark.parametrize("outcomes, reason, attempts, sleeps", [
    ([reply(429, {}, {"retry-after": "1.5"}), reply()], "", 2, [1.5]),
    ([reply(529, {})] * 3, "overloaded", 3, [0.5, 1.0]),
    ([reply(503, {}), reply(502, {}), reply()], "", 3, [0.5, 1.0]),
    ([OSError("connection refused"), reply()], "", 2, [0.5]),
    ([TimeoutError("timed out")] * 3, "timeout", 3, [0.5, 1.0]),
    ([reply(400, {"detail": "Noul question must have criteria or instructions: q"})],
     "invalid_request", 1, []),
    ([reply(400, {"detail": {"error_type": "max_tokens_exceeded"}})], "over_budget", 1, []),
    ([reply(401, {"detail": "unauthorised"})], "auth_failed", 1, []),
], ids=["429-then-ok", "529-gives-up", "5xx-then-ok", "connection-then-ok", "timeouts",
        "400-no-retry", "over-budget", "401-no-retry"])
def test_retries_follow_the_sdk_policy(policy, outcomes, reason, attempts, sleeps):
    """Mutation: retry a 4xx, ignore retry-after, or allow a third retry."""
    waited = []
    result = make(Wire(*outcomes), sleeps=waited).decide(MODEL, "x", QUESTIONS)
    assert (result.reason, result.attempts, waited) == (reason, attempts, sleeps)
    assert result.ok is (reason == "")


def test_retries_stop_at_the_time_budget(policy):
    """Mutation: retry regardless of how long the decision has already taken."""
    ticks = iter([0.0, 0.0, 0.0, 29.8])  # the breaker, the start, attempt 1's budget, then 29.8 s
    waited = []
    result = make(Wire(reply(429, {}), reply()), sleeps=waited,
                  clock=lambda: next(ticks, 29.8)).decide(MODEL, "x", QUESTIONS)
    assert result.reason == "rate_limited" and result.attempts == 1 and waited == []


@pytest.mark.parametrize("ticks, timeouts, waited", [
    # the breaker, the start, then per attempt: its budget read and the elapsed time after it
    ([0.0, 0.0, 0.0, 15.0, 15.5, 27.0, 28.0, 30.0], [10.0, 10.0, 2.0], [0.5, 1.0]),
    ([0.0, 0.0, 0.0, 15.0, 15.5, 28.6], [10.0, 10.0], [0.5]),
], ids=["third-attempt-cut-to-what-is-left", "no-third-attempt-without-a-second-left"])
def test_the_whole_decision_fits_the_thirty_second_budget(policy, ticks, timeouts, waited):
    """Mutation: give every attempt its full 10 s, or start a retry with no budget left for it.

    AGORA's concern on HAL#724: the budget was checked before the sleep but not for
    the attempt after it, so a decision could run to about 40 s."""
    clock, slept, wire = iter(ticks), [], Wire(*([reply(503, {})] * 3))
    result = make(wire, sleeps=slept, clock=lambda: next(clock, ticks[-1])).decide(MODEL, "x", QUESTIONS)
    assert [send["timeout"] for send in wire.sent] == timeouts
    assert slept == waited and result.reason == "server_error"


@pytest.mark.parametrize("outcomes, reason", [
    ([reply(503, {})] * 3, "server_error"),
    ([reply(400, {"detail": {"error_type": "api_usage_error", "message": "bad field"}})],
     "invalid_request"),
], ids=["5xx-after-retries", "400"])
def test_an_error_reply_records_nothing_and_leaves_the_source_open(policy, outcomes, reason):
    """Mutation: record an error reply as spend, or as unknown spend (which shuts the source
    for the rest of the day after one transient overload).

    AGORA's concern on HAL#725: docs/decide.md says an error reply is not recorded."""
    failed = make(Wire(*outcomes)).decide(MODEL, "x", QUESTIONS)
    entry = today() or {}
    assert failed.reason == reason
    assert entry.get("calls", 0) == 0 and entry.get("unknown", 0) == 0
    answered = make(Wire(reply())).decide(MODEL, "x", QUESTIONS)
    assert answered.ok and today()["calls"] == 1 and today()["unknown"] == 0


# --- the circuit breaker ------------------------------------------------------------

def test_three_failed_decisions_rest_the_provider_until_the_cooldown(policy):
    """Mutation: the breaker never opens, or never closes again."""
    now = [100.0]
    breaker, wire = _Breaker(), Wire(*([reply(503, {})] * 9), reply())
    provider = make(wire, breaker=breaker, clock=lambda: now[0])
    failures = [provider.decide(MODEL, "x", QUESTIONS).reason for _ in range(3)]
    assert failures == ["server_error"] * 3 and len(wire.sent) == 9
    assert provider.decide(MODEL, "x", QUESTIONS).reason == "circuit_open" and len(wire.sent) == 9
    now[0] += 61
    assert provider.decide(MODEL, "x", QUESTIONS).ok and len(wire.sent) == 10


def test_a_success_resets_the_breakers_count(policy):
    """Mutation: count failures without resetting on a success."""
    breaker = _Breaker()
    wire = Wire(*([reply(503, {})] * 6), reply(), *([reply(503, {})] * 3), reply())
    provider = make(wire, breaker=breaker)
    reasons = [provider.decide(MODEL, "x", QUESTIONS).reason for _ in range(5)]
    assert reasons == ["server_error", "server_error", "", "server_error", ""]


# --- what comes back -----------------------------------------------------------------

def test_a_reply_that_does_not_answer_the_questions_is_not_a_decision(policy):
    """Mutation: pass the vendor's answers through unchecked."""
    body = {**OK_BODY, "answers": {"urgent": OK_BODY["answers"]["urgent"]}}
    result = make(Wire(reply(body=body))).decide(MODEL, "x", QUESTIONS)
    assert result.reason == "bad_response" and result.answers == {}
    assert today()["calls"] == 1, "a reply that was billed is still recorded"


def test_a_reply_without_usage_closes_the_source_for_the_day(policy):
    """Mutation: treat an unpriceable reply as free."""
    body = {key: value for key, value in OK_BODY.items() if key != "usage"}
    wire = Wire(reply(body=body), reply())
    provider = make(wire)
    assert provider.decide(MODEL, "x", QUESTIONS).ok
    assert provider.decide(MODEL, "x", QUESTIONS).reason == "spend_refused" and len(wire.sent) == 1


# --- the key never leaves ------------------------------------------------------------

def test_the_key_never_reaches_a_result_a_log_or_a_file(policy, caplog, tmp_path):
    """Mutation: put the transport's error message, or the vendor's raw text, in the result."""
    class Leaky:
        def __call__(self, url, data, headers, timeout):
            raise OSError(f"proxy refused {headers}")

    caplog.set_level(logging.DEBUG)
    provider = make(Leaky())
    outcomes = [provider.decide(MODEL, "x", QUESTIONS),
                make(Wire(reply(401, {"detail": f"unknown key {CANARY}"}))).decide(MODEL, "x", QUESTIONS)]
    rendered = json.dumps([r.to_dict() for r in outcomes]) + "".join(repr(r) for r in outcomes)
    assert CANARY not in rendered and CANARY not in repr(provider)
    assert outcomes[1].detail == "unknown key [key]"
    assert CANARY not in caplog.text
    written = "".join(p.read_text(errors="ignore") for p in tmp_path.rglob("*") if p.is_file())
    assert CANARY not in written
