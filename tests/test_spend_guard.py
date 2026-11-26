"""The spend guard itself (lib/hal/spend_guard.py): policy, decisions, overage.

InfiniteMachine v2, Step 5h. Every test here runs under a STRICT policy and a
frozen clock: the suite-wide permissive policy from tests/conftest.py is
replaced, because these tests are what must go red when the guard stops
refusing. Credentials are obvious fakes, shaped only by their public prefixes.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from email.message import Message
from pathlib import Path

import pytest

from lib.hal import spend_guard
from lib.hal.spend_guard import API_KEY, OVERAGE, TYPESAFE_API_KEY

REPO = Path(__file__).resolve().parents[1]
BEFORE = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
LAST_SECOND = datetime(2026, 10, 10, 23, 59, 59, tzinfo=timezone.utc)
AFTER = datetime(2026, 10, 11, 0, 0, 1, tzinfo=timezone.utc)
FAKE_KEY = "sk-ant-api03-fake-test-key"
FAKE_OAUTH = "sk-ant-oat01-fake-test-token-a"
OTHER_OAUTH = "sk-ant-oat01-fake-test-token-b"
# Claude Code's session ingress; the payload is base64url for {"fake":true}.
FAKE_SESSION = "sk-ant-si-fake-header.eyJmYWtlIjp0cnVlfQ.fake-signature"


@pytest.fixture
def policy_file(tmp_path, monkeypatch):
    """Point the guard at a policy the test writes; strict until told otherwise."""
    path = tmp_path / "spend_guard.json"

    def write(opt_in=None, raw: str = None) -> Path:
        text = raw if raw is not None else json.dumps({"paid_opt_in": opt_in or []})
        path.write_text(text, encoding="utf-8")
        return path

    write()
    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(path))
    return write


@pytest.fixture
def clock(monkeypatch):
    """Freeze the guard's clock; returns a setter."""
    def at(moment: datetime) -> None:
        monkeypatch.setattr(spend_guard, "_clock", lambda: moment)
    at(BEFORE)
    return at


def test_daily_ledger_lock_timeout_refuses_without_waiting_forever(tmp_path, monkeypatch):
    """Mutation: call the unbounded ``lock`` instead of ``lock_within``.
    Survives by forcing acquisition failure and asserting the write is refused,
    remembered as unrecorded, and returns comfortably inside the lock bound."""
    monkeypatch.setenv("HAL_HOME", str(tmp_path))
    monkeypatch.setattr(spend_guard, "lock_within", lambda handle, timeout: False)
    day = "2026-11-26"
    started = time.monotonic()
    entry = spend_guard._update_entry(TYPESAFE_API_KEY, day, lambda row: None)
    elapsed = time.monotonic() - started
    assert entry is None
    assert (TYPESAFE_API_KEY, day) in spend_guard._UNRECORDED
    assert elapsed < 3.0


# --------------------------------------------------------------------------- #
# The committed policy.
# --------------------------------------------------------------------------- #
def test_the_committed_policy_opts_in_overage_typesafe_and_nothing_else(monkeypatch):
    """Overage bills only where the account's own extra usage is on; the metered
    Anthropic key stays refused (the contract's "never a silent paid key").
    TypeSafe's prepaid decision key is opted in by name, under a $0.10 daily cap the
    guard enforces: the operator's decision of 2026-10-17 (lowered from $1, set
    2026-10-10), never a silent one."""
    monkeypatch.delenv(spend_guard.CONFIG_ENV, raising=False)
    policy = spend_guard.load_policy()
    assert spend_guard.config_path() == REPO / "config" / "spend_guard.json"
    assert policy.paid_opt_in == frozenset({OVERAGE, TYPESAFE_API_KEY})
    assert policy.cap(TYPESAFE_API_KEY) == Decimal("0.1")
    assert TYPESAFE_API_KEY in spend_guard.METERED
    assert not hasattr(policy, "no_spend_until")


def test_the_committed_policy_refuses_a_paid_key_at_every_date(monkeypatch, clock):
    monkeypatch.delenv(spend_guard.CONFIG_ENV, raising=False)
    for moment in (BEFORE, LAST_SECOND, AFTER):
        clock(moment)
        assert spend_guard.decide_paid(API_KEY).allowed is False, moment
        assert spend_guard.decide_paid(OVERAGE).allowed is True, moment


# --------------------------------------------------------------------------- #
# The opt-in. There is no date rule (removed 2026-10-06).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("legacy", ["2026-10-11", "2026-10-20", "2099-01-01", "not a date", 7])
def test_a_legacy_no_spend_date_never_refuses_an_opted_in_source(policy_file, clock, legacy):
    """Regression anchor, 2026-10-06: the operator opted overage in and moved the
    date to 2026-10-20, and every request on his paid-for extra usage was refused."""
    policy_file(raw=json.dumps({"no_spend_until": legacy, "paid_opt_in": [OVERAGE]}))
    clock(BEFORE)
    verdict = spend_guard.decide_paid(OVERAGE)
    assert verdict.allowed is True, verdict.reason
    assert "before" not in verdict.reason


def test_a_source_not_opted_in_is_refused(policy_file, clock):
    policy_file(opt_in=[OVERAGE])
    verdict = spend_guard.decide_paid(API_KEY)
    assert verdict.allowed is False
    assert "paid_opt_in" in verdict.reason


def test_an_opted_in_source_is_allowed_and_only_that_one(policy_file, clock):
    policy_file(opt_in=[API_KEY])
    assert spend_guard.decide_paid(API_KEY).allowed is True
    assert spend_guard.decide_paid(OVERAGE).allowed is False   # only what was named


def test_the_wildcard_allows_every_paid_source(policy_file, clock):
    policy_file(opt_in=["*"])
    assert spend_guard.decide_paid(API_KEY).allowed is True
    assert spend_guard.decide_paid(OVERAGE).allowed is True


def test_the_clock_no_longer_decides(policy_file):
    policy_file(opt_in=[API_KEY])
    for moment in (datetime(2026, 10, 10, 23, 59), datetime(2026, 10, 11, 0, 0),
                   datetime(2000, 1, 1)):
        assert spend_guard.decide_paid(API_KEY, now=moment).allowed is True, moment


def test_require_paid_raises_with_the_decision(policy_file, clock):
    with pytest.raises(spend_guard.SpendRefused) as caught:
        spend_guard.require_paid(API_KEY)
    assert caught.value.decision.allowed is False
    assert caught.value.decision.status == 403


# --------------------------------------------------------------------------- #
# Fail-closed: every defect in the policy refuses every paid source.
# --------------------------------------------------------------------------- #
BROKEN_POLICIES = {
    "empty file": "",
    "not json": "{paid_opt_in: [*]",
    "a list": '["*"]',
    "no opt-in list": '{}',
    "only a legacy date": '{"no_spend_until": "2000-01-01"}',
    "an opt-in string": '{"paid_opt_in": "*"}',
    "a blank opt-in name": '{"paid_opt_in": ["*", " "]}',
    "a numeric opt-in name": '{"paid_opt_in": ["*", 1]}',
}


@pytest.mark.parametrize("raw", BROKEN_POLICIES.values(), ids=BROKEN_POLICIES.keys())
def test_a_broken_policy_refuses_paid(policy_file, clock, raw):
    """Each broken file would allow everything if it were read leniently (the
    wildcard). The guard must refuse instead."""
    policy_file(raw=raw)
    clock(AFTER)
    verdict = spend_guard.decide_paid(API_KEY)
    assert verdict.allowed is False
    assert "policy cannot be used" in verdict.reason


def test_a_missing_policy_refuses_paid(tmp_path, monkeypatch, clock):
    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(tmp_path / "absent.json"))
    clock(AFTER)
    assert spend_guard.decide_paid(API_KEY).allowed is False


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX file permissions require a non-root POSIX process",
)
def test_an_unreadable_policy_refuses_paid(policy_file, clock):
    path = policy_file(opt_in=["*"])
    path.chmod(0)
    try:
        clock(AFTER)
        assert spend_guard.decide_paid(API_KEY).allowed is False
    finally:
        path.chmod(0o600)


def test_any_error_inside_the_guard_refuses_paid(policy_file, clock, monkeypatch):
    policy_file(opt_in=["*"])

    def broken(*_args, **_kwargs):
        raise RuntimeError("a bug")

    monkeypatch.setattr(spend_guard, "load_policy", broken)
    clock(AFTER)
    verdict = spend_guard.decide_paid(API_KEY)
    assert verdict.allowed is False
    assert "RuntimeError" in verdict.reason


# --------------------------------------------------------------------------- #
# Reading a request's credential.
# --------------------------------------------------------------------------- #
def _message(pairs) -> Message:
    msg = Message()
    for name, value in pairs:
        msg[name] = value
    return msg


@pytest.mark.parametrize("headers, expected", [
    ([("x-api-key", FAKE_KEY)], "paid"),
    ([("X-Api-Key", "any-value-at-all")], "paid"),            # the scheme, not the value
    ([("authorization", f"Bearer {FAKE_KEY}")], "paid"),
    ([("Authorization", "bearer sk-ant-admin01-fake")], "paid"),
    ([("authorization", f"Bearer {FAKE_OAUTH}")], "subscription"),
    ([("authorization", "Bearer wire-session-token")], ""),   # not an Anthropic key
    ([("authorization", f"Bearer {FAKE_SESSION}")], ""),      # Remote Control's session token
    ([("authorization", "Bearer sk-ant-si-not-a-jwt")], "paid"),   # malformed: fails closed
    ([("authorization", "Bearer sk-ant-si-head..sig")], "paid"),   # no payload: Claude Code rejects it too
    ([("authorization", "Bearer sk-ant-si-head.%%%.sig")], "paid"),   # undecodable payload (AGORA on HAL#660)
    ([("authorization", "Bearer sk-ant-si-head.WzFd.sig")], "paid"),  # payload decodes to [1], not an object
    ([("authorization", "Bearer sk-ant-si-head.fQ.sig")], "paid"),    # payload decodes to "}", not JSON
    ([("authorization", "Bearer sk-ant-si-head.abcde.sig")], "paid"), # binascii.Error, a ValueError: caught
    # Any claim names will do: Claude Code's later session_id check is not
    # mirrored, so a renamed claim cannot refuse the bridge again.
    ([("authorization", "Bearer sk-ant-si-head.eyJzdWIiOiJ4In0.sig")], ""),
    ([("authorization", "Basic abc")], ""),
    ([], ""),
])
def test_the_credential_kind_follows_the_billing_scheme(headers, expected):
    for shape in (headers, dict(headers), _message(headers)):
        kind, _account = spend_guard.anthropic_credential(shape)
        assert kind == expected, (shape, kind)


def test_an_oauth_account_is_named_by_a_digest_never_the_token():
    kind, account = spend_guard.anthropic_credential([("authorization", f"Bearer {FAKE_OAUTH}")])
    assert kind == "subscription"
    assert len(account) == 16 and int(account, 16) >= 0
    assert FAKE_OAUTH not in account
    other = spend_guard.anthropic_credential([("authorization", f"Bearer {OTHER_OAUTH}")])[1]
    assert other != account


@pytest.mark.parametrize("method, path, expected", [
    ("POST", "/v1/messages", True),
    ("POST", "/v1/messages?beta=true", True),
    ("post", "/v1/complete", True),
    ("POST", "/v1/messages/batches", True),
    ("POST", "https://api.anthropic.com/v1/messages?beta=true", True),
    ("POST", "/v1/messages/count_tokens", False),
    ("GET", "/v1/messages", False),
    ("POST", "/v1/code/sessions", False),
    ("POST", "/api/oauth/token", False),
])
def test_billable_requests(method, path, expected):
    assert spend_guard.billable(method, path) is expected


def test_a_paid_request_is_refused_with_403_without_opt_in(policy_file, clock):
    policy_file(opt_in=[OVERAGE])
    refusal = spend_guard.check_anthropic_request(
        "POST", "/v1/messages", [("x-api-key", FAKE_KEY)])
    assert refusal is not None and refusal.status == 403
    assert refusal.source == API_KEY


def test_a_paid_request_is_sent_with_opt_in(policy_file, clock):
    policy_file(opt_in=[API_KEY])
    clock(AFTER)
    assert spend_guard.check_anthropic_request(
        "POST", "/v1/messages", [("x-api-key", FAKE_KEY)]) is None


def test_a_subscription_request_and_a_bare_request_are_sent(policy_file, clock):
    oauth = [("authorization", f"Bearer {FAKE_OAUTH}")]
    assert spend_guard.check_anthropic_request("POST", "/v1/messages", oauth) is None
    assert spend_guard.check_anthropic_request("POST", "/v1/messages", []) is None


def test_headers_that_cannot_be_read_are_treated_as_paid(policy_file, clock):
    class Exploding:
        def items(self):
            raise RuntimeError("unreadable")

    refusal = spend_guard.check_anthropic_request("POST", "/v1/messages", Exploding())
    assert refusal is not None and refusal.source == API_KEY


# --------------------------------------------------------------------------- #
# Overage (f22): a 2xx that shows it marks the account spent.
# --------------------------------------------------------------------------- #
RESET = int(BEFORE.timestamp()) + 3600


@pytest.mark.parametrize("headers, expected", [
    ({"anthropic-ratelimit-unified-status": "rejected",
      "anthropic-ratelimit-unified-overage-status": "allowed"}, True),
    ({"anthropic-ratelimit-unified-status": "rejected",
      "anthropic-ratelimit-unified-overage-status": "allowed_warning"}, True),
    ({"anthropic-ratelimit-unified-overage-in-use": "true"}, True),
    ({"Anthropic-Ratelimit-Unified-Overage-In-Use": "TRUE"}, True),
    ({"anthropic-ratelimit-unified-status": "allowed",
      "anthropic-ratelimit-unified-overage-status": "allowed"}, False),
    ({"anthropic-ratelimit-unified-status": "rejected",
      "anthropic-ratelimit-unified-overage-status": "rejected"}, False),
    ({"anthropic-ratelimit-unified-overage-in-use": "false"}, False),
    ({}, False),
])
def test_overage_in_use_matches_claude_codes_own_reading(headers, expected):
    assert spend_guard.overage_in_use(headers) is expected


def _overage(reset=RESET) -> dict:
    out = {"anthropic-ratelimit-unified-status": "rejected",
           "anthropic-ratelimit-unified-overage-status": "allowed"}
    if reset is not None:
        out["anthropic-ratelimit-unified-reset"] = str(reset)
    return out


def _oauth(token=FAKE_OAUTH):
    return [("authorization", f"Bearer {token}")]


def test_a_2xx_with_overage_spends_the_account_until_its_reset(policy_file, clock):
    until = spend_guard.observe_anthropic_response(200, _overage(), _oauth())
    assert until == RESET
    refusal = spend_guard.check_anthropic_request("POST", "/v1/messages", _oauth())
    assert refusal is not None
    assert (refusal.status, refusal.source) == (429, OVERAGE)
    assert refusal.retry_after == 3600
    assert "overage" in refusal.reason


def test_a_spent_account_still_reaches_what_does_not_bill(policy_file, clock):
    spend_guard.observe_anthropic_response(200, _overage(), _oauth())
    for method, path in (("POST", "/v1/messages/count_tokens"), ("GET", "/v1/models"),
                         ("POST", "/v1/code/sessions"), ("POST", "/api/oauth/token")):
        assert spend_guard.check_anthropic_request(method, path, _oauth()) is None, path


def test_another_account_is_not_spent(policy_file, clock):
    spend_guard.observe_anthropic_response(200, _overage(), _oauth())
    assert spend_guard.check_anthropic_request(
        "POST", "/v1/messages", _oauth(OTHER_OAUTH)) is None


def test_the_account_comes_back_at_its_reset(policy_file, clock):
    spend_guard.observe_anthropic_response(200, _overage(), _oauth())
    clock(datetime.fromtimestamp(RESET + 1, timezone.utc))
    assert spend_guard.check_anthropic_request("POST", "/v1/messages", _oauth()) is None


def test_with_no_usable_reset_the_account_is_spent_for_five_hours(policy_file, clock):
    for reset in (None, "soon", int(BEFORE.timestamp()) - 10):
        spend_guard._SPENT.clear()
        until = spend_guard.observe_anthropic_response(200, _overage(reset), _oauth())
        assert until == BEFORE.timestamp() + spend_guard.DEFAULT_SPENT_SECONDS, reset


def test_overage_needs_its_own_opt_in(policy_file, clock):
    clock(AFTER)
    account = spend_guard.anthropic_credential(_oauth())[1]
    spend_guard.mark_spent(account, AFTER.timestamp() + 60)
    policy_file(opt_in=[API_KEY])          # control: a different paid source
    assert spend_guard.check_anthropic_request("POST", "/v1/messages", _oauth()) is not None
    policy_file(opt_in=[OVERAGE])
    assert spend_guard.check_anthropic_request("POST", "/v1/messages", _oauth()) is None


@pytest.mark.parametrize("status", [429, 400, 500, "not-a-status"])
def test_only_a_2xx_marks_spent(policy_file, clock, status):
    assert spend_guard.observe_anthropic_response(status, _overage(), _oauth()) is None
    assert spend_guard.check_anthropic_request("POST", "/v1/messages", _oauth()) is None


def test_an_api_key_response_marks_nothing(policy_file, clock):
    assert spend_guard.observe_anthropic_response(
        200, _overage(), [("x-api-key", FAKE_KEY)]) is None


def test_observing_never_raises(policy_file, clock):
    assert spend_guard.observe_anthropic_response(200, object(), object()) is None


def test_the_ledger_holds_a_digest_and_never_the_token(policy_file, clock):
    spend_guard.observe_anthropic_response(200, _overage(), _oauth())
    ledger = spend_guard.ledger_path()
    text = ledger.read_text(encoding="utf-8")
    assert FAKE_OAUTH not in text and "sk-ant" not in text
    assert json.loads(text) == {spend_guard.anthropic_credential(_oauth())[1]: float(RESET)}


def test_another_process_sees_the_mark_through_the_ledger(policy_file, clock):
    spend_guard.observe_anthropic_response(200, _overage(), _oauth())
    spend_guard._SPENT.clear()            # what a second process starts with
    assert spend_guard.check_anthropic_request("POST", "/v1/messages", _oauth()) is not None


def test_a_corrupt_ledger_never_refuses_the_subscription(policy_file, clock):
    ledger = spend_guard.ledger_path()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("{not json", encoding="utf-8")
    assert spend_guard.check_anthropic_request("POST", "/v1/messages", _oauth()) is None


def test_a_mark_survives_a_ledger_that_cannot_be_written(policy_file, clock, monkeypatch):
    blocked = spend_guard.ledger_path().parent
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("a file where the state directory should be", encoding="utf-8")
    spend_guard.observe_anthropic_response(200, _overage(), _oauth())
    assert spend_guard.check_anthropic_request("POST", "/v1/messages", _oauth()) is not None
