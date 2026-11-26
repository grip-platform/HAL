"""account_switch: a spent account's turn moves to a pool account with capacity (Step 5e).

Origin, 2026-10-06: the spend guard (Step 5h) shipped before any account switch, so
when one account's weekly limit was spent its turns were refused while a second
account in the pool still had half its week left. These tests pin the order the
operator asked for: another account's subscription first, paid extra usage only as
the fallback, and never a turn moved off an account-bound route.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from lib.hal import account_switch, spend_guard

NOW = datetime(2026, 10, 6, 4, 0, tzinfo=timezone.utc)
LATER = NOW.timestamp() + 3600
OWN = "sk-ant-oat01-own-fake-token"
ALICE = "sk-ant-oat01-alice-fake-token"
BOB = "sk-ant-oat01-bob-fake-token"
POOL = [("alice", ALICE), ("bob", BOB)]
LIMIT_429 = [("anthropic-ratelimit-unified-status", "rejected"),
             ("anthropic-ratelimit-unified-reset", str(int(LATER)))]


def _auth(token: str = OWN):
    return [("authorization", f"Bearer {token}")]


def _key(token: str) -> str:
    return spend_guard.account_key(token)


@pytest.fixture(autouse=True)
def policy(monkeypatch, tmp_path):
    """Nothing opted in, no marks; returns a setter for the opt-in list."""
    account_switch.forget_exhausted()
    monkeypatch.setattr(account_switch, "_ORGS", {_key(t): _key(t) for t in (OWN, ALICE, BOB)})
    monkeypatch.setattr(account_switch, "_ORG_RETRY", {})
    monkeypatch.setattr(spend_guard, "_SPENT", {})
    path = tmp_path / "spend_guard.json"
    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(path))

    def opt_in(*names: str) -> None:
        path.write_text(json.dumps({"paid_opt_in": list(names)}), encoding="utf-8")

    opt_in()
    yield opt_in
    account_switch.forget_exhausted()


def _choose(token=OWN, pool=POOL, method="POST", target="/v1/messages?beta=true", now=NOW):
    return account_switch.choose(method, target, _auth(token), pool=lambda: pool, now=now)


def _spend(*tokens: str) -> None:
    for token in tokens:
        spend_guard.mark_spent(_key(token), LATER)


def _exhaust(*tokens: str) -> None:
    for token in tokens:
        assert account_switch.observe(429, LIMIT_429, _auth(token), now=NOW) == LATER


# --------------------------------------------------------------------------- #
# Subscription first.
# --------------------------------------------------------------------------- #
def test_an_account_with_capacity_keeps_its_own_turn():
    assert _choose() is None


def test_a_spent_account_moves_to_the_first_pool_account_with_capacity():
    _spend(OWN)
    assert _choose() == ("alice", ALICE, account_switch.SUBSCRIPTION_RANK)


def test_pool_order_is_honoured_and_spent_members_are_skipped():
    _spend(OWN, ALICE)
    assert _choose() == ("bob", BOB, account_switch.SUBSCRIPTION_RANK)


def test_an_exhausted_account_moves_too():
    _exhaust(OWN)
    assert _choose() == ("alice", ALICE, account_switch.SUBSCRIPTION_RANK)


def test_the_own_token_is_never_its_own_replacement():
    _spend(OWN)
    assert _choose(pool=[("own", OWN), ("alice", ALICE)]) == ("alice", ALICE, "subscription")


def test_subscription_elsewhere_beats_the_own_accounts_overage(policy):
    policy(spend_guard.OVERAGE)
    _spend(OWN)
    assert _choose() == ("alice", ALICE, account_switch.SUBSCRIPTION_RANK)


# --------------------------------------------------------------------------- #
# Overage is the fallback, and only when it is opted in.
# --------------------------------------------------------------------------- #
def test_without_overage_opted_in_nothing_spent_is_chosen():
    _spend(OWN, ALICE, BOB)
    assert _choose() is None


def test_with_overage_opted_in_the_own_account_keeps_its_overage(policy):
    policy(spend_guard.OVERAGE)
    _spend(OWN, ALICE, BOB)
    assert _choose() is None


def test_an_exhausted_account_moves_to_a_pool_account_whose_overage_answers(policy):
    policy(spend_guard.OVERAGE)
    _exhaust(OWN, ALICE)
    _spend(BOB)
    assert _choose() == ("bob", BOB, account_switch.OVERAGE_RANK)


def test_everything_exhausted_leaves_the_turn_to_the_guard(policy):
    policy(spend_guard.OVERAGE)
    _exhaust(OWN, ALICE, BOB)
    assert _choose() is None


# --------------------------------------------------------------------------- #
# What is never switched.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method,target", [
    ("POST", "/v1/messages/batches"), ("POST", "/v1/messages/count_tokens"),
    ("GET", "/v1/messages"), ("POST", "/v1/code/sessions"), ("POST", "/api/oauth/token"),
])
def test_only_a_messages_turn_is_ever_switched(method, target):
    _spend(OWN)
    assert _choose(method=method, target=target) is None


def test_an_absolute_form_messages_target_is_a_turn():
    _spend(OWN)
    assert _choose(target="https://api.anthropic.com/v1/messages")[0] == "alice"


@pytest.mark.parametrize("headers", [
    [("x-api-key", "sk-ant-api03-fake")], [("authorization", "Bearer sk-ant-api03-fake")], [],
])
def test_a_metered_or_absent_credential_is_never_switched(headers):
    _spend(OWN)
    assert account_switch.choose("POST", "/v1/messages", headers,
                                 pool=lambda: POOL, now=NOW) is None


def test_a_broken_pool_never_breaks_the_relay():
    _spend(OWN)

    def broken():
        raise RuntimeError("a bug")

    assert account_switch.choose("POST", "/v1/messages", _auth(), pool=broken, now=NOW) is None


# --------------------------------------------------------------------------- #
# The brake.
# --------------------------------------------------------------------------- #
def test_the_off_marker_stops_switching_and_on_resumes_it(capsys):
    _spend(OWN)
    assert account_switch.main(["off"]) == 0
    assert "off" in capsys.readouterr().out
    assert _choose() is None
    assert account_switch.main(["on"]) == 0
    assert _choose() == ("alice", ALICE, account_switch.SUBSCRIPTION_RANK)


def test_a_marker_that_cannot_be_checked_holds_the_brake(monkeypatch):
    real_stat = os.stat

    def denied(path, *args, **kwargs):
        if str(path) == str(account_switch.off_marker()):
            raise PermissionError("denied")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(account_switch.os, "stat", denied)
    assert account_switch.switching_on() is False


def test_the_cli_refuses_an_unknown_command():
    assert account_switch.main(["sideways"]) == 2


# --------------------------------------------------------------------------- #
# Observing a limit.
# --------------------------------------------------------------------------- #
def test_observe_marks_only_a_limit_429_on_a_subscription():
    assert account_switch.observe(200, LIMIT_429, _auth(), now=NOW) is None
    assert account_switch.observe(429, [("retry-after", "5")], _auth(), now=NOW) is None
    assert account_switch.observe(429, LIMIT_429, [("x-api-key", "k")], now=NOW) is None
    assert account_switch.exhausted_until(_key(OWN), now=NOW) is None
    assert account_switch.observe(429, LIMIT_429, _auth(), now=NOW) == LATER
    assert account_switch.exhausted_until(_key(OWN), now=NOW) == LATER


def test_a_429_without_a_usable_reset_holds_for_the_default():
    headers = [("anthropic-ratelimit-unified-status", "rejected"),
               ("anthropic-ratelimit-unified-reset", "soon")]
    until = account_switch.observe(429, headers, _auth(), now=NOW)
    assert until == NOW.timestamp() + account_switch.DEFAULT_EXHAUSTED_SECONDS


def test_an_exhausted_mark_expires_at_its_reset():
    _exhaust(OWN)
    after = datetime.fromtimestamp(LATER + 1, timezone.utc)
    assert account_switch.exhausted_until(_key(OWN), now=after) is None


# --------------------------------------------------------------------------- #
# Going back: a turn returns to its own account when that account resets.
# --------------------------------------------------------------------------- #
def _at(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, timezone.utc)


@pytest.mark.parametrize("mark,reader", [
    (_spend, account_switch.spent_until), (_exhaust, account_switch.exhausted_until),
], ids=["spent", "exhausted"])
def test_a_turn_returns_to_its_own_account_when_its_reset_arrives(mark, reader):
    """The approved "auto-return" extra (GRIP#8189). The wire decides every turn
    afresh, and each mark ends at the reset it carries. So a turn moved off a capped
    account goes back to it at that reset, and does not stay on the pool account it
    borrowed. This test pins HAL's boundary: the reset instant counts as renewed,
    which is also how GRIP's readers of HAL's ledger count it."""
    mark(OWN)
    assert reader(_key(OWN), now=NOW) == LATER
    assert _choose(now=_at(LATER - 1)) == ("alice", ALICE, account_switch.SUBSCRIPTION_RANK)
    assert _choose(now=_at(LATER)) is None


def test_at_its_reset_a_turn_stays_on_its_own_account_while_the_pool_is_spent(policy):
    """AGORA's falsifier on HAL#749: the return holds while every pool account is
    still spent and only its paid overage would answer. Before the reset the turn
    goes to that overage; at the reset it stays on its own account. (``None`` says
    the turn is kept, not which of the own account's allowances bills it.)"""
    policy(spend_guard.OVERAGE)
    _exhaust(OWN)
    for token in (ALICE, BOB):
        spend_guard.mark_spent(_key(token), LATER + 3600)
    assert _choose(now=_at(LATER - 1)) == ("alice", ALICE, account_switch.OVERAGE_RANK)
    assert _choose(now=_at(LATER)) is None
