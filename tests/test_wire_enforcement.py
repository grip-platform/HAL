"""The first-signal enforcement stop, end to end over loopback TLS, fake accounts only.

V>> approved "Harden, then add" for the account pool on 2026-10-08 (GRIP
``plans/ULTRAPLAN-2026-10-08-claude-capacity.md``, section 4, phase 1). A 401 or a
403 whose message names a revocation or a suspension, from ANY account, stops all
account switching and tells the operator once; a 403 is never switched; a 401 that
only says it expired keeps the dead-token switch.

Every body here is made up: Anthropic documents the error types, not the wording an
enforcement action is sent with (``lib/hal/auth_signal.py``).
"""
import json
import time

import pytest

from lib.hal import account_ledger, account_switch, auth_signal, spend_guard
from lib.hal.notification import NotificationHub
from tests import test_wire_proxy as h
from tests import test_wire_replay as replay
from tests.test_wire_proxy import (  # noqa: F401 -- fixtures pytest finds by name here:
    home, pki, make_wire,  # the wire fixture and held ask for them in this module
)
from tests.test_wire_hold import held  # noqa: F401 -- the hold-path fixture
from tests.test_wire_replay import A, B, C, OK, LIMIT, reply, request, posts


@pytest.fixture(autouse=True)
def _healthy_attempts(monkeypatch):
    from lib.hal.wire_cascade import Attempts
    monkeypatch.setattr(Attempts, '_unhealthy', False, raising=False)


def error(status, message, error_type):
    payload = {"type": "error", "error": {"type": error_type, "message": message}}
    return reply(status, json.dumps(payload).encode())


SUSPENDED = error(403, "Your account has been suspended.", "permission_error")
REVOKED = error(401, "OAuth token has been revoked.", "authentication_error")
EXPIRED = error(401, "OAuth token has expired.", "authentication_error")


class Upstream(replay.Upstream):
    """The replay tests' fake, whose organisation lookup can also be refused."""

    probes = None

    def _one(self, tls, reader, number):
        head = reader.until(b"\r\n\r\n")
        if not head:
            return False
        body = reader.exact(h._header_int(head, b"content-length"))
        self.requests.append((number, head + body))
        token = h._bearer_of(head)
        if head.startswith(b"GET /v1/models"):
            tls.sendall(self.probes.get(token) or reply(
                200, b"{}", b"anthropic-organization-id: fake-org-" + token.encode() + b"\r\n"))
        else:
            tls.sendall(self.answers.get(token, OK))
        return True


@pytest.fixture(autouse=True)
def fresh_stop(monkeypatch):
    """Every test starts with switching on, no stop recorded and nothing told."""
    monkeypatch.setattr(account_switch, "NOTICES", NotificationHub())
    account_switch._UNRECORDED_STOP.clear()
    yield
    account_switch._UNRECORDED_STOP.clear()


@pytest.fixture
def wire(make_wire, monkeypatch):
    monkeypatch.setattr(h, "FakeUpstream", Upstream)
    monkeypatch.setattr(Upstream, "answers", {A: LIMIT})
    monkeypatch.setattr(Upstream, "probes", {})
    monkeypatch.setattr(spend_guard, "_OAUTH_PREFIX", "fake-account-")
    monkeypatch.setattr(account_switch, "_ORGS", {})
    monkeypatch.setattr(account_switch, "_ORG_RETRY", {})
    monkeypatch.setattr(account_switch, "_EXHAUSTED", {})
    monkeypatch.setattr(account_switch, "_read_pool", lambda: [("B", B), ("C", C)])
    built = make_wire(pool_size=0)
    built.proxy._ledger = account_ledger.AccountLedger(resolver=account_ledger.AccountResolver(
        pool=lambda: [("A", A), ("B", B), ("C", C)]))
    return built


def rows(home, event):
    return h._wait_for(lambda: [row for row in h._rows(home) if row.get("event") == event])


def told(home):
    """What the operator was told: the notification hub's history, as kept on disk."""
    path = home / "state" / "hal" / "notifications" / "history.json"
    if not path.exists():
        return []
    return [n for n in json.loads(path.read_text()) if n["event"] == account_switch.STOP_EVENT]


def reason():
    record = account_switch.off_reason() or {}
    return {key: record.get(key) for key in ("stopped_by", "account", "status", "signal",
                                              "error_type", "word")}


def judged(status, message, error_type="permission_error"):
    payload = {"type": "error", "error": {"type": error_type, "message": message}}
    return auth_signal.classify(status, json.dumps(payload).encode())


# --------------------------------------------------------------------------- #
# Over the wire.
# --------------------------------------------------------------------------- #
def test_an_enforcement_401_stops_switching_records_why_and_the_next_turn_is_not_switched(wire):
    wire.fake.answers[A] = REVOKED
    raw = request()
    assert h._exchange(wire, raw) == REVOKED          # unchanged: no replay, no pause
    assert posts(wire) == [raw]
    assert not account_switch.switching_on()
    assert reason() == {"stopped_by": "enforcement", "account": "A", "status": 401,
                        "signal": "enforcement", "error_type": "authentication_error",
                        "word": "revoked"}
    notices = told(wire.home)
    assert len(notices) == 1 and notices[0]["severity"] == "error"
    assert "account 'A' was answered 401" in notices[0]["message"]
    signal_rows = rows(wire.home, "wire.enforcement_signal")
    assert [(r["account"], r["stopped_switching"], r["word"]) for r in signal_rows] == [
        ("A", True, "revoked")]
    # The next turn finds A's limit reached. Switching would carry it to B: not now.
    wire.fake.answers[A] = LIMIT
    assert h._exchange(wire, raw) == LIMIT
    assert posts(wire) == [raw, raw]
    kept = (account_switch.off_marker().read_text()
            + (wire.home / "state/hal/notifications/history.json").read_text()
            + json.dumps(h._rows(wire.home)))
    assert not any(token in kept for token in (A, B, C))
    assert "OAuth token has been revoked" not in kept   # the message is never kept


def test_an_enforcement_401_is_never_paused_even_with_a_spent_pool_account(wire):
    """With B merely spent to a known reset, a dead 401 would be answered with the
    honest exhaustion 429. One that names a revocation is not a dead token: it
    reaches Claude Code unchanged, never masked with a reset."""
    account_switch._EXHAUSTED[spend_guard.account_key(B)] = time.time() + 120
    wire.fake.answers[A] = REVOKED
    assert h._exchange(wire, request()) == REVOKED
    assert reason()["account"] == "A"


def test_an_enforcement_403_is_relayed_unchanged_stops_switching_and_holds_the_token_dead(wire):
    wire.fake.answers[A] = SUSPENDED
    raw = request()
    assert h._exchange(wire, raw) == SUSPENDED
    assert posts(wire) == [raw]
    assert (reason()["account"], reason()["status"], reason()["word"]) == ("A", 403, "suspended")
    assert account_ledger.LedgerView().token_health("A", A) == "dead"


def test_a_chunked_enforcement_body_is_judged_and_relayed_byte_for_byte(wire):
    payload = json.dumps({"type": "error", "error": {"type": "permission_error",
                                                     "message": "Account disabled."}}).encode()
    chunked = (b"HTTP/1.1 403 Test\r\nContent-Type: application/json\r\n"
               b"Transfer-Encoding: chunked\r\n\r\n"
               + f"{len(payload):x};keep=yes\r\n".encode() + payload + b"\r\n0\r\n\r\n")
    wire.fake.answers[A] = chunked
    assert h._exchange(wire, request()) == chunked
    assert reason()["word"] == "disabled"


def test_an_enforcement_answer_to_an_aux_call_stops_switching_too(wire):
    """The first signal counts, whichever call it came on: here a quota check."""
    wire.fake.answers[A] = SUSPENDED
    assert h._exchange(wire, request(payload=h.QUOTA)) == SUSPENDED
    assert reason()["account"] == "A" and not account_switch.switching_on()


def test_an_expiry_401_still_switches_as_before(wire):
    wire.fake.answers[A] = EXPIRED
    raw = request()
    assert h._exchange(wire, raw) == OK
    assert posts(wire) == [raw, raw.replace(A.encode(), B.encode())]
    assert account_switch.switching_on() and account_switch.off_reason() is None
    assert told(wire.home) == []
    dead = rows(wire.home, "wire.dead_token")
    assert [(r["account"], r["switched"], r["signal"]) for r in dead] == [("A", True, "expiry")]


def test_an_enforcement_answer_from_the_replacement_stops_switching(wire):
    """A's limit is reached and B is asked; B's answer names a suspension. The
    original 429 is restored, and the next limit is not carried to C."""
    wire.fake.answers[B] = SUSPENDED
    raw = request()
    assert h._exchange(wire, raw) == LIMIT
    assert posts(wire) == [raw, raw.replace(A.encode(), B.encode())]
    assert reason()["account"] == "B"
    assert h._exchange(wire, raw) == LIMIT
    assert posts(wire)[2:] == [raw]


def test_an_enforcement_answer_to_the_organisation_lookup_stops_switching(wire):
    """The free organisation lookup is the account's own answer too. B's names a
    suspension while A's turn looks for a replacement: switching stops there, and
    C, whose lookup succeeds right after, is not given the turn."""
    wire.fake.probes[B] = SUSPENDED
    raw = request()
    assert h._exchange(wire, raw) == LIMIT
    assert posts(wire) == [raw]
    assert reason()["account"] == "B"
    assert account_ledger.LedgerView().token_health("B", B) == "dead"


# --------------------------------------------------------------------------- #
# On the hold path, which Claude Code's streamed turns take.
# --------------------------------------------------------------------------- #
def _pool_of_two(held):
    held.proxy._ledger = account_ledger.AccountLedger(resolver=account_ledger.AccountResolver(
        pool=lambda: [("A", A), ("B", B)]))


def test_on_the_hold_path_an_enforcement_401_reaches_the_client_unchanged(held):
    from tests.test_wire_hold import exchange, SUCCESS, posts as held_posts
    _pool_of_two(held)
    held.fake.actions = [REVOKED, SUCCESS]
    assert exchange(held) == REVOKED
    assert [h._bearer_of(r) for r in held_posts(held)] == [A]
    assert reason()["account"] == "A" and not account_switch.switching_on()


def test_on_the_hold_path_an_enforcement_answer_from_a_replacement_stops_switching(
        held, monkeypatch):
    """The hold moves a throttled turn to B; B's answer names a suspension. The
    hold goes back to A, which is no switch, and the turn is served there."""
    from tests.test_wire_hold import exchange, SUCCESS, posts as held_posts
    from lib.hal import wire_hold
    monkeypatch.setattr(wire_hold, "BUDGET_S", 5.0)
    _pool_of_two(held)
    held.fake.actions = ([reply(429, b"throttle")] * wire_hold.THROTTLE_ATTEMPTS
                         + [SUSPENDED, SUCCESS])
    raw = exchange(held)
    assert b"answer" in raw and b"suspended" not in raw
    assert ([h._bearer_of(r) for r in held_posts(held)]
            == [A] * wire_hold.THROTTLE_ATTEMPTS + [B, A])
    assert reason()["account"] == "B" and not account_switch.switching_on()


def test_on_the_hold_path_a_later_throttle_is_not_switched_after_a_stop(held, monkeypatch):
    """AGORA's third review of HAL#703 asked for the switch decision after a stop.
    The hold moves a throttled turn to another account, whose answer names a
    suspension; back on A the turn is throttled again as many times as moved it
    before, and this time it stays on A until A serves it.

    A third account is ready throughout. With only two, the suspended account's
    token is held dead and nothing is left to switch to, so the test passed with
    the stop removed: it could not tell the stop from that dead mark."""
    from tests.test_wire_hold import exchange, SUCCESS, posts as held_posts
    from lib.hal import wire_hold
    monkeypatch.setattr(wire_hold, "BUDGET_S", 5.0)
    monkeypatch.setattr(account_switch, "_read_pool", lambda: [("A", A), ("B", B), ("C", C)])
    held.proxy._ledger = account_ledger.AccountLedger(resolver=account_ledger.AccountResolver(
        pool=lambda: [("A", A), ("B", B), ("C", C)]))
    throttles = [reply(429, b"throttle")] * wire_hold.THROTTLE_ATTEMPTS
    held.fake.actions = throttles + [SUSPENDED] + throttles + [SUCCESS]
    raw = exchange(held)
    assert b"answer" in raw
    bearers = [h._bearer_of(r) for r in held_posts(held)]
    moved_to = bearers[wire_hold.THROTTLE_ATTEMPTS]
    assert moved_to in (B, C)
    assert bearers == ([A] * wire_hold.THROTTLE_ATTEMPTS + [moved_to]
                       + [A] * (wire_hold.THROTTLE_ATTEMPTS + 1))
    assert reason()["account"] == {B: "B", C: "C"}[moved_to]


# --------------------------------------------------------------------------- #
# The stop itself, and the operator's command line.
# --------------------------------------------------------------------------- #
def test_status_prints_the_recorded_reason_off_keeps_it_and_on_clears_it(home, capsys):
    assert account_switch.stop_for_enforcement(judged(403, "Account suspended."), "bob")
    assert account_switch.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "account switching is off" in out
    assert "account 'bob' was answered 403 permission_error, naming 'suspended'" in out
    assert account_switch.main(["off"]) == 0
    assert account_switch.off_reason()["account"] == "bob"   # off kept the record
    assert account_switch.main(["on"]) == 0
    assert account_switch.switching_on() and account_switch.off_reason() is None
    assert "account switching is on" in capsys.readouterr().out


def test_the_operator_is_told_once_and_the_first_reason_stands(home):
    first = judged(401, "OAuth token revoked.", "authentication_error")
    assert account_switch.stop_for_enforcement(first, "A") is True
    assert account_switch.stop_for_enforcement(judged(403, "suspended"), "B") is False
    assert account_switch.off_reason()["account"] == "A"
    assert len(told(home)) == 1


def test_a_marker_turned_off_by_hand_still_records_the_stop_and_tells(home):
    account_switch.set_switching(False)
    assert not account_switch.switching_on() and account_switch.off_reason() is None
    assert account_switch.stop_for_enforcement(judged(403, "suspended"), "A") is True
    assert account_switch.off_reason()["account"] == "A"
    assert len(told(home)) == 1


def test_a_stop_that_cannot_be_written_still_holds_in_this_process(home, monkeypatch):
    def refuse(text):
        raise PermissionError("fake: the marker cannot be written")

    monkeypatch.setattr(account_switch, "_write_marker", refuse)
    assert account_switch.stop_for_enforcement(judged(403, "suspended"), "A") is True
    assert not account_switch.off_marker().exists()
    assert not account_switch.switching_on()
    assert account_switch.stop_for_enforcement(judged(403, "suspended"), "B") is False
    assert len(told(home)) == 1
    account_switch.set_switching(True)
    assert account_switch.switching_on()
