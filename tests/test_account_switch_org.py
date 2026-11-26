"""HAL#663's organisation rule, on HAL#662's chooser and real wire with fake TLS."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import time

import pytest

from lib.hal import account_ledger, account_switch, spend_guard
from tests.test_wire_proxy import (  # reuse the isolated TLS servers, not the legacy tests
    API, TURN, OVERAGE_REPLY, _bearer_of, _cache_loss_rows, _exchange, _wait_for,
    home, pki, make_wire, strict_spend,
)

OWN = "fake-account-A-session"
SAME = "fake-account-A-pool"
OTHER = "fake-account-B-pool"
ORGS = {OWN: "fake-org-A", SAME: "fake-org-A", OTHER: "fake-org-B"}


@pytest.fixture(autouse=True)
def identities(monkeypatch):
    # Exercise the real credential classifier with an obviously fake prefix.
    monkeypatch.setattr(spend_guard, "_OAUTH_PREFIX", "fake-account-")
    monkeypatch.setattr(account_switch, "_ORGS", {})
    monkeypatch.setattr(account_switch, "_ORG_RETRY", {})
    monkeypatch.setattr(account_switch, "_EXHAUSTED", {})
    monkeypatch.setattr(account_switch, "_read_pool", lambda: [("same", SAME), ("other", OTHER)])


def turn():
    body = json.dumps(TURN).encode()
    return (f"POST /v1/messages HTTP/1.1\r\nHost: {API}\r\nAuthorization: Bearer {OWN}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body


def bearers(wire, method):
    return [_bearer_of(raw) for _, raw in wire.fake.requests if raw.startswith(method)]


def spent_wire(make_wire):
    wire = make_wire(pool_size=0)
    wire.fake.orgs = ORGS.copy()
    wire.fake.reply_headers = OVERAGE_REPLY
    assert _exchange(wire, turn()).startswith(b"HTTP/1.1 200")
    wire.fake.reply_headers = b""
    return wire


def test_spent_moves_to_another_organisation_once_per_token(make_wire, strict_spend):
    wire = spent_wire(make_wire)
    for _ in range(2):
        assert _exchange(wire, turn()).startswith(b"HTTP/1.1 200")
    assert bearers(wire, b"POST") == [OWN, OTHER, OTHER]
    assert Counter(bearers(wire, b"GET")) == Counter({OWN: 1, SAME: 1, OTHER: 1})
    assert set(account_switch._ORGS) == {spend_guard.account_key(t) for t in ORGS}


def test_without_another_organisation_refusal_stands(make_wire, strict_spend):
    wire = spent_wire(make_wire)
    wire.fake.orgs[OTHER] = ORGS[OWN]
    assert _exchange(wire, turn()).startswith(b"HTTP/1.1 429")
    assert bearers(wire, b"POST") == [OWN]


@pytest.mark.parametrize("unknown", [OWN, OTHER])
def test_unknown_organisation_refuses_and_is_cached(make_wire, strict_spend, unknown):
    wire = spent_wire(make_wire)
    del wire.fake.orgs[unknown]
    for _ in range(2):
        assert _exchange(wire, turn()).startswith(b"HTTP/1.1 429")
    assert bearers(wire, b"POST") == [OWN]
    assert bearers(wire, b"GET").count(unknown) == 1


def test_empty_pool_sends_no_probe(make_wire, strict_spend, monkeypatch):
    wire = spent_wire(make_wire)
    monkeypatch.setattr(account_switch, "_read_pool", lambda: [])
    assert _exchange(wire, turn()).startswith(b"HTTP/1.1 429")
    assert bearers(wire, b"GET") == []


def test_limit_429_moves_this_turn(make_wire):
    wire = make_wire(pool_size=0)
    wire.fake.orgs = ORGS.copy()
    wire.fake.limit_bearers = {OWN}
    assert _exchange(wire, turn()).startswith(b"HTTP/1.1 200")
    assert _exchange(wire, turn()).startswith(b"HTTP/1.1 200")
    assert bearers(wire, b"POST") == [OWN, OTHER, OTHER]


def test_overage_also_excludes_same_organisation(make_wire, strict_spend):
    wire = make_wire(pool_size=0)
    wire.fake.orgs = ORGS.copy()
    wire.fake.limit_bearers = {OWN}
    wire.fake.orgs.pop(OTHER)  # no replacement until overage is explicitly opted in
    assert _exchange(wire, turn()).startswith(b"HTTP/1.1 429")
    wire.fake.orgs[OTHER] = ORGS[OTHER]
    account_switch._ORG_RETRY.clear()
    strict_spend(opt_in=["anthropic-overage"])
    for token in (SAME, OTHER):
        spend_guard.mark_spent(spend_guard.account_key(token), time.time() + 3600)
    assert _exchange(wire, turn()).startswith(b"HTTP/1.1 200")
    assert bearers(wire, b"POST") == [OWN, OTHER]


def test_a_mark_on_one_account_covers_every_account_of_its_known_organisation():
    """Capacity is kept by organisation (pool hardening, 2026-10-08): accounts in
    one organisation share limits, so an exhausted or a spent mark on one is a mark
    on every account whose KNOWN organisation is the same. Unknown changes nothing."""
    key = spend_guard.account_key
    account_switch._ORGS.update({key(t): org for t, org in ORGS.items()})
    until = time.time() + 3600
    account_switch._EXHAUSTED[key(OWN)] = until
    assert account_switch.exhausted_until(key(SAME)) == until
    assert account_switch.exhausted_until(key(OTHER)) is None
    assert account_switch.exhausted_until(key("fake-account-unknown-org")) is None
    spend_guard.mark_spent(key(SAME), until)
    assert account_switch.spent_until(key(OWN)) == until
    assert account_switch.spent_until(key(OTHER)) is None
    assert spend_guard.spent_until(key(OWN)) is None   # the guard's own mark is untouched


def test_choose_skips_a_pool_account_whose_organisation_is_exhausted(make_wire, monkeypatch):
    """OWN's limit is reached. P0 is exhausted, and P1, whose own marks are clean,
    is in P0's organisation: the turn goes to P2 in another organisation, not to
    P1 to be refused there."""
    p0, p1, p2 = ("fake-account-P0-pool", "fake-account-P1-pool", "fake-account-P2-pool")
    wire = make_wire(pool_size=0)
    wire.fake.orgs = {OWN: "fake-org-A", p0: "fake-org-B", p1: "fake-org-B", p2: "fake-org-C"}
    wire.fake.limit_bearers = {OWN}
    monkeypatch.setattr(account_switch, "_read_pool", lambda: [("p0", p0), ("p1", p1), ("p2", p2)])
    account_switch._EXHAUSTED[spend_guard.account_key(p0)] = time.time() + 3600
    assert _exchange(wire, turn()).startswith(b"HTTP/1.1 200")
    assert bearers(wire, b"POST") == [OWN, p2]


def delta_turn():
    """A thread delta: what Claude Code 2.1.295 sends after a thread's first request."""
    body = json.dumps({**TURN, "thread": {"type": "conversation",
                                          "previous_message_id": "msg_fake_previous"}}).encode()
    return (f"POST /v1/messages HTTP/1.1\r\nHost: {API}\r\nAuthorization: Bearer {OWN}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body


def test_ledger_rows_say_whether_a_delta_was_served_on_another_account(make_wire):
    """HAL#709 instrumentation. Whether a thread delta resolves on another account
    is unmeasured, and both switch paths move deltas today: the replay of a limit
    429, and the up-front swap once an account is marked exhausted. Each
    cache-loss row now says whether the request continued a thread and whether
    its bearer was switched, so production rows answer the question."""
    wire = make_wire(pool_size=0)
    wire.fake.orgs = ORGS.copy()
    wire.fake.limit_bearers = {OWN}
    assert _exchange(wire, delta_turn()).startswith(b"HTTP/1.1 200")   # replayed on OTHER
    assert _exchange(wire, turn()).startswith(b"HTTP/1.1 200")         # swapped up front
    assert bearers(wire, b"POST") == [OWN, OTHER, OTHER]
    _wait_for(lambda: len(_cache_loss_rows(wire.home)) >= 2)
    flags = [(r["continues_thread"], r["account_switched"]) for r in _cache_loss_rows(wire.home)]
    assert flags == [(True, True), (False, True)]


def test_ledger_rows_of_an_unswitched_turn_carry_false_flags(make_wire):
    wire = make_wire(pool_size=0)
    wire.fake.orgs = ORGS.copy()
    assert _exchange(wire, turn()).startswith(b"HTTP/1.1 200")
    _wait_for(lambda: len(_cache_loss_rows(wire.home)) >= 1)
    row = _cache_loss_rows(wire.home)[0]
    assert (row["continues_thread"], row["account_switched"]) == (False, False)


def test_choose_skips_a_pool_account_whose_organisation_is_spent():
    """The spent mark counts by organisation too. OTHER, outside the pool, billed
    paid overage, so C, in OTHER's organisation, has no subscription left either,
    and the turn goes to D."""
    key = spend_guard.account_key
    c, d = "fake-account-C-pool", "fake-account-D-pool"
    orgs = {OWN: "fake-org-A", OTHER: "fake-org-B", c: "fake-org-B", d: "fake-org-C"}
    account_switch._ORGS[key(OTHER)] = orgs[OTHER]
    until = time.time() + 3600
    spend_guard.mark_spent(key(OWN), until)
    spend_guard.mark_spent(key(OTHER), until)
    choice = account_switch.choose("POST", "/v1/messages", [("Authorization", "Bearer " + OWN)],
                                   pool=lambda: [("c", c), ("d", d)], probe_org=orgs.get)
    assert choice[:2] == ("d", d)


def test_the_holds_pool_reads_count_by_organisation(monkeypatch):
    """The hold's "all spent" check and its pause reset read a pool account's
    capacity by organisation too: B counts as spent while X, in its organisation
    but outside the pool, is."""
    from types import SimpleNamespace
    from lib.hal import wire_hold
    key = spend_guard.account_key
    b, x = "fake-account-B-pool", "fake-account-X-pool"
    monkeypatch.setattr(account_switch, "_read_pool", lambda: [("b", b)])
    account_switch._ORGS.update({key(b): "fake-org-X", key(x): "fake-org-X"})
    until = time.time() + 3600
    account_switch._EXHAUSTED[key(OWN)] = until
    spend_guard.mark_spent(key(x), until)
    proxy = SimpleNamespace(_ledger=account_ledger.AccountLedger())
    headers = [("Authorization", "Bearer " + OWN)]
    assert wire_hold.exhaustion(proxy, headers) == (True, until)
    assert wire_hold.pooled_reset(proxy, headers) == until


def test_concurrent_lookups_only_probe_once():
    calls = []

    def probe(token):
        calls.append(token)
        time.sleep(0.01)
        return ORGS[token]

    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda _: account_switch._organisation(OWN, probe), range(8)))
    assert results == [ORGS[OWN]] * 8
    assert calls == [OWN]


def test_failed_lookup_is_unknown_and_cached():
    calls = []

    def probe(token):
        calls.append(token)
        raise TimeoutError("fake upstream timed out")

    for _ in range(2):
        assert account_switch._organisation(OWN, probe) is None
    assert calls == [OWN]


def test_a_failed_lookup_is_retried_after_the_window_and_a_known_one_is_kept():
    # A lookup that fails during an outage must not disable switching for the
    # rest of the process: that is exactly when the switch is needed.
    calls = []
    answers = iter([None, ORGS[OWN]])

    def probe(token):
        calls.append(token)
        return next(answers)

    start = 1000.0
    assert account_switch._organisation(OWN, probe, now=start) is None
    assert account_switch._organisation(OWN, probe, now=start + 30) is None
    assert calls == [OWN]  # inside the window: no new probe
    later = start + account_switch.UNKNOWN_ORG_RETRY_SECONDS + 1
    assert account_switch._organisation(OWN, probe, now=later) == ORGS[OWN]
    assert account_switch._organisation(OWN, probe, now=later + 10_000) == ORGS[OWN]
    assert calls == [OWN, OWN]  # retried once after the window, then kept


def test_no_probe_reads_only_a_known_organisation_and_records_nothing():
    """HAL#770: the wire's "retry now" check passes no probe, so it never waits on the
    network. With no probe nothing was looked up, so nothing failed: no retry window
    is recorded, and the next lookup that has a probe still makes it."""
    assert account_switch._organisation(OWN, None) is None
    assert account_switch._ORG_RETRY == {}
    assert account_switch._organisation(OWN, lambda token: ORGS[OWN]) == ORGS[OWN]
    assert account_switch._organisation(OWN, None) == ORGS[OWN]


def test_probe_uses_short_timeout_and_closes(make_wire, monkeypatch):
    wire = make_wire(pool_size=0)
    observed = []

    class Socket:
        def settimeout(self, seconds):
            observed.append(("read", seconds))

        def sendall(self, data):
            assert data.startswith(b"GET /v1/models?limit=1 HTTP/1.1")

        def shutdown(self, how):
            pass

        def close(self):
            observed.append(("close", True))

    class Stream:
        def read_head(self):
            raise TimeoutError("fake timeout")

    def take(*, timeout):
        observed.append(("connect", timeout))
        from types import SimpleNamespace
        return SimpleNamespace(sock=Socket(), stream=Stream())

    monkeypatch.setattr(wire.proxy.pool, "take", take)
    assert wire.proxy._probe_org(OWN) is None
    assert observed == [("connect", 3.0), ("read", 3.0), ("close", True)] * 2
