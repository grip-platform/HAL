"""The per-account ledger (InfiniteMachine v2, Step 5b).

Headers into numbers, unknown never 0; a cooldown that lasts until the reset the
headers name; token health that follows the answers; requests in flight; GRIP's
pool read fail-closed; one entry per account written whole by one writer at a
time, across processes; and no token or header value in any sink.

Everything runs on temporary files. No test reads the operator's token store
(``tests/conftest.py`` points it at a path that does not exist) or ``~/.hal``.
Every credential here is an obvious fake.
"""
from __future__ import annotations

import itertools
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path

import pytest

from lib.hal import account_ledger as al
from lib.hal import locked_json
from lib.hal.account_ledger import (UNPOOLED, AccountLedger, AccountResolver, LedgerView,
                                    PoolAccount, read_headers)

P = al.PREFIX
NOW = 1_791_230_000          # 2026-10-05: every test runs on this fixed clock
RESET_5H = NOW + 3_600
RESET_7D = NOW + 400_000
TOKEN_A = "fake-account-A-token"
TOKEN_B = "fake-account-B-token"
TOKEN_PAID = "fake-paid-key"


class Clock:
    def __init__(self, at: float = NOW):
        self.at = at

    def __call__(self) -> float:
        return self.at


def _bearer(token: str) -> list:
    return [("authorization", f"Bearer {token}")]


def _record(data: dict, label: str, status, headers, *, now: float = NOW, digest=None) -> dict:
    return al.settle(data, label, pid=os.getpid(), started=now, now=now,
                     answer=(status, read_headers(headers, now=now)), digest=digest)


def _entry(data: dict, label: str = "acct") -> dict:
    return data["accounts"][label]


# --------------------------------------------------------------------------- #
# Reading the headers: a table, and unknown is never 0.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("headers, field, expected", [
    ([], "status", "unknown"),
    ([(al.STATUS, "allowed")], "status", "allowed"),
    ([(al.STATUS, "Allowed_Warning")], "status", "allowed_warning"),
    ([(al.STATUS, "rejected")], "status", "rejected"),
    ([(al.STATUS, "maybe")], "status", "unknown"),
    ([(al.STATUS, "")], "status", "unknown"),
    ([(al.STATUS, "allowed"), (al.STATUS, "rejected")], "status", "unknown"),
    ([(al.STATUS, "allowed"), (al.STATUS, " allowed")], "status", "allowed"),
    ([], "reset", None),
    ([(al.RESET, str(RESET_5H))], "reset", RESET_5H),
    ([(al.RESET, f"{RESET_5H}.4")], "reset", RESET_5H),
    ([(al.RESET, "0")], "reset", None),
    ([(al.RESET, "-5")], "reset", None),
    ([(al.RESET, "1e10")], "reset", None),
    ([(al.RESET, "NaN")], "reset", None),
    ([(al.RESET, "soon")], "reset", None),
    ([], "claim", None),
    ([(al.CLAIM, "five_hour")], "claim", "five_hour"),
    ([(al.CLAIM, "7d")], "claim", "seven_day"),
    ([(al.CLAIM, "seven_day_sonnet")], "claim", "seven_day_sonnet"),
    ([(al.CLAIM, "fortnightly")], "claim", "other"),
    ([], "overage_status", "unknown"),
    ([(al.OVERAGE_STATUS, "allowed")], "overage_status", "allowed"),
    ([], "overage_in_use", None),
    ([(al.OVERAGE_IN_USE, "true")], "overage_in_use", True),
    ([(al.OVERAGE_IN_USE, "FALSE")], "overage_in_use", False),
    ([(al.OVERAGE_IN_USE, "yes")], "overage_in_use", None),
    ([], "overage_disabled_reason", None),
    ([(al.OVERAGE_DISABLED_REASON, "org_spend_cap_reached")], "overage_disabled_reason",
     "org_spend_cap_reached"),
    ([(al.OVERAGE_DISABLED_REASON, "because")], "overage_disabled_reason", "other"),
    ([], "retry_after_s", None),
    ([(al.RETRY_AFTER, "37")], "retry_after_s", 37),
    ([(al.RETRY_AFTER, "-3")], "retry_after_s", None),
    ([(al.RETRY_AFTER, "later")], "retry_after_s", None),
    ([(al.RETRY_AFTER, "99999999")], "retry_after_s", None),
])
def test_each_header_is_read_and_missing_or_garbled_is_unknown(headers, field, expected):
    assert getattr(read_headers(headers, now=NOW), field) == expected


@pytest.mark.parametrize("value, expected", [
    ("0.42", 0.42), ("1", 1.0), ("0", 0.0), ("", None), ("abc", None), ("-0.1", None),
    ("inf", None), ("1e-3", None), ("101", None), ("0.5, 0.6", None),
])
def test_a_window_use_is_a_number_or_unknown_never_zero_by_default(value, expected):
    window = read_headers([(f"{P}5h-utilization", value)], now=NOW).window("five_hour")
    assert window is not None and window.utilization == expected
    assert (window.reset, window.surpassed) == (None, None)


def test_every_window_is_read_under_its_own_abbreviation():
    headers = [(f"{P}{abbreviation}-utilization", f"0.{index + 1}")
               for index, (_, abbreviation) in enumerate(al.WINDOWS)]
    headers += [(f"{P}7d-reset", str(RESET_7D)), (f"{P}7d-surpassed-threshold", "0.75")]
    reading = read_headers(headers, now=NOW)
    assert [name for name, _ in reading.windows] == [name for name, _ in al.WINDOWS]
    assert reading.window("seven_day") == al.Window(0.2, RESET_7D, 0.75)
    assert read_headers([], now=NOW).windows == ()
    assert read_headers([], now=NOW).window("five_hour") is None


def test_names_match_without_case_and_a_message_with_items_is_read_too():
    class Message:  # urllib's HTTPMessage shape
        @staticmethod
        def items():
            return [("Anthropic-RateLimit-Unified-Status", "rejected"), ("Retry-After", "5")]
    reading = read_headers(Message(), now=NOW)
    assert (reading.status, reading.retry_after_s) == ("rejected", 5)
    assert read_headers(None, now=NOW) == al.Reading()
    assert read_headers([("only-one-item",), 7], now=NOW) == al.Reading()


def test_retry_after_may_be_an_http_date():
    soon = datetime.fromtimestamp(NOW + 90, timezone.utc)
    past = datetime.fromtimestamp(NOW - 90, timezone.utc)
    assert read_headers([(al.RETRY_AFTER, format_datetime(soon, usegmt=True))],
                        now=NOW).retry_after_s == 90
    assert read_headers([(al.RETRY_AFTER, format_datetime(past, usegmt=True))],
                        now=NOW).retry_after_s == 0


def test_header_names_are_kept_and_their_values_are_not():
    reading = read_headers([(al.STATUS, "allowed"), (f"{P}brand-new-thing", "KEPT-NEVER"),
                            ("x-other", "nope")], now=NOW)
    assert reading.header_names == (f"{P}brand-new-thing", al.STATUS)
    assert "KEPT-NEVER" not in json.dumps(reading.to_json())
    assert not read_headers([("x-other", "nope")], now=NOW).has_quota


def test_the_vendor_gateway_documents_example_reads_as_spent():
    # The 429 the gateway protocol embedded in Claude Code 2.1.289 shows, in shape:
    # a cap reached, overage disabled because the spend cap is reached.
    reading = read_headers([
        ("retry-after", "37800"), ("x-should-retry", "false"), (al.STATUS, "rejected"),
        (al.RESET, "1786147200"), (al.OVERAGE_RESET, "1786147200"),
        (f"{P}overage-utilization", "1"), (f"{P}overage-surpassed-threshold", "1"),
        (f"{P}overage-period", "daily"), (al.OVERAGE_DISABLED_REASON, "org_spend_cap_reached"),
    ], now=1786147200 - 37800)
    assert (reading.status, reading.reset, reading.retry_after_s) == (
        "rejected", 1786147200, 37800)
    assert reading.window("overage") == al.Window(1.0, 1786147200, 1.0)
    assert reading.overage_disabled_reason == "org_spend_cap_reached"


# --------------------------------------------------------------------------- #
# Cooling lasts until the reset the headers name.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("headers, until, source", [
    ([(al.STATUS, "rejected"), (al.RESET, str(RESET_5H)), (al.CLAIM, "five_hour"),
      (f"{P}5h-reset", str(RESET_5H + 60)), (al.RETRY_AFTER, "30")], RESET_5H, "unified-reset"),
    ([(al.STATUS, "rejected"), (al.RESET, "garbled"), (al.CLAIM, "seven_day"),
      (f"{P}7d-reset", str(RESET_7D)), (al.RETRY_AFTER, "30")], RESET_7D, "seven_day-reset"),
    ([(al.STATUS, "rejected"), (al.CLAIM, "five_hour"), (al.RETRY_AFTER, "30")],
     NOW + 30, "retry-after"),
    ([(al.STATUS, "rejected")], None, "none"),
])
def test_a_spent_account_cools_until_the_reset_its_headers_name(headers, until, source):
    cooling = al.cooling_of(_entry(_record({}, "acct", 429, headers)), NOW)
    assert cooling is not None
    assert (cooling.until, cooling.source, cooling.reason) == (until, source, "spent")


def test_the_cooling_is_not_the_old_backoff_and_ends_when_the_reset_passes():
    entry = _entry(_record({}, "acct", 429, [(al.STATUS, "rejected"), (al.RESET, str(RESET_7D))]))
    assert al.cooling_of(entry, NOW + 3601) is not None   # the old cap was one hour
    assert al.cooling_of(entry, RESET_7D - 1) is not None
    assert al.cooling_of(entry, RESET_7D) is None


def test_a_rejection_naming_no_reset_is_open_ended_and_carries_its_age():
    cooling = al.cooling_of(_entry(_record({}, "acct", 429, [(al.STATUS, "rejected")])),
                            NOW + 10 ** 6)
    assert (cooling.until, cooling.observed_at) == (None, NOW)


@pytest.mark.parametrize("headers", [
    [(al.STATUS, "rejected"), (al.OVERAGE_STATUS, "allowed"), (al.RESET, str(RESET_5H))],
    [(al.STATUS, "allowed"), (al.OVERAGE_IN_USE, "true"), (al.RESET, str(RESET_5H))],
])
def test_a_2xx_billed_to_overage_counts_as_spent(headers):
    cooling = al.cooling_of(_entry(_record({}, "acct", 200, headers)), NOW)
    assert cooling is not None
    assert (cooling.reason, cooling.until) == ("paid-overage", RESET_5H)


def test_an_allowed_answer_clears_a_cooling_and_an_answer_without_quota_keeps_it():
    data = _record({}, "acct", 429, [(al.STATUS, "rejected"), (al.RESET, str(RESET_5H))])
    data = _record(data, "acct", 503, [("content-type", "text/plain")])
    assert al.cooling_of(_entry(data), NOW) is not None
    data = _record(data, "acct", 200, [(al.STATUS, "allowed")])
    assert al.cooling_of(_entry(data), NOW) is None


def test_a_429_without_quota_headers_cools_by_retry_after_only():
    cooling = al.cooling_of(_entry(_record({}, "acct", 429, [(al.RETRY_AFTER, "5")])), NOW)
    assert (cooling.until, cooling.reason, cooling.source) == (NOW + 5, "throttled",
                                                               "retry-after")
    cooling = al.cooling_of(_entry(_record({}, "acct", 429, [])), NOW)
    assert (cooling.until, cooling.reason) == (None, "throttled")


def test_an_overload_is_no_cooldown_for_the_account():
    assert al.cooling_of(_entry(_record({}, "acct", 529, [(al.RETRY_AFTER, "5")])), NOW) is None


# --------------------------------------------------------------------------- #
# Token health follows the answers.
# --------------------------------------------------------------------------- #
def test_token_health_follows_the_answers():
    """A 403 alone leaves the verdict where it was (amended 2026-10-08, pool
    hardening; it used to read 'dead'): it is a permission answer, a model the
    account may not use for one, not a token that needs a re-login."""
    data: dict = {}
    for status, expected in ((None, "unknown"), (200, "ok"), (429, "ok"), (401, "dead"),
                             (500, "dead"), (204, "ok"), (403, "ok")):
        if status is not None:
            data = _record(data, "acct", status, [], digest="aaaaaaaaaaaa")
        assert al.token_health_of(data.get("accounts", {}).get("acct")) == expected, status


def test_a_403_that_names_a_suspension_is_held_dead_by_its_own_write(tmp_path):
    """The wire reads the body, which the ledger never sees, and calls mark_dead
    for a 403 whose message names a suspension. The verdict is about that token:
    a re-login presents another, which reads as unknown."""
    data = _record({}, "acct", 403, [], digest=al.credential_digest(TOKEN_A))
    assert al.token_health_of(_entry(data), TOKEN_A) == "unknown"
    data = al.mark_token_dead(data, "acct", digest=al.credential_digest(TOKEN_A), now=NOW)
    assert al.token_health_of(_entry(data), TOKEN_A) == "dead"
    assert al.token_health_of(_entry(data), TOKEN_B) == "unknown"
    ledger = AccountLedger(tmp_path / "ledger.json",
                           resolver=AccountResolver(pool=lambda: [("acct", TOKEN_A)]))
    ledger.mark_dead(_bearer(TOKEN_A))
    assert LedgerView(tmp_path / "ledger.json").token_health("acct", TOKEN_A) == "dead"
    assert TOKEN_A not in (tmp_path / "ledger.json").read_text()


def test_a_verdict_is_about_one_token_and_a_new_token_starts_unknown():
    data = _record({}, "acct", 401, [], digest=al.credential_digest(TOKEN_A))
    assert al.token_health_of(_entry(data)) == "dead"
    assert al.token_health_of(_entry(data), TOKEN_A) == "dead"
    assert al.token_health_of(_entry(data), TOKEN_B) == "unknown"  # a re-login since
    data = _record(data, "acct", 429, [], digest=al.credential_digest(TOKEN_B))
    assert al.token_health_of(_entry(data)) == "unknown"
    data = _record(data, "acct", 200, [], digest=al.credential_digest(TOKEN_B))
    assert al.token_health_of(_entry(data)) == "ok"
    assert _entry(data)["token"]["digest"] == al.credential_digest(TOKEN_B)
    assert len(al.credential_digest(TOKEN_B)) == 12


# --------------------------------------------------------------------------- #
# Requests in flight.
# --------------------------------------------------------------------------- #
def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_in_flight_counts_requests_until_their_answer_begins():
    data: dict = {}
    pid = os.getpid()
    for start in (NOW, NOW + 1, NOW + 1):  # two in the same millisecond
        al.add_in_flight(data, "acct", pid=pid, started=start, now=start)
    assert al.in_flight_of(_entry(data), NOW + 2) == 3
    al.settle(data, "acct", pid=pid, started=NOW + 1, now=NOW + 2, answer=None)
    assert al.in_flight_of(_entry(data), NOW + 2) == 2
    assert _entry(data)["observations"] == 0  # no answer arrived: nothing observed


def test_in_flight_forgets_dead_processes_and_lost_requests():
    data: dict = {}
    stale = NOW - al.IN_FLIGHT_MAX_AGE_S - 1
    al.add_in_flight(data, "acct", pid=_dead_pid(), started=NOW, now=NOW)
    al.add_in_flight(data, "acct", pid=os.getpid(), started=stale, now=stale)
    assert al.in_flight_of(_entry(data), NOW) == 0
    al.add_in_flight(data, "acct", pid=os.getpid(), started=NOW, now=NOW)
    assert list(_entry(data)["in_flight"]) == [str(os.getpid())]
    assert al.in_flight_of(_entry(data), NOW) == 1


def test_a_request_a_stopped_process_left_in_flight_is_counted_once():
    """HAL#748 item 4: a wire that stopped with a request in flight forgot it. The
    running total measures the "a restart forgets sends in flight" limit."""
    data: dict = {}
    al.add_in_flight(data, "acct", pid=_dead_pid(), started=NOW, now=NOW)
    assert "orphaned_in_flight" not in _entry(data)
    al.add_in_flight(data, "acct", pid=os.getpid(), started=NOW + 1, now=NOW + 1)
    assert _entry(data)["orphaned_in_flight"] == {"count": 1, "last_at": NOW + 1}
    al.settle(data, "acct", pid=os.getpid(), started=NOW + 1, now=NOW + 2, answer=None)
    assert _entry(data)["orphaned_in_flight"] == {"count": 1, "last_at": NOW + 1}
    # A second stopped process adds to the total rather than replacing it.
    al.add_in_flight(data, "acct", pid=_dead_pid(), started=NOW + 3, now=NOW + 3)
    al.add_in_flight(data, "acct", pid=os.getpid(), started=NOW + 4, now=NOW + 4)
    assert _entry(data)["orphaned_in_flight"] == {"count": 2, "last_at": NOW + 4}


def test_a_request_that_merely_aged_out_is_not_counted_as_left_behind():
    data: dict = {}
    stale = NOW - al.IN_FLIGHT_MAX_AGE_S - 1
    al.add_in_flight(data, "acct", pid=_dead_pid(), started=stale, now=stale)
    al.add_in_flight(data, "acct", pid=os.getpid(), started=NOW, now=NOW)
    assert "orphaned_in_flight" not in _entry(data)


# --------------------------------------------------------------------------- #
# GRIP's pool: read-only, fail-closed.
# --------------------------------------------------------------------------- #
def _store(tmp_path: Path, monkeypatch, members: dict, pool=None, *, dir_mode=0o700,
           file_mode=0o600, pool_mode=0o600) -> Path:
    store = tmp_path / "store"
    store.mkdir()
    os.chmod(store, 0o700)
    active = store / al.STORE_BASENAME
    for label, token in members.items():
        member = active.with_name(f"{active.name}.{label}")
        member.write_text(token + "\n")
        os.chmod(member, file_mode)
    if pool is not False:
        pool_file = active.with_name(active.name + ".pool")
        pool_file.write_text("\n".join(pool if pool is not None else members) + "\n")
        os.chmod(pool_file, pool_mode)
    os.chmod(store, dir_mode)
    monkeypatch.setenv(al.STORE_PATH_ENV, str(active))
    return active


def test_the_pool_is_read_in_its_own_order(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch, {"acct-b": TOKEN_B, "acct-a": TOKEN_A, "acct-a-old": "x1"},
           pool=["acct-a", "acct-b"])
    assert al.pool_tokens() == [("acct-a", TOKEN_A), ("acct-b", TOKEN_B)]


def test_pool_lines_that_are_not_labels_are_ignored_and_a_token_counts_once(tmp_path,
                                                                           monkeypatch):
    _store(tmp_path, monkeypatch, {"acct-a": TOKEN_A, "acct-twin": TOKEN_A, "acct-b": TOKEN_B},
           pool=["acct-a", "pool", "../acct-b", "acct-a", "", "acct-twin", "acct-b"])
    assert [label for label, _ in al.pool_tokens()] == ["acct-a", "acct-b"]


@pytest.mark.parametrize("damage", ["no-pool-file", "dir-group-readable", "dir-symlink",
                                    "pool-file-group-readable"])
def test_a_pool_that_cannot_be_trusted_is_empty(tmp_path, monkeypatch, damage):
    members = {"acct-a": TOKEN_A}
    if damage == "no-pool-file":
        _store(tmp_path, monkeypatch, members, pool=False)
    elif damage == "dir-group-readable":
        _store(tmp_path, monkeypatch, members, dir_mode=0o750)
    elif damage == "pool-file-group-readable":
        _store(tmp_path, monkeypatch, members, pool_mode=0o640)
    else:
        active = _store(tmp_path, monkeypatch, members)
        link = tmp_path / "linked-store"
        link.symlink_to(active.parent)
        monkeypatch.setenv(al.STORE_PATH_ENV, str(link / al.STORE_BASENAME))
    assert al.pool_tokens() == []


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root reads a mode-000 directory")
def test_an_unreadable_store_is_an_empty_pool(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch, {"acct-a": TOKEN_A}, dir_mode=0o000)
    try:
        assert al.pool_tokens() == []
    finally:
        os.chmod(tmp_path / "store", 0o700)


def test_a_member_that_cannot_be_trusted_is_left_out_alone(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    active = _store(tmp_path, monkeypatch, {"acct-a": TOKEN_A, "acct-open": "fake-open-token",
                                            "acct-two": "fake two tokens", "acct-empty": "",
                                            "acct-b": TOKEN_B},
                    pool=["acct-a", "acct-open", "acct-two", "acct-empty", "acct-link",
                          "acct-missing", "acct-b"])
    os.chmod(active.with_name(f"{active.name}.acct-open"), 0o644)
    active.with_name(f"{active.name}.acct-link").symlink_to(
        active.with_name(f"{active.name}.acct-a"))
    assert al.pool_tokens() == [("acct-a", TOKEN_A), ("acct-b", TOKEN_B)]
    assert "fake-open-token" not in caplog.text and "fake two tokens" not in caplog.text


def test_a_metered_key_is_never_pool_capacity(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch, {"acct-a": TOKEN_A, "acct-paid": TOKEN_PAID})
    billing = {TOKEN_PAID: "metered"}
    assert al.pool_tokens(billing=lambda token: billing.get(token, "unknown")) == [
        ("acct-a", TOKEN_A)]


def test_a_failing_classifier_fails_the_whole_pool_closed(tmp_path, monkeypatch, caplog):
    _store(tmp_path, monkeypatch, {"acct-a": TOKEN_A})

    def broken(token):
        raise RuntimeError(f"cannot classify {token}")

    assert al.pool_tokens(billing=broken) == []
    assert "RuntimeError" in caplog.text and TOKEN_A not in caplog.text


def test_a_pool_account_never_shows_its_token():
    account = PoolAccount("acct-a", TOKEN_A)
    label, token = account
    assert (label, token, account.label, account.token) == ("acct-a", TOKEN_A, "acct-a", TOKEN_A)
    for shown in (repr(account), str(account), f"{account}", repr([account])):
        assert TOKEN_A not in shown and "acct-a" in shown


def test_no_test_can_read_the_operators_token_store(tmp_path):
    # tests/conftest.py points GRIP's store override at a path inside this test's
    # temporary directory that does not exist; nothing here may reach the real one.
    assert tmp_path in al.store_active_path().parents
    assert al.pool_tokens() == []


def test_the_store_path_follows_grips_layout(tmp_path):
    assert al.store_active_path({"XDG_CONFIG_HOME": str(tmp_path)}) == (
        tmp_path / "grip" / "claude-tokens" / "cch-oauth-token")
    assert al.store_active_path({al.STORE_PATH_ENV: str(tmp_path / "x")}) == tmp_path / "x"


# --------------------------------------------------------------------------- #
# Which account a request is on.
# --------------------------------------------------------------------------- #
def test_the_resolver_names_pool_accounts_and_buckets_the_rest():
    reads = []

    def pool():
        reads.append(1)
        return [PoolAccount("acct-a", TOKEN_A), PoolAccount("acct-b", TOKEN_B)]

    clock = Clock(0.0)
    resolver = AccountResolver(pool, ttl=30.0, clock=clock)
    assert resolver.label_for(_bearer(TOKEN_B)) == ("acct-b", al.credential_digest(TOKEN_B))
    assert resolver.label_for([("X-Api-Key", TOKEN_A)])[0] == "acct-a"
    assert resolver.label_for(_bearer("someone-else"))[0] == UNPOOLED
    assert resolver.label_for([]) == (UNPOOLED, None)
    assert len(reads) == 1
    clock.at = 31.0
    resolver.label_for(_bearer(TOKEN_A))
    assert len(reads) == 2


def test_a_failing_pool_source_puts_every_request_on_unpooled():
    def broken():
        raise OSError("store gone")
    assert AccountResolver(broken).label_for(_bearer(TOKEN_A))[0] == UNPOOLED


# --------------------------------------------------------------------------- #
# The ledger: exchanges, fail-open, bounded waits.
# --------------------------------------------------------------------------- #
@pytest.fixture
def ledger(tmp_path):
    pool = [PoolAccount("acct-a", TOKEN_A), PoolAccount("acct-b", TOKEN_B)]
    return AccountLedger(tmp_path / "state" / "account-ledger.json",
                         resolver=AccountResolver(lambda: pool), clock=Clock())


def test_an_exchange_is_in_flight_until_answered_then_recorded(ledger):
    view = ledger.view()
    with ledger.exchange(_bearer(TOKEN_A)) as watch:
        assert view.in_flight("acct-a") == 1
        watch.answered(429, [(al.STATUS, "rejected"), (al.RESET, str(RESET_5H)),
                             (al.CLAIM, "five_hour")])
    assert view.in_flight("acct-a") == 0
    cooling = view.cooling("acct-a")
    assert (cooling.until, cooling.reason, cooling.claim) == (RESET_5H, "spent", "five_hour")
    assert view.token_health("acct-a") == "unknown"
    assert set(view.accounts()) == {"acct-a"}
    assert view.accounts()["acct-a"]["http_status"] == 429


def test_an_error_inside_the_exchange_passes_through_and_ends_the_flight(ledger):
    with pytest.raises(ConnectionResetError, match="upstream went away"):
        with ledger.exchange(_bearer(TOKEN_B)):
            raise ConnectionResetError("upstream went away")
    entry = ledger.view().accounts()["acct-b"]
    assert (entry["in_flight"], entry["observations"]) == ({}, 0)


def test_a_reading_of_anything_never_raises(ledger):
    with ledger.exchange(_bearer(TOKEN_A)) as watch:
        watch.answered("not a status", 42)
    assert ledger.view().accounts()["acct-a"]["http_status"] is None


def test_a_ledger_that_cannot_be_written_never_fails_the_request(tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    ledger = AccountLedger(blocker / "ledger.json", resolver=AccountResolver(list))
    with ledger.exchange(_bearer(TOKEN_A)) as watch:
        watch.answered(200, [(al.STATUS, "allowed")])
    assert ledger.view().accounts() == {}


def _hold_lock(lock_path: Path, ready: Path) -> subprocess.Popen:
    code = ("import fcntl, os, sys, time\n"
            "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "open(sys.argv[2], 'w').close()\n"
            "time.sleep(60)\n")
    holder = subprocess.Popen([sys.executable, "-c", code, str(lock_path), str(ready)])
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists(), "the lock holder did not start"
    return holder


@pytest.mark.skipif(locked_json.fcntl is None, reason="needs fcntl")
def test_a_held_lock_skips_the_write_within_its_bound(tmp_path):
    path = tmp_path / "ledger.json"
    holder = _hold_lock(path.with_name(path.name + ".lock"), tmp_path / "ready")
    try:
        ledger = AccountLedger(path, resolver=AccountResolver(list), lock_timeout=0.2,
                               begin_timeout=0.05)
        started = time.monotonic()
        with ledger.exchange(_bearer(TOKEN_A)) as watch:
            watch.answered(200, [(al.STATUS, "allowed")])
        assert time.monotonic() - started < 1.5
        assert not path.exists()
        with pytest.raises(locked_json.LockTimeout):
            locked_json.update_json(path, dict, lock_timeout=0.05)
    finally:
        holder.kill()
        holder.wait()


# --------------------------------------------------------------------------- #
# locked_json: whole or not at all.
# --------------------------------------------------------------------------- #
def test_an_update_replaces_the_file_whole_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "deep" / "state.json"
    assert locked_json.update_json(path, lambda data: {**data, "a": 1}) == {"a": 1}
    assert locked_json.update_json(path, lambda data: {**data, "b": 2}) == {"a": 1, "b": 2}
    assert json.loads(path.read_text()) == {"a": 1, "b": 2}
    assert oct(path.stat().st_mode & 0o777) == oct(0o600)
    assert sorted(p.name for p in path.parent.iterdir()) == ["state.json", "state.json.lock"]


def test_a_failed_update_changes_nothing(tmp_path):
    path = tmp_path / "state.json"
    locked_json.update_json(path, lambda data: {"kept": True})

    def boom(_data):
        raise ValueError("refused")

    with pytest.raises(ValueError):
        locked_json.update_json(path, boom)
    with pytest.raises(TypeError):
        locked_json.update_json(path, lambda data: {"bad": object()})
    assert json.loads(path.read_text()) == {"kept": True}
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("content", [b"", b"{torn", b"[1, 2]", b"\xff\xfe"])
def test_an_unreadable_file_reads_as_empty(tmp_path, content):
    path = tmp_path / "state.json"
    path.write_bytes(content)
    assert locked_json.read_json(path) == {}
    assert locked_json.read_json(tmp_path / "missing.json") == {}


# --------------------------------------------------------------------------- #
# Concurrent writers, in separate processes.
# --------------------------------------------------------------------------- #
WRITERS, EXCHANGES = 6, 40


def _exchanges(path_text: str, count: int) -> int:
    """Run in a spawned process: ``count`` exchanges, each a count and a record."""
    ledger = AccountLedger(path_text, resolver=AccountResolver(list), lock_timeout=60.0,
                           begin_timeout=60.0)
    for _ in range(count):
        with ledger.exchange(_bearer(TOKEN_A)) as watch:
            watch.answered(200, [(al.STATUS, "allowed"), (f"{P}5h-utilization", "0.5")])
    return count


def _watch_file(path_text: str, stop_text: str) -> tuple:
    """Run in a spawned process: read the file until told to stop; count torn reads."""
    path, stop = Path(path_text), Path(stop_text)
    torn = reads = 0
    deadline = time.monotonic() + 120
    while not stop.exists() and time.monotonic() < deadline:
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            continue
        reads += 1
        try:
            json.loads(raw)
        except ValueError:
            torn += 1
    return torn, reads


def test_concurrent_writers_lose_nothing_and_no_reader_sees_half_a_file(tmp_path):
    path, stop = tmp_path / "ledger.json", tmp_path / "stop"
    with mp.get_context("spawn").Pool(WRITERS + 1) as pool:
        watcher = pool.apply_async(_watch_file, (str(path), str(stop)))
        writers = [pool.apply_async(_exchanges, (str(path), EXCHANGES)) for _ in range(WRITERS)]
        done = sum(writer.get(timeout=180) for writer in writers)
        stop.touch()
        torn, reads = watcher.get(timeout=180)
    entry = LedgerView(path).accounts()[UNPOOLED]
    assert done == WRITERS * EXCHANGES
    assert entry["observations"] == WRITERS * EXCHANGES  # not one update lost
    assert entry["in_flight"] == {}                      # every count was settled
    assert (torn, reads > 0) == (0, True)


# --------------------------------------------------------------------------- #
# No token or header value in any sink.
# --------------------------------------------------------------------------- #
SEEDED_HEADERS = [
    (al.STATUS, "SEEDED-STATUS-VALUE"), (al.CLAIM, "SEEDED-CLAIM-VALUE"),
    (al.OVERAGE_DISABLED_REASON, "SEEDED-REASON-VALUE"),
    (f"{P}5h-utilization", "SEEDED-UTIL-VALUE"), (f"{P}new-name", "SEEDED-NEW-VALUE"),
    ("www-authenticate", "SEEDED-AUTH-HEADER-VALUE"), ("request-id", "SEEDED-REQUEST-ID"),
]
SEEDED = (TOKEN_A, TOKEN_B, "seeded-client-bearer", "seeded-client-key", "fake-open-token",
          *(value for _, value in SEEDED_HEADERS))


def _every_sink(root: Path, skip: Path) -> str:
    return "".join(path.read_text(errors="replace") for path in root.rglob("*")
                   if path.is_file() and skip not in path.parents)


def _exchange_and_fail(ledger: AccountLedger, bearer: str, status: int) -> str:
    try:
        with ledger.exchange(_bearer(bearer) + [("x-api-key", "seeded-client-key")]) as watch:
            watch.answered(status, SEEDED_HEADERS)
            raise RuntimeError("the request failed")
    except RuntimeError as exc:
        return str(exc)
    return ""


def test_no_token_or_header_value_reaches_any_sink(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    active = _store(tmp_path, monkeypatch, {"acct-a": TOKEN_A, "acct-b": TOKEN_B,
                                            "acct-open": "fake-open-token"})
    os.chmod(active.with_name(f"{active.name}.acct-open"), 0o644)
    ledger = AccountLedger(tmp_path / "hal" / "account-ledger.json")  # reads the store above
    errors = [_exchange_and_fail(ledger, bearer, status) for bearer, status in
              itertools.product((TOKEN_A, "seeded-client-bearer"), (429, 401, 200))]

    def broken(_token):
        raise RuntimeError("no")

    al.pool_tokens(billing=broken)
    with AccountLedger(active / "no" / "x.json", resolver=AccountResolver(list)).exchange(
            _bearer(TOKEN_B)):
        pass
    assert ledger.view().accounts()["acct-a"]["observations"] == 3
    assert ledger.view().accounts()["acct-a"]["quota"]["header_names"]
    # Case-blind: the parser lower-cases words, so an echoed value would be lower-case.
    sinks = (_every_sink(tmp_path, active.parent) + caplog.text + "".join(errors)).lower()
    for secret in SEEDED:
        assert secret.lower() not in sinks, secret


# --------------------------------------------------------------------------- #
# The command line.
# --------------------------------------------------------------------------- #
def test_the_command_line_shows_each_account(ledger, tmp_path, capsys):
    with ledger.exchange(_bearer(TOKEN_A)) as watch:
        watch.answered(429, [(al.STATUS, "rejected"), (al.RESET, str(RESET_5H)),
                             (f"{P}5h-utilization", "1"), (f"{P}5h-reset", str(RESET_5H))])
    line = al.describe(ledger.view(), now=NOW)[0]
    assert line.startswith("acct-a  status rejected  5h 100% (resets 2026-10-05")
    assert "cooling spent until 2026-10-05" in line and "token unknown" in line
    assert "7d unknown" in line and "in flight 0" in line and "headers seen 4" in line
    assert al.main(["--path", str(ledger.path)]) == 0
    assert capsys.readouterr().out.startswith("acct-a  status rejected")
    assert al.main(["--path", str(ledger.path), "--json"]) == 0
    stored = json.loads(capsys.readouterr().out)
    assert stored["accounts"]["acct-a"]["quota"]["status"] == "rejected"
    assert al.main(["--path", str(tmp_path / "none.json")]) == 0
    assert "no account observed yet" in capsys.readouterr().out
