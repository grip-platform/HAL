"""account_switch -- carry a turn whose account is spent, or whose token is dead, on a pool account that can serve it.

The InfiniteMachine contract (``docs/infinite-machine-contract.md``, "The account
switch") puts an account switch in front of the spend guard: when one account's
limit is reached, the turn moves to another account, and paid extra usage is the
fallback, not the first answer. Until 2026-10-06 the guard shipped without it, so a
spent account simply had its turns refused. This module is the minimal switch
(Step 5e, before sending and after a rejected turn) and, since HAL#688, the
dead-token switch (Step 5c: a token that answered 401).

A 403 never switches. It is a permission answer, which can be an enforcement
action, and moving a turn to another account after one is the shape Anthropic's
Usage Policy names. An explicit revocation or suspension on ANY account, a 401 or
a 403 alike (``lib/hal/auth_signal.py``), stops all switching: see "The
enforcement stop" below.

Before the wire relays a decrypted ``POST /v1/messages`` that bills a subscription,
it asks :func:`choose` whether another account should carry it. The answer is ranked,
subscription first, as the operator's billing order puts it:

1. the request's own account while it still has subscription capacity: no switch;
2. else the first account in GRIP's pool (``grip-relay-bill pool``, in that order)
   with subscription capacity: switch to it;
3. else, only when the spend guard has overage opted in: the request's own account if
   its overage still answers (no switch), then the first pool account whose overage
   still answers;
4. else no switch, and the spend guard gives its honest refusal.

A dead token (``reason="dead_token"``) cannot be billed at all, so the own-account
steps of rules 1 and 3 never keep it: it goes straight to rule 2, then to rule 3's
pool step, in the same billing order.

Capacity is read from two marks, both keyed by a digest of the token, never the token.
The spend guard marks an account *spent* when a 2xx showed paid extra usage, so its
subscription is used up but its overage works. :func:`observe` marks an account
*exhausted* when a 429 says its limit is reached (the unified rejected status,
representative claim, or overage status): neither its subscription nor its
overage answers until the reset that 429 names. Exhausted marks live in this process.
Capacity is kept by organisation: accounts in one organisation share limits, so a
mark on one account counts for every account whose KNOWN organisation is the same
(:func:`exhausted_until`, :func:`spent_until`). An unknown organisation changes
nothing: such an account answers for its own marks only.
Token health is not one of these marks. The account ledger records, per account,
whether the token it last judged is dead (it answered 401, or a 403 that named a
suspension), with a digest of that token, never the token; a different token reads
as unknown (``account_ledger.token_health_of``). :func:`choose` reads it from there.

A replacement must have a known organisation. For a limit switch it must differ
from the request's own, because accounts in one organisation share limits, so an
unknown own organisation prevents the switch. For a dead token (it answered 401)
any known organisation will do: an authentication failure is not a shared
limit, and the dead token's own lookup fails with the same 401. The wire reads
``anthropic-organization-id`` from a free ``GET /v1/models`` through its own
upstream path, retrying once on a transport failure. A known organisation is
cached in-process by token digest; an unknown one is retried after
``UNKNOWN_ORG_RETRY_SECONDS``. An empty pool sends no probe, and a pool token the
account ledger holds as dead is never chosen.

Replay boundary and remaining limits:

* The wire can replay a limit-reached 429, or a 401 on the presented token that
  names no enforcement action, once, before any response byte reaches the client,
  with only the bearer replaced. A 403 on the presented token is never replayed:
  it reaches the client unchanged. A thinking-signature rejection, or a 401 or 403
  from the replacement, restores the original answer; thinking blocks are never
  modified.
* Holds with pings (Step 5d) live in ``lib/hal/wire_hold.py`` and use this module
  to switch. Outside a hold, without a replacement or after response bytes have
  been sent, the original response still reaches the client.
* It never touches ``/v1/messages/batches``, ``count_tokens``, ``/v1/code/*``, OAuth
  or any other account-bound route, and it never presents a metered key: the pool
  leaves metered keys out (``account_ledger.pool_tokens``).

The off-switch is a file, ``$HAL_HOME/state/hal/account-switch.off``: while it
exists nothing is switched, from the next request. A marker that cannot be checked
counts as present, so the brake holds when in doubt. ``python -m
lib.hal.account_switch off|on|status`` sets and reads it.

The enforcement stop. When any account is answered a 401 or 403 whose message
names a revocation or a suspension, :func:`stop_for_enforcement` engages that same
off-switch, writes into the marker why (the account's label, the status, the error
type, the word that matched, and when; never a token or a header value), and tells
the operator once, through HAL's notification hub (``lib/hal/notification.py``),
whose history is kept in ``$HAL_HOME/state/hal/notifications/history.json``.
Switching stays off until the operator runs ``python -m lib.hal.account_switch on``;
``status`` prints the recorded reason, and ``off`` leaves it in place. A 401 that
names only an expiry is no enforcement action and keeps the dead-token switch.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from lib.hal import spend_guard, account_ledger
from lib.hal.notification import NotificationHub
from lib.hal.wire_classify import header_value

MESSAGES_PATH = "/v1/messages"
OFF_MARKER_NAME = "account-switch.off"
#: How a choice was ranked: another account's subscription, or its overage.
SUBSCRIPTION_RANK = "subscription"
OVERAGE_RANK = "overage"

_STATUS = "anthropic-ratelimit-unified-status"
_RESET = "anthropic-ratelimit-unified-reset"
#: How long an exhausted mark holds when the 429 named no usable reset.
DEFAULT_EXHAUSTED_SECONDS = spend_guard.DEFAULT_SPENT_SECONDS

_LOCK = threading.Lock()
#: account key -> Unix seconds until which that account answers 429.
_EXHAUSTED: dict = {}

_ORG_LOCK = threading.Lock()
#: account key -> its known organisation, kept for the life of the process.
_ORGS: dict[str, Optional[str]] = {}
#: account key -> monotonic time after which a failed lookup may be retried.
_ORG_RETRY: dict[str, float] = {}
#: How long a failed or empty organisation lookup stands before it is retried.
UNKNOWN_ORG_RETRY_SECONDS = 60.0

#: How an enforcement stop names itself in the off marker.
ENFORCEMENT_STOP = "enforcement"
#: The notification an enforcement stop sends through HAL's hub.
STOP_EVENT = "account_switch.stopped"
_OFF_TEXT = "account switching is off; remove this file to turn it on\n"
_STOP_LOCK = threading.Lock()
#: Set when a stop could not be written to the marker. Nothing on disk would then
#: hold the brake, so this process keeps switching off until it restarts.
_UNRECORDED_STOP = threading.Event()
#: HAL's notification hub (``lib/hal/notification.py``): a stop is told through it.
NOTICES = NotificationHub()

Choice = Tuple[str, str, str]   # (label, token, rank)


def _read_pool() -> List[Tuple[str, str]]:
    """GRIP's pool, in order. Lazy, like the wire's other ledger imports."""
    from lib.hal.account_ledger import pool_tokens
    return [(account[0], account[1]) for account in pool_tokens()]


# --------------------------------------------------------------------------- #
# The off-switch, and the enforcement stop.
# --------------------------------------------------------------------------- #
def off_marker() -> Path:
    """``$HAL_HOME/state/hal/account-switch.off``, resolved at call time."""
    home = os.environ.get("HAL_HOME") or os.path.expanduser("~/.hal")
    return Path(home) / "state" / "hal" / OFF_MARKER_NAME


def switching_on() -> bool:
    """True unless the off marker exists, or this process made a stop it could not
    write down. ``os.stat``, not ``Path.exists``: Python 3.14's ``exists`` returns
    False on a permission error, which would release the brake exactly when its
    state cannot be read."""
    if _UNRECORDED_STOP.is_set():
        return False
    try:
        os.stat(off_marker())
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _write_marker(text: str) -> None:
    """Write the marker whole: a reader sees the old one or the new, never half."""
    marker = off_marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    partial = marker.with_name(f".{marker.name}.{os.getpid()}.{threading.get_ident()}")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, marker)


def set_switching(on: bool) -> None:
    """Remove (on) or create (off) the marker. Turning switching off by hand keeps
    an enforcement stop's record in place, so its reason is never lost."""
    if on:
        _UNRECORDED_STOP.clear()
        try:
            off_marker().unlink()
        except FileNotFoundError:
            pass
        return
    if off_reason() is None:
        _write_marker(_OFF_TEXT)


def off_reason() -> Optional[dict]:
    """The enforcement stop the marker records, or None: switching is on, it was
    turned off by hand, or the marker cannot be read as such a record."""
    try:
        record = json.loads(off_marker().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(record, dict) and record.get("stopped_by") == ENFORCEMENT_STOP:
        return record
    return None


def describe_stop(record: dict) -> str:
    """A recorded stop, in two sentences for the operator."""
    word = f", naming '{record['word']}'" if record.get("word") else ""
    return (f"Account switching stopped at {record.get('at', 'an unknown time')}: account "
            f"'{record.get('account', 'unknown')}' was answered {record.get('status')} "
            f"{record.get('error_type')}{word}, which may be an enforcement action. Check "
            "that account, then turn switching back on: python -m lib.hal.account_switch on")


def _stamp(now: Optional[datetime]) -> str:
    moment = now if now is not None else datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stop_for_enforcement(signal: Any, account: str, *,
                         now: Optional[datetime] = None) -> bool:
    """Stop all account switching because ``account`` was answered ``signal``: a 401
    or 403 whose message names a revocation or a suspension (``lib.hal.auth_signal``).
    True when this call made the stop.

    The marker records why from closed lists and the account's label only: never a
    token or a header value. The operator is told once per stop: a marker that
    already records one is left alone, so its first reason stands, and the stop lasts
    until ``python -m lib.hal.account_switch on``. A marker written by hand (``off``)
    is replaced by the record, so the operator still hears of it. If the marker
    cannot be written, this process keeps switching off until it restarts. Never
    raises: a stop must never break the relay that saw it.
    """
    try:
        with _STOP_LOCK:
            if off_reason() is not None or _UNRECORDED_STOP.is_set():
                return False
            record = {"stopped_by": ENFORCEMENT_STOP, "account": str(account)[:64],
                      **signal.fields(), "at": _stamp(now)}
            try:
                _write_marker(json.dumps(record, sort_keys=True) + "\n")
            except OSError:
                _UNRECORDED_STOP.set()
        _notify(record)
        return True
    except Exception:  # noqa: BLE001 -- see the docstring; the brake still holds
        _UNRECORDED_STOP.set()
        return False


def _notify(record: dict) -> None:
    """Tell the operator through HAL's notification hub and keep its history under
    ``$HAL_HOME``. Never raises: telling must never break the relay."""
    try:
        NOTICES.notify(STOP_EVENT, describe_stop(record), severity="error", **record)
        NOTICES.save_history()
    except Exception:  # noqa: BLE001 -- see the docstring
        pass


# --------------------------------------------------------------------------- #
# Exhausted accounts.
# --------------------------------------------------------------------------- #
def _seconds(now: Optional[datetime]) -> float:
    return now.timestamp() if now is not None else time.time()


def _reset_epoch(headers: Any, current: float) -> float:
    try:
        reset = float(header_value(headers, _RESET).strip())
    except ValueError:
        reset = 0.0
    return reset if reset > current else current + DEFAULT_EXHAUSTED_SECONDS


def limit_reached(status: Any, headers: Any) -> bool:
    """The shared usage-limit predicate; a plain throttle is not exhaustion."""
    return int(status) == 429 and bool(
        header_value(headers, _STATUS).strip().lower() == "rejected"
        or header_value(headers, "anthropic-ratelimit-unified-representative-claim")
        or header_value(headers, "anthropic-ratelimit-unified-overage-status"))


def observe(status: Any, response_headers: Any, request_headers: Any, *,
            now: Optional[datetime] = None) -> Optional[float]:
    """Mark the request's account exhausted when a 429 says its limit is reached.

    Returns the Unix second the mark holds until, or None. Never raises: an
    observation must never break the relay that called it.
    """
    try:
        if not limit_reached(status, response_headers):
            return None
        kind, account = spend_guard.anthropic_credential(request_headers)
        if kind != spend_guard.SUBSCRIPTION:
            return None
        until = _reset_epoch(response_headers, _seconds(now))
        with _LOCK:
            _EXHAUSTED[account] = max(until, _EXHAUSTED.get(account, 0.0))
        return until
    except Exception:  # noqa: BLE001 -- see the docstring
        return None


def _org_siblings(account: str) -> List[str]:
    """``account``, then every account whose KNOWN organisation is its own. Accounts
    in one organisation share limits, so a mark on any of them counts for all; an
    unknown organisation changes nothing."""
    with _ORG_LOCK:
        org = _ORGS.get(account)
        if org is None:
            return [account]
        return [account] + [key for key, value in _ORGS.items()
                            if value == org and key != account]


def exhausted_until(account: str, *, now: Optional[datetime] = None) -> Optional[float]:
    """Until when ``account`` answers 429: the latest exhausted mark on it or on any
    account of its known organisation, or None."""
    keys = _org_siblings(account)
    with _LOCK:
        until = max(_EXHAUSTED.get(key, 0.0) for key in keys)
    return until if until > _seconds(now) else None


def spent_until(account: str, *, now: Optional[datetime] = None) -> Optional[float]:
    """The spend guard's spent mark on ``account`` or on any account of its known
    organisation: the latest, or None."""
    marks = [spend_guard.spent_until(key, now=now) for key in _org_siblings(account)]
    marks = [mark for mark in marks if mark is not None]
    return max(marks) if marks else None


def forget_exhausted() -> None:
    """Drop every exhausted mark (tests, and an operator who knows better)."""
    with _LOCK:
        _EXHAUSTED.clear()


def _has_subscription(account: str, now: Optional[datetime]) -> bool:
    return (spent_until(account, now=now) is None
            and exhausted_until(account, now=now) is None)


def _overage_answers(account: str, now: Optional[datetime]) -> bool:
    return exhausted_until(account, now=now) is None


# --------------------------------------------------------------------------- #
# The choice.
# --------------------------------------------------------------------------- #
def _is_turn(method: str, target: str) -> bool:
    if str(method).upper() != "POST":
        return False
    text = str(target)
    bare = urlsplit(text).path if "://" in text else text.split("?", 1)[0]
    return bare == MESSAGES_PATH


def _bearer(headers: Any) -> str:
    auth = header_value(headers, "authorization").strip()
    return auth[7:].strip() if auth[:7].lower() == "bearer " else ""


def _organisation(token: str, probe: Optional[Callable[[str], Optional[str]]],
                  now: Optional[float] = None) -> Optional[str]:
    """The organisation behind ``token``, or None when unknown; keyed by digest only.

    A known organisation is read once per token and kept. A failed or empty
    lookup is unknown, never different, and stands for
    ``UNKNOWN_ORG_RETRY_SECONDS`` before it is retried: a lookup that failed
    during an outage must not disable switching for that token for the rest
    of the process, which is when switching matters most. With no ``probe``
    only a known organisation is returned, and nothing is recorded: no lookup
    was made, so none failed, and a later probe is not held back (HAL#770).
    """
    key = spend_guard.account_key(token)
    clock = time.monotonic() if now is None else now
    with _ORG_LOCK:
        if key in _ORGS:
            return _ORGS[key]
        if probe is None or clock < _ORG_RETRY.get(key, float("-inf")):
            return None
        try:
            found = (probe(token) or "").strip() or None
        except Exception:  # a failed lookup is unknown, never different
            found = None
        if found is None:
            _ORG_RETRY[key] = clock + UNKNOWN_ORG_RETRY_SECONDS
        else:
            _ORGS[key] = found
            _ORG_RETRY.pop(key, None)
        return found


def _first(candidates: Iterable[Tuple[str, str]], usable: Callable[[str], bool],
           rank: str) -> Optional[Choice]:
    for label, token in candidates:
        if usable(spend_guard.account_key(token)):
            return label, token, rank
    return None


def _keeps_own(kind: str, account: str, dead: bool, after_throttle: bool,
               now: Optional[datetime]) -> bool:
    """True when the request must stay on its own credential: it is not a
    subscription, or a limit switch outside a throttle finds its own subscription
    still has capacity. A dead token cannot be billed at all, so its account's
    remaining capacity never keeps it."""
    if kind != spend_guard.SUBSCRIPTION:
        return True
    return not dead and not after_throttle and _has_subscription(account, now)


def _replacements(own: str, dead: bool,
                  pool: Optional[Callable[[], Sequence[Tuple[str, str]]]],
                  probe_org: Optional[Callable[[str], Optional[str]]]) -> List[Tuple[str, str]]:
    """The pool tokens that may serve instead, in pool order.

    Never the request's own token, and never one the account ledger holds as dead
    (it answered 401, or a 403 that named a suspension); a re-login presents a
    different token, which the ledger reads as unknown, so the account returns on
    its own. A replacement must belong to a known organisation: for a limit
    switch one other than the
    request's own, because accounts in one organisation share limits; for a dead
    token any, because an authentication failure is not a shared limit (and the
    dead token's own lookup fails with the same 401). The request's own
    organisation is probed only when a candidate exists.
    """
    view = account_ledger.LedgerView()
    others = [(label, token) for label, token in (pool or _read_pool)()
              if token != own and view.token_health(label, token) != "dead"]
    if not others:
        return []
    excluded = {None}
    if not dead:
        own_org = _organisation(own, probe_org)
        if own_org is None:
            return []
        excluded.add(own_org)
    return [(label, token) for label, token in others
            if _organisation(token, probe_org) not in excluded]


def _by_billing_order(others: Sequence[Tuple[str, str]], account: str, dead: bool,
                      now: Optional[datetime]) -> Optional[Choice]:
    """Subscription capacity first. Overage only where the spend guard opts it in,
    and, for a limit switch, only when the request's own overage cannot answer."""
    found = _first(others, lambda key: _has_subscription(key, now), SUBSCRIPTION_RANK)
    if found is not None:
        return found
    if not spend_guard.decide_paid(spend_guard.OVERAGE, now=now).allowed:
        return None
    if not dead and _overage_answers(account, now):
        return None
    return _first(others, lambda key: _overage_answers(key, now), OVERAGE_RANK)


def choose(method: str, target: str, headers: Any, *,
           pool: Optional[Callable[[], Sequence[Tuple[str, str]]]] = None,
           after_throttle: bool = False,
           reason: str = "limit",
           now: Optional[datetime] = None,
           probe_org: Optional[Callable[[str], Optional[str]]] = None) -> Optional[Choice]:
    """``(label, token, rank)`` to bill this request instead, or None to keep it.

    ``reason`` is ``"limit"`` (the default: the account is spent) or
    ``"dead_token"`` (its token answered a 401 that named no enforcement action);
    any other keeps the request. A 403 is never a reason: the wire never asks.
    ``after_throttle`` lets the wire's bounded hold try another subscription after
    repeated headerless 429s without marking the account spent; ``probe_org``
    reads a token's organisation. Rules: :func:`_keeps_own`, :func:`_replacements`
    and :func:`_by_billing_order`. The off-switch is read again before a choice is
    returned: an organisation lookup on the way can itself meet an enforcement
    answer and stop switching. Never raises: on any surprise the request keeps
    its own account, and the spend guard still decides whether it may be sent.
    """
    try:
        if (reason not in ("limit", "dead_token") or not _is_turn(method, target)
                or not switching_on()):
            return None
        dead = reason == "dead_token"
        kind, account = spend_guard.anthropic_credential(headers)
        if _keeps_own(kind, account, dead, after_throttle, now):
            return None
        others = _replacements(_bearer(headers), dead, pool, probe_org)
        choice = _by_billing_order(others, account, dead, now) if others else None
        return choice if switching_on() else None
    except Exception:  # noqa: BLE001 -- see the docstring
        return None


# --------------------------------------------------------------------------- #
# The brake's command line.
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "status"
    if command in ("on", "off"):
        set_switching(command == "on")
    elif command != "status":
        print("usage: python -m lib.hal.account_switch [on|off|status]", file=sys.stderr)
        return 2
    print(f"account switching is {'on' if switching_on() else 'off'} ({off_marker()})")
    reason = off_reason()
    if reason is not None:
        print(describe_stop(reason))
    return 0


if __name__ == "__main__":
    sys.exit(main())
