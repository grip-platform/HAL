"""account_ledger -- each Claude account's limits, as its own answers report them.

InfiniteMachine v2, Step 5b (``docs/infinite-machine-contract.md``, "The per-account
ledger"). api.anthropic.com answers every subscription request, a success or a
failure, with ``anthropic-ratelimit-unified-*`` headers (fleet findings f19, f22).
This module reads them into numbers and keeps one entry per account:

- status (``allowed``, ``allowed_warning``, ``rejected``), the representative claim
  (which limit speaks for the account) and the reset it names;
- each window's use, reset and warning threshold: 5-hour, 7-day, the 7-day window
  that includes overage, and the overage window itself (paid extra usage);
- overage state: its status, reset, whether it is in use, and why it is disabled;
- token health: ``ok`` after a 2xx, ``dead`` (needs a re-login) after a 401,
  ``unknown`` before either and whenever a different token is presented. A 403
  alone changes nothing: it is a permission answer (a model the account may not
  use, for one), not a dead token. The wire holds a 403's token dead only when
  its message names a suspension (:meth:`AccountLedger.mark_dead`);
- requests in flight: sent on the account, answer not yet begun -- the requests a
  hold would be waiting on;
- cooling: until the reset the headers name.

It observes and records. It decides nothing: which account serves a request, and
when to switch, is Step 5c, which reads this ledger through :class:`LedgerView`.

The header names
================
Taken from Claude Code 2.1.289's own parser of these headers (the function that
reads every streamed reply's headers, and its per-window helper, read from the
installed binary on 2026-10-05) and from the gateway protocol document embedded in
the same binary, which says they are "the same ``anthropic-ratelimit-unified-*``
headers api.anthropic.com sends its subscribers". The per-window headers are built
from an abbreviation: ``-5h-``, ``-7d-``, ``-7d_oi-``, ``-overage-``. No live
response was read for this module. To make the names checkable against real
traffic, each entry records which unified header NAMES its last answer carried
(``quota.header_names``; never a value).

Unknown is never 0
==================
A missing header, a value outside its closed vocabulary, a number that does not
parse, is out of range, or is sent twice with two different values -- each is
recorded as unknown (``None``, or ``"unknown"``/``"other"`` for words), never as
0 and never as the raw text. Claude Code itself reads a missing status as
``allowed`` (f19); the ledger does not.

Accounts
========
An account is a label in GRIP's token store (``lib/claude_token_store.py`` in GRIP
owns its layout): ``cch-oauth-token.<label>`` per account and ``cch-oauth-token.pool``
for the pool's order. :func:`pool_tokens` reads it, read-only, and fails closed. A
request is put on the account whose pool token it presents; any other credential --
usually Claude Code's own login -- is put on :data:`UNPOOLED`.

Custody
=======
No token, bearer or header value is written to any log, file or exception text.
The ledger keeps parsed numbers and closed-vocabulary words, header names, and
twelve hex characters of a token's SHA-256 (which token a health verdict is about;
GRIP's ``grip-claude-as --check`` shows the operator the same digest). A
:class:`PoolAccount`'s ``repr`` withholds its token.

Writes go through ``lib/hal/locked_json.py``: whole or not at all, one writer at a
time, across threads and processes. Every write on a request path is fail-open and
bounded in time: the ledger never fails, and never stalls, the traffic it records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import stat
import sys
import threading
import time
import urllib.error
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from lib.hal.locked_json import read_json, update_json

_log = logging.getLogger("hal.account_ledger")

# --------------------------------------------------------------------------- #
# The header names.
# --------------------------------------------------------------------------- #
PREFIX = "anthropic-ratelimit-unified-"
STATUS = PREFIX + "status"
RESET = PREFIX + "reset"
CLAIM = PREFIX + "representative-claim"
OVERAGE_STATUS = PREFIX + "overage-status"
OVERAGE_RESET = PREFIX + "overage-reset"
OVERAGE_IN_USE = PREFIX + "overage-in-use"
OVERAGE_DISABLED_REASON = PREFIX + "overage-disabled-reason"
RETRY_AFTER = "retry-after"
#: Each window, and the abbreviation in its three headers:
#: ``<PREFIX><abbreviation>-utilization``, ``-reset`` and ``-surpassed-threshold``.
WINDOWS = (("five_hour", "5h"), ("seven_day", "7d"),
           ("seven_day_overage_included", "7d_oi"), ("overage", "overage"))

UNKNOWN = "unknown"
OTHER = "other"
STATUSES = frozenset({"allowed", "allowed_warning", "rejected"})
#: The representative claim, by its own name or by its window's abbreviation.
CLAIMS = {"five_hour": "five_hour", "seven_day": "seven_day",
          "seven_day_opus": "seven_day_opus", "seven_day_sonnet": "seven_day_sonnet",
          "seven_day_overage_included": "seven_day_overage_included",
          "overage": "overage",
          **{abbreviation: window for window, abbreviation in WINDOWS}}
#: Why overage is disabled: every value Claude Code 2.1.289 names.
DISABLED_REASONS = frozenset({
    "out_of_credits", "overage_not_provisioned", "org_level_disabled",
    "org_level_disabled_until", "org_service_level_disabled", "org_spend_cap_reached",
    "seat_tier_level_disabled", "seat_tier_zero_credit_limit", "member_level_disabled",
    "member_zero_credit_limit", "group_zero_credit_limit", "no_limits_configured",
    "fetch_error", "unknown"})

_NUMBER = re.compile(r"[0-9]{1,12}(?:\.[0-9]{1,12})?\Z")
_SECONDS = re.compile(r"[0-9]{1,9}\Z")
_HEADER_NAME = re.compile(r"anthropic-ratelimit-unified-[a-z0-9_-]{1,64}\Z")
_MAX_VALUE = 64
_MAX_NAMES = 40
_EPOCH_RANGE = (1_000_000_000, 100_000_000_000)  # 2001 to the year 5138
_FRACTION_RANGE = (0.0, 100.0)
_MAX_RETRY_AFTER_S = 30 * 86400


# --------------------------------------------------------------------------- #
# Reading one answer's headers.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Window:
    """One limit window: use as a fraction, its reset (epoch seconds), its warning level."""

    utilization: Optional[float] = None
    reset: Optional[int] = None
    surpassed: Optional[float] = None

    def to_json(self) -> dict:
        return {"utilization": self.utilization, "reset": self.reset,
                "surpassed": self.surpassed}


@dataclass(frozen=True)
class Reading:
    """What one answer's headers said about its account. Every field may be unknown."""

    status: str = UNKNOWN
    reset: Optional[int] = None
    claim: Optional[str] = None
    windows: Tuple[Tuple[str, Window], ...] = ()
    overage_status: str = UNKNOWN
    overage_reset: Optional[int] = None
    overage_in_use: Optional[bool] = None
    overage_disabled_reason: Optional[str] = None
    retry_after_s: Optional[int] = None
    header_names: Tuple[str, ...] = ()

    @property
    def has_quota(self) -> bool:
        """True when the answer carried any unified rate-limit header at all."""
        return bool(self.header_names)

    def window(self, name: Optional[str]) -> Optional[Window]:
        return dict(self.windows).get(name) if name else None

    def to_json(self) -> dict:
        return {"status": self.status, "reset": self.reset, "claim": self.claim,
                "windows": {name: window.to_json() for name, window in self.windows},
                "overage": {"status": self.overage_status, "reset": self.overage_reset,
                            "in_use": self.overage_in_use,
                            "disabled_reason": self.overage_disabled_reason},
                "retry_after_s": self.retry_after_s,
                "header_names": list(self.header_names)}


def _pairs(headers: Any) -> Iterator[Tuple[str, str]]:
    """(lower-cased name, value) from a list of pairs or anything with ``items()``."""
    items = getattr(headers, "items", None)
    source = items() if callable(items) else (headers or ())
    for item in source:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            yield str(item[0]).strip().lower(), str(item[1])


def _collect(headers: Any) -> Dict[str, List[str]]:
    """The rate-limit headers only, every value of each. Nothing else is kept."""
    values: Dict[str, List[str]] = {}
    for name, value in _pairs(headers):
        if name.startswith(PREFIX) or name == RETRY_AFTER:
            values.setdefault(name, []).append(value)
    return values


def _single(values: Dict[str, List[str]], name: str) -> Optional[str]:
    """One header's value. None when absent, sent twice with two values, or unusable."""
    found = {value.strip() for value in values.get(name, ())}
    if len(found) != 1:
        return None
    text = found.pop()
    return text if 0 < len(text) <= _MAX_VALUE and text.isprintable() else None


def _number(text: Optional[str], low: float, high: float) -> Optional[float]:
    if text is None or not _NUMBER.match(text):
        return None
    value = float(text)
    return value if low <= value <= high else None


def _epoch(text: Optional[str]) -> Optional[int]:
    value = _number(text, *_EPOCH_RANGE)
    return None if value is None else int(round(value))


def _fraction(text: Optional[str]) -> Optional[float]:
    return _number(text, *_FRACTION_RANGE)


def _status(text: Optional[str]) -> str:
    word = (text or "").lower()
    return word if word in STATUSES else UNKNOWN


def _claim(text: Optional[str]) -> Optional[str]:
    return None if text is None else CLAIMS.get(text.lower(), OTHER)


def _disabled_reason(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    word = text.lower()
    return word if word in DISABLED_REASONS else OTHER


def _flag(text: Optional[str]) -> Optional[bool]:
    return {"true": True, "false": False}.get((text or "").lower())


def _retry_after(text: Optional[str], now: float) -> Optional[int]:
    """Seconds to wait: delta-seconds, or an HTTP date (a past date is 0)."""
    if text is None:
        return None
    seconds = int(text) if _SECONDS.match(text) else _seconds_until(text, now)
    if seconds is None or not 0 <= seconds <= _MAX_RETRY_AFTER_S:
        return None
    return seconds


def _seconds_until(text: str, now: float) -> Optional[int]:
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if when is None or when.tzinfo is None:
        return None
    return max(0, int(when.timestamp() - now))


def _windows(values: Dict[str, List[str]]) -> Tuple[Tuple[str, Window], ...]:
    found = []
    for window, abbreviation in WINDOWS:
        base = f"{PREFIX}{abbreviation}-"
        names = [base + part for part in ("utilization", "reset", "surpassed-threshold")]
        if not any(name in values for name in names):
            continue
        found.append((window, Window(utilization=_fraction(_single(values, names[0])),
                                     reset=_epoch(_single(values, names[1])),
                                     surpassed=_fraction(_single(values, names[2])))))
    return tuple(found)


def _names(values: Dict[str, List[str]]) -> Tuple[str, ...]:
    return tuple(sorted(name for name in values if _HEADER_NAME.match(name))[:_MAX_NAMES])


def read_headers(headers: Any, now: Optional[float] = None) -> Reading:
    """One answer's rate-limit headers as a :class:`Reading`.

    ``headers`` is a list of ``(name, value)`` pairs or anything with ``items()``.
    Names are matched without regard to case. Missing or garbled values are unknown,
    never 0, and no raw value is kept.
    """
    now = time.time() if now is None else now
    values = _collect(headers)

    def one(name: str) -> Optional[str]:
        return _single(values, name)

    return Reading(status=_status(one(STATUS)), reset=_epoch(one(RESET)),
                   claim=_claim(one(CLAIM)), windows=_windows(values),
                   overage_status=_status(one(OVERAGE_STATUS)),
                   overage_reset=_epoch(one(OVERAGE_RESET)),
                   overage_in_use=_flag(one(OVERAGE_IN_USE)),
                   overage_disabled_reason=_disabled_reason(one(OVERAGE_DISABLED_REASON)),
                   retry_after_s=_retry_after(one(RETRY_AFTER), now),
                   header_names=_names(values))


# --------------------------------------------------------------------------- #
# GRIP's pool of accounts, read-only.
# --------------------------------------------------------------------------- #
#: GRIP's override, honoured here too: it names the ACTIVE token file directly.
STORE_PATH_ENV = "GRIP_RELAY_CCH_OAUTH_PATH"
STORE_BASENAME = "cch-oauth-token"
_LABEL = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_RESERVED_LABELS = frozenset({"pool", "migrate", "forget"})
_MAX_POOL = 16
_MAX_FILE_BYTES = 4096
_GROUP_OTHER = 0o077


class PoolAccount(tuple):
    """``(label, token)``. Its ``repr`` and ``str`` withhold the token."""

    __slots__ = ()

    def __new__(cls, label: str, token: str):
        return super().__new__(cls, (label, token))

    @property
    def label(self) -> str:
        return self[0]

    @property
    def token(self) -> str:
        return self[1]

    def __repr__(self) -> str:
        return f"PoolAccount(label={self[0]!r}, token=<withheld>)"


def store_active_path(env: Optional[Mapping[str, str]] = None) -> Path:
    """GRIP's active token file: the override when set, else
    ``${XDG_CONFIG_HOME:-~/.config}/grip/claude-tokens/cch-oauth-token``."""
    env = os.environ if env is None else env
    override = (env.get(STORE_PATH_ENV) or "").strip()
    if override:
        return Path(override).expanduser()
    base = (env.get("XDG_CONFIG_HOME") or "").strip()
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "grip" / "claude-tokens" / STORE_BASENAME


def pool_tokens(env: Optional[Mapping[str, str]] = None, *,
                billing: Optional[Callable[[str], str]] = None) -> List[PoolAccount]:
    """``(label, token)`` for each account in GRIP's pool, in the pool's order. Read-only.

    Fails closed: ``[]`` when the pool cannot be trusted -- its directory or pool file
    is missing, a symlink, owned by someone else, or open to group or others -- and on
    any error. A member whose own file fails those checks, holds no credential, or
    holds a metered key (pay-per-token: never pool capacity) is left out on its own.
    The same token under two labels counts once, under the first. ``billing`` is the
    credential classifier (default: the one the signed thread rows use). Nothing read
    here is ever logged: a member left out is named by label and reason only.
    """
    try:
        return _pool(store_active_path(env), billing or _billing)
    except Exception as exc:  # noqa: BLE001 -- any surprise reads as no pool, never a crash
        _log.warning("account pool not read (%s); no pool account is available",
                     type(exc).__name__)
        return []


def _billing(token: str) -> str:
    from lib.hal.infinite_cost import billing_from_headers  # lazy: keeps imports light
    return billing_from_headers((("authorization", f"Bearer {token}"),))


def _pool(active: Path, billing: Callable[[str], str]) -> List[PoolAccount]:
    accounts: List[PoolAccount] = []
    seen = set()
    for label in _pool_labels(active):
        token = _member_token(active, label, billing)
        if token is not None and token not in seen:
            seen.add(token)
            accounts.append(PoolAccount(label, token))
    return accounts


def _pool_labels(active: Path) -> List[str]:
    text = _private_text(active.with_name(active.name + ".pool"))
    labels: List[str] = []
    for line in (text or "").splitlines():
        label = line.strip()
        if _LABEL.match(label) and label not in _RESERVED_LABELS and label not in labels:
            labels.append(label)
    return labels[:_MAX_POOL]


def _member_token(active: Path, label: str, billing: Callable[[str], str]) -> Optional[str]:
    token = (_private_text(active.with_name(f"{active.name}.{label}")) or "").strip()
    if not token or not token.isprintable() or any(ch.isspace() for ch in token):
        _log.info("pool account %r left out: no private file holding one credential", label)
        return None
    if billing(token) == "metered":
        _log.info("pool account %r left out: a metered key is never pool capacity", label)
        return None
    return token


def _owned(st: os.stat_result) -> bool:
    getuid = getattr(os, "getuid", None)  # absent on Windows: nothing to compare
    return getuid is None or st.st_uid == getuid()


def _private_dir(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return (stat.S_ISDIR(st.st_mode) and _owned(st)
            and not stat.S_IMODE(st.st_mode) & _GROUP_OTHER)


def _private_text(path: Path) -> Optional[str]:
    """The file's text when it and its directory are private to this user, else None.

    The file is opened without following a symlink and judged by the descriptor it
    opened, so it cannot be swapped between the check and the read.
    """
    if not _private_dir(path.parent):
        return None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not (stat.S_ISREG(st.st_mode) and _owned(st)
                and not stat.S_IMODE(st.st_mode) & _GROUP_OTHER):
            return None
        data = os.read(fd, _MAX_FILE_BYTES + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    if len(data) > _MAX_FILE_BYTES:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Which account a request is on.
# --------------------------------------------------------------------------- #
#: The bucket for any credential that is not a pool token.
UNPOOLED = "unpooled"


def credential_of(headers: Any) -> str:
    """The credential a request presents: its Bearer token, else its ``x-api-key``."""
    bearer = key = ""
    for name, value in _pairs(headers):
        text = value.strip()
        if name == "authorization" and not bearer and text.lower().startswith("bearer "):
            bearer = text[7:].strip()
        elif name == "x-api-key" and not key:
            key = text
    return bearer or key


def credential_digest(credential: str) -> str:
    """Twelve hex characters of SHA-256: which token a verdict is about, never the token."""
    return _sha256(credential)[:12]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AccountResolver:
    """Names the account a request's credential belongs to: a pool label, or UNPOOLED.

    The pool is read at most once per ``ttl`` seconds. Only SHA-256 digests of its
    tokens are kept between reads.
    """

    def __init__(self, pool: Callable[[], Sequence[Tuple[str, str]]] = pool_tokens, *,
                 ttl: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self._pool = pool
        self._ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._labels: Dict[str, str] = {}
        self._read_at: Optional[float] = None

    def label_for(self, headers: Any) -> Tuple[str, Optional[str]]:
        """``(label, digest)`` for the credential ``headers`` present."""
        credential = credential_of(headers)
        if not credential:
            return UNPOOLED, None
        full = _sha256(credential)
        return self._digests().get(full, UNPOOLED), full[:12]

    def _digests(self) -> Dict[str, str]:
        with self._lock:
            now = self._clock()
            if self._read_at is None or now - self._read_at >= self._ttl:
                self._labels = self._read_pool()
                self._read_at = now
            return self._labels

    def _read_pool(self) -> Dict[str, str]:
        try:
            pairs = list(self._pool())
        except Exception as exc:  # noqa: BLE001 -- a broken pool source means no pool
            _note("pool", exc)
            return {}
        labels: Dict[str, str] = {}
        for label, token in pairs:
            labels.setdefault(_sha256(token), label)
        return labels


# --------------------------------------------------------------------------- #
# The ledger's state, as pure functions of the stored object.
# --------------------------------------------------------------------------- #
LEDGER_FILE = "account-ledger.json"
SCHEMA_VERSION = 1
#: A request counted in flight longer than this is taken to be lost (its process
#: died or never closed it): longer than the gateway's 600 s upstream timeout and
#: the wire's 900 s idle timeout.
IN_FLIGHT_MAX_AGE_S = 900.0
#: A 401 says the token needs a re-login. A 403 alone does not: see the docstring.
_DEAD_STATUSES = frozenset({401})


def _new_entry() -> dict:
    return {"observed_at": None, "http_status": None, "observations": 0, "quota": None,
            "cooling": None, "token": {"health": UNKNOWN, "since": None, "digest": None},
            "in_flight": {}}


def _entry(data: dict, label: str) -> dict:
    data["version"] = SCHEMA_VERSION
    accounts = data.get("accounts")
    if not isinstance(accounts, dict):
        accounts = data["accounts"] = {}
    entry = accounts.get(label)
    if not isinstance(entry, dict):
        entry = accounts[label] = _new_entry()
    for key, value in _new_entry().items():
        entry.setdefault(key, value)
    return entry


def _flights(entry: dict) -> Dict[str, list]:
    flights = entry.get("in_flight")
    if not isinstance(flights, dict):
        flights = entry["in_flight"] = {}
    return flights


def _alive(pid_text: str) -> bool:
    """Whether the process that counted a request still runs.

    Never signals on Windows, where ``os.kill`` would end the process: there an
    entry is retired by age alone.
    """
    try:
        pid = int(pid_text)
    except ValueError:
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _live_starts(starts: Any, now: float) -> list:
    if not isinstance(starts, list):
        return []
    return [start for start in starts
            if isinstance(start, (int, float)) and now - start < IN_FLIGHT_MAX_AGE_S]


def _prune_in_flight(entry: dict, now: float) -> None:
    flights = _flights(entry)
    for pid in list(flights):
        kept = _live_starts(flights[pid], now)
        if kept and _alive(pid):
            flights[pid] = kept
        else:
            if kept:
                _count_orphans(entry, len(kept), now)
            del flights[pid]


def _count_orphans(entry: dict, count: int, now: float) -> None:
    """Requests still in flight for a process that has stopped: a wire restart (or
    crash) forgot them. A running total per account, kept so the "a restart forgets
    sends in flight" limit can be measured; it decides nothing (HAL#748 item 4)."""
    orphans = entry.get("orphaned_in_flight")
    if not isinstance(orphans, dict):
        orphans = entry["orphaned_in_flight"] = {"count": 0, "last_at": None}
    previous = orphans.get("count")
    orphans["count"] = (previous if isinstance(previous, int) else 0) + count
    orphans["last_at"] = round(now, 3)


def add_in_flight(data: dict, label: str, *, pid: int, started: float, now: float) -> dict:
    """Count one request in flight on ``label``, for process ``pid``."""
    entry = _entry(data, label)
    _prune_in_flight(entry, now)
    _flights(entry).setdefault(str(pid), []).append(round(started, 3))
    return data


def _drop_in_flight(entry: dict, pid: int, started: float) -> None:
    flights = _flights(entry)
    starts = flights.get(str(pid))
    mark = round(started, 3)
    if isinstance(starts, list) and mark in starts:
        starts.remove(mark)
    if not starts:
        flights.pop(str(pid), None)


def settle(data: dict, label: str, *, pid: int, started: float, now: float,
           answer: Optional[Tuple[Optional[int], Optional[Reading]]],
           digest: Optional[str] = None) -> dict:
    """End one request's flight and record its answer (``None``: it got none)."""
    entry = _entry(data, label)
    _drop_in_flight(entry, pid, started)
    _prune_in_flight(entry, now)
    if answer is not None:
        _record_answer(entry, answer[0], answer[1], digest, now)
    return data


def _record_answer(entry: dict, status: Optional[int], reading: Optional[Reading],
                   digest: Optional[str], now: float) -> None:
    entry["observations"] = int(entry.get("observations") or 0) + 1
    entry["observed_at"] = round(now, 3)
    entry["http_status"] = status
    _record_token(entry, status, digest, now)
    if reading is not None and reading.has_quota:
        entry["quota"] = reading.to_json()
        entry["cooling"] = _cooling_from_quota(reading, now)
    elif status == 429:
        entry["cooling"] = _throttled(reading, now)


def _record_token(entry: dict, status: Optional[int], digest: Optional[str],
                  now: float) -> None:
    token = entry.get("token")
    if not isinstance(token, dict):
        token = entry["token"] = {"health": UNKNOWN, "since": None, "digest": None}
    if digest and token.get("digest") not in (None, digest):
        _set_health(token, UNKNOWN, now)  # a different token from the one judged
    if digest:
        token["digest"] = digest
    verdict = _health_after(status)
    if verdict is not None:
        _set_health(token, verdict, now)


def _health_after(status: Optional[int]) -> Optional[str]:
    if status in _DEAD_STATUSES:
        return "dead"
    if isinstance(status, int) and 200 <= status < 300:
        return "ok"
    return None


def _set_health(token: dict, health: str, now: float) -> None:
    if token.get("health") != health:
        token["health"] = health
        token["since"] = round(now, 3)


def mark_token_dead(data: dict, label: str, *, digest: Optional[str], now: float) -> dict:
    """Hold the token ``digest`` names on ``label`` as dead (needs a re-login), for
    an answer whose status alone does not say so: a 403 whose message names a
    suspension. A verdict about a different token is replaced, as any answer's is."""
    entry = _entry(data, label)
    _record_token(entry, None, digest, now)
    _set_health(entry["token"], "dead", now)
    return data


def _cooling_from_quota(reading: Reading, now: float) -> Optional[dict]:
    """Spent -- rejected, or billed to overage -- until the reset the headers name."""
    if reading.status != "rejected" and reading.overage_in_use is not True:
        return None
    until, source = _binding_reset(reading, now)
    return {"until": until, "reason": _spent_reason(reading), "source": source,
            "claim": reading.claim}


def _spent_reason(reading: Reading) -> str:
    """``paid-overage`` when the answer was being billed to extra usage (f22)."""
    paid = reading.overage_in_use is True or (
        reading.status == "rejected"
        and reading.overage_status in ("allowed", "allowed_warning"))
    return "paid-overage" if paid else "spent"


def _binding_reset(reading: Reading, now: float) -> Tuple[Optional[int], str]:
    window = reading.window(reading.claim)
    retry = None if reading.retry_after_s is None else int(now) + reading.retry_after_s
    for source, value in (("unified-reset", reading.reset),
                          (f"{reading.claim}-reset", window.reset if window else None),
                          ("retry-after", retry)):
        if value is not None:
            return value, source
    return None, "none"


def _throttled(reading: Optional[Reading], now: float) -> dict:
    """A 429 that carried no quota headers: its scope is uncertain."""
    seconds = reading.retry_after_s if reading is not None else None
    if seconds is None:
        return {"until": None, "reason": "throttled", "source": "none", "claim": None}
    return {"until": int(now) + seconds, "reason": "throttled", "source": "retry-after",
            "claim": None}


@dataclass(frozen=True)
class Cooling:
    """An account the latest evidence says is cooling.

    ``until`` is the epoch second the headers named; ``None`` when they named none --
    the ledger does not invent a time, and Step 5c chooses how to re-probe (the
    evidence's age is ``now - observed_at``). ``reason`` is ``spent``,
    ``paid-overage`` or ``throttled``; ``source`` names the header the time came from.
    """

    until: Optional[int]
    reason: str
    source: str
    claim: Optional[str]
    observed_at: Optional[float]


def cooling_of(entry: Optional[dict], now: float) -> Optional[Cooling]:
    """The account's cooling at ``now``, or None (none recorded, or its reset has passed)."""
    raw = entry.get("cooling") if isinstance(entry, dict) else None
    if not isinstance(raw, dict):
        return None
    until = raw.get("until") if type(raw.get("until")) is int else None
    if until is not None and until <= now:
        return None
    return Cooling(until=until, reason=str(raw.get("reason") or UNKNOWN),
                   source=str(raw.get("source") or "none"), claim=raw.get("claim"),
                   observed_at=entry.get("observed_at"))


def in_flight_of(entry: Optional[dict], now: float) -> int:
    """Requests counted in flight by processes that still run, none older than the cap."""
    flights = entry.get("in_flight") if isinstance(entry, dict) else None
    if not isinstance(flights, dict):
        return 0
    return sum(len(_live_starts(starts, now)) for pid, starts in flights.items()
               if _alive(pid))


def token_health_of(entry: Optional[dict], token: Optional[str] = None) -> str:
    """``ok``, ``dead`` or ``unknown``. Given the token about to be used, a verdict
    about a different token (a re-login since) reads as ``unknown``."""
    info = entry.get("token") if isinstance(entry, dict) else None
    if not isinstance(info, dict):
        return UNKNOWN
    judged = info.get("digest")
    if token is not None and judged is not None and judged != credential_digest(token):
        return UNKNOWN
    return str(info.get("health") or UNKNOWN)


# --------------------------------------------------------------------------- #
# The ledger file: observed by AccountLedger, read through LedgerView.
# --------------------------------------------------------------------------- #
def default_path() -> Path:
    """``$HAL_HOME/state/hal/account-ledger.json``, read when called (each test has its own)."""
    home = os.environ.get("HAL_HOME") or os.path.expanduser("~/.hal")
    return Path(home) / "state" / "hal" / LEDGER_FILE


def _ledger_path(path: Optional[Path]) -> Path:
    return path if path is not None else default_path()


_NOTED: set = set()
_NOTED_GUARD = threading.Lock()


def _note(what: str, exc: BaseException) -> None:
    """A skipped ledger step, logged once per kind and exception type. Type only."""
    key = (what, type(exc).__name__)
    with _NOTED_GUARD:
        first = key not in _NOTED
        _NOTED.add(key)
    log = _log.warning if first else _log.debug
    log("account ledger: %s skipped (%s)", what, type(exc).__name__)


def _answer_status(resp: Any) -> Optional[int]:
    status = getattr(resp, "status", None) or getattr(resp, "code", None)
    return status if isinstance(status, int) else None


class LedgerView:
    """Reads the ledger. Every query reads the file afresh; nothing is written."""

    def __init__(self, path: Optional[os.PathLike] = None, *,
                 clock: Callable[[], float] = time.time):
        self._path = Path(path) if path is not None else None
        self._clock = clock

    @property
    def path(self) -> Path:
        return _ledger_path(self._path)

    def accounts(self) -> Dict[str, dict]:
        """Every account's stored entry, by label."""
        accounts = read_json(self.path).get("accounts")
        if not isinstance(accounts, dict):
            return {}
        return {label: entry for label, entry in accounts.items() if isinstance(entry, dict)}

    def cooling(self, label: str, now: Optional[float] = None) -> Optional[Cooling]:
        """Cooling until the reset its headers named, or None when it is not cooling."""
        return cooling_of(self.accounts().get(label), self._at(now))

    def in_flight(self, label: str, now: Optional[float] = None) -> int:
        return in_flight_of(self.accounts().get(label), self._at(now))

    def token_health(self, label: str, token: Optional[str] = None) -> str:
        return token_health_of(self.accounts().get(label), token)

    def _at(self, now: Optional[float]) -> float:
        return self._clock() if now is None else now


class Watch:
    """One request on one account, from sent until its answer begins."""

    def __init__(self, ledger: "AccountLedger", label: str, digest: Optional[str],
                 started: float):
        self.label = label
        self._ledger = ledger
        self._digest = digest
        self._started = started
        self._answer: Optional[Tuple[Optional[int], Reading]] = None
        self._closed = False

    def answered(self, status: Any, headers: Any) -> None:
        """The answer's head arrived: keep its status and its reading, never its values."""
        try:
            code = status if isinstance(status, int) else None
            self._answer = (code, read_headers(headers, self._ledger._clock()))
        except Exception as exc:  # noqa: BLE001 -- observing never fails the request
            _note("reading", exc)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ledger._settle(self.label, self._started, self._answer, self._digest)


class AccountLedger:
    """Observes requests and their answers, and writes the ledger.

    ``path`` defaults to :func:`default_path`, resolved at each use. ``resolver`` puts
    a request on its account (default: GRIP's pool, read-only). A write on a request
    path waits at most ``lock_timeout`` seconds for the lock (``begin_timeout`` for
    the count taken before a request is sent), then is skipped and logged by type.
    """

    def __init__(self, path: Optional[os.PathLike] = None, *,
                 resolver: Optional[AccountResolver] = None,
                 clock: Callable[[], float] = time.time,
                 lock_timeout: float = 0.25, begin_timeout: float = 0.05):
        self._path = Path(path) if path is not None else None
        self._resolver = resolver
        self._clock = clock
        self._lock_timeout = lock_timeout
        self._begin_timeout = begin_timeout

    @property
    def path(self) -> Path:
        return _ledger_path(self._path)

    def view(self) -> LedgerView:
        """A reader of the same file, on the same clock."""
        return LedgerView(self._path, clock=self._clock)

    @contextmanager
    def exchange(self, request_headers: Any) -> Iterator[Watch]:
        """Count one request in flight on its account until its answer begins.

        Call ``watch.answered(status, headers)`` when the answer's head arrives; on
        leaving the block the flight ends and the answer, if any, is recorded.
        Nothing the ledger does can raise into the request.
        """
        watch = self._begin(request_headers)
        try:
            yield watch
        finally:
            watch.close()

    def observing(self, opener: Callable) -> Callable:
        """Wrap a gateway opener (``gateway.build_default_opener``'s contract) so that
        every answer it gives -- each transport retry's too -- is observed."""
        def observed(method, url, headers, body):
            with self.exchange(headers) as watch:
                try:
                    resp = opener(method, url, headers, body)
                except urllib.error.HTTPError as answer:  # an answer raised, not returned
                    watch.answered(_answer_status(answer), getattr(answer, "headers", None))
                    raise
                watch.answered(_answer_status(resp), getattr(resp, "headers", None))
                return resp
        return observed

    def _begin(self, request_headers: Any) -> Watch:
        label, digest = self._label_for(request_headers)
        started = self._clock()
        self._write("count", lambda data: add_in_flight(
            data, label, pid=os.getpid(), started=started, now=started),
            self._begin_timeout)
        return Watch(self, label, digest, started)

    def _settle(self, label: str, started: float, answer, digest: Optional[str]) -> None:
        now = self._clock()
        self._write("record", lambda data: settle(
            data, label, pid=os.getpid(), started=started, now=now, answer=answer,
            digest=digest), self._lock_timeout)

    def mark_dead(self, request_headers: Any) -> None:
        """Hold the token these headers present as dead, for a 403 whose message
        names a suspension (:func:`mark_token_dead`). Fail-open and bounded in
        time, like every other write on a request path."""
        label, digest = self._label_for(request_headers)
        now = self._clock()
        self._write("health", lambda data: mark_token_dead(
            data, label, digest=digest, now=now), self._lock_timeout)

    def _label_for(self, request_headers: Any) -> Tuple[str, Optional[str]]:
        try:
            if self._resolver is None:
                self._resolver = AccountResolver()
            return self._resolver.label_for(request_headers)
        except Exception as exc:  # noqa: BLE001
            _note("account", exc)
            return UNPOOLED, None

    def _write(self, what: str, mutate: Callable[[dict], dict], timeout: float) -> None:
        try:
            update_json(self.path, mutate, lock_timeout=timeout)
        except Exception as exc:  # noqa: BLE001 -- observability never subtracts availability
            _note(what, exc)


# --------------------------------------------------------------------------- #
# The command line: show the ledger.
# --------------------------------------------------------------------------- #
def _when(epoch: Any) -> str:
    if type(epoch) is not int:
        return "unknown"
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _window_text(name: str, window: Any) -> str:
    if not isinstance(window, dict):
        return f"{name} unknown"
    use = window.get("utilization")
    share = f"{use * 100:.0f}%" if isinstance(use, (int, float)) else "unknown"
    return f"{name} {share} (resets {_when(window.get('reset'))})"


def _cooling_text(cooling: Optional[Cooling]) -> str:
    if cooling is None:
        return "cooling no"
    return f"cooling {cooling.reason} until {_when(cooling.until)}"


def describe(view: LedgerView, now: Optional[float] = None) -> List[str]:
    """One line per account: status, windows, cooling, token health, in flight."""
    now = time.time() if now is None else now
    lines = []
    for label, entry in sorted(view.accounts().items()):
        quota = entry.get("quota") if isinstance(entry.get("quota"), dict) else {}
        windows = quota.get("windows") if isinstance(quota.get("windows"), dict) else {}
        lines.append("  ".join([
            label, f"status {quota.get('status', UNKNOWN)}",
            _window_text("5h", windows.get("five_hour")),
            _window_text("7d", windows.get("seven_day")),
            _cooling_text(cooling_of(entry, now)),
            f"token {token_health_of(entry)}",
            f"in flight {in_flight_of(entry, now)}",
            f"headers seen {len(quota.get('header_names') or [])}"]))
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lib.hal.account_ledger",
        description="Show what each Claude account's own answers said about its limits.")
    parser.add_argument("--path", help="the ledger file (default: $HAL_HOME/state/hal/"
                                       f"{LEDGER_FILE})")
    parser.add_argument("--json", action="store_true", help="print the stored ledger as JSON")
    args = parser.parse_args(argv)
    view = LedgerView(args.path)
    if args.json:
        print(json.dumps(read_json(view.path), indent=2, sort_keys=True))
        return 0
    lines = describe(view)
    print("\n".join(lines) if lines else f"no account observed yet ({view.path})")
    return 0


__all__ = [
    "AccountLedger",
    "AccountResolver",
    "Cooling",
    "LedgerView",
    "PoolAccount",
    "Reading",
    "UNPOOLED",
    "Watch",
    "Window",
    "add_in_flight",
    "cooling_of",
    "credential_digest",
    "credential_of",
    "default_path",
    "describe",
    "in_flight_of",
    "mark_token_dead",
    "pool_tokens",
    "read_headers",
    "settle",
    "store_active_path",
    "token_health_of",
]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
