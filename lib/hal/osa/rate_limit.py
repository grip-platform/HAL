"""osa γ-2b — per-script rate limit subsystem.

Closes Gap 3 from the γ-2 broly MESH RETHINK (2026-04-19): a tier-2+
capability token is single-use but does not cap per-script dispatch
frequency. A stuck autonomous loop could fire 1 000 iMessages in 60 s
using 1 000 freshly-minted tokens. This module enforces a rolling
per-``(script_name)`` window count via a SQLite counter that is
durable across processes.

Design:

* :class:`RateLimit` — ``(window_s, max_calls)`` declared on the
  :class:`Script` registry entry.
* :func:`check_and_record` — called by :mod:`dispatch.osa` BEFORE
  ``_authorise_with_transport``. Expires old rows, counts remaining
  dispatches in the window, raises :class:`RateLimitExceeded` if at
  or above the cap, otherwise records this call and returns.
* Counter storage: ``$HAL_HOME/state/osa-rate-limits.db`` (SQLite with
  a single ``dispatches(script_name TEXT, ts REAL)`` table).
  Overridable for tests via ``db_path`` argument.

Rejection path: the dispatcher catches :class:`RateLimitExceeded`,
writes one audit entry with ``error="ratelimit_exceeded"`` and the
original provenance, then re-raises so the caller sees the refusal.

Hypothesis H154 (2026-05-19): zero ratelimit overruns over 30 days
and no silent pass-through — if a rate-limited script dispatches
beyond its cap, the counter DB or this module is wrong.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = [
    "RateLimit",
    "RateLimitExceeded",
    "check_and_record",
    "default_db_path",
    "_reset_for_tests",
]


@dataclass(frozen=True)
class RateLimit:
    """Declarative per-script rate limit.

    ``window_s`` is the rolling window in seconds. ``max_calls`` is the
    maximum number of dispatches allowed within any such window. A call
    at time ``t`` counts against every window ``(t - window_s, t]``.
    """
    window_s: int
    max_calls: int

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ValueError(
                f"RateLimit.window_s must be positive, got {self.window_s}"
            )
        if self.max_calls <= 0:
            raise ValueError(
                f"RateLimit.max_calls must be positive, got {self.max_calls}"
            )


class RateLimitExceeded(RuntimeError):
    """Raised when a dispatch would exceed the script's rolling-window cap."""


def default_db_path() -> Path:
    """Return ``$HAL_HOME/state/osa-rate-limits.db`` (parent auto-created)."""
    hal_home = os.environ.get("HAL_HOME", str(Path.home() / ".hal"))
    db = Path(hal_home) / "state" / "osa-rate-limits.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    return db


_SCHEMA = """
CREATE TABLE IF NOT EXISTS dispatches (
    script_name TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_dispatches_script_ts
    ON dispatches(script_name, ts);
"""


# Serialises connection setup in this process: threads that opened the database
# at the same moment raced to change its journal mode, and one could fail with
# "database is locked" instead of getting a result (HAL#919).
_SETUP_LOCK = threading.Lock()


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with schema applied, in WAL mode where SQLite allows it.

    WAL mode is stored in the database file, so it is set only when the file
    is not in WAL yet.
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=2.0)
    try:
        with _SETUP_LOCK:
            (mode,) = conn.execute("PRAGMA journal_mode").fetchone()
            if str(mode).lower() != "wal":
                conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
    except BaseException:
        conn.close()  # the caller's finally never sees a connection that failed here
        raise
    return conn


def check_and_record(
    script_name: str,
    rate_limit: RateLimit,
    *,
    db_path: Optional[Path] = None,
    now: Optional[float] = None,
) -> None:
    """Raise :class:`RateLimitExceeded` if over cap, otherwise record this call.

    Single transaction: expires old rows, counts remaining, either inserts
    the new row (under cap) or raises (at/over cap). The check AND the
    insert happen under the same SQLite lock so two concurrent dispatchers
    cannot race past the cap.
    """
    effective_db = db_path if db_path is not None else default_db_path()
    effective_now = time.time() if now is None else now
    cutoff = effective_now - rate_limit.window_s
    conn = _connect(effective_db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM dispatches WHERE script_name = ? AND ts < ?",
            (script_name, cutoff),
        )
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM dispatches WHERE script_name = ?",
            (script_name,),
        ).fetchone()
        if count >= rate_limit.max_calls:
            conn.execute("COMMIT")
            raise RateLimitExceeded(
                f"rate limit exceeded for {script_name!r}: "
                f"{count}/{rate_limit.max_calls} calls in last "
                f"{rate_limit.window_s}s window"
            )
        conn.execute(
            "INSERT INTO dispatches(script_name, ts) VALUES (?, ?)",
            (script_name, effective_now),
        )
        conn.execute("COMMIT")
    finally:
        conn.close()


def _reset_for_tests(db_path: Optional[Path] = None) -> None:
    """Remove the rate-limit DB (test isolation)."""
    target = db_path if db_path is not None else default_db_path()
    if target.exists():
        target.unlink()
