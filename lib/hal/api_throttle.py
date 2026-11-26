"""Throttle for the one public HAL endpoint: POST /api/chat.

/api/chat is deliberately left public (the public chat demo) while PR #382 gates the
rest of /api/*. Because it is reachable from the internet via the
hal.grip-web.com Cloudflare tunnel, an unthrottled endpoint lets anyone drain
the operator's prepaid Claude (CCH) subscription. This module adds two
independent, stdlib-only, thread-safe defences:

1. Per-client-IP sliding-window rate limit (default 10 requests / 60s / IP).
2. A global daily budget circuit breaker. It tracks cumulative /api/chat cost
   (real response tokens when available, else 1 unit per request as a proxy)
   and OPENS at the configured cap until the next UTC-midnight reset.

Fail-open-within-limit contract
--------------------------------
The endpoint must never crash because of the throttle. ``ChatThrottle.check``
catches any internal error and returns ALLOW (a single request slips through),
trusting the per-IP limiter as the floor. We never fail toward DENY: a buggy
throttle taking the public demo offline is a worse outcome than one request
escaping the cap. The clock is injected so tests advance time without sleeping.

State is in-memory and single-process (HAL serves /api/chat from one process).
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, Optional, Tuple

_DEFAULT_RL_PER_MIN = 10
_DEFAULT_DAILY_BUDGET = 2000.0
_WINDOW_SECONDS = 60.0
_RESERVE_UNIT = 1.0  # budget units reserved per ALLOW; settled by record_cost
_SWEEP_INTERVAL_S = 300.0  # prune stale per-IP windows at most this often
_MAX_TRACKED_IPS = 50_000  # hard cap on distinct IP windows (flood guard)


def _utcnow() -> datetime:
    """Injectable wall clock (UTC). Patched in tests to advance time."""
    return datetime.now(timezone.utc)


def _env_float(name: str, default: float) -> float:
    """Read a float from env, falling back to default on missing/garbage."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def client_ip_from_headers(headers, peer: str = "") -> str:
    """Resolve the real client IP behind Cloudflare.

    Precedence: CF-Connecting-IP (Cloudflare's authoritative client header) ->
    first hop of X-Forwarded-For -> socket peer. ``headers`` is any object with
    a ``.get(name)`` method (e.g. http.server's ``self.headers``).
    """
    try:
        cf = (headers.get("CF-Connecting-IP") or "").strip()
        if cf:
            return cf
        xff = (headers.get("X-Forwarded-For") or "").strip()
        if xff:
            return xff.split(",", 1)[0].strip()
    except (AttributeError, TypeError):
        # Expected for None / non-mapping headers. Narrow on purpose: any OTHER
        # error is a genuine bug and propagates to _chat_throttle_check's outer
        # fail-open guard rather than being silently swallowed here.
        pass
    return peer or "unknown"


class ChatThrottle:
    """Per-IP sliding-window limiter + global daily budget breaker.

    Thread-safe: all mutable state is guarded by a single lock. The clock is
    injected via ``now`` so tests can advance time deterministically.
    """

    def __init__(
        self,
        per_min: Optional[int] = None,
        daily_budget: Optional[float] = None,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        # Clamp to >= 1: a 0/negative per_min makes _rate_limited's
        # `len(hits) >= per_min` always true, 429-ing every request — the
        # throttle would DoS its own endpoint on a misconfigured env (AGORA #384 r4).
        self._per_min = max(1, per_min if per_min is not None else int(_env_float(
            "HAL_CHAT_RL_PER_MIN", _DEFAULT_RL_PER_MIN)))
        self._daily_budget = daily_budget if daily_budget is not None else _env_float(
            "HAL_CHAT_DAILY_BUDGET", _DEFAULT_DAILY_BUDGET)
        self._now = now
        self._lock = threading.Lock()
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._spent = 0.0
        self._budget_day = self._day_key(now())
        self._last_sweep = now().timestamp()

    @staticmethod
    def _day_key(ts: datetime) -> str:
        return ts.astimezone(timezone.utc).strftime("%Y-%m-%d")

    def _roll_budget_day(self, ts: datetime) -> None:
        """Reset cumulative spend when the UTC day rolls over. Lock held."""
        day = self._day_key(ts)
        if day != self._budget_day:
            self._budget_day = day
            self._spent = 0.0

    def _rate_limited(self, ip: str, ts: datetime) -> bool:
        """True if this IP exceeded its window allowance. Lock held."""
        now_s = ts.timestamp()
        hits = self._hits[ip]
        cutoff = now_s - _WINDOW_SECONDS
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self._per_min:
            return True
        hits.append(now_s)
        self._enforce_ip_cap(ip)
        return False

    def _enforce_ip_cap(self, current_ip: str) -> None:
        """O(1) hard bound on tracked IPs (AGORA #384 r4): if over
        ``_MAX_TRACKED_IPS``, evict the oldest-inserted window (FIFO) — never the
        IP being served. Replaces a prior O(n log n) sort-under-lock so a flood
        at the cap cannot create a latency spike that itself triggers more 429s."""
        if len(self._hits) <= _MAX_TRACKED_IPS:
            return
        oldest = next(iter(self._hits))
        if oldest != current_ip:
            del self._hits[oldest]

    def _sweep_stale_ips(self, now_s: float) -> None:
        """Drop per-IP windows with no live hits, at most every
        ``_SWEEP_INTERVAL_S``. Proactively reclaims windows for IPs that stopped
        appearing; the absolute count bound is enforced cheaply on insertion by
        :meth:`_enforce_ip_cap` (AGORA #384). Lock held."""
        if now_s - self._last_sweep < _SWEEP_INTERVAL_S:
            return
        self._last_sweep = now_s
        cutoff = now_s - _WINDOW_SECONDS
        stale = [ip for ip, hits in self._hits.items()
                 if not hits or hits[-1] <= cutoff]
        for ip in stale:
            del self._hits[ip]

    def check(self, ip: str) -> Tuple[bool, int, str]:
        """Gate one /api/chat request from ``ip``.

        Returns ``(allowed, http_code, reason)``. On allow: ``(True, 200, "")``.

        Budget accounting is reserve-then-settle, closing a check-then-act gap:
        an ALLOW atomically reserves one budget unit under the lock, so a burst
        of requests that all pass BEFORE any :meth:`record_cost` lands cannot
        collectively overshoot the cap (correct even if the server later becomes
        multi-threaded). :meth:`record_cost` settles that unit up to the real
        token cost. Fail-open: any internal error allows the request.
        """
        try:
            with self._lock:
                ts = self._now()
                now_s = ts.timestamp()
                self._roll_budget_day(ts)
                self._sweep_stale_ips(now_s)
                if self._spent >= self._daily_budget:
                    return False, 429, (
                        "Daily usage budget for the public demo is exhausted. "
                        "It resets at 00:00 UTC."
                    )
                if self._rate_limited(ip, ts):
                    return False, 429, (
                        f"Rate limit exceeded: max {self._per_min} requests per "
                        "minute. Please slow down."
                    )
                self._spent += _RESERVE_UNIT  # reserve atomically; settled later
                return True, 200, ""
        except Exception:
            return True, 200, ""

    def record_cost(self, cost: float = 1.0) -> None:
        """Settle a completed /api/chat call against the daily total.

        :meth:`check` already reserved one budget unit on ALLOW; this settles it
        up to the real cost by adding ``cost - one unit`` (clamped >= 0, so a
        sub-unit cost never refunds below the reservation and spend never goes
        negative). Pass the real token count when reachable; the default leaves
        the request at its already-counted unit. Fail-open: a recording error
        never propagates to the endpoint.
        """
        try:
            with self._lock:
                self._roll_budget_day(self._now())
                self._spent += max(0.0, float(cost) - _RESERVE_UNIT)
        except Exception:
            pass

    def release_reservation(self) -> None:
        """Refund a unit reserved by :meth:`check` when the request did NOT
        complete (empty body, model error, timeout). Without it, a failed
        request leaks its reserved unit into the daily total until the UTC reset
        (AGORA #384). Subtracts one unit, clamped >= 0. Fail-open: a release
        error never propagates to the endpoint."""
        try:
            with self._lock:
                self._roll_budget_day(self._now())
                self._spent = max(0.0, self._spent - _RESERVE_UNIT)
        except Exception:
            pass

    @property
    def spent(self) -> float:
        """Current daily spend (read-only, lock-guarded) for tests + monitoring,
        so assertions need not poke the private field (AGORA #384 suggestion)."""
        with self._lock:
            return self._spent


# Process-global instance bound lazily so env vars are read at first use.
_INSTANCE: Optional[ChatThrottle] = None
_INSTANCE_LOCK = threading.Lock()


def get_chat_throttle() -> ChatThrottle:
    """Return the process-wide ChatThrottle, constructing it once."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = ChatThrottle()
    return _INSTANCE
