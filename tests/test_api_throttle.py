"""Goodhart-proof tests for lib/hal/api_throttle (HAL-VPS W0).

Each test fails if the defence is a no-op:
- The rate-limit test asserts request N allowed, N+1 in the SAME window blocked,
  then allowed again AFTER the window via the injected clock. A no-op limiter
  (always allow) fails the "N+1 blocked" assertion.
- The breaker test asserts it opens EXACTLY at budget and resets after the UTC
  day rolls. A no-op breaker (always allow) fails the "at budget -> blocked"
  assertion; a breaker that never resets fails the "after reset -> allowed" one.
- The reserve test asserts check() ALONE counts toward budget (a burst with no
  record_cost cannot overshoot) — fails the old check-then-act code.
- The IP-resolver test asserts a bad headers object never raises — fails if the
  resolver drops its defensive guard (which would crash _api_chat).

No real sleeps: a mutable clock cell is advanced explicitly.
"""

from datetime import datetime, timedelta, timezone

from lib.hal.api_throttle import ChatThrottle, client_ip_from_headers


def _clock(start: datetime):
    """Return (now_fn, advance_fn) over a mutable cell — deterministic, no sleep."""
    cell = {"t": start}

    def now():
        return cell["t"]

    def advance(seconds):
        cell["t"] = cell["t"] + timedelta(seconds=seconds)

    return now, advance


_T0 = datetime(2026, 5, 29, 12, 0, 0, tzinfo=timezone.utc)


class _Headers:
    """Minimal stand-in for http.server's self.headers (.get interface)."""

    def __init__(self, mapping):
        self._m = mapping

    def get(self, key, default=None):
        return self._m.get(key, default)


# --- Per-IP rate limiter ---------------------------------------------------

def test_rate_limiter_blocks_over_limit_then_recovers_after_window():
    now, advance = _clock(_T0)
    t = ChatThrottle(per_min=3, daily_budget=10_000, now=now)
    ip = "203.0.113.7"

    # First 3 in the window are allowed.
    for i in range(3):
        allowed, code, _ = t.check(ip)
        assert allowed is True, f"request {i + 1} should be allowed"
        assert code == 200

    # 4th in the SAME window is blocked with 429 (fails if limiter is a no-op).
    allowed, code, reason = t.check(ip)
    assert allowed is False, "4th request in window must be blocked"
    assert code == 429
    assert "rate limit" in reason.lower()

    # Advance past the 60s window -> allowed again (proves it is a real window).
    advance(61)
    allowed, code, _ = t.check(ip)
    assert allowed is True, "request after window must be allowed again"
    assert code == 200


def test_rate_limiter_is_per_ip_not_global():
    now, _ = _clock(_T0)
    t = ChatThrottle(per_min=1, daily_budget=10_000, now=now)

    assert t.check("198.51.100.1")[0] is True
    # Same IP again -> blocked.
    assert t.check("198.51.100.1")[0] is False
    # Different IP -> still allowed (limiter must key on IP).
    assert t.check("198.51.100.2")[0] is True


# --- Global daily budget circuit breaker -----------------------------------

def test_breaker_opens_exactly_at_budget_and_resets_next_day():
    now, advance = _clock(_T0)
    # High per-min so the limiter never interferes with the budget assertions.
    t = ChatThrottle(per_min=10_000, daily_budget=3.0, now=now)
    ip = "203.0.113.9"

    # Spend exactly the budget (3 units): each request allowed, each records 1.
    for i in range(3):
        allowed, _, _ = t.check(ip)
        assert allowed is True, f"request {i + 1} within budget should pass"
        t.record_cost(1.0)

    # Now spent == budget -> breaker OPEN -> next request blocked with 429.
    allowed, code, reason = t.check(ip)
    assert allowed is False, "breaker must open exactly at budget"
    assert code == 429
    assert "budget" in reason.lower()

    # Roll to the next UTC day -> breaker resets -> allowed again.
    advance(24 * 3600)
    allowed, code, _ = t.check(ip)
    assert allowed is True, "breaker must reset after UTC-midnight"
    assert code == 200


def test_breaker_uses_real_token_cost_when_recorded():
    now, _ = _clock(_T0)
    t = ChatThrottle(per_min=10_000, daily_budget=100.0, now=now)
    ip = "203.0.113.10"

    assert t.check(ip)[0] is True
    t.record_cost(99.0)          # big call: reserved 1 + settle 98 -> spent 99
    assert t.check(ip)[0] is True
    t.record_cost(1.0)           # now exactly at budget (100)
    assert t.check(ip)[0] is False, "breaker opens once recorded cost hits cap"


def test_check_reserves_budget_so_a_burst_cannot_overshoot():
    # Reserve-then-settle fix (AGORA #384 REQUEST_CHANGES, concern 2): check()
    # reserves a budget unit atomically, so a burst of allowed requests landing
    # BEFORE any record_cost() cannot collectively overshoot the cap. The OLD
    # code (check did not mutate _spent) let unlimited requests pass until
    # record_cost ran -> this test fails the old code.
    now, _ = _clock(_T0)
    t = ChatThrottle(per_min=10_000, daily_budget=3.0, now=now)
    ip = "203.0.113.20"

    assert t.check(ip)[0] is True   # reserves -> spent 1
    assert t.check(ip)[0] is True   # reserves -> spent 2
    assert t.check(ip)[0] is True   # reserves -> spent 3
    allowed, code, _ = t.check(ip)  # no record_cost yet; budget already reserved
    assert allowed is False, "check() must reserve budget; a burst cannot overshoot"
    assert code == 429


# --- Client IP resolution (Cloudflare-aware) -------------------------------

def test_client_ip_precedence_cf_then_xff_then_peer():
    # CF-Connecting-IP wins.
    h = _Headers({"CF-Connecting-IP": "1.1.1.1", "X-Forwarded-For": "2.2.2.2"})
    assert client_ip_from_headers(h, "9.9.9.9") == "1.1.1.1"

    # No CF header -> first hop of X-Forwarded-For.
    h = _Headers({"X-Forwarded-For": "2.2.2.2, 3.3.3.3"})
    assert client_ip_from_headers(h, "9.9.9.9") == "2.2.2.2"

    # Neither -> socket peer.
    h = _Headers({})
    assert client_ip_from_headers(h, "9.9.9.9") == "9.9.9.9"


def test_client_ip_falls_back_on_expected_bad_headers():
    # Narrowed catch (AGORA #384 round-2): None / non-mapping headers fall back
    # to peer then "unknown". TRULY-unexpected errors are NOT swallowed here —
    # they propagate to _chat_throttle_check's outer fail-open guard (covered in
    # tests/test_api_chat_reservation.py).
    assert client_ip_from_headers(None, "9.9.9.9") == "9.9.9.9"   # None.get -> AttributeError -> peer
    assert client_ip_from_headers(None, "") == "unknown"
    assert client_ip_from_headers(12345, "9.9.9.9") == "9.9.9.9"  # int has no .get -> AttributeError -> peer


# --- Fail-open contract -----------------------------------------------------

def test_check_fails_open_on_internal_error():
    now, _ = _clock(_T0)
    t = ChatThrottle(per_min=1, daily_budget=10_000, now=now)

    # Corrupt internal state so the window scan raises; check must ALLOW, not crash.
    t._hits = None  # type: ignore[assignment]
    allowed, code, _ = t.check("203.0.113.11")
    assert allowed is True, "throttle must fail OPEN, never crash the endpoint"
    assert code == 200


# --- Env-configured defaults -----------------------------------------------

def test_env_overrides_apply(monkeypatch):
    # No importlib.reload (AGORA #384 concern 3): ChatThrottle.__init__ reads env
    # when args are None, so monkeypatch.setenv + a fresh instance picks up the
    # override without mutating module state (which reload would leak across tests).
    monkeypatch.setenv("HAL_CHAT_RL_PER_MIN", "2")
    monkeypatch.setenv("HAL_CHAT_DAILY_BUDGET", "5")
    t = ChatThrottle()  # reads env via defaults
    ip = "203.0.113.12"
    assert t.check(ip)[0] is True
    assert t.check(ip)[0] is True
    assert t.check(ip)[0] is False, "HAL_CHAT_RL_PER_MIN=2 must block the 3rd"


# --- AGORA #384 round-2: leak-free reservation + bounded memory --------------

def test_release_reservation_refunds_a_unit():
    # A request reserved by check() but not completed must be refunded, else the
    # unit leaks into the daily total until the UTC reset.
    now, _ = _clock(_T0)
    t = ChatThrottle(per_min=10_000, daily_budget=10.0, now=now)
    ip = "203.0.113.30"
    assert t.check(ip)[0] is True          # reserves 1 -> spent 1
    assert t.spent == 1.0
    t.release_reservation()                # request failed -> refund
    assert t.spent == 0.0, "release_reservation must refund the reserved unit"
    t.release_reservation()                # never goes negative on an extra release
    assert t.spent == 0.0


def test_zero_token_cost_keeps_reservation_floor():
    # A 0-token successful call settles to the already-reserved 1 unit (a
    # completed call still consumes a slot) — never negative, never double-counted.
    now, _ = _clock(_T0)
    t = ChatThrottle(per_min=10_000, daily_budget=10.0, now=now)
    ip = "203.0.113.31"
    assert t.check(ip)[0] is True          # reserves 1 -> spent 1
    t.record_cost(0.0)                     # zero-token settle
    assert t.spent == 1.0, "a 0-token completed call stays counted at its reserved unit"


def test_sweep_prunes_stale_ips():
    # Per-IP windows for IPs that stop appearing must be pruned, else _hits grows
    # unbounded on a public endpoint. After the sweep interval, a check() drops
    # windows whose hits have all expired.
    now, advance = _clock(_T0)
    t = ChatThrottle(per_min=10_000, daily_budget=10_000, now=now)
    for i in range(200):                   # 200 one-off IPs
        t.check(f"198.51.100.{i}")
    assert len(t._hits) == 200
    advance(400)                           # past the 60s window AND the 300s sweep interval
    t.check("203.0.113.99")                # triggers the sweep
    assert len(t._hits) == 1, "stale per-IP windows must be pruned; only the live IP remains"


def test_sweep_hard_caps_tracked_ips(monkeypatch):
    # Flood guard (AGORA #384 r3): even within the sweep interval, _hits is
    # bounded ABSOLUTELY by _MAX_TRACKED_IPS (oldest-active evicted), not just
    # time-pruned. Without the cap, 20 one-off IPs would all persist -> fails.
    monkeypatch.setattr("lib.hal.api_throttle._MAX_TRACKED_IPS", 5)
    now, _ = _clock(_T0)
    t = ChatThrottle(per_min=10_000, daily_budget=10_000, now=now)
    for i in range(20):
        t.check(f"198.51.100.{i}")
    assert len(t._hits) <= 6, "hard cap must bound tracked IPs (cap=5, +1 in-flight)"


def test_per_min_zero_does_not_self_dos():
    # AGORA #384 r4: per_min<=0 must clamp to >=1, else len(hits) >= 0 is always
    # True and the throttle 429s every request (self-DoS on a misconfigured env).
    now, _ = _clock(_T0)
    assert ChatThrottle(per_min=0, daily_budget=10_000, now=now).check("203.0.113.40")[0] is True, \
        "per_min=0 must clamp to >=1, not 429 every request"
    assert ChatThrottle(per_min=-5, daily_budget=10_000, now=now).check("203.0.113.41")[0] is True, \
        "negative per_min must clamp too"
