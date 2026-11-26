"""TypeSafe's System One API (the Jev model) as a HAL decision provider.

Native HTTP to ``POST /v1/systemone``. TypeSafe's SDK is not used: it is 0.x,
and its versions before 0.7.1 could put the key into logged exceptions. Every
decision goes through these steps, in order, and stops at the first that says no:

1. the circuit breaker is closed (three failed sends in a row rest it for 60 s);
2. HAL's spend guard allows ``typesafe-api-key`` today, under its daily cap;
3. a key resolves, from keychain item ``grip-typesafe`` or the environment;
4. GRIP's decision shield scrubs the body, refusing the send if it cannot;
5. the scrubbed body is no larger than the provider's byte ceiling;
6. it goes out with proxies disabled, so no TLS-intercepting hop sees the key,
   retried the way TypeSafe's own SDK retries: at most twice, on a connection
   failure, 408, 429 or any 5xx, honouring ``retry-after``, with backoff from
   0.5 s doubling to 5 s less up to 25% jitter, 10 s per attempt and 30 s in all.
   Redirects are refused, so the key never follows a 3xx to another host: a 3xx
   comes back as an answer with reason ``bad_response`` and is never retried;
7. the answers are checked against the questions, and the input tokens are
   priced and recorded against the daily cap.

Measured against the live API on 2026-10-10 (docs/decide.md): a malformed
request is HTTP 400, not the documented 422; an over-budget one is refused with
``max_tokens_exceeded``, never truncated. Answers are NOT repeatable: on
2026-10-12 one request sent six times came back as six different answer sets
(a Noul between 0.91 and 0.94), so a caller must never repeat a request just
to get a reading that passes. The key is a local variable for one decision. It is never stored on
the provider, and never put in a result, a log line or an exception.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from decimal import Decimal
from typing import Any, Callable, Mapping, Optional, Tuple

from lib.hal import spend_guard
from lib.hal.decide import shield as decision_shield
from lib.hal.decide.contract import (DecisionModel, DecisionResult, InvalidDecision,
                                     ProviderSpec, read_answers)

MAX_RETRIES = 2
INITIAL_BACKOFF_S, MAX_BACKOFF_S, JITTER = 0.5, 5.0, 0.25
ATTEMPT_TIMEOUT_S, TOTAL_BUDGET_S = 10.0, 30.0
#: A retry starts only when at least this much of the budget would be left for it,
#: and each attempt's timeout is cut to what is left, so 30 s is the whole decision.
MIN_ATTEMPT_S = 1.0
BREAKER_THRESHOLD, BREAKER_COOLDOWN_S = 3, 60.0
#: TypeSafe's key shape, scrubbed from any vendor text a result repeats.
_KEY_SHAPE = re.compile(r"apikey_[A-Za-z0-9_]+")
_STATUS_REASONS = {401: "auth_failed", 403: "auth_failed", 408: "timeout",
                   429: "rate_limited", 529: "overloaded"}

#: ``(status, headers, body)`` for an HTTP answer; ``(None, {}, kind)`` when none came.
Outcome = Tuple[Optional[int], dict, Any]


class _Refusal(Exception):
    """A step that said no before anything was sent."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


class _Breaker:
    """Opens after ``threshold`` failed decisions in a row, for ``cooldown`` seconds."""

    def __init__(self, threshold: int = BREAKER_THRESHOLD, cooldown: float = BREAKER_COOLDOWN_S):
        self.threshold, self.cooldown = threshold, cooldown
        self._lock = threading.Lock()
        self._failures, self._open_until = 0, 0.0

    def is_open(self, now: float) -> bool:
        with self._lock:
            return now < self._open_until

    def failed(self, now: float) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.threshold:
                self._open_until, self._failures = now + self.cooldown, 0

    def succeeded(self) -> None:
        with self._lock:
            self._failures = 0


_BREAKER = _Breaker()


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect instead of following it. urllib copies ordinary request
    headers, the bearer key among them, onto a 301, 302 or 303 to ANY host, and the
    TypeSafe endpoint is fixed, so it never has a legitimate reason to redirect. The
    3xx then comes back as an HTTP answer like any other non-2xx, and is not retried."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _no_proxy_transport(url: str, data: bytes, headers: dict, timeout: float) -> Outcome:
    """POST ``data`` with proxies disabled and redirects refused. Any HTTP answer is
    returned; no answer raises."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _RefuseRedirects())
    request = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


def _resolve_key(source: str) -> Optional[str]:
    """The shared resolver: keychain first, then the environment (lib/hal/providers/openai_compat.py)."""
    from lib.hal.providers.openai_compat import _resolve_api_key
    key = _resolve_api_key(source)
    return key if key and key != source and key.isprintable() and " " not in key else None


class TypeSafeDecisionProvider:
    """Answers decisions through TypeSafe's API. Every collaborator can be replaced, for tests."""

    name = "typesafe"

    def __init__(self, spec: ProviderSpec, *, transport: Optional[Callable] = None,
                 shield: Optional[Callable] = None, key_resolver: Optional[Callable] = None,
                 sleep: Callable = time.sleep, clock: Callable = time.monotonic,
                 jitter: Callable = random.random, breaker: Optional[_Breaker] = None):
        self.spec = spec
        self._transport = transport or _no_proxy_transport
        self._shield = shield or decision_shield.scrub
        self._resolve_key = key_resolver or _resolve_key
        self._sleep, self._clock, self._jitter = sleep, clock, jitter
        self._breaker = breaker or _BREAKER

    def __repr__(self) -> str:
        return f"TypeSafeDecisionProvider({self.spec.endpoint!r})"

    def decide(self, model: DecisionModel, state: Any,
               questions: Mapping[str, dict]) -> DecisionResult:
        base = {"provider": self.name, "pinned": model.pinned}
        try:
            key = self._preflight()
            data = self._body(model, state, questions)
        except _Refusal as refusal:
            return DecisionResult.unavailable(refusal.reason, refusal.detail, **base)
        outcome, attempts, latency_ms = self._send(data, key)
        return self._result(outcome, questions, {**base, "attempts": attempts,
                                                 "latency_ms": latency_ms})

    # --- before the send -------------------------------------------------------

    def _preflight(self) -> str:
        if self._breaker.is_open(self._clock()):
            raise _Refusal("circuit_open", "recent sends kept failing; resting before the next")
        verdict = spend_guard.decide_within_cap(self.spec.spend_source)
        if not verdict.allowed:
            raise _Refusal("spend_refused", verdict.reason)
        key = self._resolve_key(self.spec.key_source)
        if not key:
            raise _Refusal("no_key", f"no key from {self.spec.key_source} or its environment fallback")
        return key

    def _body(self, model: DecisionModel, state: Any, questions: Mapping[str, dict]) -> bytes:
        body = {"state": state, "model": model.id, "questions": dict(questions)}
        try:
            scrubbed = self._shield(body, to=self.spec.endpoint)
        except decision_shield.ShieldUnavailable as exc:
            raise _Refusal("shield_unavailable", str(exc)) from None
        except OSError as exc:
            raise _Refusal("shield_refused", f"{type(exc).__name__}: {exc}"[:200]) from None
        data = json.dumps(scrubbed, ensure_ascii=False).encode("utf-8")
        if len(data) > self.spec.max_request_bytes:
            raise _Refusal("over_budget", f"{len(data)} bytes is over the "
                                          f"{self.spec.max_request_bytes}-byte ceiling")
        return data

    # --- the send --------------------------------------------------------------

    def _send(self, data: bytes, key: str) -> Tuple[Outcome, int, int]:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                   "Accept": "application/json", "User-Agent": "hal-decide/1"}
        started, attempt = self._clock(), 0
        while True:
            attempt += 1
            left = TOTAL_BUDGET_S - (self._clock() - started)
            outcome = self._attempt(data, headers, min(ATTEMPT_TIMEOUT_S, left))
            delay = self._retry_delay(outcome, attempt, self._clock() - started)
            if delay is None:
                return outcome, attempt, int((self._clock() - started) * 1000)
            self._sleep(delay)

    def _attempt(self, data: bytes, headers: dict, timeout: float) -> Outcome:
        try:
            return self._transport(self.spec.endpoint, data, headers, timeout)
        except Exception as exc:  # noqa: BLE001 -- a message could carry the request; keep only the kind
            timed_out = isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)
            return None, {}, "timeout" if timed_out else "unreachable"

    def _retry_delay(self, outcome: Outcome, attempt: int, elapsed: float) -> Optional[float]:
        status, headers, _ = outcome
        if attempt > MAX_RETRIES or not _retryable(status):
            return None
        delay = _retry_after(headers)
        if delay is None:
            backoff = min(INITIAL_BACKOFF_S * 2 ** (attempt - 1), MAX_BACKOFF_S)
            delay = backoff * (1 - JITTER * self._jitter())
        return None if elapsed + delay + MIN_ATTEMPT_S > TOTAL_BUDGET_S else delay

    # --- after the send --------------------------------------------------------

    def _result(self, outcome: Outcome, questions: Mapping[str, dict], extra: dict) -> DecisionResult:
        status, _, raw = outcome
        if status != 200:
            if status is None or status in (408, 429) or status >= 500:
                self._breaker.failed(self._clock())
            return DecisionResult.unavailable(_reason(status, raw), _detail(status, raw), **extra)
        self._breaker.succeeded()
        payload = _json(raw)
        usage = _usage(payload)
        cost = self._record(usage)
        try:
            answers = read_answers(questions, payload.get("answers"))
        except InvalidDecision as exc:
            return DecisionResult.unavailable("bad_response", str(exc), usage=usage,
                                              cost_usd=cost, **extra)
        return DecisionResult(ok=True, answers=answers, model=str(payload.get("model") or ""),
                              usage=usage, cost_usd=cost, **extra)

    def _record(self, usage: dict) -> str:
        """Price the reply and add it to today's spend; an unpriceable reply closes the source for the day."""
        source = self.spec.spend_source
        if "input_tokens" not in usage:
            spend_guard.record_unknown(source, "a TypeSafe reply carried no token usage")
            return ""
        cost = (Decimal(usage["input_tokens"]) * self.spec.usd_per_million_input
                + Decimal(usage.get("output_tokens", 0)) * self.spec.usd_per_million_output
                ) / Decimal(1_000_000)
        spend_guard.record_spend(source, cost, usage=usage)
        return spend_guard.usd_text(cost)


def _retryable(status: Optional[int]) -> bool:
    return status is None or status in (408, 429) or status >= 500


def _retry_after(headers: Mapping) -> Optional[float]:
    """The server's own wait, from ``retry-after-ms`` or ``retry-after`` in seconds; None when absent."""
    lowered = {str(k).lower(): v for k, v in (headers or {}).items()}
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            seconds = float(lowered[name]) * scale
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= seconds <= 60:
            return seconds
    return None


def _json(raw: Any) -> dict:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _usage(payload: dict) -> dict:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {name: usage[name] for name in ("input_tokens", "output_tokens")
            if type(usage.get(name)) is int and usage[name] >= 0}


def _vendor_error(raw: Any) -> Tuple[str, str]:
    """``(error_type, message)`` from an error body: ``{"detail": {...}}`` or ``{"detail": "..."}``."""
    detail = _json(raw).get("detail")
    if isinstance(detail, dict):
        return str(detail.get("error_type") or ""), str(detail.get("message") or "")
    return "", str(detail or "")


def _reason(status: Optional[int], raw: Any) -> str:
    if status is None:
        return raw if raw in ("timeout", "unreachable") else "unreachable"
    if status in (400, 422):
        return "over_budget" if _vendor_error(raw)[0] == "max_tokens_exceeded" else "invalid_request"
    if status in _STATUS_REASONS:
        return _STATUS_REASONS[status]
    return "server_error" if status >= 500 else "bad_response"


def _detail(status: Optional[int], raw: Any) -> str:
    """The vendor's own explanation, short and with anything key-shaped removed."""
    if status is None:
        return ""
    text = ": ".join(part for part in _vendor_error(raw) if part) or f"HTTP {status}"
    return _KEY_SHAPE.sub("[key]", text)[:200]
