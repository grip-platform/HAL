"""HAL Gateway — failover + cost/audit layer (P3).

Layers two HAL value-adds on top of the P1/P2 gateway, transparently to every
harness:

* **Same-format failover** — an ordered list of upstreams; if the primary
  exhausts its transport retries (raises) or returns a failover-class status
  (429 / 5xx), the next upstream is tried. The caller receives ONE
  :class:`~lib.hal.gateway.UpstreamResponse` — the failover is invisible to the
  harness, which sees a single continuous response (the H-GW-3 claim).
* **Per-call audit** — every attempt is recorded to an append-only JSONL sink
  (which upstream, status, latency, whether a failover occurred). Fail-open: a
  telemetry failure can never break a request.

**Why same-format only (load-bearing correctness constraint):** P1's whole
value is *verbatim* forwarding that preserves Anthropic ``thinking`` /
``redacted_thinking`` signatures (#3274). Failing a ``/v1/messages`` call over
to a *different provider* would change the wire format and destroy that
guarantee. So a failover target is always another **Anthropic-compatible**
upstream, never a cross-provider swap. Cross-provider routing is a deliberate
non-goal of P3 (it would belong to a translated path, and is gated as future
work). Every target must also pass the spend guard (``lib/hal/spend_guard.py``):
each upstream is a :class:`~lib.hal.gateway.GatewayProxy`, which asks it before
sending, so a paid key is never a failover target before the operator allows
one. The env-built paid rung that used to sit here is removed
(:data:`REMOVED_FALLBACK_ENV`).

One auth scheme (InfiniteMachine v2, Step 5b): an upstream that carries its own
credential sends that credential alone, as ``Authorization``; the client's
credential never travels beside it (:func:`_upstream_headers`).

Every answer the default upstream proxies receive -- each transport retry's
included -- is read into the per-account ledger (``lib/hal/account_ledger.py``):
observe only, fail-open, nothing decided here.

Backward compatible: with a single upstream configured, :class:`FailoverProxy`
behaves byte-identically to a bare :class:`GatewayProxy` — it duck-types the
same ``forward(...)`` contract, so the P1/P2 handlers use it unchanged.

Scope: P3 only. P4 (Fern SDK) and P5 (folded kernel) stay frozen per the plan's
demand-gate.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from lib.hal.account_ledger import AccountLedger
from lib.hal.spend_guard import API_KEY, OVERAGE, SpendRefused

from lib.hal.gateway import (
    DEFAULT_UPSTREAM,
    GatewayProxy,
    GatewayTransportError,
    GatewayUpstreamError,
    UpstreamResponse,
    build_default_opener,
    is_transport_error,
    redact,
)



class AccountsSpent(SpendRefused):
    """Every configured account refused before a response was returned."""


def _finish_refusal(exc, spent, paid_refused, total, last_exc):
    """Only observed limits plus guarded paid sources exhaust authorised capacity."""
    if spent and len(spent) + paid_refused == total:
        earliest = min(spent, key=lambda d: d.retry_after or float("inf"))
        raise AccountsSpent(earliest) from exc
    if spent:
        # A transient fault elsewhere is not evidence of pool exhaustion.
        raise last_exc or GatewayTransportError(
            "no available upstream; some candidates failed transiently")
    raise exc


# Statuses that trigger failover to the NEXT same-format upstream — only AFTER
# the inner proxy has exhausted its own transport retries on the current one.
_FAILOVER_STATUS = frozenset({429, 500, 502, 503, 504, 529})

# The two headers that carry an Anthropic credential. One request carries one.
_AUTH_HEADERS = frozenset({"authorization", "x-api-key"})


@dataclass(frozen=True)
class UpstreamSpec:
    """One same-format upstream in the failover chain.

    ``header_overrides`` inject this upstream's own credential over the client's,
    so a fallback can carry a different Anthropic credential than the primary.
    The credential travels as one ``Authorization`` header, never as
    ``x-api-key``, and replaces the client's whole (see :func:`_upstream_headers`).
    """
    name: str
    base_url: str = DEFAULT_UPSTREAM
    header_overrides: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        names = [name.lower() for name, _ in self.header_overrides]
        if "x-api-key" in names or names.count("authorization") > 1:
            raise ValueError(f"upstream {self.name!r}: a credential override travels as "
                             "one Authorization header, never as x-api-key")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_sink(event: dict) -> None:
    """Append one audit event to ``$HAL_HOME/logs/gateway-audit.jsonl``."""
    home = Path(os.environ.get("HAL_HOME", str(Path.home() / ".hal")))
    path = home / "logs" / "gateway-audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, separators=(",", ":")) + "\n")


class GatewayTelemetry:
    """Append-only per-call audit/cost sink.

    Fail-open by construction: :meth:`record` never raises into the request
    path — a telemetry/disk failure must not break a gateway call (the gateway
    is monotone-toward-safe; observability is never allowed to subtract
    availability).
    """

    def __init__(self, sink: Optional[Callable[[dict], None]] = None):
        self._sink = sink or _default_sink

    def record(self, event: dict) -> None:
        try:
            self._sink({"ts": _now_iso(), **event})
        except Exception:
            pass


def _merge_headers(
    base: Iterable[Tuple[str, str]], overrides: Iterable[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """Merge header lists (case-insensitive); ``overrides`` win, order kept."""
    merged: dict = {}
    order: List[str] = []
    for name, value in list(base) + list(overrides):
        low = name.lower()
        if low not in merged:
            order.append(low)
        merged[low] = (name, value)
    return [merged[low] for low in order]


def _upstream_headers(
    base: Iterable[Tuple[str, str]], overrides: Iterable[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """The headers one upstream receives.

    A spec that carries its own credential replaces the client's whole: the
    client's ``Authorization`` AND ``x-api-key`` are both dropped before the
    spec's is added, so exactly one credential travels -- the spec's. (Before
    Step 5b only a same-named header was replaced, so an OAuth client reached the
    env fallback carrying its own bearer beside the fallback's key: two
    credentials on one request, and the client's bearer sent to whatever
    ``HAL_GATEWAY_FALLBACK_UPSTREAM`` names.)
    """
    overrides = list(overrides)
    base = list(base)
    if any(name.lower() in _AUTH_HEADERS for name, _ in overrides):
        base = [(name, value) for name, value in base if name.lower() not in _AUTH_HEADERS]
    return _merge_headers(base, overrides)


def _default_proxy_factory(
    spec: UpstreamSpec, *, max_retries: int, sleep: Callable[[float], None],
    ledger: AccountLedger,
) -> GatewayProxy:
    """A verbatim proxy for ``spec`` whose every upstream answer -- each transport
    retry's too -- passes through the per-account ledger (observe only, fail-open)."""
    return GatewayProxy(upstream_base=spec.base_url, max_retries=max_retries, sleep=sleep,
                        opener=ledger.observing(build_default_opener()))


class FailoverProxy:
    """Ordered same-format upstreams; returns the first that does not fail over.

    Duck-types :meth:`GatewayProxy.forward`, so the P1/P2 handlers use it
    unchanged. With one spec it is behaviourally identical to a bare
    ``GatewayProxy``. ``ledger`` is the per-account ledger the default proxies
    report every answer to (default: the one under ``$HAL_HOME``).
    """

    def __init__(
        self,
        specs: Iterable[UpstreamSpec],
        *,
        proxy_factory: Optional[Callable[[UpstreamSpec], object]] = None,
        telemetry: Optional[GatewayTelemetry] = None,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        ledger: Optional[AccountLedger] = None,
    ):
        self._specs: List[UpstreamSpec] = list(specs)
        if not self._specs:
            raise ValueError("FailoverProxy requires at least one UpstreamSpec")
        ledger = ledger or AccountLedger()
        factory = proxy_factory or (
            lambda s: _default_proxy_factory(s, max_retries=max_retries, sleep=sleep,
                                             ledger=ledger))
        self._proxies = [factory(s) for s in self._specs]
        self._telemetry = telemetry or GatewayTelemetry()

    def forward(
        self,
        path: str,
        body: Optional[bytes],
        client_headers: Iterable[Tuple[str, str]],
        *,
        method: str = "POST",
        max_retries: Optional[int] = None,
    ) -> UpstreamResponse:
        """Try each upstream in order; return the first non-failover response.

        Failover triggers on a raised ``GatewayUpstreamError`` (transport
        exhausted) or a failover-class status, but ONLY while a next upstream
        remains — the last upstream's response (even a 429) is relayed so the
        harness sees the real upstream error rather than a synthesised one.
        """
        base = list(client_headers)
        last = len(self._specs) - 1
        last_exc: Optional[GatewayUpstreamError] = None
        spent = []
        paid_refused = 0
        for index, (spec, proxy) in enumerate(zip(self._specs, self._proxies)):
            headers = _upstream_headers(base, spec.header_overrides)
            start = time.monotonic()
            try:
                resp = proxy.forward(path, body, headers, method=method)
            except SpendRefused as exc:
                if exc.decision.status == 429:
                    spent.append(exc.decision)
                elif (exc.decision.status == 403
                      and exc.decision.source in (API_KEY, OVERAGE)):
                    # A guarded paid source is unavailable, not authorised
                    # capacity. Never send to it without the existing opt-in.
                    paid_refused += 1
                else:
                    raise
                self._record(spec, path, status=exc.decision.status, served=False, started=start,
                             failed_over=index > 0, note="spend_refused")
                if index == last:
                    _finish_refusal(exc, spent, paid_refused, len(self._specs), last_exc)
                continue
            except GatewayUpstreamError as exc:
                last_exc = exc
                self._record(spec, path, status=None, served=False, started=start,
                             failed_over=index > 0, note="unreachable")
                if index == last:
                    # Keep the kind: a network failure on every upstream is
                    # still a transport fault, which a client is sent as 503.
                    kind = (GatewayTransportError if is_transport_error(exc)
                            else GatewayUpstreamError)
                    raise kind(
                        f"all {len(self._specs)} upstream(s) exhausted: "
                        f"{redact(str(exc))}") from exc
                continue
            if resp.status in _FAILOVER_STATUS and index < last:
                self._record(spec, path, status=resp.status, served=False,
                             started=start, failed_over=index > 0, note="failover")
                resp.close()
                continue
            self._record(spec, path, status=resp.status, served=True,
                         started=start, failed_over=index > 0)
            return resp
        # Unreachable: the loop returns or raises on the final spec.
        raise GatewayUpstreamError("failover chain exhausted without a response")

    def _record(self, spec, path, *, status, served, started, failed_over, note=""):
        self._telemetry.record({
            "event": note or "served",
            "path": path,
            "upstream": spec.name,
            "status": status,
            "served": served,
            "failed_over": failed_over,
            "latency_ms": round((time.monotonic() - started) * 1000),
        })


#: Env vars that once appended a second upstream to the chain, its only shape a
#: paid ``x-api-key`` override. Removed with the spend guard (InfiniteMachine
#: Step 5h): a paid key never counts as capacity, and a fallback HAL chooses
#: must ask ``lib/hal/spend_guard.py`` first, which an env var cannot. Set,
#: they are ignored, with one line saying so.
REMOVED_FALLBACK_ENV = ("HAL_GATEWAY_FALLBACK_UPSTREAM", "HAL_GATEWAY_FALLBACK_KEY")


def _specs_from_env(upstream_base: Optional[str] = None) -> List[UpstreamSpec]:
    """The process chain: the primary upstream only (``HAL_GATEWAY_UPSTREAM``,
    default Anthropic). The env fallback rung is gone (:data:`REMOVED_FALLBACK_ENV`);
    account failover arrives in Step 5, through the spend guard."""
    ignored = [name for name in REMOVED_FALLBACK_ENV if os.environ.get(name)]
    if ignored:
        sys.stderr.write(f"  HAL Gateway: {', '.join(ignored)} ignored: the env "
                         "fallback chain was removed (spend guard, Step 5h)\n")
    return [UpstreamSpec(
        name="anthropic-primary",
        base_url=(upstream_base if upstream_base is not None else
                  os.environ.get("HAL_GATEWAY_UPSTREAM", DEFAULT_UPSTREAM)))]


def build_failover_proxy(telemetry: Optional[GatewayTelemetry] = None, *,
                         upstream_base: Optional[str] = None) -> FailoverProxy:
    """Construct the process gateway proxy (failover-aware, env-configured)."""
    return FailoverProxy(_specs_from_env(upstream_base), telemetry=telemetry)
