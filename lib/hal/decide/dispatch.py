"""``decide()``: one call per state, with every question batched into it.

Batching is where a decision model pays: one request carries a fixed overhead
of about 270 tokens and each further yes/no question about 15 more (measured
2026-10-10), so eight questions cost about 1,420 tokens in one call and about
10,500 in eight.

When no decision model can answer (a malformed request, an unknown model, a
closed spend guard, a missing key or shield, a failed send), ``decide()``
returns an explicit unavailable result with a reason code, never a guess. Each
caller keeps its own fallback for that case.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from lib.hal.decide import catalog
from lib.hal.decide.contract import DecisionProvider, DecisionResult, InvalidDecision, validate
from lib.hal.decide.typesafe import TypeSafeDecisionProvider

DEFAULT_MODEL = "jev-latest"
_PROVIDERS = {"typesafe": TypeSafeDecisionProvider}


def decide(state: Any, questions: Mapping[str, dict], *, model: str = DEFAULT_MODEL,
           provider: Optional[DecisionProvider] = None) -> DecisionResult:
    """Ask ``model`` every question about ``state`` in one call. Never raises."""
    try:
        validate(state, questions)
    except InvalidDecision as exc:
        return DecisionResult.unavailable("invalid_request", str(exc))
    try:
        entry, spec = catalog.resolve(model)
    except KeyError:
        return DecisionResult.unavailable("unknown_model", f"{model!r} is not in {catalog.config_path()}")
    except catalog.CatalogError as exc:
        return DecisionResult.unavailable("catalog_unavailable", str(exc))
    chosen = provider if provider is not None else _provider_for(spec)
    if chosen is None:
        return DecisionResult.unavailable("catalog_unavailable", f"no provider class for {spec.name!r}")
    try:
        return chosen.decide(entry, state, questions)
    except Exception as exc:  # noqa: BLE001 -- the contract is an answer or a reason, never a raise
        return DecisionResult.unavailable("provider_error", type(exc).__name__,
                                          provider=getattr(chosen, "name", ""))


def _provider_for(spec) -> Optional[DecisionProvider]:
    factory = _PROVIDERS.get(spec.name)
    return factory(spec) if factory is not None else None
