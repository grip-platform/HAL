"""Anchor tests: the Router stamps a short ``reason`` on every response it
returns, and a direct ``registry.chat`` caller leaves it None.

Origin: AGORA DEFER on PR #763 — ``ChatResponse.reason`` was added and stamped
by each Router strategy, but nothing consumed it and no test anchored the
per-strategy strings. This file locks the stamping contract so a strategy that
forgets (or mis-labels) its reason fails CI.

Goodhart-proof: asserts the EXACT per-strategy string (not a prefix), so a
renamed or dropped reason fails; and asserts a direct chat call leaves
``reason=None``, pinning the "None for direct callers" contract.
"""
from __future__ import annotations

from lib.hal.providers.openai_compat import OpenAICompatError
from lib.hal.router import Router
from lib.hal.types import ChatResponse, RoutingStrategy


def _chat_response(address: str) -> ChatResponse:
    provider, _, model = address.partition("/")
    return ChatResponse(text="ok", model=model, provider=provider)


def _mk_registry(
    *,
    fallback_chain=None,
    quality_cost_tiers=None,
    tier_chains=None,
    fail_first: str | None = None,
):
    """Stub registry whose ``chat`` returns a real ChatResponse (reason=None by
    default), optionally raising OpenAICompatError for ``fail_first`` so the
    fallback chain advances. No network, no providers.yaml.
    """
    class _Routing:
        default_model = "p/m"
        default_strategy = RoutingStrategy.FIXED

    routing = _Routing()
    routing.fallback_chain = fallback_chain or []
    routing.quality_cost_tiers = quality_cost_tiers or {}
    routing.tier_chains = tier_chains or {}

    class _Stub:
        def __init__(self):
            self.routing = routing

        def chat(self, address, prompt, **kwargs):
            if fail_first is not None and address == fail_first:
                raise OpenAICompatError("simulated provider failure")
            return _chat_response(address)

    return _Stub()


def test_route_fixed_stamps_reason_fixed():
    router = Router(_mk_registry())
    resp = router._route_fixed("hi", model="p/m")
    assert resp.reason == "fixed"


def test_route_fallback_primary_stamps_fixed():
    # i == 0 is the chain's primary — still the operator's first choice.
    router = Router(_mk_registry(fallback_chain=["p/m"]))
    resp = router._route_fallback("hi")
    assert resp.reason == "fixed"


def test_route_fallback_secondary_stamps_fallback():
    # i == 1 is a genuine fallback after the primary failed.
    router = Router(_mk_registry(fallback_chain=["p/a", "p/b"], fail_first="p/a"))
    resp = router._route_fallback("hi")
    assert resp.reason == "fallback"
    assert resp.provider == "p" and resp.model == "b"


def test_route_quality_cost_stamps_tier():
    router = Router(
        _mk_registry(
            quality_cost_tiers={"simple": "p/m", "standard": "p/s", "complex": "p/c"},
        )
    )
    resp = router._route_quality_cost("hi")  # short → "simple"
    assert resp.reason == "quality-cost:simple"


def test_route_tier_chain_stamps_fallback():
    # _detect_cost_tier("p/a") → "mid" (fake model, no pricing) → tier_chains["mid"].
    router = Router(_mk_registry(tier_chains={"mid": ["p/b"]}))
    resp = router._route_tier_chain("hi", "p/a")
    assert resp.reason == "fallback"


def test_direct_chat_leaves_reason_none():
    registry = _mk_registry()
    resp = registry.chat("p/m", "hi")
    assert resp.reason is None
