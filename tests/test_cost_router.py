"""Tests for HAL cost-aware router."""
import os
import sys
import unittest
from unittest.mock import MagicMock, call

_WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HAL_HOME = os.path.expanduser("~/.hal")

# Insert worktree first; evict any stale lib.hal cache from earlier test files.
# Save/restore pattern: only evict, never permanently replace stdlib modules.
if sys.path[:1] != [_WORKTREE]:
    sys.path.insert(0, _WORKTREE)
if _HAL_HOME not in sys.path:
    sys.path.append(_HAL_HOME)


def _cached_lib_is_stale() -> bool:
    """True when a cached ``lib`` package was loaded from outside this checkout,
    the only case the eviction below is for. Evicting a ``lib`` this checkout
    loaded split the module objects between test files: a file imported earlier
    kept the old ``lib.hal.infinite_cost`` while ``gateway``'s lazy import found
    a new one, so its patched observer never ran (HAL#800)."""
    cached = sys.modules.get("lib")
    if cached is None:
        return False
    here = os.path.realpath(_WORKTREE) + os.sep
    return not any(os.path.realpath(p).startswith(here)
                   for p in getattr(cached, "__path__", []))


if _cached_lib_is_stale():
    for _m in [m for m in sys.modules if m == "lib" or m.startswith("lib.hal")]:
        del sys.modules[_m]

from lib.hal.cost_router import (
    CostRouter,
    RouteDecision,
    _assess_complexity,
    _classify_tier,
    _partition_by_tier,
)
from lib.hal.providers.openai_compat import OpenAICompatError
from lib.hal.router import Router
from lib.hal.types import ChatResponse, TokenUsage


class TestAssessComplexity(unittest.TestCase):

    def test_simple_query(self):
        self.assertEqual(_assess_complexity("What time is it?"), "simple")

    def test_medium_with_code(self):
        self.assertEqual(_assess_complexity("Fix this: def broken(): pass"), "medium")

    def test_complex_query(self):
        result = _assess_complexity("Architect a microservices migration plan for the auth system")
        self.assertEqual(result, "complex")

    def test_long_query_is_complex(self):
        result = _assess_complexity("word " * 250)
        self.assertEqual(result, "complex")


class TestClassifyTier(unittest.TestCase):

    def test_free(self):
        self.assertEqual(_classify_tier(0.0), "free")

    def test_cheap(self):
        self.assertEqual(_classify_tier(0.5), "cheap")

    def test_mid(self):
        self.assertEqual(_classify_tier(3.0), "mid")

    def test_premium(self):
        self.assertEqual(_classify_tier(15.0), "premium")


class TestPartitionByTier(unittest.TestCase):

    def test_known_models(self):
        models = ["groq/llama-3.3-70b-versatile", "anthropic/claude-opus-4-7"]
        tiers = _partition_by_tier(models)
        self.assertIn("groq/llama-3.3-70b-versatile", tiers["cheap"])
        # claude-opus-4-7 is $5/mtok input — exactly at premium boundary
        self.assertIn("anthropic/claude-opus-4-7", tiers["premium"])

    def test_unknown_model_goes_to_mid(self):
        tiers = _partition_by_tier(["provider/unknown-model-xyz"])
        self.assertIn("provider/unknown-model-xyz", tiers["mid"])

    def test_empty(self):
        tiers = _partition_by_tier([])
        self.assertEqual(sum(len(v) for v in tiers.values()), 0)


class TestCostRouterNoRegistry(unittest.TestCase):

    def test_no_providers(self):
        router = CostRouter()
        decision = router.select("Hello")
        self.assertEqual(decision.tier, "none")

    def test_route_decision_fields(self):
        d = RouteDecision(model="groq/llama", tier="cheap", reason="test", estimated_cost_usd=0.001)
        self.assertEqual(d.model, "groq/llama")
        self.assertEqual(d.tier, "cheap")


class TestTierChainUsedOnFailure(unittest.TestCase):
    """Tier chain is tried when the primary cheap model fails."""

    def _make_registry(self, tier_chains):
        registry = MagicMock()
        # Use a plain namespace to avoid RoutingConfig version skew in discovery.
        routing = MagicMock()
        routing.quality_cost_tiers = {
            "simple": "qwen/qwen3.5-flash",
            "standard": "groq/llama-3.3-70b-versatile",
            "complex": "gemini/gemini-3.1-pro-preview",
        }
        routing.tier_chains = tier_chains
        routing.fallback_chain = []
        routing.default_model = "groq/llama-3.3-70b-versatile"
        registry.routing = routing
        return registry

    def test_tier_chain_used_on_failure(self):
        """When qwen3.5-flash fails, the router tries the next cheap-tier model."""
        ok_response = ChatResponse(
            text="ok", model="groq/llama-3.3-70b-versatile",
            provider="groq", usage=TokenUsage(),
        )
        registry = self._make_registry({
            "cheap": ["qwen/qwen3.5-flash", "openrouter/google/gemma-4-31b-it", "groq/llama-3.3-70b-versatile"],
        })

        def chat_side_effect(address, prompt, **kwargs):
            if address == "qwen/qwen3.5-flash":
                raise OpenAICompatError("rate limited")
            return ok_response

        registry.chat.side_effect = chat_side_effect
        router = Router(registry)
        result = router._route_quality_cost("what time is it")

        # Second call must be the next cheap-tier model, not a premium one
        calls = [c[0][0] for c in registry.chat.call_args_list]
        self.assertIn("openrouter/google/gemma-4-31b-it", calls)
        premium_models = {"gemini/gemini-3.1-pro-preview", "deepseek/deepseek-reasoner"}
        self.assertTrue(premium_models.isdisjoint(set(calls)),
                        f"Premium model was tried before exhausting cheap tier: {calls}")
        self.assertEqual(result.text, "ok")

    def test_tier_chains_empty_falls_back_to_global(self):
        """Empty tier_chains dict preserves existing behaviour — tries global standard tier."""
        ok_response = ChatResponse(
            text="ok", model="groq/llama-3.3-70b-versatile",
            provider="groq", usage=TokenUsage(),
        )
        registry = self._make_registry({})  # empty tier_chains

        def chat_side_effect(address, prompt, **kwargs):
            if address == "qwen/qwen3.5-flash":
                raise OpenAICompatError("rate limited")
            return ok_response

        registry.chat.side_effect = chat_side_effect
        router = Router(registry)
        result = router._route_quality_cost("what time is it")

        calls = [c[0][0] for c in registry.chat.call_args_list]
        # With no tier chain, must fall back to the global standard model
        self.assertIn("groq/llama-3.3-70b-versatile", calls)
        self.assertEqual(result.text, "ok")


class TestStaleLibEviction(unittest.TestCase):
    """HAL#800: the import-time eviction runs only for a ``lib`` cached from
    outside this checkout, so test files share one set of module objects."""

    def _stale_with_cached_lib(self, path):
        fake = type(sys)("lib")
        fake.__path__ = [path]
        saved = sys.modules.get("lib")
        sys.modules["lib"] = fake
        try:
            return _cached_lib_is_stale()
        finally:
            if saved is None:
                sys.modules.pop("lib", None)
            else:
                sys.modules["lib"] = saved

    def test_a_lib_from_this_checkout_is_kept(self):
        self.assertFalse(self._stale_with_cached_lib(os.path.join(_WORKTREE, "lib")))

    def test_a_lib_from_elsewhere_is_evicted(self):
        self.assertTrue(self._stale_with_cached_lib("/nonexistent-elsewhere/hal/lib"))

    def test_no_cached_lib_means_nothing_to_evict(self):
        saved = sys.modules.pop("lib", None)
        try:
            self.assertFalse(_cached_lib_is_stale())
        finally:
            if saved is not None:
                sys.modules["lib"] = saved


if __name__ == "__main__":
    unittest.main()
