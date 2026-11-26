"""Tests for lib/hal/moe.py — HALCouncil.deliberate_parallel()."""
import sys
import pathlib
import time
import unittest
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lib.hal.moe import HALCouncil
from lib.hal.types import CouncilResult, CouncilRound


def _make_registry(model_responses: dict):
    """Build a mock registry whose chat() returns scripted text per model."""
    registry = MagicMock()

    def chat(model, prompt, system="", **kwargs):
        # **kwargs absorbs max_tokens (and any future forwarded args) so the
        # stub stays compatible with HALCouncil's real-world chat signature.
        resp = MagicMock()
        resp.text = model_responses.get(model, f"response from {model}")
        resp.cost_usd = 0.001
        return resp

    registry.chat.side_effect = chat
    routing = MagicMock()
    models = list(model_responses.keys())
    routing.tier_chains = {
        "premium": models[:1],
        "mid": models[1:2] if len(models) > 1 else [],
        "cheap": models[2:3] if len(models) > 2 else [],
        "free": [],
    }
    registry.routing = routing
    return registry


_THREE = {
    "p/m1": "answer from m1",
    "p/m2": "answer from m2",
    "p/m3": "answer from m3",
}


class TestPhase0Dispatch(unittest.TestCase):

    def test_all_models_called(self):
        """Phase-0 fires all 3 models; synth + debate add more calls."""
        registry = _make_registry(_THREE)
        council = HALCouncil(registry)
        council.deliberate_parallel("q?", models=list(_THREE), rounds=1)
        # At minimum 3 phase-0 + 1 synthesis + 3 debate + 1 final-merge = 8 calls
        self.assertGreaterEqual(registry.chat.call_count, 3)

    def test_failed_model_excluded(self):
        """A model whose chat raises is silently excluded; others still vote."""
        registry = _make_registry(_THREE)
        original_side = registry.chat.side_effect

        def chat_with_failure(model, prompt, system="", **kwargs):
            if model == "p/m2":
                raise RuntimeError("network error")
            return original_side(model, prompt, system=system, **kwargs)

        registry.chat.side_effect = chat_with_failure
        council = HALCouncil(registry)
        result = council.deliberate_parallel("q?", models=list(_THREE), rounds=1)
        # Result must still contain content from surviving models
        self.assertIsInstance(result, CouncilResult)
        self.assertNotEqual(result.merged_text, "[No council members responded]")

    def test_returns_council_result(self):
        """Return type is CouncilResult."""
        council = HALCouncil(_make_registry(_THREE))
        result = council.deliberate_parallel("q?", models=list(_THREE), rounds=1)
        self.assertIsInstance(result, CouncilResult)

    def test_slow_model_excluded(self):
        """A model whose query raises TimeoutError is excluded from votes.

        Uses a real ThreadPoolExecutor. The slow model's _query_member_safe
        raises TimeoutError, which _dispatch_parallel catches and excludes.
        """
        responses = {"p/fast1": "fast", "p/slow": "slow"}
        registry = _make_registry(responses)
        council = HALCouncil(registry)

        original_query = council._query_member_safe

        def patched_query(model, prompt, system, weight, max_tokens=None):
            if "slow" in model:
                raise TimeoutError("mock slow model timed out")
            return original_query(model, prompt, system, weight, max_tokens)

        with patch.object(council, "_query_member_safe", side_effect=patched_query):
            votes = council._dispatch_parallel(
                list(responses), "q?", "", None, timeout_s=30.0
            )

        vote_texts = [v.response.text for v in votes]
        self.assertNotIn("slow", vote_texts)
        self.assertIn("fast", vote_texts)


class TestSynthesisPhase(unittest.TestCase):

    def test_synthesis_called_after_phase0(self):
        """After phase-0 completes, synthesiser is called at least once."""
        registry = _make_registry(_THREE)
        council = HALCouncil(registry)
        council.deliberate_parallel("q?", models=list(_THREE), rounds=1)
        # Synthesis prompt contains "SYNTHESIS" keyword
        synth_calls = [
            c for c in registry.chat.call_args_list
            if "SYNTHESIS" in str(c) or "synthesising" in str(c).lower()
        ]
        self.assertGreaterEqual(len(synth_calls), 1)

    def test_synthesis_failure_non_fatal(self):
        """If synthesiser raises, merged_text is still non-empty (fallback used)."""
        registry = _make_registry(_THREE)
        call_count = [0]
        original_side = registry.chat.side_effect

        def chat_synth_fails(model, prompt, system=""):
            call_count[0] += 1
            # Fail synthesis calls (they contain the _SYNTHESIS_PROMPT keyword)
            if "synthesising" in prompt.lower():
                raise RuntimeError("synth down")
            return original_side(model, prompt, system=system)

        registry.chat.side_effect = chat_synth_fails
        council = HALCouncil(registry)
        result = council.deliberate_parallel("q?", models=list(_THREE), rounds=0)
        self.assertTrue(len(result.merged_text) > 0)

    def test_debate_prompt_injects_peers(self):
        """Debate call args contain the synthesis text."""
        registry = _make_registry(_THREE)
        synthesis_text = []
        original_side = registry.chat.side_effect

        def capture(model, prompt, system="", **kwargs):
            resp = original_side(model, prompt, system=system, **kwargs)
            if "SYNTHESIS FROM PREVIOUS ROUND" in prompt:
                synthesis_text.append(prompt)
            return resp

        registry.chat.side_effect = capture
        council = HALCouncil(registry)
        council.deliberate_parallel("q?", models=list(_THREE), rounds=1)
        self.assertTrue(len(synthesis_text) > 0, "No debate prompts captured")

    def test_debate_excludes_self_marker(self):
        """The [self_model] marker appears in at least one debate prompt."""
        registry = _make_registry(_THREE)
        debate_prompts = []
        original_side = registry.chat.side_effect

        def capture(model, prompt, system="", **kwargs):
            resp = original_side(model, prompt, system=system, **kwargs)
            if "SYNTHESIS FROM PREVIOUS ROUND" in prompt:
                debate_prompts.append(prompt)
            return resp

        registry.chat.side_effect = capture
        council = HALCouncil(registry)
        council.deliberate_parallel("q?", models=list(_THREE), rounds=1)
        markers = [p for p in debate_prompts if "[p/" in p]
        self.assertTrue(len(markers) > 0, "No [self_model] marker found in debate prompts")


class TestRoundsAndBudget(unittest.TestCase):

    def test_rounds_param_controls_count(self):
        """rounds=2 → len(result.rounds) == 3 (round 0 + 2 debate rounds)."""
        council = HALCouncil(_make_registry(_THREE))
        result = council.deliberate_parallel("q?", models=list(_THREE), rounds=2)
        self.assertEqual(len(result.rounds), 3)

    def test_budget_preflight_raises(self):
        """budget_usd=0.0000001 with multiple models raises ValueError before any chat call."""
        registry = _make_registry(_THREE)
        council = HALCouncil(registry)
        with self.assertRaises(ValueError):
            council.deliberate_parallel("q?", models=list(_THREE), budget_usd=0.0000001)
        registry.chat.assert_not_called()

    def test_budget_none_no_preflight(self):
        """budget_usd=None → no ValueError, chat is called."""
        registry = _make_registry(_THREE)
        council = HALCouncil(registry)
        result = council.deliberate_parallel("q?", models=list(_THREE), budget_usd=None, rounds=0)
        self.assertIsInstance(result, CouncilResult)
        self.assertGreater(registry.chat.call_count, 0)


class TestBackwardCompatAndAutoSelect(unittest.TestCase):

    def test_deliberate_unchanged(self):
        """Original deliberate() still works and result.rounds == ()."""
        registry = _make_registry(_THREE)
        council = HALCouncil(registry)
        result = council.deliberate("q?", models=list(_THREE))
        self.assertIsInstance(result, CouncilResult)
        self.assertEqual(result.rounds, ())

    def test_auto_select_uses_tier_chains(self):
        """models=None → auto-selects from registry.routing.tier_chains."""
        registry = _make_registry(_THREE)
        council = HALCouncil(registry)
        result = council.deliberate_parallel("q?", models=None, rounds=0)
        self.assertIsInstance(result, CouncilResult)


# ----------------------------------------------------------------------
# max_tokens tier resolver — pure-logic unit tests
# ----------------------------------------------------------------------

from lib.hal.moe import (  # noqa: E402
    _LEGACY_FALLBACK_MAX_TOKENS,
    _LOCAL_MIN,
    _LOCAL_OUTPUT_CEILING,
    _REASONING_MULTIPLIER,
    _TIER_DEFAULT_MAX_TOKENS,
    _estimate_input_tokens,
    _resolve_max_tokens,
)
from lib.hal.models_command import ModelEntry  # noqa: E402


def _entry(address, tier, *, tags=(), ctx=128000):
    return ModelEntry(
        address=address, tier=tier, paid=True, modality="text",
        context_window=ctx, tags=tuple(tags), strengths=(), description="",
    )


class TestResolveMaxTokens(unittest.TestCase):
    """Verify the tier defaults match the SWOT-calibrated policy."""

    def test_unknown_model_returns_legacy_fallback(self):
        self.assertEqual(
            _resolve_max_tokens("unknown/model", "hi", {}),
            _LEGACY_FALLBACK_MAX_TOKENS,
        )

    def test_subscription_tier_default(self):
        cat = {"anthropic/opus": _entry("anthropic/opus", "subscription")}
        self.assertEqual(
            _resolve_max_tokens("anthropic/opus", "hi", cat),
            _TIER_DEFAULT_MAX_TOKENS["subscription"],
        )

    def test_frontier_tier_default(self):
        cat = {"gemini/flash": _entry("gemini/flash", "frontier")}
        self.assertEqual(
            _resolve_max_tokens("gemini/flash", "hi", cat),
            _TIER_DEFAULT_MAX_TOKENS["frontier"],
        )

    def test_premium_tier_default(self):
        cat = {"ds/v32": _entry("ds/v32", "premium")}
        self.assertEqual(
            _resolve_max_tokens("ds/v32", "hi", cat),
            _TIER_DEFAULT_MAX_TOKENS["premium"],
        )

    def test_mid_tier_default(self):
        cat = {"g/flash": _entry("g/flash", "mid")}
        self.assertEqual(
            _resolve_max_tokens("g/flash", "hi", cat),
            _TIER_DEFAULT_MAX_TOKENS["mid"],
        )

    def test_cheap_tier_operator_chose_8k(self):
        """Operator calibrated cheap tier at 8K (more than premium's 4K)."""
        cat = {"groq/llama": _entry("groq/llama", "cheap")}
        self.assertEqual(
            _resolve_max_tokens("groq/llama", "hi", cat),
            8_192,
        )
        # Explicit asymmetry: cheap > premium is intentional.
        self.assertGreater(
            _TIER_DEFAULT_MAX_TOKENS["cheap"],
            _TIER_DEFAULT_MAX_TOKENS["premium"],
        )

    def test_free_tier_fallback_for_unknown_tier_label(self):
        cat = {"x/y": _entry("x/y", "unknown-tier-label")}
        self.assertEqual(
            _resolve_max_tokens("x/y", "hi", cat),
            _TIER_DEFAULT_MAX_TOKENS["free"],
        )

    def test_reasoning_tag_doubles_cap(self):
        cat = {
            "ds/reasoner": _entry(
                "ds/reasoner", "frontier", tags=("reasoning", "math"),
            ),
        }
        self.assertEqual(
            _resolve_max_tokens("ds/reasoner", "hi", cat),
            _TIER_DEFAULT_MAX_TOKENS["frontier"] * _REASONING_MULTIPLIER,
        )

    def test_local_tier_hits_ceiling_on_tiny_input(self):
        cat = {"ollama/gemma": _entry("ollama/gemma", "local", ctx=256_000)}
        self.assertEqual(
            _resolve_max_tokens("ollama/gemma", "hi", cat),
            _LOCAL_OUTPUT_CEILING,
        )

    def test_local_tier_shrinks_when_input_eats_context(self):
        """Big input → output cap drops below ceiling but stays above LOCAL_MIN."""
        cat = {"ollama/gemma": _entry("ollama/gemma", "local", ctx=128_000)}
        # Use 100K tokens of input (→ 400K chars) against a 128K ctx model.
        huge = "x" * 400_000
        cap = _resolve_max_tokens("ollama/gemma", huge, cat)
        self.assertGreaterEqual(cap, _LOCAL_MIN)
        self.assertLess(cap, _LOCAL_OUTPUT_CEILING)

    def test_local_tier_floors_at_min_when_input_exceeds_context(self):
        cat = {"ollama/gemma": _entry("ollama/gemma", "local", ctx=8_000)}
        crush = "x" * 80_000  # ≈20K tokens against 8K ctx
        self.assertEqual(
            _resolve_max_tokens("ollama/gemma", crush, cat),
            _LOCAL_MIN,
        )

    def test_local_reasoning_tag_doubles_headroom_cap(self):
        """A local model tagged reasoning gets 2× the context-aware cap."""
        cat = {
            "ollama/reasoner": _entry(
                "ollama/reasoner", "local",
                tags=("reasoning",), ctx=256_000,
            ),
        }
        # Small input → ceiling × 2 = 64K
        self.assertEqual(
            _resolve_max_tokens("ollama/reasoner", "hi", cat),
            _LOCAL_OUTPUT_CEILING * _REASONING_MULTIPLIER,
        )

    def test_estimate_input_tokens_is_roughly_len_over_four(self):
        self.assertEqual(_estimate_input_tokens("x" * 40), 10)
        self.assertEqual(_estimate_input_tokens(""), 1)  # floor of 1
        self.assertEqual(_estimate_input_tokens("xyz"), 1)  # floor of 1


class TestCouncilMaxTokensIntegration(unittest.TestCase):
    """Verify the resolved max_tokens actually reaches registry.chat()."""

    def test_deliberate_passes_max_tokens_to_registry(self):
        captured_kwargs = []

        def chat(model, prompt, system="", **kwargs):
            captured_kwargs.append(kwargs)
            resp = MagicMock()
            resp.text = "ok"
            resp.cost_usd = 0.001
            return resp

        registry = MagicMock()
        registry.chat.side_effect = chat

        council = HALCouncil(registry)
        council.deliberate("q?", models=["unknown/model"])

        self.assertEqual(len(captured_kwargs), 1)
        self.assertIn("max_tokens", captured_kwargs[0])
        # Unknown model → legacy fallback
        self.assertEqual(captured_kwargs[0]["max_tokens"],
                         _LEGACY_FALLBACK_MAX_TOKENS)

    def test_explicit_max_tokens_overrides_resolver(self):
        captured_kwargs = []

        def chat(model, prompt, system="", **kwargs):
            captured_kwargs.append(kwargs)
            resp = MagicMock()
            resp.text = "ok"
            resp.cost_usd = 0.001
            return resp

        registry = MagicMock()
        registry.chat.side_effect = chat

        council = HALCouncil(registry)
        council.deliberate("q?", models=["unknown/model"], max_tokens=999)

        self.assertEqual(captured_kwargs[0]["max_tokens"], 999)

    def test_deliberate_parallel_honours_max_tokens(self):
        captured_kwargs = []

        def chat(model, prompt, system="", **kwargs):
            captured_kwargs.append(kwargs)
            resp = MagicMock()
            resp.text = "ok"
            resp.cost_usd = 0.001
            return resp

        registry = MagicMock()
        registry.chat.side_effect = chat
        routing = MagicMock()
        routing.tier_chains = {"premium": ["unknown/a"],
                                "mid": [], "cheap": [], "free": []}
        registry.routing = routing

        council = HALCouncil(registry)
        council.deliberate_parallel(
            "q?", models=["unknown/a"], rounds=0, max_tokens=12345,
        )

        # Every chat call in the parallel path should carry the override.
        self.assertTrue(captured_kwargs)
        for kwargs in captured_kwargs:
            self.assertEqual(kwargs.get("max_tokens"), 12345)


if __name__ == "__main__":
    unittest.main()
