"""Unit tests for the num_ctx policy decision layer (#1228, W3).

The policy module is a pure-function layer — no I/O, no Ollama dependency —
so every decision path can be exercised in-memory with no mocks. These
tests lock down the decision matrix for:

- ``select_num_ctx(estimated, current, model_max)`` — the plan-compliant
  single-integer API (returns current when no reload, next tier otherwise)
- ``pick_ctx_tier(estimated, arch_max)`` — the underlying tier picker
- ``estimate_token_count(messages)`` — the chars/4 heuristic
- ``plan_ctx_for_request(...)`` dataclass surface when no network

Each test names the specific contract condition it guards so a regression
in ``_CTX_TIERS`` ordering, ``_CTX_SAFETY`` threshold, or the clamp logic
surfaces as one obvious failure rather than a tangled cluster.
"""
import os
import sys
import unittest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.providers.num_ctx_policy import (  # noqa: E402
    _CTX_SAFETY,
    _CTX_TIERS,
    CtxDecision,
    estimate_token_count,
    format_reload_log,
    pick_ctx_tier,
    select_num_ctx,
)


class TestSelectNumCtxSafetyBand(unittest.TestCase):
    """``select_num_ctx`` returns ``current`` when within the safety band."""

    def test_far_below_safety_returns_current(self):
        # 100 tokens << 80% of 4096 -> no reload
        self.assertEqual(select_num_ctx(100, 4096, 131072), 4096)

    def test_just_at_safety_boundary_returns_current(self):
        # int(0.8 * 4096) = 3276, so 3276 still fits -> no reload
        self.assertEqual(select_num_ctx(3276, 4096, 131072), 4096)

    def test_one_over_safety_boundary_triggers_reload(self):
        # 3277 > int(0.8 * 4096) = 3276 -> reload to next tier (16K)
        self.assertEqual(select_num_ctx(3277, 4096, 131072), 16384)


class TestSelectNumCtxTierEscalation(unittest.TestCase):
    """Escalation picks the smallest tier that fits estimated / safety."""

    def test_escalates_from_4k_to_16k(self):
        # estimated 5000 -> target = 6250 -> 16K tier
        self.assertEqual(select_num_ctx(5000, 4096, 131072), 16384)

    def test_escalates_to_32k(self):
        # 20K tokens -> target = 25000 -> 32K tier
        self.assertEqual(select_num_ctx(20000, 4096, 131072), 32768)

    def test_escalates_to_64k(self):
        # 40K tokens -> target = 50000 -> 64K tier
        self.assertEqual(select_num_ctx(40000, 4096, 131072), 65536)

    def test_escalates_to_128k(self):
        # 100K tokens -> target = 125000 -> 128K tier
        self.assertEqual(select_num_ctx(100000, 4096, 131072), 131072)


class TestSelectNumCtxClamp(unittest.TestCase):
    """``select_num_ctx`` clamps at ``model_max`` when the ask exceeds it."""

    def test_beyond_model_max_returns_model_max(self):
        # Model caps at 8192 but we need a 64K tier -> clamp
        self.assertEqual(select_num_ctx(40000, 4096, 8192), 8192)

    def test_model_max_smaller_than_smallest_tier(self):
        # Weirdly small model max still clamps, doesn't crash
        self.assertEqual(select_num_ctx(100, 0, 2048), 2048)


class TestSelectNumCtxColdLoad(unittest.TestCase):
    """``current <= 0`` means no model loaded; skip safety gate."""

    def test_cold_load_zero_current_picks_smallest_fitting_tier(self):
        # current=0 -> skip safety band, pick tier for 100 tokens -> 4K
        self.assertEqual(select_num_ctx(100, 0, 131072), 4096)

    def test_cold_load_negative_current_treated_as_zero(self):
        self.assertEqual(select_num_ctx(100, -1, 131072), 4096)


class TestSelectNumCtxZeroEstimated(unittest.TestCase):
    """Degenerate empty-message path: estimated=0 fits any current."""

    def test_zero_estimated_with_loaded_model_returns_current(self):
        self.assertEqual(select_num_ctx(0, 32768, 131072), 32768)

    def test_zero_estimated_cold_load_picks_smallest_tier(self):
        self.assertEqual(select_num_ctx(0, 0, 131072), 4096)


class TestPickCtxTier(unittest.TestCase):
    """Underlying tier picker: smallest tier >= estimated / _CTX_SAFETY."""

    def test_estimated_zero_returns_smallest_tier(self):
        self.assertEqual(pick_ctx_tier(0), _CTX_TIERS[0])

    def test_estimated_just_below_smallest_returns_smallest(self):
        # target for est=3000 = 3751 -> still 4K
        self.assertEqual(pick_ctx_tier(3000), 4096)

    def test_estimated_above_largest_clamps_to_arch_max(self):
        self.assertEqual(pick_ctx_tier(500000, arch_max=131072), 131072)


class TestEstimateTokenCount(unittest.TestCase):
    """Heuristic: chars / _CHARS_PER_TOKEN, aggregated over all messages."""

    def test_single_string_content(self):
        # 4 chars -> 1 token
        self.assertEqual(estimate_token_count([{"role": "u", "content": "abcd"}]), 1)

    def test_multiple_messages_summed(self):
        self.assertEqual(
            estimate_token_count([
                {"role": "u", "content": "a" * 100},
                {"role": "a", "content": "b" * 200},
            ]),
            75,  # 300 chars / 4
        )

    def test_list_of_parts_content(self):
        msg = {"role": "u", "content": [
            {"type": "text", "text": "a" * 40},
            {"type": "text", "text": "b" * 40},
        ]}
        self.assertEqual(estimate_token_count([msg]), 20)  # 80 / 4

    def test_none_and_empty_content_contribute_zero(self):
        self.assertEqual(estimate_token_count([
            {"role": "u", "content": None},
            {"role": "a", "content": ""},
            {"role": "u", "content": []},
        ]), 0)


class TestCtxDecisionDataclass(unittest.TestCase):
    """The ``CtxDecision`` dataclass is the public return type of
    :func:`plan_ctx_for_request`. Guard its field set so adapters don't
    start reading fields that don't exist.
    """

    def test_ctx_decision_fields(self):
        d = CtxDecision(
            reload=True, new_ctx=16384, old_ctx=4096,
            estimated_tokens=5000, trigger_reason="ctx_upgrade",
            keep_alive="2026-04-17T10:00:00Z",
        )
        self.assertTrue(d.reload)
        self.assertEqual(d.new_ctx, 16384)
        self.assertEqual(d.old_ctx, 4096)
        self.assertEqual(d.estimated_tokens, 5000)
        self.assertEqual(d.trigger_reason, "ctx_upgrade")
        self.assertEqual(d.keep_alive, "2026-04-17T10:00:00Z")

    def test_ctx_decision_is_frozen(self):
        d = CtxDecision(
            reload=False, new_ctx=4096, old_ctx=4096,
            estimated_tokens=100, trigger_reason="within_safety_band",
            keep_alive=None,
        )
        with self.assertRaises(Exception):  # FrozenInstanceError subclasses AttributeError
            d.new_ctx = 65536


class TestFormatReloadLog(unittest.TestCase):
    """Log format: machine-parseable key=value single line."""

    def test_log_contains_all_expected_keys(self):
        d = CtxDecision(
            reload=True, new_ctx=16384, old_ctx=4096,
            estimated_tokens=5000, trigger_reason="ctx_upgrade",
            keep_alive="29m54s",
        )
        line = format_reload_log(d, model="gemma4:e4b")
        for key in (
            "model=gemma4:e4b", "reload=True", "old_ctx=4096",
            "new_ctx=16384", "est_tokens=5000", "trigger=ctx_upgrade",
            "keep_alive=29m54s",
        ):
            self.assertIn(key, line)


class TestConstants(unittest.TestCase):
    """Lock down the tier ladder + safety fraction — regressions here
    silently change which requests trigger a reload."""

    def test_ctx_tiers_ordered_and_complete(self):
        self.assertEqual(_CTX_TIERS, (4096, 16384, 32768, 65536, 131072))
        # Ordering invariant: strictly increasing
        for i in range(len(_CTX_TIERS) - 1):
            self.assertLess(_CTX_TIERS[i], _CTX_TIERS[i + 1])

    def test_ctx_safety_is_eighty_percent(self):
        self.assertEqual(_CTX_SAFETY, 0.8)


if __name__ == "__main__":
    unittest.main()
