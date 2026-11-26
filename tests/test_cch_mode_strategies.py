"""Tests for D5 CCH mode → context-pruning strategy table (TF>> liberation matrix, 2026-04-20).

Verifies that ``cch.py`` exposes ``resolve_context_strategy(mode)`` backed by
a TOML table under ``configs/mode_context_strategies.toml``. Mirrors TF>>'s
declarative pattern and HAL's D2 pattern, but for context-pruning kwargs
rather than model capabilities.

Falsifiable claims:
- The TOML file ships at ``HAL_HOME/configs/mode_context_strategies.toml``.
- ``resolve_context_strategy("analysis")`` returns ``clear_tool_results=True``
  and the analysis-tuned token thresholds (regression baseline).
- ``resolve_context_strategy("rsi")`` returns BOTH clear flags True
  (autonomous loops prune aggressively).
- ``resolve_context_strategy("plan")`` returns BOTH clear flags False
  (plan mode is read-only and budget-generous).
- An unknown mode falls back to ``[modes."default"]`` from the TOML when
  present, else to ``_DEFAULT_CONTEXT_STRATEGY`` from cch.py.
- The returned dict is a fresh copy — mutating it does NOT leak into the
  next call's result.
- The loader filters unknown TOML keys, so a typo in ``mode_context_strategies.toml``
  cannot inject arbitrary kwargs into the API call.
- Missing TOML file degrades to ``_DEFAULT_CONTEXT_STRATEGY`` (permissive,
  unlike the model registry which raises ImportError).

Goodhart guard (Rule 14): tests assert the LITERAL token thresholds and
clear-flag values per mode, not just "strategy returned a dict". A test
that only asserted ``isinstance(result, dict)`` would pass for any dict
including an empty one.
"""
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.utils.cch import (
    _DEFAULT_CONTEXT_STRATEGY,
    _MODE_STRATEGIES_TOML,
    _VALID_STRATEGY_KEYS,
    _load_mode_strategies,
    resolve_context_strategy,
)


class TestStrategyTomlShipped(unittest.TestCase):
    def test_toml_file_exists(self):
        self.assertTrue(
            _MODE_STRATEGIES_TOML.is_file(),
            f"configs/mode_context_strategies.toml must exist at {_MODE_STRATEGIES_TOML}",
        )


class TestResolveKnownModes(unittest.TestCase):
    """Each known mode must return its tuned strategy verbatim."""

    def test_analysis_mode_prunes_results_only(self):
        s = resolve_context_strategy("analysis")
        self.assertTrue(s["clear_tool_results"])
        self.assertFalse(s["clear_tool_uses"])
        self.assertEqual(s["max_input_tokens"], 150_000)
        self.assertEqual(s["target_input_tokens"], 60_000)

    def test_code_mode_prunes_results_keeps_uses(self):
        s = resolve_context_strategy("code")
        self.assertTrue(s["clear_tool_results"])
        self.assertFalse(s["clear_tool_uses"])
        self.assertEqual(s["max_input_tokens"], 180_000)
        self.assertEqual(s["target_input_tokens"], 50_000)

    def test_rsi_mode_prunes_both(self):
        """Autonomous loops fire many tools per turn — prune both kinds."""
        s = resolve_context_strategy("rsi")
        self.assertTrue(s["clear_tool_results"])
        self.assertTrue(s["clear_tool_uses"])
        self.assertEqual(s["target_input_tokens"], 30_000)

    def test_ultrado_mode_matches_rsi(self):
        """Ultrado and RSI are both autonomous — same strategy shape."""
        rsi = resolve_context_strategy("rsi")
        ultrado = resolve_context_strategy("ultrado")
        self.assertEqual(rsi["clear_tool_results"], ultrado["clear_tool_results"])
        self.assertEqual(rsi["clear_tool_uses"], ultrado["clear_tool_uses"])

    def test_plan_mode_no_pruning(self):
        """Plan mode is read-only; pruning offers no benefit."""
        s = resolve_context_strategy("plan")
        self.assertFalse(s["clear_tool_results"])
        self.assertFalse(s["clear_tool_uses"])
        self.assertEqual(s["max_input_tokens"], 200_000)


class TestResolveFallback(unittest.TestCase):
    def test_unknown_mode_uses_table_default(self):
        """Unknown mode falls through to [modes.\"default\"] in TOML."""
        s = resolve_context_strategy("totally-fake-mode-xyz")
        # The shipped default has both clear flags False (conservative).
        self.assertFalse(s["clear_tool_results"])
        self.assertFalse(s["clear_tool_uses"])
        self.assertEqual(s["max_input_tokens"], 180_000)

    def test_none_mode_uses_table_default(self):
        s = resolve_context_strategy(None)
        self.assertFalse(s["clear_tool_results"])
        self.assertFalse(s["clear_tool_uses"])

    def test_returned_dict_is_a_copy(self):
        """Mutating the result must not leak into the next call."""
        s1 = resolve_context_strategy("rsi")
        s1["clear_tool_results"] = False
        s2 = resolve_context_strategy("rsi")
        self.assertTrue(s2["clear_tool_results"], "subsequent call leaked mutated state")


class TestLoaderFiltersInvalidKeys(unittest.TestCase):
    """A typo in the TOML must not silently inject bogus kwargs."""

    def test_unknown_keys_dropped(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False,
        ) as fh:
            fh.write(textwrap.dedent("""
                [modes."test-mode"]
                clear_tool_results = true
                bogus_typo = true
                clear_tool_uses_typo = true
                max_input_tokens = 100000
            """).strip())
            tmp = Path(fh.name)
        try:
            table = _load_mode_strategies(tmp)
            self.assertIn("test-mode", table)
            self.assertNotIn("bogus_typo", table["test-mode"])
            self.assertNotIn("clear_tool_uses_typo", table["test-mode"])
            self.assertEqual(table["test-mode"]["clear_tool_results"], True)
            self.assertEqual(table["test-mode"]["max_input_tokens"], 100_000)
        finally:
            tmp.unlink()

    def test_only_valid_keys_pass(self):
        """All four declared schema keys must round-trip through the loader."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False,
        ) as fh:
            fh.write(textwrap.dedent("""
                [modes."test-mode"]
                clear_tool_results = true
                clear_tool_uses = true
                max_input_tokens = 123456
                target_input_tokens = 7890
            """).strip())
            tmp = Path(fh.name)
        try:
            table = _load_mode_strategies(tmp)
            self.assertEqual(set(table["test-mode"].keys()), _VALID_STRATEGY_KEYS)
        finally:
            tmp.unlink()


class TestLoaderPermissive(unittest.TestCase):
    """Unlike the model registry, missing/empty strategy table is OK."""

    def test_missing_file_returns_empty_dict(self):
        nonexistent = Path("/tmp/hal_strategies_does_not_exist_d5.toml")
        if nonexistent.exists():
            nonexistent.unlink()
        self.assertEqual(_load_mode_strategies(nonexistent), {})

    def test_no_modes_section_returns_empty(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False,
        ) as fh:
            fh.write("[other]\nkey = 'value'\n")
            tmp = Path(fh.name)
        try:
            self.assertEqual(_load_mode_strategies(tmp), {})
        finally:
            tmp.unlink()


class TestDefaultStrategy(unittest.TestCase):
    def test_default_has_all_four_keys(self):
        self.assertEqual(set(_DEFAULT_CONTEXT_STRATEGY.keys()), _VALID_STRATEGY_KEYS)

    def test_default_is_conservative(self):
        """Both clear flags must default to False (no pruning)."""
        self.assertFalse(_DEFAULT_CONTEXT_STRATEGY["clear_tool_results"])
        self.assertFalse(_DEFAULT_CONTEXT_STRATEGY["clear_tool_uses"])


if __name__ == "__main__":
    unittest.main()
