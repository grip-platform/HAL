"""Tests for HAL model-capability registry (D2 from TF>> liberation matrix, 2026-04-20).

Verifies cch.py loads MODEL_CAPABILITIES from configs/models.toml at import
time, replacing the hardcoded dict that previously lived inline. Mirrors
TF>>'s declarative model-registry pattern from src/utils/model/configs.ts.

Falsifiable claims:
- configs/models.toml exists at HAL_HOME and parses as valid TOML.
- The loader returns at least the three currently-shipping models
  (claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5-20251001) with
  the same capability flags they had pre-D2 (regression baseline).
- supports_capability() and filter_beta_header() behaviour is unchanged
  from the inline-dict implementation (the refactor is data-source-only).
- A missing TOML raises ImportError, NOT a silent empty-dict fallback —
  the empty-dict failure mode would degrade every model to conservative
  defaults without surfacing the regression.
- Malformed TOML (no [models.*] sections) raises ImportError for the
  same reason.

Goodhart guard (Rule 14): tests assert specific model identities and
specific capability values, not just dict-non-empty. A test that only
asserted len(MODEL_CAPABILITIES) > 0 would pass even if every entry
shipped with all-false capabilities.
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
    MODEL_CAPABILITIES,
    _MODELS_TOML,
    _load_model_capabilities,
    filter_beta_header,
    supports_capability,
)


class TestRegistryLoadedFromToml(unittest.TestCase):
    def test_toml_file_exists(self):
        self.assertTrue(
            _MODELS_TOML.is_file(),
            f"configs/models.toml must exist at {_MODELS_TOML}",
        )

    def test_three_known_models_present(self):
        for model in (
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ):
            self.assertIn(
                model, MODEL_CAPABILITIES,
                f"{model} must be present in the registry (regression baseline)",
            )

    def test_capability_baseline_unchanged(self):
        """Pre-D2 capability flags must round-trip via TOML unchanged."""
        baseline = {
            "claude-opus-4-7":          {"adaptive_thinking": True,  "context_management": True},
            "claude-sonnet-4-6":         {"adaptive_thinking": True,  "context_management": True},
            "claude-haiku-4-5-20251001": {"adaptive_thinking": False, "context_management": True},
        }
        for model, caps in baseline.items():
            for cap, expected in caps.items():
                self.assertEqual(
                    MODEL_CAPABILITIES[model][cap], expected,
                    f"{model}.{cap}: TOML registry must match pre-D2 baseline",
                )


class TestSupportsCapabilityRegression(unittest.TestCase):
    """The refactor is data-source-only; behaviour must be unchanged."""

    def test_known_true_capability(self):
        self.assertTrue(supports_capability("claude-opus-4-7", "adaptive_thinking"))

    def test_known_false_capability(self):
        self.assertFalse(supports_capability("claude-haiku-4-5-20251001", "adaptive_thinking"))

    def test_unknown_model_falls_to_conservative(self):
        self.assertFalse(supports_capability("claude-imaginary-9-9", "adaptive_thinking"))
        self.assertFalse(supports_capability("claude-imaginary-9-9", "context_management"))

    def test_unknown_capability_returns_false(self):
        self.assertFalse(supports_capability("claude-opus-4-7", "made_up_feature"))


class TestFilterBetaHeaderRegression(unittest.TestCase):
    def test_haiku_drops_adaptive_thinking_token(self):
        beta_in = "claude-code-20250219,adaptive-thinking-2026-01-28,context-management-2025-06-27"
        result = filter_beta_header(beta_in, model="claude-haiku-4-5-20251001")
        self.assertNotIn("adaptive-thinking-2026-01-28", result)
        self.assertIn("context-management-2025-06-27", result)
        self.assertIn("claude-code-20250219", result)

    def test_opus_keeps_all_tokens(self):
        beta_in = "claude-code-20250219,adaptive-thinking-2026-01-28,context-management-2025-06-27"
        result = filter_beta_header(beta_in, model="claude-opus-4-7")
        self.assertIn("adaptive-thinking-2026-01-28", result)
        self.assertIn("context-management-2025-06-27", result)


class TestLoaderFailsLoud(unittest.TestCase):
    """Negative tests for the loader's failure modes."""

    def test_missing_file_raises_import_error(self):
        nonexistent = Path("/tmp/hal_registry_does_not_exist_d2.toml")
        if nonexistent.exists():
            nonexistent.unlink()
        with self.assertRaises(ImportError):
            _load_model_capabilities(nonexistent)

    def test_empty_models_section_raises_import_error(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False,
        ) as fh:
            fh.write("# no [models.*] sections\n[other]\nkey = 'value'\n")
            tmp_path = Path(fh.name)
        try:
            with self.assertRaises(ImportError):
                _load_model_capabilities(tmp_path)
        finally:
            tmp_path.unlink()

    def test_loader_coerces_non_bool_caps_to_skipped(self):
        """Defensive: only bool fields land in the returned caps dict."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False,
        ) as fh:
            fh.write(textwrap.dedent("""
                [models."test-model"]
                adaptive_thinking = true
                bogus_string = "not a bool"
                bogus_int = 42
            """).strip())
            tmp_path = Path(fh.name)
        try:
            caps = _load_model_capabilities(tmp_path)
            self.assertEqual(caps["test-model"]["adaptive_thinking"], True)
            self.assertNotIn("bogus_string", caps["test-model"])
            self.assertNotIn("bogus_int", caps["test-model"])
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
