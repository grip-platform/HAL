"""Tests for MODEL_CAPABILITIES in CCHSigner (W4, #1229).

Haiku 4.5 was 400-ing on the adaptive-thinking beta header because the
CCH signer unconditionally sent it. W4 introduces a per-model capability
map and gates the header + body field behind it so:

- Opus 4.7 / Sonnet 4.6 still get adaptive_thinking + context_management
  (unchanged behaviour)
- Haiku 4.5 gets neither adaptive_thinking body field nor the
  adaptive-thinking-2026-01-28 beta token (fixes the 400)
- Unknown models degrade to the conservative default (both capabilities
  off) rather than 400-ing on beta tokens they don't recognise

Tests exercise:
- supports_capability() for known + unknown models
- filter_beta_header() preserves unrelated tokens and strips only
  capability-gated ones
- CCHSigner.build_body() emits or omits 'thinking' based on capability
- CCHSigner.build_body() emits or omits 'context_management' likewise
- CCHSigner.build_request() passes a filtered anthropic-beta header
- The adaptive thinking token is dropped for Haiku 4.5 and for unknown
  models
"""
import os
import sys
import unittest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.utils.cch import (  # noqa: E402
    CCHSigner,
    MODEL_CAPABILITIES,
    filter_beta_header,
    supports_capability,
)


class TestSupportsCapability(unittest.TestCase):
    def test_opus_supports_adaptive_thinking(self):
        self.assertTrue(supports_capability("claude-opus-4-7", "adaptive_thinking"))

    def test_sonnet_supports_adaptive_thinking(self):
        self.assertTrue(supports_capability("claude-sonnet-4-6", "adaptive_thinking"))

    def test_haiku_does_not_support_adaptive_thinking(self):
        self.assertFalse(
            supports_capability("claude-haiku-4-5-20251001", "adaptive_thinking"),
        )

    def test_haiku_supports_context_management(self):
        self.assertTrue(
            supports_capability("claude-haiku-4-5-20251001", "context_management"),
        )

    def test_unknown_model_adaptive_thinking_defaults_false(self):
        self.assertFalse(supports_capability("some-future-model-xyz", "adaptive_thinking"))

    def test_unknown_model_context_management_defaults_false(self):
        self.assertFalse(supports_capability("some-future-model-xyz", "context_management"))

    def test_unknown_capability_defaults_false(self):
        self.assertFalse(supports_capability("claude-opus-4-7", "telekinesis"))


class TestModelCapabilitiesMap(unittest.TestCase):
    def test_known_model_entries_present(self):
        self.assertIn("claude-opus-4-7", MODEL_CAPABILITIES)
        self.assertIn("claude-sonnet-4-6", MODEL_CAPABILITIES)
        self.assertIn("claude-haiku-4-5-20251001", MODEL_CAPABILITIES)

    def test_each_entry_has_both_flags(self):
        for model, caps in MODEL_CAPABILITIES.items():
            with self.subTest(model=model):
                self.assertIn("adaptive_thinking", caps)
                self.assertIn("context_management", caps)


class TestFilterBetaHeader(unittest.TestCase):
    FULL_BETA = (
        "claude-code-20250219,"
        "oauth-2025-04-20,"
        "adaptive-thinking-2026-01-28,"
        "research-preview-2026-02-01,"
        "context-management-2025-06-27"
    )

    def test_opus_preserves_all_tokens(self):
        filtered = filter_beta_header(self.FULL_BETA, model="claude-opus-4-7")
        self.assertIn("adaptive-thinking-2026-01-28", filtered)
        self.assertIn("context-management-2025-06-27", filtered)
        self.assertIn("claude-code-20250219", filtered)

    def test_haiku_strips_adaptive_thinking(self):
        filtered = filter_beta_header(
            self.FULL_BETA, model="claude-haiku-4-5-20251001",
        )
        self.assertNotIn("adaptive-thinking-2026-01-28", filtered)
        # Haiku DOES support context management — it must survive.
        self.assertIn("context-management-2025-06-27", filtered)
        # Non-capability-gated tokens must pass through.
        self.assertIn("claude-code-20250219", filtered)
        self.assertIn("oauth-2025-04-20", filtered)
        self.assertIn("research-preview-2026-02-01", filtered)

    def test_unknown_model_strips_both_capability_gated_tokens(self):
        filtered = filter_beta_header(self.FULL_BETA, model="future-model")
        self.assertNotIn("adaptive-thinking-2026-01-28", filtered)
        self.assertNotIn("context-management-2025-06-27", filtered)
        # Ungated tokens still pass through.
        self.assertIn("claude-code-20250219", filtered)

    def test_empty_beta_value_returns_empty(self):
        self.assertEqual(filter_beta_header("", model="claude-opus-4-7"), "")

    def test_whitespace_only_tokens_dropped(self):
        filtered = filter_beta_header("  ,  ,a,  ,b", model="claude-opus-4-7")
        self.assertEqual(filtered.split(","), ["a", "b"])


class TestBuildBodyAdaptiveThinking(unittest.TestCase):
    def test_opus_body_includes_adaptive_thinking(self):
        signer = CCHSigner()
        body, _ = signer.build_body("hi", model="claude-opus-4-7")
        self.assertIn('"thinking":{"type":"adaptive"}', body)

    def test_haiku_body_omits_adaptive_thinking(self):
        signer = CCHSigner()
        body, _ = signer.build_body("hi", model="claude-haiku-4-5-20251001")
        self.assertNotIn('"thinking"', body)

    def test_unknown_model_omits_adaptive_thinking(self):
        signer = CCHSigner()
        body, _ = signer.build_body("hi", model="future-model")
        self.assertNotIn('"thinking"', body)


class TestBuildBodyContextManagement(unittest.TestCase):
    def test_opus_includes_context_management(self):
        signer = CCHSigner()
        body, _ = signer.build_body("hi", model="claude-opus-4-7")
        self.assertIn('"context_management"', body)

    def test_haiku_omits_context_management_body_field(self):
        """Haiku 4.5's MODEL_CAPABILITIES declares context_management=True,
        but the body field is a thinking-clearance edit — if the model
        emits no thinking (adaptive_thinking=False), there's nothing to
        clear, so the field is correctly omitted. The capability flag
        exists to survive beta-header filtering, not to force an empty
        edit payload."""
        signer = CCHSigner()
        body, _ = signer.build_body("hi", model="claude-haiku-4-5-20251001")
        self.assertNotIn('"context_management"', body)

    def test_unknown_model_omits_context_management(self):
        signer = CCHSigner()
        body, _ = signer.build_body("hi", model="future-model")
        self.assertNotIn('"context_management"', body)


class TestBuildRequestHeaderFiltering(unittest.TestCase):
    def test_haiku_request_header_omits_adaptive_thinking(self):
        signer = CCHSigner()
        headers, _ = signer.build_request(
            "hi",
            oauth_token="sk-x",
            model="claude-haiku-4-5-20251001",
        )
        self.assertNotIn("adaptive-thinking-2026-01-28", headers["anthropic-beta"])
        self.assertIn("context-management-2025-06-27", headers["anthropic-beta"])

    def test_opus_request_header_preserves_adaptive_thinking(self):
        signer = CCHSigner()
        headers, _ = signer.build_request(
            "hi",
            oauth_token="sk-x",
            model="claude-opus-4-7",
        )
        self.assertIn("adaptive-thinking-2026-01-28", headers["anthropic-beta"])


if __name__ == "__main__":
    unittest.main()
