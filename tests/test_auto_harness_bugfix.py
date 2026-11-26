"""Regression tests for auto_harness bug fixes (fix/auto-harness-metadata-and-max-tokens).

Two bugs prevented AutoHarness from ever returning features end-to-end:

1. _fetch_metadata requested the gh CLI field "topics", which does not exist
   (the correct name is repositoryTopics). Any unknown field makes gh reject
   the whole --json query with a non-zero exit, so _fetch_metadata silently
   returned {} and the council prompt lost all context. The fix: remove
   the unused field entirely (downstream code never reads topics).

2. _analyse_with_council invoked council.deliberate() which routes each
   member query and the synthesis step through registry.chat() with the
   registry's default max_tokens. That default is tight enough to truncate
   a multi-feature JSON array mid-element (observed ~387 chars, no closing
   "]"), making _parse_features return []. The fix: single capable model
   via registry.chat() with an explicit max_tokens=2048.

Each test here asserts a concrete observable - Goodhart-safe. A regression
that re-adds "topics" or drops the max_tokens override would fail at least
one test.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal import auto_harness  # noqa: E402


_VALID_FEATURE_JSON = """[
  {"name": "Cache layer", "description": "Hot reads", "applicability": 8,
   "complexity": 3, "novelty": 5, "burden": 3,
   "recommendation": "absorb", "hal_module": "memory_engine",
   "rationale": "Reduces redundant reads."}
]"""


class TestFetchMetadataFieldList(unittest.TestCase):
    """Bug 1: gh --json query must NOT include unknown fields."""

    def test_topics_not_in_gh_query(self):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = "{}"
            return mock

        harness = auto_harness.AutoHarness(registry=None)
        with patch.object(auto_harness.subprocess, "run", side_effect=fake_run):
            harness._fetch_metadata("owner/repo")

        # Find the --json argument value
        argv = captured["argv"]
        self.assertIn("--json", argv)
        json_fields = argv[argv.index("--json") + 1]
        self.assertNotIn("topics", json_fields.split(","),
                         f"topics must not be in gh --json query; got {json_fields}")

    def test_required_fields_present_in_gh_query(self):
        """Downstream code reads description/primaryLanguage/stargazerCount."""
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = "{}"
            return mock

        harness = auto_harness.AutoHarness(registry=None)
        with patch.object(auto_harness.subprocess, "run", side_effect=fake_run):
            harness._fetch_metadata("owner/repo")

        json_fields = captured["argv"][captured["argv"].index("--json") + 1].split(",")
        for required in ("description", "primaryLanguage", "stargazerCount"):
            self.assertIn(required, json_fields)


class TestAnalyseMaxTokens(unittest.TestCase):
    """Bug 2: registry.chat must receive an ample max_tokens kwarg."""

    def _build_harness_with_mock_registry(self, chat_return_text):
        registry = MagicMock()
        provider = MagicMock()
        provider.name = "groq"
        provider.models = ["llama-3.3-70b-versatile"]
        registry.list_providers.return_value = [provider]
        response = MagicMock()
        response.text = chat_return_text
        registry.chat.return_value = response
        return auto_harness.AutoHarness(registry=registry), registry

    def test_chat_called_with_max_tokens_at_least_2048(self):
        harness, registry = self._build_harness_with_mock_registry(_VALID_FEATURE_JSON)
        harness._analyse_with_council(
            repo="owner/repo", metadata={}, structure=[], readme="",
        )
        registry.chat.assert_called_once()
        kwargs = registry.chat.call_args.kwargs
        self.assertIn("max_tokens", kwargs,
                      "registry.chat must be called with explicit max_tokens")
        self.assertGreaterEqual(kwargs["max_tokens"], 2048,
                                "max_tokens must be at least 2048 to fit multi-feature JSON")

    def test_valid_json_produces_features(self):
        harness, _ = self._build_harness_with_mock_registry(_VALID_FEATURE_JSON)
        features = harness._analyse_with_council(
            repo="owner/repo", metadata={}, structure=[], readme="",
        )
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].name, "Cache layer")
        self.assertEqual(features[0].recommendation, "absorb")

    def test_truncated_json_returns_empty_not_crash(self):
        # Mid-element truncation — exactly what the old deliberate() path produced
        truncated = _VALID_FEATURE_JSON[:60]
        harness, _ = self._build_harness_with_mock_registry(truncated)
        features = harness._analyse_with_council(
            repo="owner/repo", metadata={}, structure=[], readme="",
        )
        self.assertEqual(features, [],
                         "Truncated JSON must degrade gracefully, not raise")

    def test_no_models_returns_empty(self):
        registry = MagicMock()
        registry.list_providers.return_value = []
        harness = auto_harness.AutoHarness(registry=registry)
        self.assertEqual(
            harness._analyse_with_council(
                repo="x/y", metadata={}, structure=[], readme="",
            ),
            [],
        )
        registry.chat.assert_not_called()

    def test_chat_exception_returns_empty(self):
        registry = MagicMock()
        provider = MagicMock()
        provider.name = "groq"
        provider.models = ["m"]
        registry.list_providers.return_value = [provider]
        registry.chat.side_effect = RuntimeError("network")
        harness = auto_harness.AutoHarness(registry=registry)
        self.assertEqual(
            harness._analyse_with_council(
                repo="x/y", metadata={}, structure=[], readme="",
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
