"""Tests for HAL Warm Model Router."""
import io
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.hal.warm_router import OllamaLivenessProbe, WarmModel, WarmModelRouter

_TIER_CHAINS = {
    "free": ["ollama/gemma4:e4b"],
    "cheap": ["qwen/qwen3.5-flash", "ollama/gemma4:e4b"],
    "mid": ["gemini/gemini-2.5-flash"],
    "premium": ["gemini/gemini-3.1-pro-preview"],
}


def _urlopen_ctx(models: list[dict]):
    """Return a context manager mock that yields a readable JSON response."""
    body = json.dumps({"models": models}).encode()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=body)))
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


class TestOllamaLivenessProbe(unittest.TestCase):

    @patch("lib.hal.warm_router.urllib.request.urlopen")
    def test_warm_models_returns_list(self, mock_urlopen):
        mock_urlopen.return_value = _urlopen_ctx([
            {"name": "gemma4:e4b", "size": 15_000_000_000},
        ])
        probe = OllamaLivenessProbe()
        warm = probe.warm_models()
        self.assertEqual(len(warm), 1)
        self.assertEqual(warm[0].name, "gemma4:e4b")

    @patch("lib.hal.warm_router.urllib.request.urlopen")
    def test_warm_models_empty_on_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("timeout")
        probe = OllamaLivenessProbe()
        self.assertEqual(probe.warm_models(), [])

    @patch("lib.hal.warm_router.urllib.request.urlopen")
    def test_is_warm_exact_match(self, mock_urlopen):
        mock_urlopen.return_value = _urlopen_ctx([{"name": "gemma4:e4b", "size": 0}])
        probe = OllamaLivenessProbe()
        self.assertTrue(probe.is_warm("gemma4:e4b"))

    @patch("lib.hal.warm_router.urllib.request.urlopen")
    def test_is_warm_base_name_match(self, mock_urlopen):
        mock_urlopen.return_value = _urlopen_ctx([{"name": "gemma4:latest", "size": 0}])
        probe = OllamaLivenessProbe()
        self.assertTrue(probe.is_warm("gemma4:e4b"))

    @patch("lib.hal.warm_router.urllib.request.urlopen")
    def test_is_warm_false_when_down(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionRefusedError()
        probe = OllamaLivenessProbe()
        self.assertFalse(probe.is_warm("gemma4:e4b"))


class TestWarmModelRouter(unittest.TestCase):

    def _probe(self, warm_names: list[str]) -> OllamaLivenessProbe:
        probe = MagicMock(spec=OllamaLivenessProbe)
        probe.warm_models.return_value = [WarmModel(name=n) for n in warm_names]
        return probe

    def test_routes_free_tier_to_warm_ollama(self):
        router = WarmModelRouter(
            probe=self._probe(["gemma4:e4b"]),
            catalog_models={},
            tier_chains=_TIER_CHAINS,
        )
        result = router.route("free")
        self.assertEqual(result, "ollama/gemma4:e4b")

    def test_routes_cheap_tier_to_warm_ollama(self):
        router = WarmModelRouter(
            probe=self._probe(["gemma4:e4b"]),
            catalog_models={},
            tier_chains=_TIER_CHAINS,
        )
        result = router.route("cheap")
        self.assertEqual(result, "ollama/gemma4:e4b")

    def test_skips_premium_tier(self):
        router = WarmModelRouter(
            probe=self._probe(["gemma4:e4b"]),
            catalog_models={},
            tier_chains=_TIER_CHAINS,
        )
        self.assertIsNone(router.route("premium"))

    def test_returns_none_when_no_warm_models(self):
        router = WarmModelRouter(
            probe=self._probe([]),
            catalog_models={},
            tier_chains=_TIER_CHAINS,
        )
        self.assertIsNone(router.route("free"))

    def test_returns_none_when_warm_model_not_in_tier_chain(self):
        router = WarmModelRouter(
            probe=self._probe(["llama3.2:3b"]),  # not in tier chains
            catalog_models={},
            tier_chains=_TIER_CHAINS,
        )
        self.assertIsNone(router.route("free"))

    def test_prefers_ollama_over_api_in_cheap_tier(self):
        # cheap chain = [qwen, ollama]. If ollama is warm, skip qwen.
        router = WarmModelRouter(
            probe=self._probe(["gemma4:e4b"]),
            catalog_models={},
            tier_chains=_TIER_CHAINS,
        )
        result = router.route("cheap")
        self.assertEqual(result, "ollama/gemma4:e4b")


if __name__ == "__main__":
    unittest.main()
