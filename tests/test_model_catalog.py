"""Tests for HAL model catalog."""
import os, sys, unittest
from lib.hal.model_catalog import ModelCatalog, ModelInfo

class TestModelCatalog(unittest.TestCase):
    def test_get_known(self):
        m = ModelCatalog().get("claude-opus-4-7")
        self.assertIsNotNone(m)
        self.assertEqual(m.provider, "anthropic")

    def test_get_unknown(self):
        self.assertIsNone(ModelCatalog().get("nonexistent"))

    def test_find_by_context(self):
        results = ModelCatalog().find(min_context=500_000)
        self.assertTrue(all(m.context_window >= 500_000 for m in results))

    def test_find_by_cost(self):
        results = ModelCatalog().find(max_cost=1.0)
        self.assertTrue(all(m.input_cost_per_mtok <= 1.0 for m in results))

    def test_find_by_provider(self):
        results = ModelCatalog().find(provider="anthropic")
        self.assertTrue(all(m.provider == "anthropic" for m in results))

    def test_list_all(self):
        self.assertGreater(len(ModelCatalog().list_all()), 5)

    def test_providers(self):
        p = ModelCatalog().providers()
        self.assertIn("anthropic", p)
        self.assertIn("groq", p)

    def test_is_free(self):
        free = [m for m in ModelCatalog().list_all() if m.is_free]
        self.assertIsInstance(free, list)

    def test_gemma4_26b_moe_entry(self):
        """Gemma 4 26B MoE — primary local model — exists with correct metadata."""
        m = ModelCatalog().get("gemma4:26b")
        self.assertIsNotNone(m, "gemma4:26b missing from catalog")
        self.assertEqual(m.provider, "ollama")
        self.assertEqual(m.context_window, 256_000)
        self.assertTrue(m.is_free)
        self.assertIn("multimodal", m.strengths)
        self.assertIn("speed", m.strengths)

if __name__ == "__main__":
    unittest.main()
