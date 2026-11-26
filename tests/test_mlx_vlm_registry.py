"""Wave 1b: mlx_vlm provider registry address-parsing tests.

Verifies that parse_model_address() correctly identifies mlx_vlm as a known
provider prefix, returning (provider, model) tuples, and that existing
providers are unaffected.
"""
import unittest
from pathlib import Path

from lib.hal.registry import parse_model_address

# The checkout under test, never the home install: run from a worktree or a CI
# checkout, ~/.hal is other code or absent (on 2026-11-26 the live ~/.hal was 56
# commits behind main, and with an empty HOME these tests failed).
REPO = Path(__file__).resolve().parent.parent


class TestMlxVlmAddressParsing(unittest.TestCase):

    def test_qwen_vl_address(self):
        provider, model = parse_model_address("mlx_vlm/mlx-community/Qwen2-VL-7B-Instruct-4bit")
        self.assertEqual(provider, "mlx_vlm")
        self.assertEqual(model, "mlx-community/Qwen2-VL-7B-Instruct-4bit")

    def test_phi4_vision_address(self):
        provider, model = parse_model_address("mlx_vlm/phi-4-vision-instruct")
        self.assertEqual(provider, "mlx_vlm")
        self.assertEqual(model, "phi-4-vision-instruct")

    def test_falcon_ocr_address(self):
        provider, model = parse_model_address("mlx_vlm/falcon-ocr")
        self.assertEqual(provider, "mlx_vlm")
        self.assertEqual(model, "falcon-ocr")

    def test_sam_address(self):
        provider, model = parse_model_address("mlx_vlm/sam-3.1")
        self.assertEqual(provider, "mlx_vlm")
        self.assertEqual(model, "sam-3.1")


class TestMlxVlmYamlParsing(unittest.TestCase):
    """Verify that the mlx_vlm block in providers.yaml parses correctly."""

    def _load_mlx_config(self):
        from lib.hal.registry import _load_yaml_config
        path = str(REPO / "config" / "providers.yaml")
        configs, _ = _load_yaml_config(path)
        return next((c for c in configs if c.name == "mlx_vlm"), None)

    def test_mlx_vlm_config_present(self):
        config = self._load_mlx_config()
        self.assertIsNotNone(config, "mlx_vlm must be present in providers.yaml")

    def test_mlx_vlm_timeout_s_parsed(self):
        config = self._load_mlx_config()
        self.assertIsNotNone(config)
        self.assertEqual(config.timeout_s, 180.0)

    def test_mlx_vlm_type_is_local(self):
        from lib.hal.types import ProviderType
        config = self._load_mlx_config()
        self.assertIsNotNone(config)
        self.assertEqual(config.type, ProviderType.LOCAL)

    def test_mlx_vlm_auto_discover_models(self):
        config = self._load_mlx_config()
        self.assertIsNotNone(config)
        self.assertTrue(config.auto_discover_models)


class TestExistingProvidersUnaffected(unittest.TestCase):

    def test_groq_still_works(self):
        provider, model = parse_model_address("groq/llama-3.3-70b-versatile")
        self.assertEqual(provider, "groq")
        self.assertEqual(model, "llama-3.3-70b-versatile")

    def test_gemini_still_works(self):
        provider, model = parse_model_address("gemini/gemini-2.5-flash")
        self.assertEqual(provider, "gemini")
        self.assertEqual(model, "gemini-2.5-flash")

    def test_unknown_prefix_returns_empty(self):
        provider, model = parse_model_address("unknownprovider/some-model")
        self.assertEqual(provider, "")

    def test_no_slash_returns_empty_provider(self):
        provider, model = parse_model_address("llama-3.3-70b")
        self.assertEqual(provider, "")
        self.assertEqual(model, "llama-3.3-70b")


if __name__ == "__main__":
    unittest.main()
