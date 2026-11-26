"""Wave 2: mlx-vlm discovery port-8080 disambiguation tests.

Port 8080 is shared between llamacpp and mlx-vlm. Discovery probes the
/v1/models response and uses VLM model-name patterns to decide which
provider to register — zero config required from the user.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.discovery import (
    _is_vlm_server,
    _probe,
    discover_local_providers,
)
from lib.hal.types import ProviderType


class TestIsVlmServer(unittest.TestCase):
    """Unit tests for the model-name pattern discriminator."""

    def test_qwen_vl_detected(self):
        self.assertTrue(_is_vlm_server(["mlx-community/Qwen2-VL-7B-Instruct-4bit"]))

    def test_qwen3_vl_detected(self):
        self.assertTrue(_is_vlm_server(["mlx-community/Qwen3-VL-7B-Instruct-4bit"]))

    def test_phi4_vision_detected(self):
        self.assertTrue(_is_vlm_server(["phi-4-vision-instruct"]))

    def test_gemma_vision_detected(self):
        self.assertTrue(_is_vlm_server(["gemma-3-vision-9b-instruct"]))

    def test_granite_vision_detected(self):
        self.assertTrue(_is_vlm_server(["granite-vision-3.2-2b"]))

    def test_falcon_ocr_detected(self):
        self.assertTrue(_is_vlm_server(["falcon-ocr-2b"]))

    def test_llava_detected(self):
        # "llava" matches at any position in the model ID — intentional, covers all LLaVA variants
        self.assertTrue(_is_vlm_server(["llava-1.6-mistral-7b"]))

    def test_idefics_detected(self):
        self.assertTrue(_is_vlm_server(["idefics3-8b-llama3"]))

    def test_sam_detected(self):
        self.assertTrue(_is_vlm_server(["sam-3.1-hiera-tiny"]))

    def test_rf_detr_detected(self):
        self.assertTrue(_is_vlm_server(["rf-detr-base"]))

    def test_gemma4_detected(self):
        """Gemma 4 is natively multimodal — no 'vision' suffix needed."""
        self.assertTrue(_is_vlm_server(["mlx-community/gemma-4-27b-it-4bit"]))

    def test_gemma4_ollama_format_detected(self):
        self.assertTrue(_is_vlm_server(["gemma4:27b"]))

    def test_gemma4_26b_detected(self):
        self.assertTrue(_is_vlm_server(["gemma-4-26b-a4b-it"]))

    def test_text_only_model_not_vlm(self):
        self.assertFalse(_is_vlm_server(["llama-3.1-8b-instruct"]))

    def test_mistral_not_vlm(self):
        self.assertFalse(_is_vlm_server(["mistral-7b-instruct"]))

    def test_empty_model_list_not_vlm(self):
        self.assertFalse(_is_vlm_server([]))

    def test_mixed_list_vlm_wins(self):
        # If any model is VLM, the server is a VLM server
        self.assertTrue(_is_vlm_server(["llama-3.1-8b", "Qwen2-VL-7B"]))


class TestDiscoverMlxVlmVsCpp(unittest.TestCase):
    """Integration tests for port-8080 discrimination in discover_local_providers."""

    def _make_registry(self):
        registry = MagicMock()
        registry.get.return_value = None  # Nothing pre-registered
        return registry

    @patch("lib.hal.discovery._fetch_models",
           return_value=["mlx-community/Qwen2-VL-7B-Instruct-4bit"])
    @patch("lib.hal.discovery._probe", return_value=True)
    def test_registers_as_mlx_vlm_when_vlm_models(self, _mock_probe, _mock_fetch):
        registry = self._make_registry()
        result = discover_local_providers(registry)
        registered_names = [c.args[0].name for c in registry.register.call_args_list]
        self.assertIn("mlx_vlm", registered_names)
        self.assertNotIn("llamacpp", registered_names)
        self.assertIn("mlx_vlm", result)

    @patch("lib.hal.discovery._fetch_models",
           return_value=["llama-3.1-8b-instruct"])
    @patch("lib.hal.discovery._probe", return_value=True)
    def test_registers_as_llamacpp_when_text_models(self, _mock_probe, _mock_fetch):
        registry = self._make_registry()
        result = discover_local_providers(registry)
        registered_names = [c.args[0].name for c in registry.register.call_args_list]
        self.assertIn("llamacpp", registered_names)
        self.assertNotIn("mlx_vlm", registered_names)

    @patch("lib.hal.discovery._probe", return_value=False)
    def test_no_registration_when_port_8080_down(self, _mock_probe):
        registry = self._make_registry()
        result = discover_local_providers(registry)
        self.assertNotIn("mlx_vlm", result)
        self.assertNotIn("llamacpp", result)

    @patch("lib.hal.discovery._fetch_models",
           return_value=["mlx-community/Qwen2-VL-7B-Instruct-4bit"])
    @patch("lib.hal.discovery._probe", return_value=True)
    def test_skips_if_mlx_vlm_already_registered(self, _mock_probe, _mock_fetch):
        registry = self._make_registry()
        # mlx_vlm already registered — should skip port 8080
        registry.get.side_effect = lambda name: MagicMock() if name == "mlx_vlm" else None
        result = discover_local_providers(registry)
        self.assertNotIn("mlx_vlm", result)

    @patch("lib.hal.discovery._fetch_models", return_value=[])
    @patch("lib.hal.discovery._probe", return_value=True)
    def test_empty_models_falls_back_to_llamacpp(self, _mock_probe, _mock_fetch):
        registry = self._make_registry()
        result = discover_local_providers(registry)
        registered_names = [c.args[0].name for c in registry.register.call_args_list]
        self.assertIn("llamacpp", registered_names)


if __name__ == "__main__":
    unittest.main()
