"""Wave 3: mlx-vlm cost tracker free-tier tests.

All mlx-vlm v0.4.3 models must resolve to $0 (free local inference)
and classify as the "free" cost tier.
"""
import os
import sys
import unittest

from lib.hal.cost_tracker import calculate_cost, _resolve_pricing
from lib.hal.cost_router import _classify_tier

_MLX_MODELS = [
    "mlx-community/Qwen2-VL-7B-Instruct-4bit",
    "mlx-community/Qwen3-VL-7B-Instruct-4bit",
    "mlx-community/phi-4-vision-instruct-4bit",
    "mlx-community/gemma-4-vision-9b-4bit",
    "falcon-ocr",
    "falcon-perception",
    "granite-vision-3.2-2b",
    "sam-3.1",
    "rf-detr",
]


class TestMlxVlmFreePricing(unittest.TestCase):

    def test_all_models_resolve_pricing(self):
        for model in _MLX_MODELS:
            pricing = _resolve_pricing(model)
            self.assertIsNotNone(pricing, f"No pricing found for {model}")

    def test_all_models_zero_input_cost(self):
        for model in _MLX_MODELS:
            pricing = _resolve_pricing(model)
            self.assertEqual(pricing[0], 0.0, f"{model} input cost should be $0")

    def test_all_models_zero_output_cost(self):
        for model in _MLX_MODELS:
            pricing = _resolve_pricing(model)
            self.assertEqual(pricing[1], 0.0, f"{model} output cost should be $0")

    def test_calculate_cost_returns_zero(self):
        for model in _MLX_MODELS:
            cost = calculate_cost(model, input_tokens=100_000, output_tokens=10_000)
            self.assertEqual(cost, 0.0, f"{model} total cost should be $0")

    def test_all_models_free_tier(self):
        for model in _MLX_MODELS:
            pricing = _resolve_pricing(model)
            tier = _classify_tier(pricing[0])
            self.assertEqual(tier, "free", f"{model} should be free tier")

    def test_gemma4_vision_free(self):
        cost = calculate_cost("mlx-community/gemma-4-vision-9b-4bit", input_tokens=50_000, output_tokens=5_000)
        self.assertEqual(cost, 0.0)

    def test_cloud_models_still_have_cost(self):
        """Regression: cloud models must not accidentally become free."""
        pricing = _resolve_pricing("claude-sonnet-4-6")
        self.assertGreater(pricing[0], 0.0)


if __name__ == "__main__":
    unittest.main()
