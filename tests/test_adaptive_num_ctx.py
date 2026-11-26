"""Tests for Phase γ W6 adaptive num_ctx (H-CTX)."""
from __future__ import annotations

import unittest

from lib.hal.adaptive_num_ctx import compute_num_ctx


class TestComputeNumCtx(unittest.TestCase):
    def test_small_prompt_clamps_to_floor(self):
        # 50 prompt + 100 response + 512 margin = 662 < 2048 floor.
        self.assertEqual(compute_num_ctx(50, 100, model_max=32_768), 2048)

    def test_medium_prompt_passes_through(self):
        # 2000 + 1000 + 512 = 3512, well below 32K.
        self.assertEqual(compute_num_ctx(2000, 1000, model_max=32_768), 3512)

    def test_large_prompt_capped_at_model_max(self):
        self.assertEqual(
            compute_num_ctx(30_000, 10_000, model_max=32_768), 32_768,
        )

    def test_exact_match_to_model_max(self):
        self.assertEqual(
            compute_num_ctx(31_256, 1000, model_max=32_768, safety_margin=512),
            32_768,
        )

    def test_negative_inputs_clamped(self):
        # Negatives treated as 0; ideal = 0 < floor → return floor (or model_max).
        self.assertEqual(compute_num_ctx(-1, -1, model_max=32_768), 2048)

    def test_model_max_below_floor_returns_model_max(self):
        # Tiny model_max should win — we never exceed it even when floor says otherwise.
        self.assertEqual(compute_num_ctx(0, 0, model_max=1024), 1024)

    def test_custom_floor(self):
        self.assertEqual(
            compute_num_ctx(50, 50, model_max=32_768, floor=4096), 4096,
        )

    def test_custom_safety_margin(self):
        self.assertEqual(
            compute_num_ctx(2000, 1000, model_max=32_768, safety_margin=1000),
            4000,
        )

    def test_invalid_model_max_raises(self):
        with self.assertRaises(ValueError):
            compute_num_ctx(100, 100, model_max=0)
        with self.assertRaises(ValueError):
            compute_num_ctx(100, 100, model_max=-1)


if __name__ == "__main__":
    unittest.main()
