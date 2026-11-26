"""Tests for Phase γ W5 free-tier cascade (H-FREECAS)."""
from __future__ import annotations

import unittest

from lib.hal.free_tier_cascade import (
    DEFAULT_CASCADE,
    CascadeTier,
    RateLimitError,
    free_only,
    next_provider,
)


class TestDefaultCascade(unittest.TestCase):
    def test_first_three_are_free(self):
        for tier in DEFAULT_CASCADE[:3]:
            self.assertEqual(tier.cost_cls, "free")

    def test_cascade_non_empty(self):
        self.assertGreater(len(DEFAULT_CASCADE), 0)

    def test_gemini_flash_is_first(self):
        self.assertEqual(DEFAULT_CASCADE[0].provider, "gemini")


class TestNextProvider(unittest.TestCase):
    def test_none_returns_first_tier(self):
        self.assertEqual(next_provider(None), DEFAULT_CASCADE[0])

    def test_second_step(self):
        self.assertEqual(next_provider(DEFAULT_CASCADE[0]), DEFAULT_CASCADE[1])

    def test_last_tier_returns_none(self):
        self.assertIsNone(next_provider(DEFAULT_CASCADE[-1]))

    def test_unknown_tier_returns_none(self):
        fake = CascadeTier("unknown", "model", "free")
        self.assertIsNone(next_provider(fake))

    def test_empty_cascade_returns_none(self):
        self.assertIsNone(next_provider(None, cascade=()))

    def test_custom_cascade(self):
        custom = (
            CascadeTier("a", "m1", "free"),
            CascadeTier("b", "m2", "cheap"),
        )
        self.assertEqual(next_provider(None, cascade=custom), custom[0])
        self.assertEqual(next_provider(custom[0], cascade=custom), custom[1])
        self.assertIsNone(next_provider(custom[1], cascade=custom))


class TestFreeOnly(unittest.TestCase):
    def test_returns_only_free_tiers(self):
        free = free_only()
        self.assertTrue(all(t.cost_cls == "free" for t in free))
        self.assertGreater(len(free), 0)

    def test_empty_cascade_returns_empty(self):
        self.assertEqual(free_only(()), ())


class TestRateLimitError(unittest.TestCase):
    def test_raises_and_catches(self):
        with self.assertRaises(RateLimitError):
            raise RateLimitError("429")


if __name__ == "__main__":
    unittest.main()
