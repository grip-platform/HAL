"""Tests for HAL token budget."""
import os
import sys
import unittest

from lib.hal.budget import Budget, BudgetStatus


class TestBudget(unittest.TestCase):

    def test_initial_state(self):
        b = Budget(max_usd=1.0)
        self.assertFalse(b.exhausted)
        self.assertAlmostEqual(b.remaining, 1.0)

    def test_spend_tracks(self):
        b = Budget(max_usd=1.0)
        b.spend(0.3)
        self.assertAlmostEqual(b.remaining, 0.7)

    def test_exhausted_when_spent(self):
        b = Budget(max_usd=0.50)
        b.spend(0.50)
        self.assertTrue(b.exhausted)

    def test_spend_returns_budget_ok(self):
        b = Budget(max_usd=1.0)
        self.assertTrue(b.spend(0.3))

    def test_spend_returns_exhausted(self):
        b = Budget(max_usd=0.10)
        self.assertFalse(b.spend(0.15))

    def test_can_afford(self):
        b = Budget(max_usd=1.0)
        b.spend(0.80)
        self.assertTrue(b.can_afford(0.15))
        self.assertFalse(b.can_afford(0.25))

    def test_status(self):
        b = Budget(max_usd=2.0)
        b.spend(0.50)
        b.spend(0.25)
        s = b.status()
        self.assertEqual(s.max_usd, 2.0)
        self.assertAlmostEqual(s.spent_usd, 0.75)
        self.assertEqual(s.queries, 2)
        self.assertFalse(s.exhausted)

    def test_reset(self):
        b = Budget(max_usd=1.0)
        b.spend(0.90)
        b.reset()
        self.assertFalse(b.exhausted)
        self.assertAlmostEqual(b.remaining, 1.0)

    def test_reset_with_new_max(self):
        b = Budget(max_usd=1.0)
        b.reset(new_max=5.0)
        self.assertAlmostEqual(b.remaining, 5.0)


class TestBudgetStatus(unittest.TestCase):

    def test_frozen(self):
        s = BudgetStatus(1.0, 0.5, 0.5, False, 0.5, 3)
        with self.assertRaises(AttributeError):
            s.max_usd = 2.0


if __name__ == "__main__":
    unittest.main()
