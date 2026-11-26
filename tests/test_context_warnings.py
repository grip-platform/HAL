"""Tests for HAL graduated context warnings."""
import os
import sys
import unittest

from lib.hal.context_warnings import (
    ContextMonitor,
    Severity,
    estimate_turn_tokens,
)


class TestContextMonitor(unittest.TestCase):

    def test_no_warning_at_start(self):
        monitor = ContextMonitor(max_tokens=100_000)
        self.assertIsNone(monitor.check())

    def test_info_at_80_pct(self):
        monitor = ContextMonitor(max_tokens=100_000)
        monitor.add_tokens(80_000)
        warning = monitor.check()
        self.assertIsNotNone(warning)
        self.assertEqual(warning.severity, Severity.INFO)

    def test_warning_at_85_pct(self):
        monitor = ContextMonitor(max_tokens=100_000)
        monitor.add_tokens(85_000)
        warning = monitor.check()
        self.assertIsNotNone(warning)
        # Should jump to WARNING (skipping INFO since 85% > 80%)
        self.assertEqual(warning.severity, Severity.WARNING)

    def test_critical_at_95_pct(self):
        monitor = ContextMonitor(max_tokens=100_000)
        monitor.add_tokens(95_000)
        warning = monitor.check()
        self.assertIsNotNone(warning)
        self.assertEqual(warning.severity, Severity.CRITICAL)

    def test_escalation_only(self):
        """Once a threshold fires, lower thresholds don't re-fire."""
        monitor = ContextMonitor(max_tokens=100_000)
        monitor.add_tokens(85_000)
        w1 = monitor.check()
        self.assertEqual(w1.severity, Severity.WARNING)
        # Second check at same level returns None (already warned)
        self.assertIsNone(monitor.check())

    def test_escalation_to_critical(self):
        monitor = ContextMonitor(max_tokens=100_000)
        monitor.add_tokens(85_000)
        monitor.check()  # Fires WARNING
        monitor.add_tokens(10_000)  # Now at 95%
        w2 = monitor.check()
        self.assertIsNotNone(w2)
        self.assertEqual(w2.severity, Severity.CRITICAL)

    def test_warning_has_suggested_actions(self):
        monitor = ContextMonitor(max_tokens=100_000)
        monitor.add_tokens(95_000)
        warning = monitor.check()
        self.assertGreater(len(warning.suggested_actions), 0)

    def test_tokens_remaining(self):
        monitor = ContextMonitor(max_tokens=100_000)
        monitor.add_tokens(60_000)
        self.assertEqual(monitor.tokens_remaining, 40_000)

    def test_usage_pct(self):
        monitor = ContextMonitor(max_tokens=200_000)
        monitor.add_tokens(100_000)
        self.assertAlmostEqual(monitor.usage_pct, 0.5)

    def test_set_tokens(self):
        monitor = ContextMonitor(max_tokens=100_000)
        monitor.set_tokens(50_000)
        self.assertEqual(monitor.tokens_used, 50_000)

    def test_reset(self):
        monitor = ContextMonitor(max_tokens=100_000)
        monitor.add_tokens(90_000)
        monitor.check()
        monitor.reset()
        self.assertEqual(monitor.tokens_used, 0)
        self.assertIsNone(monitor.check())

    def test_force_check_ignores_previous(self):
        monitor = ContextMonitor(max_tokens=100_000)
        monitor.add_tokens(85_000)
        monitor.check()  # Fires WARNING
        # check(force=True) should still return the current state
        w = monitor.check(force=True)
        self.assertIsNotNone(w)
        self.assertEqual(w.severity, Severity.WARNING)


class TestEstimateTurnTokens(unittest.TestCase):

    def test_estimates_tokens(self):
        tokens = estimate_turn_tokens("Hello world", "Hi there, how can I help?")
        self.assertGreater(tokens, 0)

    def test_empty_messages(self):
        tokens = estimate_turn_tokens("", "")
        self.assertEqual(tokens, 0)


class TestSeverity(unittest.TestCase):

    def test_values(self):
        self.assertEqual(Severity.NONE.value, "none")
        self.assertEqual(Severity.INFO.value, "info")
        self.assertEqual(Severity.WARNING.value, "warning")
        self.assertEqual(Severity.CRITICAL.value, "critical")


if __name__ == "__main__":
    unittest.main()
