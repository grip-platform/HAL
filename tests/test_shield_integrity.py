"""Tests for self-healing shield integrity model in ProviderMonitor."""
import os
import sys
import unittest

from lib.hal.provider_monitor import ProviderHealth, ProviderMonitor


class TestProviderHealthIntegrity(unittest.TestCase):
    """Test the target/actual integrity convergence model."""

    def test_initial_integrity_equals_target(self):
        h = ProviderHealth(provider="groq")
        self.assertEqual(h.actual_integrity, 1.0)
        self.assertEqual(h.target_integrity, 1.0)

    def test_failure_decays_integrity(self):
        h = ProviderHealth(provider="groq", decay_rate=0.20)
        h.actual_integrity = max(0.0, h.actual_integrity - h.decay_rate)
        self.assertAlmostEqual(h.actual_integrity, 0.80)

    def test_integrity_never_below_zero(self):
        h = ProviderHealth(provider="groq", actual_integrity=0.1, decay_rate=0.20)
        h.actual_integrity = max(0.0, h.actual_integrity - h.decay_rate)
        self.assertAlmostEqual(h.actual_integrity, 0.0)

    def test_integrity_never_exceeds_target(self):
        h = ProviderHealth(provider="groq", target_integrity=0.8,
                           actual_integrity=0.75, regen_rate=0.10)
        h.actual_integrity = min(h.target_integrity,
                                 h.actual_integrity + h.regen_rate)
        self.assertAlmostEqual(h.actual_integrity, 0.8)

    def test_to_dict_includes_integrity(self):
        h = ProviderHealth(provider="groq", actual_integrity=0.75)
        d = h.to_dict()
        self.assertEqual(d["integrity"], 0.75)
        self.assertEqual(d["target_integrity"], 1.0)


class TestProviderMonitorHealing(unittest.TestCase):
    """Test heal_tick() and record_failure() on ProviderMonitor."""

    def test_heal_tick_regenerates(self):
        monitor = ProviderMonitor()
        monitor._health["groq"] = ProviderHealth(
            provider="groq", actual_integrity=0.60, regen_rate=0.05,
        )
        monitor.heal_tick()
        self.assertAlmostEqual(monitor._health["groq"].actual_integrity, 0.65)

    def test_multiple_heal_ticks_converge_to_target(self):
        monitor = ProviderMonitor()
        monitor._health["groq"] = ProviderHealth(
            provider="groq", actual_integrity=0.0, target_integrity=1.0,
            regen_rate=0.05,
        )
        for _ in range(25):
            monitor.heal_tick()
        self.assertAlmostEqual(monitor._health["groq"].actual_integrity, 1.0)

    def test_record_failure_decays_and_tracks(self):
        monitor = ProviderMonitor()
        monitor._health["groq"] = ProviderHealth(provider="groq")
        monitor.record_failure("groq")
        h = monitor._health["groq"]
        self.assertAlmostEqual(h.actual_integrity, 0.80)
        self.assertEqual(h.consecutive_failures, 1)
        self.assertEqual(h.total_failures, 1)

    def test_record_failure_creates_health_if_missing(self):
        monitor = ProviderMonitor()
        monitor.record_failure("new-provider")
        self.assertIn("new-provider", monitor._health)
        self.assertAlmostEqual(monitor._health["new-provider"].actual_integrity, 0.80)

    def test_three_failures_marks_unhealthy(self):
        monitor = ProviderMonitor()
        monitor._health["groq"] = ProviderHealth(provider="groq")
        for _ in range(3):
            monitor.record_failure("groq")
        self.assertFalse(monitor._health["groq"].healthy)

    def test_full_decay_then_full_recovery(self):
        """5 failures then enough heal ticks to fully recover."""
        monitor = ProviderMonitor()
        monitor._health["groq"] = ProviderHealth(
            provider="groq", decay_rate=0.20, regen_rate=0.05,
        )
        for _ in range(5):
            monitor.record_failure("groq")
        self.assertAlmostEqual(monitor._health["groq"].actual_integrity, 0.0)
        for _ in range(20):
            monitor.heal_tick()
        self.assertAlmostEqual(monitor._health["groq"].actual_integrity, 1.0)

    def test_get_integrity_returns_value(self):
        monitor = ProviderMonitor()
        monitor._health["groq"] = ProviderHealth(
            provider="groq", actual_integrity=0.42,
        )
        self.assertAlmostEqual(monitor.get_integrity("groq"), 0.42)

    def test_get_integrity_missing_returns_one(self):
        monitor = ProviderMonitor()
        self.assertEqual(monitor.get_integrity("unknown"), 1.0)


if __name__ == "__main__":
    unittest.main()
