"""Tests for HAL provider monitor."""
import os
import sys
import unittest
from unittest.mock import MagicMock

from lib.hal.provider_monitor import MonitorReport, ProviderHealth, ProviderMonitor


class TestProviderHealth(unittest.TestCase):

    def test_uptime(self):
        self.assertAlmostEqual(ProviderHealth("t", total_checks=10, total_failures=0).uptime_pct, 100.0)

    def test_to_dict(self):
        self.assertIn("uptime_pct", ProviderHealth("groq").to_dict())


class TestMonitorReport(unittest.TestCase):

    def test_all_healthy(self):
        self.assertTrue(MonitorReport(providers=[ProviderHealth("a", healthy=True)]).all_healthy)

    def test_not_all(self):
        self.assertFalse(MonitorReport(providers=[ProviderHealth("a", healthy=False)]).all_healthy)


class TestProviderMonitor(unittest.TestCase):

    def test_no_registry(self):
        self.assertEqual(len(ProviderMonitor().check_all().providers), 0)

    def test_healthy(self):
        p = MagicMock()
        p.name = "groq"
        p.models = ["llama"]
        p.healthy = True
        reg = MagicMock()
        reg.list_providers.return_value = [p]
        self.assertTrue(ProviderMonitor(reg).check_all().providers[0].healthy)

    def test_get_healthy(self):
        p = MagicMock()
        p.name = "groq"
        p.models = ["llama"]
        p.healthy = True
        reg = MagicMock()
        reg.list_providers.return_value = [p]
        mon = ProviderMonitor(reg)
        mon.check_all()
        self.assertIn("groq", mon.get_healthy_providers())


if __name__ == "__main__":
    unittest.main()
