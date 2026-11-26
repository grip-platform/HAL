"""Tests for HAL retry policy."""
import os, sys, unittest
from lib.hal.retry_policy import ExponentialBackoff, LinearBackoff, FixedDelay

class TestExponentialBackoff(unittest.TestCase):
    def test_delays(self):
        b = ExponentialBackoff(base_s=1.0, max_s=30.0)
        self.assertAlmostEqual(b.delay(0), 1.0)
        self.assertAlmostEqual(b.delay(1), 2.0)
        self.assertAlmostEqual(b.delay(2), 4.0)

    def test_capped(self):
        b = ExponentialBackoff(base_s=1.0, max_s=10.0)
        self.assertAlmostEqual(b.delay(10), 10.0)

class TestLinearBackoff(unittest.TestCase):
    def test_delays(self):
        b = LinearBackoff(base_s=2.0)
        self.assertAlmostEqual(b.delay(0), 2.0)
        self.assertAlmostEqual(b.delay(1), 4.0)

class TestFixedDelay(unittest.TestCase):
    def test_constant(self):
        d = FixedDelay(5.0)
        self.assertAlmostEqual(d.delay(0), 5.0)
        self.assertAlmostEqual(d.delay(99), 5.0)

if __name__ == "__main__":
    unittest.main()
