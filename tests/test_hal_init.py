"""Tests for HAL init bootstrap."""
import os, sys, unittest
from lib.hal.hal_init import HALRuntime
from lib.hal.system_info import HAL_VERSION

class TestHALRuntime(unittest.TestCase):
    def test_not_ready_empty(self):
        r = HALRuntime()
        self.assertFalse(r.ready)
        self.assertEqual(r.version, HAL_VERSION)

    def test_ready_with_registry(self):
        r = HALRuntime(registry=object())
        self.assertTrue(r.ready)

if __name__ == "__main__":
    unittest.main()
