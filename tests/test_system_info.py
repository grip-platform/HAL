"""Tests for HAL system info."""
import os, sys, unittest
from lib.hal.system_info import HAL_VERSION, get_system_info, get_version_string

class TestSystemInfo(unittest.TestCase):
    def test_version(self):
        self.assertEqual(HAL_VERSION, "1.0.0")

    def test_get_system_info(self):
        info = get_system_info()
        self.assertIn("hal_version", info)
        self.assertIn("python_version", info)
        self.assertIn("modules", info)
        self.assertGreater(info["modules"], 50)

    def test_version_string(self):
        v = get_version_string()
        self.assertIn("HAL", v)
        self.assertIn("1.0.0", v)

if __name__ == "__main__":
    unittest.main()
