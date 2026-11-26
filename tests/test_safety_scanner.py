"""Tests for HAL safety scanner."""
import os, sys, unittest
from lib.hal.safety_scanner import SafetyScanner

class TestSafetyScanner(unittest.TestCase):
    def test_clean_code(self):
        code = "def add(a, b):\n    return a + b"
        self.assertEqual(len(SafetyScanner().scan_code(code)), 0)

    def test_detects_eval(self):
        code = "result = eval(user_input)"
        issues = SafetyScanner().scan_code(code)
        self.assertGreater(len(issues), 0)
        self.assertEqual(issues[0].category, "injection")

    def test_detects_os_system(self):
        code = "os.system(command)"
        issues = SafetyScanner().scan_code(code)
        self.assertGreater(len(issues), 0)

    def test_detects_pickle(self):
        code = "data = pickle.loads(raw)"
        issues = SafetyScanner().scan_code(code)
        self.assertGreater(len(issues), 0)
        self.assertEqual(issues[0].category, "deserialization")

    def test_non_python(self):
        self.assertEqual(len(SafetyScanner().scan_code("code", "javascript")), 0)

    def test_severity(self):
        issues = SafetyScanner().scan_code("eval(x)")
        self.assertEqual(issues[0].severity, "high")

if __name__ == "__main__":
    unittest.main()
