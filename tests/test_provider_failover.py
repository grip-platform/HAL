"""Tests for HAL provider failover."""
import os, sys, unittest
from lib.hal.provider_failover import FailoverChain

class TestFailoverChain(unittest.TestCase):
    def test_primary(self):
        c = FailoverChain(["a", "b", "c"])
        self.assertEqual(c.primary, "a")

    def test_next_available(self):
        r = FailoverChain(["a", "b"]).next_available()
        self.assertEqual(r.model, "a")
        self.assertFalse(r.is_fallback)

    def test_skip_failed(self):
        r = FailoverChain(["a", "b", "c"]).next_available(failed={"a"})
        self.assertEqual(r.model, "b")
        self.assertTrue(r.is_fallback)

    def test_mark_failed(self):
        c = FailoverChain(["a", "b"])
        c.mark_failed("a")
        self.assertEqual(c.next_available().model, "b")

    def test_mark_recovered(self):
        c = FailoverChain(["a", "b"])
        c.mark_failed("a")
        c.mark_recovered("a")
        self.assertEqual(c.next_available().model, "a")

    def test_all_failed(self):
        r = FailoverChain(["a"]).next_available(failed={"a"})
        self.assertEqual(r.model, "")

    def test_healthy(self):
        c = FailoverChain(["a", "b", "c"])
        c.mark_failed("b")
        self.assertEqual(c.healthy, ["a", "c"])

if __name__ == "__main__":
    unittest.main()
