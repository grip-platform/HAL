"""Tests for HAL progress tracker."""
import os, sys, unittest
from lib.hal.progress_tracker import ProgressTracker

class TestProgressTracker(unittest.TestCase):
    def test_advance(self):
        t = ProgressTracker("build", 5)
        t.advance("Step 1")
        s = t.status
        self.assertEqual(s.completed, 1)
        self.assertAlmostEqual(s.pct, 20.0)

    def test_done(self):
        t = ProgressTracker("t", 2)
        t.advance()
        self.assertFalse(t.done)
        t.advance()
        self.assertTrue(t.done)

    def test_summary(self):
        t = ProgressTracker("build", 10)
        t.advance("A")
        self.assertIn("10%", t.status.summary)

    def test_set_progress(self):
        t = ProgressTracker("t", 100)
        t.set_progress(50)
        self.assertAlmostEqual(t.status.pct, 50.0)

    def test_steps(self):
        t = ProgressTracker("t", 3)
        t.advance("A")
        t.advance("B")
        self.assertEqual(t.steps_completed, ["A", "B"])

    def test_cant_exceed_total(self):
        t = ProgressTracker("t", 2)
        for _ in range(5):
            t.advance()
        self.assertEqual(t.status.completed, 2)

if __name__ == "__main__":
    unittest.main()
