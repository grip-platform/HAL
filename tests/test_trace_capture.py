"""Tests for HAL ExecutionTrace and TraceCapture."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.hal.trace_capture import ExecutionTrace, TraceCapture, TraceStep


class TestExecutionTrace(unittest.TestCase):

    def test_empty_trace_not_succeeded(self):
        trace = ExecutionTrace()
        self.assertFalse(trace.succeeded)

    def test_all_successful_steps_succeeded(self):
        trace = ExecutionTrace()
        trace.record("step1", "ok", True)
        trace.record("step2", "ok", True)
        self.assertTrue(trace.succeeded)
        self.assertIsNone(trace.failure_point)

    def test_failure_point_set_on_first_failure(self):
        trace = ExecutionTrace()
        trace.record("step1", "ok", True)
        trace.record("step2", "fail", False)
        trace.record("step3", "ok", True)
        self.assertEqual(trace.failure_point, 1)

    def test_summary_with_failure(self):
        trace = ExecutionTrace()
        trace.record("build", "ok", True)
        trace.record("test", "assertion error", False)
        summary = trace.summary()
        self.assertIn("step 1", summary)
        self.assertIn("test", summary)

    def test_summary_all_pass(self):
        trace = ExecutionTrace()
        trace.record("build", "ok", True)
        summary = trace.summary()
        self.assertIn("1/1", summary)


class TestTraceCapture(unittest.TestCase):

    def test_context_manager_captures_duration(self):
        with TraceCapture() as tc:
            pass
        self.assertGreaterEqual(tc.trace.total_duration_ms, 0.0)

    def test_exception_recorded_in_trace(self):
        try:
            with TraceCapture() as tc:
                raise ValueError("boom")
        except ValueError:
            pass
        self.assertEqual(tc.trace.failure_point, 0)
        self.assertIn("boom", tc.trace.steps[0].result)

    def test_exception_not_suppressed(self):
        with self.assertRaises(RuntimeError):
            with TraceCapture():
                raise RuntimeError("propagated")


class TestTraceStep(unittest.TestCase):

    def test_step_fields(self):
        step = TraceStep(step=0, action="compile", result="ok", success=True, duration_ms=12.5)
        self.assertEqual(step.action, "compile")
        self.assertTrue(step.success)
        self.assertAlmostEqual(step.duration_ms, 12.5)


if __name__ == "__main__":
    unittest.main()
