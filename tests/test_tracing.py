"""Tests for HAL tracing/observability."""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from lib.hal import tracing
from lib.hal.tracing import Span, Tracer


class TestSpan(unittest.TestCase):

    def test_duration(self):
        s = Span(name="test", trace_id="t", span_id="s", start_time=100.0, end_time=100.5)
        self.assertAlmostEqual(s.duration_ms, 500.0)

    def test_set_attribute(self):
        s = Span(name="test", trace_id="t", span_id="s")
        s.set_attribute("model", "groq/llama")
        self.assertEqual(s.attributes["model"], "groq/llama")

    def test_set_error(self):
        s = Span(name="test", trace_id="t", span_id="s")
        s.set_error("connection refused")
        self.assertEqual(s.status, "error")
        self.assertIn("refused", s.error_message)

    def test_to_dict(self):
        s = Span(name="test", trace_id="t", span_id="s", start_time=0, end_time=1)
        d = s.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertIn("duration_ms", d)


class TestTracer(unittest.TestCase):

    def test_span_context_manager(self):
        tracer = Tracer("test")
        with tracer.span("operation") as s:
            s.set_attribute("key", "value")
        self.assertEqual(len(tracer.spans), 1)
        self.assertEqual(tracer.spans[0].name, "operation")
        self.assertGreater(tracer.spans[0].duration_ms, 0)

    def test_span_captures_error(self):
        tracer = Tracer("test")
        with self.assertRaises(ValueError):
            with tracer.span("failing") as s:
                raise ValueError("boom")
        self.assertEqual(tracer.spans[0].status, "error")
        self.assertIn("boom", tracer.spans[0].error_message)

    def test_record(self):
        tracer = Tracer("test")
        tracer.record("manual_span", duration_ms=150, model="groq")
        self.assertEqual(len(tracer.spans), 1)
        self.assertAlmostEqual(tracer.spans[0].duration_ms, 150, delta=50)

    def test_exported_span_timestamps_agree_with_duration(self):
        # A coarse wall clock (Python 3.12 on Windows) can read the same instant
        # at both ends while perf_counter measures a real interval.
        tracer = Tracer("test")
        ticks = iter([5.0, 5.25])
        fake_clock = SimpleNamespace(time=lambda: 1000.0, perf_counter=lambda: next(ticks))
        with mock.patch.object(tracing, "time", fake_clock):
            with tracer.span("operation"):
                pass
        exported = tracer.spans[0].to_dict()
        self.assertAlmostEqual(exported["duration_ms"], 250.0)
        self.assertAlmostEqual((exported["end_time"] - exported["start_time"]) * 1000, 250.0, places=6)

    def test_record_keeps_the_given_duration_exactly(self):
        tracer = Tracer("test")
        tracer.record("manual_span", duration_ms=150)
        span = tracer.spans[0]
        self.assertEqual(span.duration_ms, 150.0)
        self.assertAlmostEqual((span.end_time - span.start_time) * 1000, 150.0, places=2)

    def test_summary(self):
        tracer = Tracer("test-svc")
        with tracer.span("a"):
            pass
        with tracer.span("b"):
            pass
        summary = tracer.summary()
        self.assertEqual(summary["span_count"], 2)
        self.assertEqual(summary["service"], "test-svc")

    def test_export_json(self):
        tracer = Tracer("export-test")
        with tracer.span("op"):
            pass
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tracer.export_json(path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["service"], "export-test")
            self.assertEqual(len(data["spans"]), 1)
        finally:
            os.unlink(path)

    def test_trace_id(self):
        tracer = Tracer("my-service")
        self.assertTrue(tracer.trace_id.startswith("my-service-"))

    def test_multiple_spans(self):
        tracer = Tracer("multi")
        for i in range(5):
            with tracer.span(f"step_{i}"):
                pass
        self.assertEqual(len(tracer.spans), 5)
        self.assertEqual(tracer.summary()["error_count"], 0)


if __name__ == "__main__":
    unittest.main()
