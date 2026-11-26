"""Tests for HAL session replay timeline."""
import json
import os
import sys
import tempfile
import time
import unittest

from lib.hal.session_replay import (
    ReplayTimeline,
    SessionRecorder,
    TimelineEvent,
)


class TestTimelineEvent(unittest.TestCase):

    def test_to_dict(self):
        event = TimelineEvent(
            timestamp=1000.0, event_type="turn",
            role="user", content="Hello",
        )
        d = event.to_dict()
        self.assertEqual(d["event_type"], "turn")
        self.assertEqual(d["role"], "user")

    def test_content_truncated_in_dict(self):
        long_content = "x" * 1000
        event = TimelineEvent(timestamp=0.0, event_type="turn", content=long_content)
        d = event.to_dict()
        self.assertLessEqual(len(d["content"]), 500)


class TestSessionRecorder(unittest.TestCase):

    def test_generates_session_id(self):
        recorder = SessionRecorder()
        self.assertTrue(recorder.session_id.startswith("hal-"))

    def test_custom_session_id(self):
        recorder = SessionRecorder(session_id="test-123")
        self.assertEqual(recorder.session_id, "test-123")

    def test_record_turn(self):
        recorder = SessionRecorder()
        recorder.record("turn", "Hello", role="user")
        recorder.record("turn", "Hi!", role="assistant")
        timeline = recorder.to_timeline()
        self.assertEqual(timeline.turn_count, 1)
        self.assertEqual(len(timeline.events), 2)

    def test_record_tool(self):
        recorder = SessionRecorder()
        recorder.record("tool", "Read", metadata={"file": "test.py"}, duration_ms=50)
        timeline = recorder.to_timeline()
        self.assertEqual(timeline.tool_count, 1)

    def test_record_milestone(self):
        recorder = SessionRecorder()
        recorder.record("milestone", "PR merged")
        timeline = recorder.to_timeline()
        self.assertEqual(len(timeline.milestones()), 1)

    def test_record_error(self):
        recorder = SessionRecorder()
        recorder.record("error", "Connection refused")
        timeline = recorder.to_timeline()
        self.assertEqual(len(timeline.errors()), 1)

    def test_annotate_last(self):
        recorder = SessionRecorder()
        recorder.record("turn", "Hello", role="user")
        recorder.annotate_last("Key question")
        timeline = recorder.to_timeline()
        self.assertEqual(timeline.events[0].annotation, "Key question")

    def test_annotate_empty_noop(self):
        recorder = SessionRecorder()
        recorder.annotate_last("nothing")  # Should not raise


class TestReplayTimeline(unittest.TestCase):

    def test_duration(self):
        events = [
            TimelineEvent(timestamp=100.0, event_type="turn"),
            TimelineEvent(timestamp=110.0, event_type="turn"),
        ]
        timeline = ReplayTimeline(session_id="t", events=events, start_time=100.0)
        self.assertAlmostEqual(timeline.duration_s, 10.0)

    def test_summary(self):
        recorder = SessionRecorder(session_id="test")
        recorder.record("turn", "Q1", role="user")
        recorder.record("turn", "A1", role="assistant")
        recorder.record("tool", "Read")
        summary = recorder.to_timeline().summary
        self.assertIn("1 turns", summary)
        self.assertIn("1 tool", summary)

    def test_export_json(self):
        recorder = SessionRecorder(session_id="export-test")
        recorder.record("turn", "Hello", role="user")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            recorder.to_timeline().export_json(path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["session_id"], "export-test")
            self.assertEqual(len(data["events"]), 1)
        finally:
            os.unlink(path)

    def test_export_markdown(self):
        recorder = SessionRecorder(session_id="md-test")
        recorder.record("turn", "Fix bug", role="user")
        recorder.record("tool", "Read", duration_ms=30)
        recorder.record("turn", "Fixed", role="assistant")
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            recorder.to_timeline().export_markdown(path)
            with open(path) as f:
                content = f.read()
            self.assertIn("md-test", content)
            self.assertIn("Fix bug", content)
        finally:
            os.unlink(path)

    def test_to_dict(self):
        timeline = ReplayTimeline(session_id="t", events=[], start_time=0.0)
        d = timeline.to_dict()
        self.assertEqual(d["session_id"], "t")
        self.assertIn("events", d)


class TestHalPortableFormat(unittest.TestCase):
    """Tests for .hal portable session format export/import."""

    def test_export_hal_format(self):
        from lib.hal.session_replay import export_hal_format
        recorder = SessionRecorder(session_id="portable-test")
        recorder.record("turn", "Hello", role="user")
        recorder.record("tool", "Read", duration_ms=15)
        recorder.record("milestone", "Bug fixed")
        timeline = recorder.to_timeline()

        with tempfile.NamedTemporaryFile(suffix=".hal", delete=False) as f:
            path = f.name
        try:
            export_hal_format(timeline, path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["format"], "hal-session-v1")
            self.assertEqual(data["version"], 1)
            self.assertEqual(data["session_id"], "portable-test")
            self.assertEqual(data["summary"]["turns"], 1)
            self.assertEqual(data["summary"]["tool_calls"], 1)
            self.assertEqual(data["summary"]["milestones"], 1)
            self.assertEqual(len(data["events"]), 3)
        finally:
            os.unlink(path)

    def test_import_hal_format_roundtrip(self):
        from lib.hal.session_replay import export_hal_format, import_hal_format
        recorder = SessionRecorder(session_id="roundtrip")
        recorder.record("turn", "Test message", role="user")
        recorder.record("turn", "Response", role="assistant")
        timeline = recorder.to_timeline()

        with tempfile.NamedTemporaryFile(suffix=".hal", delete=False) as f:
            path = f.name
        try:
            export_hal_format(timeline, path)
            imported = import_hal_format(path)
            self.assertIsNotNone(imported)
            self.assertEqual(imported.session_id, "roundtrip")
            self.assertEqual(imported.turn_count, 1)
            self.assertEqual(len(imported.events), 2)
        finally:
            os.unlink(path)

    def test_import_nonexistent_returns_none(self):
        from lib.hal.session_replay import import_hal_format
        self.assertIsNone(import_hal_format("/nonexistent/session.hal"))

    def test_import_wrong_format_returns_none(self):
        from lib.hal.session_replay import import_hal_format
        with tempfile.NamedTemporaryFile(suffix=".hal", delete=False, mode="w") as f:
            json.dump({"format": "unknown"}, f)
            path = f.name
        try:
            self.assertIsNone(import_hal_format(path))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
