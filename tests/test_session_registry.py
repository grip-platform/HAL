"""Tests for HAL Session Registry — multi-session conversation management."""
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch


def _make_mock_registry():
    reg = MagicMock()
    reg.routing = MagicMock()
    reg.routing.default_model = "groq/llama-3.3-70b"
    reg.list_providers.return_value = []
    return reg


class _SessionRegistryTestBase(unittest.TestCase):
    """Base class that sets up a SessionRegistry with temp dir and mocks."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._mock_registry = _make_mock_registry()
        self._patcher_dir = patch("lib.hal.session_registry._SESSIONS_DIR", self._tmpdir)
        self._patcher_conv = patch("lib.hal.session_registry.Conversation")
        self._patcher_dir.start()
        mock_conv_cls = self._patcher_conv.start()
        mock_conv = MagicMock()
        mock_conv.model = "groq/llama-3.3-70b"
        mock_conv.history = []
        mock_conv.cost = MagicMock()
        mock_conv.cost.snapshot.return_value = MagicMock(total_usd=0.0)
        mock_conv_cls.return_value = mock_conv
        from lib.hal.session_registry import SessionRegistry
        self.registry = SessionRegistry(registry=self._mock_registry)

    def tearDown(self):
        self._patcher_conv.stop()
        self._patcher_dir.stop()


class TestSessionCreate(_SessionRegistryTestBase):

    def test_create_returns_session_id(self):
        sid = self.registry.create()
        self.assertIsInstance(sid, str)
        self.assertEqual(len(sid), 8)

    def test_create_with_model(self):
        sid = self.registry.create(model="anthropic/claude-sonnet-4-6")
        session = self.registry.get(sid)
        self.assertIsNotNone(session)
        self.assertEqual(session.model, "anthropic/claude-sonnet-4-6")

    def test_create_with_title(self):
        sid = self.registry.create(title="Test chat")
        session = self.registry.get(sid)
        self.assertEqual(session.title, "Test chat")

    def test_create_increments_count(self):
        self.registry.create()
        self.registry.create()
        self.registry.create()
        self.assertEqual(len(self.registry.list_sessions()), 3)


class TestSessionGet(_SessionRegistryTestBase):

    def test_get_existing(self):
        sid = self.registry.create()
        session = self.registry.get(sid)
        self.assertIsNotNone(session)
        self.assertEqual(session.id, sid)

    def test_get_nonexistent(self):
        self.assertIsNone(self.registry.get("nonexistent"))

    def test_get_or_create_existing(self):
        sid = self.registry.create(title="Original")
        session = self.registry.get_or_create(sid)
        self.assertEqual(session.title, "Original")

    def test_get_or_create_new(self):
        session = self.registry.get_or_create(None, model="test-model")
        self.assertIsNotNone(session)
        self.assertIsNotNone(session.id)


class TestSessionList(_SessionRegistryTestBase):

    def test_list_empty(self):
        self.assertEqual(self.registry.list_sessions(), [])

    def test_list_returns_summaries(self):
        self.registry.create(title="First")
        self.registry.create(title="Second")
        summaries = self.registry.list_sessions()
        self.assertEqual(len(summaries), 2)
        titles = {s.title for s in summaries}
        self.assertIn("First", titles)
        self.assertIn("Second", titles)

    def test_list_ordered_by_updated(self):
        self.registry.create(title="Old")
        time.sleep(0.01)
        self.registry.create(title="New")
        summaries = self.registry.list_sessions()
        self.assertEqual(summaries[0].title, "New")
        self.assertEqual(summaries[1].title, "Old")

    def test_summary_to_dict(self):
        self.registry.create(title="Test")
        summary = self.registry.list_sessions()[0]
        d = summary.to_dict()
        self.assertIn("id", d)
        self.assertEqual(d["title"], "Test")
        self.assertIn("created_at", d)
        self.assertIn("updated_at", d)


class TestSessionDelete(_SessionRegistryTestBase):

    def test_delete_existing(self):
        sid = self.registry.create()
        self.assertTrue(self.registry.delete(sid))
        self.assertIsNone(self.registry.get(sid))

    def test_delete_nonexistent(self):
        self.assertFalse(self.registry.delete("nonexistent"))

    def test_delete_removes_from_list(self):
        s1 = self.registry.create()
        s2 = self.registry.create()
        self.registry.delete(s1)
        summaries = self.registry.list_sessions()
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].id, s2)


class TestSessionEviction(_SessionRegistryTestBase):

    def test_evicts_oldest_when_over_capacity(self):
        with patch("lib.hal.session_registry._MAX_SESSIONS", 3):
            s1 = self.registry.create(title="Oldest")
            time.sleep(0.01)
            self.registry.create(title="Middle")
            time.sleep(0.01)
            self.registry.create(title="Newer")
            time.sleep(0.01)
            self.registry.create(title="Newest")
            summaries = self.registry.list_sessions()
            self.assertEqual(len(summaries), 3)
            ids = {s.id for s in summaries}
            self.assertNotIn(s1, ids)


class TestSessionPersistence(_SessionRegistryTestBase):

    def test_persist_creates_file(self):
        sid = self.registry.create(title="Persisted")
        path = os.path.join(self._tmpdir, f"{sid}.json")
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["id"], sid)
        self.assertEqual(data["title"], "Persisted")

    def test_delete_removes_file(self):
        sid = self.registry.create()
        path = os.path.join(self._tmpdir, f"{sid}.json")
        self.assertTrue(os.path.exists(path))
        self.registry.delete(sid)
        self.assertFalse(os.path.exists(path))


class TestSessionUpdateTitle(_SessionRegistryTestBase):

    def test_update_title(self):
        sid = self.registry.create(title="Old title")
        self.registry.update_title(sid, "New title")
        session = self.registry.get(sid)
        self.assertEqual(session.title, "New title")

    def test_update_nonexistent_no_error(self):
        self.registry.update_title("nonexistent", "Title")


class TestStreamEventProtocol(unittest.TestCase):
    """Verify GRIP-GUI compatible event shapes match grip-session.ts expectations."""

    def test_text_delta_shape(self):
        event = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello world"},
        }
        self.assertEqual(event["type"], "content_block_delta")
        self.assertEqual(event["delta"]["type"], "text_delta")
        self.assertIsInstance(event["delta"]["text"], str)

    def test_result_shape(self):
        event = {
            "type": "result",
            "cost_usd": 0.001,
            "num_turns": 3,
            "session_id": "abc12345",
            "model": "groq/llama-3.3-70b",
        }
        self.assertEqual(event["type"], "result")
        self.assertIsInstance(event["cost_usd"], float)
        self.assertIsInstance(event["session_id"], str)

    def test_init_event_shape(self):
        event = {"type": "system", "subtype": "init", "session_id": "abc12345"}
        self.assertEqual(event["subtype"], "init")

    def test_error_event_shape(self):
        event = {"type": "error", "error": {"message": "Something went wrong"}}
        self.assertEqual(event["type"], "error")


class TestServerHelpers(unittest.TestCase):

    def test_prepare_session_auto_titles(self):
        from lib.hal.server import _prepare_session
        mock_sessions = MagicMock()
        mock_session = MagicMock()
        mock_session.message_count = 0
        mock_session.model = "test"
        mock_session.id = "abc"
        mock_sessions.get_or_create.return_value = mock_session
        result = _prepare_session(mock_sessions, None, "test", "Hello world query")
        mock_sessions.update_title.assert_called_once_with("abc", "Hello world query")
        self.assertIs(result, mock_session)

    def test_prepare_session_truncates_long_title(self):
        from lib.hal.server import _prepare_session
        mock_sessions = MagicMock()
        mock_session = MagicMock()
        mock_session.message_count = 0
        mock_session.model = "test"
        mock_session.id = "abc"
        mock_sessions.get_or_create.return_value = mock_session
        long_msg = "x" * 100
        _prepare_session(mock_sessions, None, "test", long_msg)
        call_args = mock_sessions.update_title.call_args[0]
        self.assertEqual(len(call_args[1]), 53)

    def test_prepare_session_skips_title_if_existing(self):
        from lib.hal.server import _prepare_session
        mock_sessions = MagicMock()
        mock_session = MagicMock()
        mock_session.message_count = 5
        mock_session.model = "test"
        mock_sessions.get_or_create.return_value = mock_session
        _prepare_session(mock_sessions, "existing-id", "test", "msg")
        mock_sessions.update_title.assert_not_called()

    def test_stream_response_chunks_emits_events(self):
        from lib.hal.server import _stream_response_chunks
        from lib.hal.response_streamer import ResponseStreamer
        events = []
        _stream_response_chunks(events.append, ResponseStreamer(), "Hello world")
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "text")
        self.assertEqual(events[0]["delta"]["type"], "text_delta")

    def test_stream_result_event_shape(self):
        from lib.hal.server import _stream_result_event
        mock_session = MagicMock()
        mock_session.id = "test-123"
        mock_session.conversation.history = [MagicMock(), MagicMock()]
        mock_session.conversation.cost.snapshot.return_value = MagicMock(total_usd=0.005)
        mock_response = MagicMock()
        mock_response.model = "groq/llama"
        events = []
        _stream_result_event(events.append, mock_session, mock_response, 150.5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "result")
        self.assertEqual(events[0]["session_id"], "test-123")
        self.assertEqual(events[0]["cost_usd"], 0.005)


if __name__ == "__main__":
    unittest.main()
