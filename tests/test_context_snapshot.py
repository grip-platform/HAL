"""Tests for HAL context snapshot."""
import json, os, sys, tempfile, unittest
from lib.hal.context_snapshot import ContextSnapshot
from lib.hal.types import ChatMessage

class TestContextSnapshot(unittest.TestCase):
    def test_save_and_load(self):
        snap = ContextSnapshot(
            session_id="test", model="groq/llama",
            history=[ChatMessage(role="user", content="Hello")],
            total_cost_usd=0.01,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            snap.save(path)
            loaded = ContextSnapshot.load(path)
            self.assertEqual(loaded.session_id, "test")
            self.assertEqual(loaded.model, "groq/llama")
            self.assertEqual(len(loaded.history), 1)
            self.assertEqual(loaded.history[0].content, "Hello")
        finally:
            os.unlink(path)

    def test_turn_count(self):
        snap = ContextSnapshot(history=[
            ChatMessage(role="user", content="Q1"),
            ChatMessage(role="assistant", content="A1"),
            ChatMessage(role="user", content="Q2"),
        ])
        self.assertEqual(snap.turn_count, 2)

    def test_empty_snapshot(self):
        snap = ContextSnapshot()
        self.assertEqual(snap.turn_count, 0)

    def test_list_snapshots(self):
        result = ContextSnapshot.list_snapshots()
        self.assertIsInstance(result, list)

if __name__ == "__main__":
    unittest.main()
