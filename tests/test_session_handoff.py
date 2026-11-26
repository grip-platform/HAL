"""Tests for HAL session handoff."""
import json, os, sys, tempfile, unittest
from lib.hal.session_handoff import HandoffPackage
from lib.hal.types import ChatMessage

class TestHandoffPackage(unittest.TestCase):
    def test_save_and_load(self):
        pkg = HandoffPackage(
            session_id="s1", model="groq/llama",
            history=[ChatMessage(role="user", content="Hello")],
            context_summary="Debugging auth", cost_usd=0.05,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            pkg.save(path)
            loaded = HandoffPackage.load(path)
            self.assertEqual(loaded.session_id, "s1")
            self.assertEqual(loaded.model, "groq/llama")
            self.assertEqual(len(loaded.history), 1)
        finally:
            os.unlink(path)

    def test_turn_count(self):
        pkg = HandoffPackage(session_id="t", model="m", history=[
            ChatMessage(role="user", content="Q"),
            ChatMessage(role="assistant", content="A"),
        ])
        self.assertEqual(pkg.turn_count, 1)

    def test_empty(self):
        self.assertEqual(HandoffPackage(session_id="t", model="m").turn_count, 0)

if __name__ == "__main__":
    unittest.main()
