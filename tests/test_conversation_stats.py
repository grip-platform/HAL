"""Tests for HAL conversation stats."""
import os, sys, unittest
from lib.hal.conversation_stats import ConversationStats
from lib.hal.types import ChatMessage

def _msg(role, content):
    return ChatMessage(role=role, content=content)

class TestConversationStats(unittest.TestCase):
    def test_basic(self):
        stats = ConversationStats.from_history([
            _msg("user", "How do I sort a list?"),
            _msg("assistant", "Use the sorted() function in Python."),
        ])
        self.assertEqual(stats.user_messages, 1)
        self.assertEqual(stats.assistant_messages, 1)
        self.assertEqual(stats.questions_asked, 1)

    def test_code_blocks(self):
        stats = ConversationStats.from_history([
            _msg("assistant", "Here:\n```python\nx = 1\n```\nDone"),
        ])
        self.assertEqual(stats.code_blocks, 1)

    def test_response_ratio(self):
        stats = ConversationStats.from_history([
            _msg("user", "Hi"),
            _msg("assistant", "Hello there how are you doing today friend"),
        ])
        self.assertGreater(stats.response_ratio, 1.0)

    def test_empty(self):
        stats = ConversationStats.from_history([])
        self.assertEqual(stats.total_messages, 0)

    def test_to_dict(self):
        d = ConversationStats.from_history([_msg("user", "Hi")]).to_dict()
        self.assertIn("total_messages", d)

    def test_top_words(self):
        stats = ConversationStats.from_history([
            _msg("user", "python python python code code function"),
        ])
        self.assertGreater(len(stats.top_words), 0)

if __name__ == "__main__":
    unittest.main()
