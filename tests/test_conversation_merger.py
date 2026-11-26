"""Tests for HAL conversation merger."""
import os, sys, unittest
from lib.hal.conversation_merger import merge_conversations
from lib.hal.types import ChatMessage

def _msg(role, content):
    return ChatMessage(role=role, content=content)

class TestMerge(unittest.TestCase):
    def test_append(self):
        a = [_msg("user", "A1")]
        b = [_msg("user", "B1")]
        merged = merge_conversations(a, b, "append")
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].content, "A1")
        self.assertEqual(merged[1].content, "B1")

    def test_interleave(self):
        a = [_msg("user", "A1"), _msg("user", "A2")]
        b = [_msg("user", "B1"), _msg("user", "B2")]
        merged = merge_conversations(a, b, "interleave")
        self.assertEqual(len(merged), 4)
        self.assertEqual(merged[0].content, "A1")
        self.assertEqual(merged[1].content, "B1")

    def test_empty(self):
        self.assertEqual(len(merge_conversations([], [])), 0)

    def test_unequal_lengths(self):
        a = [_msg("user", "A1"), _msg("user", "A2"), _msg("user", "A3")]
        b = [_msg("user", "B1")]
        merged = merge_conversations(a, b, "interleave")
        self.assertEqual(len(merged), 4)

if __name__ == "__main__":
    unittest.main()
