"""Tests for empty-message guard in lib/hal/utils/cch.py.

Covers _is_empty_content, _filter_empty_messages, and the wire-up
inside CCHSigner.build_body that drops empty turns before Anthropic
serialisation (fixes the 'messages.N: user messages must have
non-empty content' 400 error on CCH subscription path).
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.types import ChatMessage
from lib.hal.utils.cch import CCHSigner, CCHError, _is_empty_content, _filter_empty_messages


class TestIsEmptyContent(unittest.TestCase):

    def test_none_is_empty(self):
        self.assertTrue(_is_empty_content(None))

    def test_empty_string_is_empty(self):
        self.assertTrue(_is_empty_content(""))

    def test_whitespace_string_is_empty(self):
        self.assertTrue(_is_empty_content("   \n\t  "))

    def test_nonempty_string_not_empty(self):
        self.assertFalse(_is_empty_content("hello"))

    def test_empty_list_is_empty(self):
        self.assertTrue(_is_empty_content([]))

    def test_list_blank_text_only_is_empty(self):
        self.assertTrue(_is_empty_content([{"type": "text", "text": "   "}]))

    def test_list_with_real_text_not_empty(self):
        self.assertFalse(_is_empty_content([{"type": "text", "text": "hi"}]))

    def test_image_block_not_empty(self):
        self.assertFalse(_is_empty_content([{"type": "image", "source": {}}]))

    def test_tool_use_block_not_empty(self):
        self.assertFalse(_is_empty_content([{"type": "tool_use", "id": "x"}]))

    def test_unknown_block_type_treated_as_nonempty(self):
        self.assertFalse(_is_empty_content([{"type": "future_block"}]))


class TestFilterEmptyMessages(unittest.TestCase):

    def test_drops_empty_string_messages(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": ""},
        ]
        self.assertEqual(_filter_empty_messages(msgs), [msgs[0]])

    def test_preserves_order(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        self.assertEqual(_filter_empty_messages(msgs), msgs)

    def test_all_empty_returns_empty_list(self):
        msgs = [{"role": "user", "content": ""}]
        self.assertEqual(_filter_empty_messages(msgs), [])


class TestEmptyMessageFiltering(unittest.TestCase):
    """Guards against Anthropic 'messages.N: must have non-empty content' 400."""

    def _make_msgs(self, pairs):
        return [ChatMessage(role=r, content=c) for r, c in pairs]

    def test_empty_string_user_dropped(self):
        signer = CCHSigner()
        msgs = self._make_msgs([
            ("user", "hello"),
            ("assistant", "hi"),
            ("user", ""),
            ("user", "follow-up"),
        ])
        body, _ = signer.build_body(messages=msgs, model="claude-opus-4-7")
        payload = json.loads(body)
        contents = [m["content"] for m in payload["messages"]]
        self.assertNotIn("", contents)
        self.assertEqual(len(payload["messages"]), 3)

    def test_whitespace_only_dropped(self):
        signer = CCHSigner()
        msgs = self._make_msgs([("user", "hello"), ("user", "   \n\t  ")])
        body, _ = signer.build_body(messages=msgs, model="claude-opus-4-7")
        payload = json.loads(body)
        self.assertEqual(len(payload["messages"]), 1)

    def test_empty_list_dropped(self):
        signer = CCHSigner()
        msgs = self._make_msgs([("user", "hello"), ("user", [])])
        body, _ = signer.build_body(messages=msgs, model="claude-opus-4-7")
        payload = json.loads(body)
        self.assertEqual(len(payload["messages"]), 1)

    def test_image_only_preserved(self):
        signer = CCHSigner()
        msgs = [ChatMessage(role="user", content=[
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
        ])]
        body, _ = signer.build_body(messages=msgs, model="claude-opus-4-7")
        payload = json.loads(body)
        self.assertEqual(len(payload["messages"]), 1)

    def test_all_empty_raises(self):
        signer = CCHSigner()
        msgs = [ChatMessage(role="user", content="")]
        with self.assertRaises(CCHError):
            signer.build_body(messages=msgs, model="claude-opus-4-7")

    def test_single_turn_prompt_unchanged(self):
        signer = CCHSigner()
        body, _ = signer.build_body(prompt="hi", model="claude-opus-4-7")
        payload = json.loads(body)
        self.assertEqual(payload["messages"][0]["content"], "hi")


if __name__ == "__main__":
    unittest.main()
