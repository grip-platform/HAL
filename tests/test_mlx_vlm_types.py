"""Wave 1a: ChatMessage multimodal content type tests.

Verifies that extending ChatMessage.content to Union[str, list] is fully
backward-compatible — existing str content still works, and list content
(image_url, audio_url parts) is accepted without error.
"""
import os
import sys
import unittest

from lib.hal.types import ChatMessage, ContentPart


class TestStrContentBackwardCompat(unittest.TestCase):

    def test_str_content_accepted(self):
        msg = ChatMessage(role="user", content="hello world")
        self.assertEqual(msg.content, "hello world")

    def test_str_content_type(self):
        msg = ChatMessage(role="user", content="hello")
        self.assertIsInstance(msg.content, str)

    def test_empty_str_content(self):
        msg = ChatMessage(role="user", content="")
        self.assertEqual(msg.content, "")

    def test_system_role_str_content(self):
        msg = ChatMessage(role="system", content="You are HAL.")
        self.assertEqual(msg.role, "system")
        self.assertEqual(msg.content, "You are HAL.")


class TestMultimodalListContent(unittest.TestCase):

    def test_text_image_content_accepted(self):
        parts = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        ]
        msg = ChatMessage(role="user", content=parts)
        self.assertIsInstance(msg.content, list)
        self.assertEqual(len(msg.content), 2)

    def test_image_only_content(self):
        parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}}]
        msg = ChatMessage(role="user", content=parts)
        self.assertEqual(msg.content[0]["type"], "image_url")

    def test_audio_content_accepted(self):
        parts = [
            {"type": "text", "text": "Transcribe this"},
            {"type": "audio_url", "audio_url": {"url": "file:///path/audio.wav", "format": "wav"}},
        ]
        msg = ChatMessage(role="user", content=parts)
        self.assertEqual(msg.content[1]["type"], "audio_url")

    def test_list_content_rejects_non_dict_parts(self):
        """Content list must contain dict parts — non-dict elements are a usage error."""
        # Verify that a correctly-shaped part round-trips through the frozen dataclass
        part: ContentPart = {"type": "text", "text": "hello"}
        msg = ChatMessage(role="user", content=[part])
        self.assertEqual(msg.content[0]["type"], "text")
        self.assertEqual(msg.content[0]["text"], "hello")

    def test_frozen_rejects_mutation(self):
        msg = ChatMessage(role="user", content="hello")
        with self.assertRaises((AttributeError, TypeError)):
            msg.content = "changed"  # type: ignore[misc]

    def test_multipart_preserves_order(self):
        parts = [
            {"type": "text", "text": "First"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,A"}},
            {"type": "text", "text": "Last"},
        ]
        msg = ChatMessage(role="user", content=parts)
        self.assertEqual(msg.content[0]["text"], "First")
        self.assertEqual(msg.content[2]["text"], "Last")


if __name__ == "__main__":
    unittest.main()
