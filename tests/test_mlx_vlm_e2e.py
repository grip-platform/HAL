"""Wave 8: openai_compat.py zero-change proof — multimodal wire format.

Hypothesis H7: _message_to_dict() passes list content through unchanged,
producing the correct OpenAI content-array wire format for multimodal
messages. No code changes to openai_compat.py are required.

These tests inspect the HTTP request body directly to confirm the
serialisation path is already correct.
"""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.providers.openai_compat import (
    OpenAICompatProvider,
    _message_to_dict,
)
from lib.hal.types import ChatMessage


class TestMessageToDictMultimodal(unittest.TestCase):
    """_message_to_dict passes list content through without modification."""

    def test_string_content_unchanged(self):
        msg = ChatMessage(role="user", content="Hello")
        d = _message_to_dict(msg)
        self.assertEqual(d["content"], "Hello")
        self.assertIsInstance(d["content"], str)

    def test_list_content_passed_through_as_list(self):
        parts = [
            {"type": "text", "text": "Describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        ]
        msg = ChatMessage(role="user", content=parts)
        d = _message_to_dict(msg)
        self.assertIs(d["content"], parts)  # identity: same object, no copy
        self.assertIsInstance(d["content"], list)

    def test_list_content_serialises_to_json_array(self):
        parts = [
            {"type": "text", "text": "What is shown?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
        ]
        msg = ChatMessage(role="user", content=parts)
        d = _message_to_dict(msg)
        body = json.dumps({"messages": [d]})
        parsed = json.loads(body)
        self.assertIsInstance(parsed["messages"][0]["content"], list)
        self.assertEqual(parsed["messages"][0]["content"][0]["type"], "text")
        self.assertEqual(parsed["messages"][0]["content"][1]["type"], "image_url")

    def test_provider_name_preserved_in_response(self):
        """ChatResponse.provider must equal the provider's name attribute."""
        provider = OpenAICompatProvider(
            name="mlx_vlm",
            base_url="http://localhost:8080",
        )
        fake_response = {
            "choices": [{"message": {"content": "a cat"}, "finish_reason": "stop"}],
            "model": "mlx-community/Qwen2-VL-7B-Instruct-4bit",
            "usage": {},
            "_headers": {},
        }
        with patch.object(provider, "_do_request", return_value=fake_response):
            resp = provider.chat(
                "Describe this image",
                model="mlx-community/Qwen2-VL-7B-Instruct-4bit",
            )
        self.assertEqual(resp.provider, "mlx_vlm")
        self.assertEqual(resp.text, "a cat")

    def test_http_body_contains_content_array_for_list_messages(self):
        """Confirms the actual bytes sent over the wire contain a content array."""
        provider = OpenAICompatProvider(
            name="mlx_vlm",
            base_url="http://localhost:8080",
        )
        parts = [
            {"type": "text", "text": "What colour is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j"}},
        ]
        captured_bodies: list[bytes] = []

        original_do_request = provider._do_request

        def capture(method, url, body):
            if body:
                captured_bodies.append(json.dumps(body).encode())
            return {
                "choices": [{"message": {"content": "red"}, "finish_reason": "stop"}],
                "model": "mlx-community/Qwen2-VL-7B-Instruct-4bit",
                "usage": {},
                "_headers": {},
            }

        with patch.object(provider, "_do_request", side_effect=capture):
            provider.chat(
                "",
                model="mlx-community/Qwen2-VL-7B-Instruct-4bit",
                messages=[ChatMessage(role="user", content=parts)],
            )

        self.assertEqual(len(captured_bodies), 1)
        payload = json.loads(captured_bodies[0])
        content = payload["messages"][0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
