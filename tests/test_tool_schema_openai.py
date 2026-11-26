"""Regression tests for issue #200 — /tools on schema mismatch on OpenAI-compat.

Scope:
  * ``_anthropic_tools_to_openai`` converts Anthropic-shaped tool specs to
    OpenAI-compat shape ({type: function, function: {...}}).
  * Conversion is idempotent — OpenAI-shaped input passes through unchanged.
  * ``make_openai_compat_chat_fn`` emits OpenAI-shaped tools on the wire
    (Goodhart guard — reverting the conversion call breaks this test).
  * ``_provider_label_from_url`` maps URLs to human-readable provider names.
  * ``AnthropicChatError`` renders provider-specific messages so error
    triage points at the real offending API.
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.tool_runtime import (
    AnthropicChatError,
    ChatAPIError,
    _anthropic_tools_to_openai,
    _provider_label_from_url,
    make_openai_compat_chat_fn,
)


# Sample input: Anthropic shape as produced by `to_anthropic_tools()`.
ANTHROPIC_TOOL = {
    "name": "bash",
    "description": "Execute a bash command and return stdout.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
        },
        "required": ["command"],
    },
}

# Same tool in OpenAI-compat shape — what Groq/OpenAI/Gemini actually accept.
OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command and return stdout.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
        },
    },
}


class TestAnthropicToolsToOpenAI(unittest.TestCase):

    def test_wraps_with_type_function(self):
        """Anthropic -> OpenAI: adds type + function wrapper."""
        out = _anthropic_tools_to_openai([ANTHROPIC_TOOL])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["type"], "function")
        self.assertIn("function", out[0])

    def test_preserves_name_description_schema(self):
        """All three fields must survive the shape change."""
        out = _anthropic_tools_to_openai([ANTHROPIC_TOOL])[0]
        self.assertEqual(out["function"]["name"], "bash")
        self.assertEqual(
            out["function"]["description"],
            "Execute a bash command and return stdout.",
        )
        self.assertEqual(
            out["function"]["parameters"],
            ANTHROPIC_TOOL["input_schema"],
        )

    def test_idempotent_on_openai_input(self):
        """Pre-shaped OpenAI input passes through untouched."""
        out = _anthropic_tools_to_openai([OPENAI_TOOL])
        self.assertEqual(out, [OPENAI_TOOL])

    def test_handles_missing_description(self):
        """Anthropic specs sometimes omit description — don't crash."""
        stripped = {k: v for k, v in ANTHROPIC_TOOL.items() if k != "description"}
        out = _anthropic_tools_to_openai([stripped])[0]
        self.assertEqual(out["function"]["description"], "")

    def test_empty_list_passthrough(self):
        self.assertEqual(_anthropic_tools_to_openai([]), [])

    def test_does_not_mutate_input(self):
        spec = dict(ANTHROPIC_TOOL)
        _ = _anthropic_tools_to_openai([spec])
        # Original should still be in Anthropic shape
        self.assertIn("input_schema", spec)
        self.assertNotIn("type", spec)
        self.assertNotIn("function", spec)


class TestOpenAICompatChatFnWireShape(unittest.TestCase):
    """Goodhart guard — if the conversion call is removed from
    ``make_openai_compat_chat_fn``, this test fails because the wire body
    still carries Anthropic-shaped tools.
    """

    def test_posts_openai_shaped_tools(self):
        captured: dict = {}

        def fake_post(*, url, api_key, body, timeout_s, extra_headers):
            captured["body"] = body
            # Minimal valid OpenAI response so the normaliser doesn't raise
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
                "model": "llama-3.3-70b-versatile",
            }

        chat_fn = make_openai_compat_chat_fn(
            base_url="https://api.groq.com/openai/v1",
            api_key="test",
            model="llama-3.3-70b-versatile",
            max_tokens=100,
        )

        with patch("lib.hal.tool_runtime._post_openai_chat", side_effect=fake_post):
            chat_fn(messages=[{"role": "user", "content": "hi"}],
                    tools=[ANTHROPIC_TOOL])

        self.assertIn("tools", captured["body"])
        wire_tool = captured["body"]["tools"][0]
        self.assertEqual(wire_tool["type"], "function",
                         "tools.0.type missing — issue #200 regression")
        self.assertEqual(wire_tool["function"]["name"], "bash")
        self.assertEqual(captured["body"]["tool_choice"], "auto")

    def test_no_tools_field_when_list_empty(self):
        """Matches Anthropic chat_fn contract — empty tools means no tools key."""
        captured: dict = {}

        def fake_post(*, url, api_key, body, timeout_s, extra_headers):
            captured["body"] = body
            return {
                "choices": [{"message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "model": "m",
            }

        chat_fn = make_openai_compat_chat_fn(
            base_url="https://api.groq.com/openai/v1",
            api_key="test", model="m", max_tokens=100,
        )

        with patch("lib.hal.tool_runtime._post_openai_chat", side_effect=fake_post):
            chat_fn(messages=[{"role": "user", "content": "hi"}], tools=[])

        self.assertNotIn("tools", captured["body"])
        self.assertNotIn("tool_choice", captured["body"])


class TestProviderLabel(unittest.TestCase):

    def test_groq(self):
        self.assertEqual(
            _provider_label_from_url("https://api.groq.com/openai/v1/chat/completions"),
            "Groq",
        )

    def test_gemini(self):
        self.assertEqual(
            _provider_label_from_url(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            ),
            "Gemini",
        )

    def test_openai(self):
        self.assertEqual(
            _provider_label_from_url("https://api.openai.com/v1/chat/completions"),
            "OpenAI",
        )

    def test_ollama_localhost(self):
        self.assertEqual(
            _provider_label_from_url("http://localhost:11434/v1/chat/completions"),
            "Ollama",
        )

    def test_unknown_defaults_to_provider(self):
        self.assertEqual(
            _provider_label_from_url("https://example.invalid/v1"),
            "Provider",
        )


class TestAnthropicChatErrorMessage(unittest.TestCase):

    def test_default_provider_label(self):
        """Backwards compat: no provider arg -> 'Anthropic API returned ...'"""
        err = AnthropicChatError(400, '{"error":"x"}')
        self.assertIn("Anthropic API returned 400", str(err))
        self.assertEqual(err.provider, "Anthropic")

    def test_provider_override_in_message(self):
        err = AnthropicChatError(400, '{"error":"x"}', provider="Groq")
        self.assertIn("Groq API returned 400", str(err))
        self.assertNotIn("Anthropic API", str(err))
        self.assertEqual(err.provider, "Groq")

    def test_chat_api_error_alias(self):
        """ChatAPIError is an alias for the same class (migration path)."""
        self.assertIs(ChatAPIError, AnthropicChatError)


class TestOpenAICompatRaiseSite(unittest.TestCase):
    """When the OpenAI-compat POST 4xx's, the raised error must carry the
    correct provider label (not the misleading 'Anthropic')."""

    def test_groq_400_raises_with_groq_label(self):
        import urllib.error
        import io

        class _FakeHTTPError(urllib.error.HTTPError):
            def __init__(self):
                super().__init__(
                    "https://api.groq.com/openai/v1/chat/completions",
                    400, "Bad Request", {},
                    io.BytesIO(b'{"error":"tools.0.type missing"}'),
                )

        with patch(
            "lib.hal.tool_runtime.urllib.request.urlopen",
            side_effect=_FakeHTTPError(),
        ):
            from lib.hal.tool_runtime import _post_openai_chat
            with self.assertRaises(AnthropicChatError) as cm:
                _post_openai_chat(
                    url="https://api.groq.com/openai/v1/chat/completions",
                    api_key="test",
                    body={"model": "m", "messages": []},
                    timeout_s=5,
                )
        self.assertEqual(cm.exception.provider, "Groq")
        self.assertIn("Groq API returned 400", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
