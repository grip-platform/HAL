"""Integration test — Issue #200.

Verify that :func:`make_openai_compat_chat_fn` wraps incoming
Anthropic-shape tools into the OpenAI ``{type: "function", function: {...}}``
envelope before POSTing to ``/chat/completions``. Also verifies the
provider-neutral error label derivation.

The test intercepts :func:`lib.hal.tool_runtime._post_openai_chat` and
asserts on the ``body`` dict it receives. This is the exact seam the
Groq/OpenAI request body flows through, so asserting on the body there
is equivalent to asserting on the wire format that ultimately reaches
the provider.
"""
from __future__ import annotations

import importlib.util
import pathlib
import unittest
from unittest.mock import patch

_REPO = pathlib.Path(__file__).parent.parent


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_runtime = _load("lib/hal/tool_runtime.py", "_wt_tool_runtime_post")
_tool_adapter = _load("lib/hal/tool_adapter.py", "_wt_tool_adapter_post")
make_openai_compat_chat_fn = _runtime.make_openai_compat_chat_fn
to_anthropic_tools = _tool_adapter.to_anthropic_tools
_derive_provider_label = _runtime._derive_provider_label
_fix_openai_compat_body = _runtime._fix_openai_compat_body
AnthropicChatError = _runtime.AnthropicChatError


_FAKE_OPENAI_RESPONSE: dict = {
    "id": "chatcmpl-stub",
    "model": "llama-3.3-70b-versatile",
    "choices": [{
        "message": {"role": "assistant", "content": "stub response"},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}


class OpenAiCompatToolsSchemaTests(unittest.TestCase):
    """Issue #200 anchor — tools must be OpenAI-wrapped on POST."""

    def test_groq_receives_type_function_envelope(self):
        """The exact scenario from Issue #200: Groq + HAL's 8 built-ins."""
        captured: dict = {}

        def _fake_post(*, url, api_key, body, timeout_s, extra_headers=None):
            captured["url"] = url
            captured["body"] = body
            return _FAKE_OPENAI_RESPONSE

        with patch.object(_runtime, "_post_openai_chat", side_effect=_fake_post):
            chat_fn = make_openai_compat_chat_fn(
                base_url="https://api.groq.com/openai/v1",
                api_key="sk-stub",
                model="llama-3.3-70b-versatile",
                max_tokens=512,
            )
            chat_fn(messages=[{"role": "user", "content": "hi"}], tools=to_anthropic_tools())

        body = captured["body"]
        self.assertIn("tools", body, "tools must be in body when non-empty")
        self.assertEqual(len(body["tools"]), 9)  # 8 core + osa_run (PR #226)
        for entry in body["tools"]:
            # The bug this test protects against: missing "type" on a tool entry.
            self.assertEqual(entry["type"], "function",
                             "every tool MUST have type=function for OpenAI-compat")
            self.assertIn("function", entry)
            self.assertIn("name", entry["function"])
            self.assertIn("description", entry["function"])
            self.assertIn("parameters", entry["function"])
            # Anthropic-specific keys must NOT be at the top level.
            self.assertNotIn("input_schema", entry)

    def test_tool_choice_auto_added_alongside_tools(self):
        captured: dict = {}

        def _fake_post(*, url, api_key, body, timeout_s, extra_headers=None):
            captured["body"] = body
            return _FAKE_OPENAI_RESPONSE

        with patch.object(_runtime, "_post_openai_chat", side_effect=_fake_post):
            chat_fn = make_openai_compat_chat_fn(
                base_url="https://api.openai.com/v1",
                api_key="sk-stub",
                model="gpt-4",
            )
            chat_fn(messages=[{"role": "user", "content": "hi"}], tools=to_anthropic_tools())

        self.assertEqual(captured["body"].get("tool_choice"), "auto")

    def test_no_tools_omits_tools_key(self):
        """Empty tools list must NOT set body['tools'] — minimal request."""
        captured: dict = {}

        def _fake_post(*, url, api_key, body, timeout_s, extra_headers=None):
            captured["body"] = body
            return _FAKE_OPENAI_RESPONSE

        with patch.object(_runtime, "_post_openai_chat", side_effect=_fake_post):
            chat_fn = make_openai_compat_chat_fn(
                base_url="https://api.groq.com/openai/v1",
                api_key="sk-stub",
                model="any-model",
            )
            chat_fn(messages=[{"role": "user", "content": "hi"}], tools=[])

        self.assertNotIn("tools", captured["body"])
        self.assertNotIn("tool_choice", captured["body"])

    def test_gpt5_remaps_max_tokens_for_happi_path(self):
        body = {"model": "gpt-5.5", "messages": [], "max_tokens": 123}
        _fix_openai_compat_body(body)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["max_completion_tokens"], 123)

    def test_gpt4_keeps_max_tokens_for_happi_path(self):
        body = {"model": "gpt-4o", "messages": [], "max_tokens": 123}
        _fix_openai_compat_body(body)
        self.assertEqual(body["max_tokens"], 123)
        self.assertNotIn("max_completion_tokens", body)

    def test_gpt5_factory_posts_max_completion_tokens(self):
        captured: dict = {}

        def _fake_post(*, url, api_key, body, timeout_s, extra_headers=None):
            captured["body"] = body
            return _FAKE_OPENAI_RESPONSE

        with patch.object(_runtime, "_post_openai_chat", side_effect=_fake_post):
            chat_fn = make_openai_compat_chat_fn(
                base_url="https://api.openai.com/v1",
                api_key="sk-stub",
                model="gpt-5.5",
                max_tokens=321,
            )
            chat_fn(messages=[{"role": "user", "content": "hi"}], tools=[])

        self.assertNotIn("max_tokens", captured["body"])
        self.assertEqual(captured["body"]["max_completion_tokens"], 321)


class ProviderLabelTests(unittest.TestCase):
    """The derived provider label replaces the misleading "Anthropic" default."""

    def test_groq_url_yields_groq_label(self):
        self.assertEqual(_derive_provider_label("https://api.groq.com/openai/v1"), "Groq")

    def test_openai_url_yields_openai_label(self):
        self.assertEqual(_derive_provider_label("https://api.openai.com/v1"), "OpenAI")

    def test_gemini_url_yields_gemini_label(self):
        self.assertEqual(
            _derive_provider_label("https://generativelanguage.googleapis.com/v1beta/openai"),
            "Gemini",
        )

    def test_localhost_yields_ollama_label(self):
        self.assertEqual(_derive_provider_label("http://localhost:11434/v1"), "Ollama")
        self.assertEqual(_derive_provider_label("http://127.0.0.1:11434/v1"), "Ollama")

    def test_anthropic_url_yields_anthropic_label(self):
        self.assertEqual(_derive_provider_label("https://api.anthropic.com/v1/messages"), "Anthropic")

    def test_unknown_yields_generic_label(self):
        self.assertEqual(_derive_provider_label("https://custom.example.com/v1"), "Provider")


class ChatTransportErrorAliasTests(unittest.TestCase):
    """The new alias points at the same class — isinstance checks keep working."""

    def test_chat_transport_error_is_alias_of_anthropic_chat_error(self):
        self.assertIs(_runtime.ChatTransportError, AnthropicChatError)

    def test_error_carries_provider_attribute(self):
        err = AnthropicChatError(400, "body", provider="Groq")
        self.assertEqual(err.provider, "Groq")
        self.assertEqual(err.status, 400)
        # Default message uses the provider label.
        self.assertIn("Groq", str(err))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
