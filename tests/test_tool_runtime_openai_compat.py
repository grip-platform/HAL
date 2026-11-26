"""Tests for the HAL OpenAI-compat chat_fn factory (W1).

Verifies:
- ``make_openai_compat_chat_fn`` returns a closure with the
  ``(*, messages, tools) -> dict`` contract matching
  :func:`make_anthropic_chat_fn`, so it is a drop-in replacement from
  :func:`run_tool_loop`'s perspective.
- Request body carries ``model``, ``messages``, ``max_tokens``,
  ``temperature`` and only includes ``tools`` + ``tool_choice`` when a
  non-empty list is supplied.
- ``base_url`` is post-fixed with ``/chat/completions`` regardless of
  trailing slash.
- Authorisation header is ``Bearer <api_key>`` and any extra headers are
  merged verbatim.
- HTTP errors propagate as :class:`AnthropicChatError` (reused across
  providers so the loop has one exception type to catch).
- The returned dict is already normalised to Anthropic shape — the
  loop must not need to know which provider is behind the closure.

Patches :func:`lib.hal.tool_runtime._post_openai_chat` at a single seam.
"""
import os
import sys
import unittest
from unittest import mock

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.tool_runtime import (  # noqa: E402
    AnthropicChatError,
    make_openai_compat_chat_fn,
)


def _openai_text_response(text: str = "hi", model: str = "m") -> dict:
    return {
        "id": "chatcmpl-1",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }


class TestFactoryContract(unittest.TestCase):
    def test_returns_callable_with_messages_and_tools_kwargs(self):
        chat_fn = make_openai_compat_chat_fn(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="gemma4:e4b",
        )
        self.assertTrue(callable(chat_fn))

    def test_url_appends_chat_completions(self):
        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            return_value=_openai_text_response(),
        ) as post:
            chat_fn = make_openai_compat_chat_fn(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                model="gemma4:e4b",
            )
            chat_fn(messages=[{"role": "user", "content": "hi"}], tools=[])
            self.assertEqual(
                post.call_args.kwargs["url"],
                "http://localhost:11434/v1/chat/completions",
            )

    def test_trailing_slash_on_base_url_normalised(self):
        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            return_value=_openai_text_response(),
        ) as post:
            chat_fn = make_openai_compat_chat_fn(
                base_url="http://localhost:11434/v1/",
                api_key="ollama",
                model="gemma4:e4b",
            )
            chat_fn(messages=[], tools=[])
            self.assertEqual(
                post.call_args.kwargs["url"],
                "http://localhost:11434/v1/chat/completions",
            )


class TestBodyShape(unittest.TestCase):
    def test_body_has_model_messages_max_tokens_temperature(self):
        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            return_value=_openai_text_response(),
        ) as post:
            chat_fn = make_openai_compat_chat_fn(
                base_url="http://x/v1", api_key="k",
                model="gemma4:e4b", max_tokens=2048, temperature=0.3,
            )
            msgs = [{"role": "user", "content": "hi"}]
            chat_fn(messages=msgs, tools=[])
            body = post.call_args.kwargs["body"]
            self.assertEqual(body["model"], "gemma4:e4b")
            self.assertEqual(body["messages"], msgs)
            self.assertEqual(body["max_tokens"], 2048)
            self.assertEqual(body["temperature"], 0.3)

    def test_tools_omitted_when_empty(self):
        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            return_value=_openai_text_response(),
        ) as post:
            chat_fn = make_openai_compat_chat_fn(
                base_url="http://x/v1", api_key="k", model="m",
            )
            chat_fn(messages=[], tools=[])
            body = post.call_args.kwargs["body"]
            self.assertNotIn("tools", body)
            self.assertNotIn("tool_choice", body)

    def test_tools_and_tool_choice_included_when_non_empty(self):
        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            return_value=_openai_text_response(),
        ) as post:
            chat_fn = make_openai_compat_chat_fn(
                base_url="http://x/v1", api_key="k", model="m",
            )
            tools = [{"type": "function", "function": {"name": "Read"}}]
            chat_fn(messages=[], tools=tools)
            body = post.call_args.kwargs["body"]
            self.assertEqual(body["tools"], tools)
            self.assertEqual(body["tool_choice"], "auto")


class TestHeaders(unittest.TestCase):
    def test_api_key_forwarded(self):
        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            return_value=_openai_text_response(),
        ) as post:
            chat_fn = make_openai_compat_chat_fn(
                base_url="http://x/v1", api_key="gsk_test", model="m",
            )
            chat_fn(messages=[], tools=[])
            self.assertEqual(post.call_args.kwargs["api_key"], "gsk_test")

    def test_extra_headers_forwarded_verbatim(self):
        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            return_value=_openai_text_response(),
        ) as post:
            chat_fn = make_openai_compat_chat_fn(
                base_url="http://x/v1", api_key="k", model="m",
                extra_headers={"x-ollama-keep-alive": "30m"},
            )
            chat_fn(messages=[], tools=[])
            self.assertEqual(
                post.call_args.kwargs["extra_headers"],
                {"x-ollama-keep-alive": "30m"},
            )


class TestResponseNormalisation(unittest.TestCase):
    def test_returned_dict_is_anthropic_shape_text(self):
        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            return_value=_openai_text_response(text="hello"),
        ):
            chat_fn = make_openai_compat_chat_fn(
                base_url="http://x/v1", api_key="k", model="m",
            )
            result = chat_fn(messages=[], tools=[])
        self.assertIsInstance(result["content"], list)
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertEqual(result["content"][0]["text"], "hello")
        self.assertEqual(result["usage"]["input_tokens"], 3)
        self.assertEqual(result["usage"]["output_tokens"], 1)


class TestErrorPropagation(unittest.TestCase):
    def test_http_error_bubbles_as_anthropic_chat_error(self):
        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            side_effect=AnthropicChatError(429, "rate limited"),
        ):
            chat_fn = make_openai_compat_chat_fn(
                base_url="http://x/v1", api_key="k", model="m",
            )
            with self.assertRaises(AnthropicChatError) as ctx:
                chat_fn(messages=[], tools=[])
        self.assertEqual(ctx.exception.status, 429)


def _responses_reply(model: str = "gpt-6-astra") -> dict:
    """A Responses-wire object carrying one text part and one function call."""
    return {
        "id": "resp_1",
        "model": model,
        "status": "completed",
        "output": [
            {"type": "message",
             "content": [{"type": "output_text", "text": "checking"}]},
            {"type": "function_call", "call_id": "call_1", "name": "Read",
             "arguments": "{\"path\": \"/tmp/x\"}"},
        ],
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }


_ANTHROPIC_TOOLS = [{
    "name": "Read", "description": "read a file",
    "input_schema": {"type": "object", "properties": {}},
}]


class TestResponsesWire(unittest.TestCase):
    """gpt-6 family (Astra) must take /v1/responses, never /chat/completions.

    Vendor: "tool calling requires the Responses API" — tools on
    /chat/completions answer HTTP 400 (measured live 2026-09-10), and the
    HAL chat/REPL always attaches its tools, so the wire is chosen per model
    family via model_quirks.tools_require_responses_api, mirroring
    gateway_egress's early return. Anchors the 2026-09-13 gap where the
    direct chat path posted gpt-6-astra to /chat/completions anyway.
    """

    def _fn(self, model="gpt-6-astra"):
        return make_openai_compat_chat_fn(
            base_url="http://x/v1", api_key="k", model=model,
        )

    def test_gpt6_posts_to_responses_url(self):
        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            return_value=_responses_reply(),
        ) as post:
            self._fn()(messages=[{"role": "user", "content": "hi"}],
                       tools=list(_ANTHROPIC_TOOLS))
            self.assertEqual(post.call_args.kwargs["url"],
                             "http://x/v1/responses")

    def test_gpt6_tool_free_turn_also_takes_responses_wire(self):
        # Family-wide choice, mirroring the gateway: one predicate, one wire.
        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            return_value=_responses_reply(),
        ) as post:
            self._fn()(messages=[{"role": "user", "content": "hi"}], tools=[])
            self.assertEqual(post.call_args.kwargs["url"],
                             "http://x/v1/responses")
            self.assertNotIn("tools", post.call_args.kwargs["body"])

    def test_gpt6_request_is_responses_shape_without_locked_fields(self):
        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            return_value=_responses_reply(),
        ) as post:
            self._fn()(messages=[{"role": "user", "content": "hi"}],
                       tools=list(_ANTHROPIC_TOOLS))
            body = post.call_args.kwargs["body"]
            self.assertIn("input", body)
            self.assertNotIn("messages", body)
            # gpt-6 locks temperature/top_p: omitted, never forwarded.
            self.assertNotIn("temperature", body)
            self.assertNotIn("top_p", body)
            self.assertIs(body["stream"], False)
            self.assertIs(body["store"], False)
            # Flat Responses function tool (name/parameters at top level).
            self.assertEqual(body["tools"][0]["name"], "Read")
            self.assertIn("parameters", body["tools"][0])
            self.assertEqual(body["tool_choice"], "auto")

    def test_gpt6_response_translated_back_to_anthropic_shape(self):
        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            return_value=_responses_reply(),
        ):
            result = self._fn()(messages=[{"role": "user", "content": "hi"}],
                                tools=list(_ANTHROPIC_TOOLS))
        self.assertEqual(result["content"][0],
                         {"type": "text", "text": "checking"})
        call = result["content"][1]
        self.assertEqual(call["type"], "tool_use")
        self.assertEqual(call["id"], "call_1")
        self.assertEqual(call["name"], "Read")
        self.assertEqual(call["input"], {"path": "/tmp/x"})
        self.assertEqual(result["stop_reason"], "tool_use")
        self.assertEqual(result["usage"]["input_tokens"], 5)
        self.assertEqual(result["usage"]["output_tokens"], 2)

    def test_non_gpt6_model_stays_on_chat_completions(self):
        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            return_value=_openai_text_response(),
        ) as post:
            self._fn(model="gpt-5.4")(
                messages=[{"role": "user", "content": "hi"}],
                tools=list(_ANTHROPIC_TOOLS))
            self.assertEqual(post.call_args.kwargs["url"],
                             "http://x/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
