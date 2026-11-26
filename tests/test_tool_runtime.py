"""Tests for the HAL Anthropic chat_fn factory + extractor (W3).

Verifies:
- ``make_anthropic_chat_fn`` returns a closure that posts to the Messages
  API with the expected shape (model, max_tokens, messages, optional
  system, tools only when non-empty, extra_headers forwarded)
- The closure returns the parsed response dict (as produced by
  :func:`_post_messages`) unchanged, so downstream callers can feed it
  straight into :func:`parse_tool_use_blocks` and :func:`run_tool_loop`
- Non-2xx responses raise :class:`AnthropicChatError` carrying the
  status code and body so the caller can render a useful message
- ``extract_final_text`` concatenates only text blocks and degrades
  safely on malformed input

Tests patch :func:`lib.hal.tool_runtime._post_messages` directly rather
than mocking urllib — a single seam per file keeps the tests resilient
to stdlib internals and free of network timing assumptions.
"""
import os
import sys
import unittest
from unittest import mock

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.tool_runtime import (
    AnthropicChatError,
    _recover_llama_textcall,
    _tool_call_to_block,
    extract_final_text,
    make_anthropic_chat_fn,
)


class TestChatFnBody(unittest.TestCase):
    def test_body_includes_model_messages_and_max_tokens(self):
        with mock.patch("lib.hal.tool_runtime._post_messages") as post:
            post.return_value = {"content": [], "stop_reason": "end_turn"}
            chat_fn = make_anthropic_chat_fn(
                api_key="sk-ant-test", model="claude-opus-4-7", max_tokens=1024,
            )
            chat_fn(messages=[{"role": "user", "content": "hi"}], tools=[])
            body = post.call_args.kwargs["body"]
            self.assertEqual(body["model"], "claude-opus-4-7")
            self.assertEqual(body["max_tokens"], 1024)
            self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])

    def test_tools_only_included_when_non_empty(self):
        with mock.patch("lib.hal.tool_runtime._post_messages") as post:
            post.return_value = {"content": [], "stop_reason": "end_turn"}
            chat_fn = make_anthropic_chat_fn(api_key="k", model="m")
            chat_fn(messages=[], tools=[])
            self.assertNotIn("tools", post.call_args.kwargs["body"])

    def test_tools_forwarded_when_non_empty(self):
        with mock.patch("lib.hal.tool_runtime._post_messages") as post:
            post.return_value = {"content": [], "stop_reason": "end_turn"}
            chat_fn = make_anthropic_chat_fn(api_key="k", model="m")
            tools = [{"name": "Read", "description": "", "input_schema": {}}]
            chat_fn(messages=[], tools=tools)
            self.assertEqual(post.call_args.kwargs["body"]["tools"], tools)

    def test_system_only_included_when_set(self):
        with mock.patch("lib.hal.tool_runtime._post_messages") as post:
            post.return_value = {"content": [], "stop_reason": "end_turn"}
            make_anthropic_chat_fn(api_key="k", model="m")(messages=[], tools=[])
            self.assertNotIn("system", post.call_args.kwargs["body"])

    def test_system_forwarded_when_provided(self):
        with mock.patch("lib.hal.tool_runtime._post_messages") as post:
            post.return_value = {"content": [], "stop_reason": "end_turn"}
            make_anthropic_chat_fn(api_key="k", model="m", system="You are HAL.")(
                messages=[], tools=[],
            )
            self.assertEqual(
                post.call_args.kwargs["body"]["system"], "You are HAL.",
            )


class TestChatFnHeaders(unittest.TestCase):
    def test_api_key_and_version_passed(self):
        with mock.patch("lib.hal.tool_runtime._post_messages") as post:
            post.return_value = {"content": [], "stop_reason": "end_turn"}
            make_anthropic_chat_fn(
                api_key="sk-ant-xyz", model="m", api_version="2099-01-01",
            )(messages=[], tools=[])
            self.assertEqual(post.call_args.kwargs["api_key"], "sk-ant-xyz")
            self.assertEqual(post.call_args.kwargs["api_version"], "2099-01-01")

    def test_extra_headers_forwarded_verbatim(self):
        with mock.patch("lib.hal.tool_runtime._post_messages") as post:
            post.return_value = {"content": [], "stop_reason": "end_turn"}
            make_anthropic_chat_fn(
                api_key="k", model="m",
                extra_headers={"anthropic-beta": "tools-2024-04-04"},
            )(messages=[], tools=[])
            self.assertEqual(
                post.call_args.kwargs["extra_headers"],
                {"anthropic-beta": "tools-2024-04-04"},
            )


class TestChatFnReturnValue(unittest.TestCase):
    def test_closure_returns_post_messages_result_unchanged(self):
        fake_response = {
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        with mock.patch("lib.hal.tool_runtime._post_messages",
                        return_value=fake_response):
            result = make_anthropic_chat_fn(api_key="k", model="m")(
                messages=[], tools=[],
            )
        self.assertIs(result, fake_response)


class TestErrorHandling(unittest.TestCase):
    def test_http_error_propagates_as_anthropic_chat_error(self):
        with mock.patch(
            "lib.hal.tool_runtime._post_messages",
            side_effect=AnthropicChatError(400, "bad request"),
        ):
            chat_fn = make_anthropic_chat_fn(api_key="k", model="m")
            with self.assertRaises(AnthropicChatError) as ctx:
                chat_fn(messages=[], tools=[])
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.body, "bad request")

    def test_error_message_truncates_long_body(self):
        long_body = "x" * 500
        err = AnthropicChatError(500, long_body)
        self.assertLess(len(str(err)), 300)


class TestExtractFinalText(unittest.TestCase):
    def test_single_text_block(self):
        response = {"content": [{"type": "text", "text": "hello"}]}
        self.assertEqual(extract_final_text(response), "hello")

    def test_multiple_text_blocks_concatenated(self):
        response = {
            "content": [
                {"type": "text", "text": "foo "},
                {"type": "text", "text": "bar"},
            ]
        }
        self.assertEqual(extract_final_text(response), "foo bar")

    def test_tool_use_blocks_ignored(self):
        response = {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
                {"type": "text", "text": "done"},
            ]
        }
        self.assertEqual(extract_final_text(response), "done")

    def test_thinking_blocks_ignored(self):
        response = {
            "content": [
                {"type": "thinking", "thinking": "reasoning..."},
                {"type": "text", "text": "answer"},
            ]
        }
        self.assertEqual(extract_final_text(response), "answer")

    def test_unknown_block_types_ignored(self):
        response = {"content": [{"type": "alien", "payload": "x"}]}
        self.assertEqual(extract_final_text(response), "")

    def test_non_dict_input_returns_empty_string(self):
        self.assertEqual(extract_final_text(None), "")
        self.assertEqual(extract_final_text("text"), "")
        self.assertEqual(extract_final_text([]), "")

    def test_missing_content_returns_empty_string(self):
        self.assertEqual(extract_final_text({}), "")

    def test_non_list_content_returns_empty_string(self):
        self.assertEqual(extract_final_text({"content": "nope"}), "")

    def test_text_field_missing_defaults_to_empty(self):
        response = {"content": [{"type": "text"}]}
        self.assertEqual(extract_final_text(response), "")

    def test_non_string_text_field_skipped(self):
        response = {"content": [{"type": "text", "text": 42}]}
        self.assertEqual(extract_final_text(response), "")

    def test_non_dict_block_entries_skipped(self):
        response = {
            "content": [
                "string",
                {"type": "text", "text": "ok"},
                42,
            ]
        }
        self.assertEqual(extract_final_text(response), "ok")


class TestIntegrationWithToolLoop(unittest.TestCase):
    """End-to-end: make_anthropic_chat_fn + run_tool_loop sharing one mocked transport."""

    def test_round_trip_with_stub_dispatcher(self):
        from lib.hal.tool_loop import run_tool_loop

        class _StubToolResult:
            def __init__(self, output="", error="", blocked=False, reason=""):
                self.output = output
                self.error = error
                self.blocked = blocked
                self.reason = reason

        class _StubDispatcher:
            def execute(self, name, tool_input):
                return _StubToolResult(output=f"{name}:{tool_input.get('x', '')}")

        responses = [
            {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read",
                     "input": {"x": "42"}},
                ],
                "stop_reason": "tool_use",
            },
            {
                "content": [{"type": "text", "text": "final"}],
                "stop_reason": "end_turn",
            },
        ]

        with mock.patch(
            "lib.hal.tool_runtime._post_messages",
            side_effect=lambda **kw: responses.pop(0),
        ):
            chat_fn = make_anthropic_chat_fn(api_key="k", model="m")
            result = run_tool_loop(
                messages=[{"role": "user", "content": "go"}],
                chat_fn=chat_fn,
                tools=[{"name": "Read", "description": "", "input_schema": {}}],
                dispatcher=_StubDispatcher(),
            )

        self.assertEqual(result.iterations, 2)
        self.assertFalse(result.hit_iteration_cap)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["tool_result"]["content"], "Read:42")
        self.assertEqual(extract_final_text(result.final_response), "final")


class TestRecoverLlamaTextcall(unittest.TestCase):
    """Unit tests for _recover_llama_textcall."""

    def test_clean_name_returns_unchanged(self):
        self.assertEqual(_recover_llama_textcall("Bash"), ("Bash", ""))

    def test_non_string_returns_unchanged(self):
        self.assertEqual(_recover_llama_textcall(None), (None, ""))

    def test_embedded_json_splits_correctly(self):
        name, args = _recover_llama_textcall('Bash{"command": "ls"}')
        self.assertEqual(name, "Bash")
        self.assertEqual(args, '{"command": "ls"}')

    def test_recovered_args_are_valid_json(self):
        import json
        _, args = _recover_llama_textcall('Read{"path": "/tmp/x"}')
        self.assertEqual(json.loads(args), {"path": "/tmp/x"})

    def test_empty_candidate_name_returns_original(self):
        # name starts with '{' — no valid tool name before brace
        original = '{"command": "ls"}'
        self.assertEqual(_recover_llama_textcall(original), (original, ""))

    def test_args_not_closing_brace_returns_original(self):
        original = 'Bash{"command": "ls"'  # truncated — no closing brace
        self.assertEqual(_recover_llama_textcall(original), (original, ""))

    def test_invalid_json_in_args_returns_original(self):
        original = 'Bash{not valid json}'
        self.assertEqual(_recover_llama_textcall(original), (original, ""))

    def test_no_brace_returns_unchanged(self):
        self.assertEqual(_recover_llama_textcall("Write"), ("Write", ""))


class TestToolCallToBlockLlamaRecovery(unittest.TestCase):
    """Integration tests: _tool_call_to_block with LLaMA text-mode input."""

    def test_normal_call_unchanged(self):
        call = {"id": "c1", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}
        block = _tool_call_to_block(call)
        self.assertEqual(block["name"], "Bash")
        self.assertEqual(block["input"], {"command": "ls"})
        self.assertNotIn("_parse_error", block)

    def test_llama_textmode_call_recovered(self):
        # LLaMA 3.3 70B text-mode: name carries embedded JSON, arguments empty
        call = {"id": "c2", "function": {"name": 'Bash{"command": "echo hi"}', "arguments": ""}}
        block = _tool_call_to_block(call)
        self.assertEqual(block["name"], "Bash")
        self.assertEqual(block["input"], {"command": "echo hi"})
        self.assertNotIn("_parse_error", block)

    def test_llama_recovery_only_fires_when_arguments_empty(self):
        # arguments is non-empty — recovery must NOT fire even if name has '{'
        call = {
            "id": "c3",
            "function": {
                "name": 'WeirdName{"key": "ignored"}',
                "arguments": '{"real": "args"}',
            },
        }
        block = _tool_call_to_block(call)
        # name is kept verbatim because arguments was present
        self.assertIn("{", block["name"])
        self.assertEqual(block["input"], {"real": "args"})

    def test_non_dict_call_returns_none(self):
        self.assertIsNone(_tool_call_to_block("not a dict"))

    def test_missing_function_key_produces_empty_block(self):
        block = _tool_call_to_block({"id": "c4"})
        self.assertEqual(block["name"], "")
        self.assertEqual(block["input"], {})


if __name__ == "__main__":
    unittest.main()
