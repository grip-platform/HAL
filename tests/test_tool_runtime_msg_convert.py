"""Tests for _convert_messages_for_openai — Anthropic → OpenAI message shaping.

Groq and Ollama return HTTP 400 when run_tool_loop appends Anthropic-shaped
assistant messages (content as a list of blocks) and tool-result user messages
to the history, then sends that history back unchanged on the next request.

This module tests the converter that re-shapes those multi-turn histories
so OpenAI-compat endpoints receive their native format.
"""
import json
import os
import sys
import unittest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.tool_runtime import _convert_messages_for_openai


class TestPassThrough(unittest.TestCase):
    def test_plain_string_user_message_unchanged(self):
        msgs = [{"role": "user", "content": "hello"}]
        self.assertEqual(_convert_messages_for_openai(msgs), msgs)

    def test_plain_string_assistant_unchanged(self):
        msgs = [{"role": "assistant", "content": "hi there"}]
        result = _convert_messages_for_openai(msgs)
        self.assertEqual(result[0]["content"], "hi there")

    def test_empty_list_returns_empty(self):
        self.assertEqual(_convert_messages_for_openai([]), [])

    def test_none_content_normalised_to_empty_string(self):
        result = _convert_messages_for_openai([{"role": "user", "content": None}])
        self.assertEqual(result[0]["content"], "")


class TestAssistantToolUse(unittest.TestCase):
    def _tool_use_msg(self, tool_id="t1", name="Write", inp=None):
        return {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tool_id, "name": name,
                 "input": inp or {"file_path": "/tmp/x", "content": "hello"}},
            ],
        }

    def test_tool_use_becomes_tool_calls(self):
        result = _convert_messages_for_openai([self._tool_use_msg()])
        calls = result[0]["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "t1")
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "Write")

    def test_arguments_are_json_string(self):
        inp = {"file_path": "/tmp/x", "content": "hello"}
        result = _convert_messages_for_openai([self._tool_use_msg(inp=inp)])
        args = result[0]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(args, str)
        self.assertEqual(json.loads(args), inp)

    def test_content_is_null_when_no_text(self):
        result = _convert_messages_for_openai([self._tool_use_msg()])
        self.assertIsNone(result[0]["content"])

    def test_text_and_tool_use_both_preserved(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me write that."},
                {"type": "tool_use", "id": "t1", "name": "Write",
                 "input": {"file_path": "x.txt", "content": "y"}},
            ],
        }
        result = _convert_messages_for_openai([msg])
        self.assertEqual(result[0]["content"], "Let me write that.")
        self.assertEqual(len(result[0]["tool_calls"]), 1)

    def test_multiple_tool_uses_in_one_turn(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Read",
                 "input": {"file_path": "a.txt"}},
                {"type": "tool_use", "id": "t2", "name": "Grep",
                 "input": {"pattern": "TODO"}},
            ],
        }
        result = _convert_messages_for_openai([msg])
        names = [c["function"]["name"] for c in result[0]["tool_calls"]]
        self.assertEqual(names, ["Read", "Grep"])


class TestToolResults(unittest.TestCase):
    def test_tool_result_becomes_role_tool(self):
        msg = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
        ]}
        result = _convert_messages_for_openai([msg])
        self.assertEqual(result[0]["role"], "tool")
        self.assertEqual(result[0]["content"], "ok")
        self.assertEqual(result[0]["tool_call_id"], "t1")

    def test_multiple_results_expand_to_separate_messages(self):
        msg = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "r2"},
        ]}
        result = _convert_messages_for_openai([msg])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["tool_call_id"], "t1")
        self.assertEqual(result[1]["tool_call_id"], "t2")

    def test_error_result_content_preserved(self):
        msg = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": "permission denied", "is_error": True},
        ]}
        result = _convert_messages_for_openai([msg])
        self.assertEqual(result[0]["content"], "permission denied")


class TestFullRoundTrip(unittest.TestCase):
    def test_groq_rejection_shape_no_longer_produced(self):
        """Groq rejected messages[3].content as non-string (Issue #285).

        After conversion the assistant turn has content=None (or string),
        never a list of blocks.
        """
        messages = [
            {"role": "user", "content": "output a poem to a file"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Write",
                 "input": {"file_path": "poem.md", "content": "roses..."}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
            ]},
        ]
        result = _convert_messages_for_openai(messages)
        # Assistant content must be str | None, not a list
        self.assertNotIsInstance(result[1]["content"], list)
        self.assertIn("tool_calls", result[1])

    def test_full_three_turn_shape(self):
        messages = [
            {"role": "user", "content": "write hello to /tmp/x.txt"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Writing now."},
                {"type": "tool_use", "id": "toolu_abc", "name": "Write",
                 "input": {"file_path": "/tmp/x.txt", "content": "hello"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_abc",
                 "content": "wrote 5 bytes"},
            ]},
        ]
        result = _convert_messages_for_openai(messages)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[1]["role"], "assistant")
        self.assertEqual(result[1]["tool_calls"][0]["id"], "toolu_abc")
        self.assertEqual(result[2]["role"], "tool")
        self.assertEqual(result[2]["tool_call_id"], "toolu_abc")


if __name__ == "__main__":
    unittest.main()
