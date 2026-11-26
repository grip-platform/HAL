"""Tests for ANSI tool-call rendering (W7 integration sprint).

Verifies:
- ansi_render_tool_call formats success + error cases correctly
- colour=False produces plain text (log/CI-safe)
- Truncation of long input values
- Malformed tool_use/tool_result degrade safely
- send_with_tools_to_cli writes rendered lines to the given stream
  and forwards ToolLoopResult intact
"""
import io
import os
import sys
import unittest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.conversation import Conversation
from lib.hal.tool_loop import ToolLoopResult
from lib.hal.tool_render import ansi_render_tool_call, send_with_tools_to_cli


class _StubToolResult:
    def __init__(self, output="", error="", blocked=False, reason=""):
        self.output = output
        self.error = error
        self.blocked = blocked
        self.reason = reason


class _StubDispatcher:
    def __init__(self, handlers):
        self._handlers = handlers

    def execute(self, name, tool_input):
        if name in self._handlers:
            return self._handlers[name](tool_input)
        return _StubToolResult(error=f"unknown: {name}")


def _tool_use(tid, name, inp):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def _text(text):
    return {"type": "text", "text": text}


def _response(content, stop_reason="end_turn"):
    return {"content": content, "stop_reason": stop_reason}


class TestAnsiRenderSuccess(unittest.TestCase):
    def test_plain_text_happy_path(self):
        line = ansi_render_tool_call(
            {"name": "Read", "input": {"file_path": "/tmp/x"}},
            {"is_error": False},
            colour=False,
        )
        self.assertIn("Read", line)
        self.assertIn("file_path=/tmp/x", line)
        self.assertIn("\u25B8", line)  # ▸ success arrow

    def test_coloured_output_contains_ansi_codes(self):
        line = ansi_render_tool_call(
            {"name": "Read", "input": {"file_path": "/tmp/x"}},
            {"is_error": False},
            colour=True,
        )
        self.assertIn("\033[", line)  # ANSI escape present

    def test_no_input_renders_name_only(self):
        line = ansi_render_tool_call(
            {"name": "Grep", "input": {}},
            {"is_error": False},
            colour=False,
        )
        # No trailing args
        self.assertTrue(line.endswith("Grep"))


class TestAnsiRenderError(unittest.TestCase):
    def test_error_arrow_and_reason(self):
        line = ansi_render_tool_call(
            {"name": "Bash", "input": {"command": "rm -rf /"}},
            {"is_error": True, "content": "blocked by guard"},
            colour=False,
        )
        self.assertIn("\u2716", line)  # ✖ error arrow
        self.assertIn("Bash", line)
        self.assertIn("blocked by guard", line)

    def test_error_without_content_still_renders(self):
        line = ansi_render_tool_call(
            {"name": "Read", "input": {}},
            {"is_error": True},
            colour=False,
        )
        self.assertIn("\u2716", line)
        self.assertIn("Read", line)

    def test_multiline_error_content_takes_first_line(self):
        line = ansi_render_tool_call(
            {"name": "Bash", "input": {}},
            {"is_error": True, "content": "line 1\nline 2\nline 3"},
            colour=False,
        )
        self.assertIn("line 1", line)
        self.assertNotIn("line 2", line)


class TestInputFormatting(unittest.TestCase):
    def test_multiple_kwargs_space_separated(self):
        line = ansi_render_tool_call(
            {"name": "Grep", "input": {"pattern": "foo", "path": "/x"}},
            {"is_error": False},
            colour=False,
        )
        self.assertIn("pattern=foo", line)
        self.assertIn("path=/x", line)

    def test_long_value_truncated(self):
        long = "x" * 200
        line = ansi_render_tool_call(
            {"name": "Write", "input": {"content": long}},
            {"is_error": False},
            colour=False,
        )
        self.assertIn("...", line)
        self.assertLess(len(line), 200)

    def test_dict_value_rendered_as_json(self):
        line = ansi_render_tool_call(
            {"name": "X", "input": {"config": {"k": "v"}}},
            {"is_error": False},
            colour=False,
        )
        self.assertIn('{"k":"v"}', line)


class TestMalformedInput(unittest.TestCase):
    def test_missing_name_renders_empty(self):
        line = ansi_render_tool_call(
            {"input": {}},
            {"is_error": False},
            colour=False,
        )
        # Does not raise; name absent but arrow still present
        self.assertIn("\u25B8", line)

    def test_non_dict_tool_use_does_not_raise(self):
        line = ansi_render_tool_call("not a dict", {"is_error": False}, colour=False)
        self.assertIsInstance(line, str)

    def test_non_dict_tool_result_does_not_raise(self):
        line = ansi_render_tool_call(
            {"name": "X", "input": {}}, "not a dict", colour=False,
        )
        self.assertIsInstance(line, str)

    def test_non_dict_input_falls_back_to_repr(self):
        line = ansi_render_tool_call(
            {"name": "X", "input": "raw string"},
            {"is_error": False},
            colour=False,
        )
        self.assertIn("'raw string'", line)


class TestSendWithToolsToCli(unittest.TestCase):
    def _conv(self):
        return Conversation(model="", registry=None)

    def test_tool_calls_rendered_to_provided_stream(self):
        conv = self._conv()
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output="x"),
        })
        responses = [
            _response([_tool_use("t1", "Read", {"file_path": "/x"})], "tool_use"),
            _response([_text("done")]),
        ]
        chat = lambda **kw: responses.pop(0)

        buf = io.StringIO()
        result = send_with_tools_to_cli(
            conv, "go",
            chat_fn=chat,
            tools=[{"name": "Read"}],
            dispatcher=dispatcher,
            stream=buf,
            colour=False,
        )

        output = buf.getvalue()
        self.assertIn("Read", output)
        self.assertIn("file_path=/x", output)
        self.assertIsInstance(result, ToolLoopResult)

    def test_no_tool_calls_writes_nothing_to_stream(self):
        conv = self._conv()
        buf = io.StringIO()
        send_with_tools_to_cli(
            conv, "hi",
            chat_fn=lambda **kw: _response([_text("hello")]),
            tools=[],
            dispatcher=_StubDispatcher({}),
            stream=buf,
            colour=False,
        )
        self.assertEqual(buf.getvalue(), "")

    def test_blocked_call_rendered_with_error_arrow(self):
        conv = self._conv()
        dispatcher = _StubDispatcher({
            "Bash": lambda inp: _StubToolResult(
                blocked=True, reason="destructive cmd",
            ),
        })
        responses = [
            _response([_tool_use("t1", "Bash", {"command": "rm"})], "tool_use"),
            _response([_text("cannot")]),
        ]
        chat = lambda **kw: responses.pop(0)

        buf = io.StringIO()
        send_with_tools_to_cli(
            conv, "go",
            chat_fn=chat,
            tools=[{"name": "Bash"}],
            dispatcher=dispatcher,
            stream=buf,
            colour=False,
        )
        output = buf.getvalue()
        self.assertIn("\u2716", output)  # ✖
        self.assertIn("destructive cmd", output)

    def test_conversation_history_updated_after_call(self):
        conv = self._conv()
        send_with_tools_to_cli(
            conv, "hi",
            chat_fn=lambda **kw: _response([_text("hi back")]),
            tools=[],
            dispatcher=_StubDispatcher({}),
            stream=io.StringIO(),
            colour=False,
        )
        self.assertEqual(len(conv._history), 2)
        self.assertEqual(conv._history[-1].content, "hi back")


if __name__ == "__main__":
    unittest.main()
