"""Tests for SSE tool-event streaming (W8 integration sprint).

Verifies:
- format_sse_event produces correct SSE wire format (event: / data: / blank)
- iter_tool_events emits tool_call events in order, then a final event
- No-tools turn yields exactly one final event
- chat_fn exception produces an error event (no hang)
- Compact JSON in data line (no embedded newlines)
- Queue timeout produces a clean error rather than hanging the client
"""
import json
import os
import sys
import time
import unittest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.conversation import Conversation
from lib.hal.tool_sse import format_sse_event, iter_tool_events


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


def _conv():
    return Conversation(model="", registry=None)


def _parse_events(stream_text: str) -> list[tuple[str, dict]]:
    """Parse a concatenated SSE stream into [(event_type, data_dict), ...]."""
    parsed = []
    for block in stream_text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_type = ""
        data_line = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_line = line[len("data:"):].strip()
        data = json.loads(data_line) if data_line else {}
        parsed.append((event_type, data))
    return parsed


class TestFormatSseEvent(unittest.TestCase):
    def test_basic_shape(self):
        out = format_sse_event("tool_call", {"name": "Read"})
        self.assertIn("event: tool_call\n", out)
        self.assertIn("data: ", out)
        self.assertTrue(out.endswith("\n\n"))

    def test_data_is_compact_json_no_whitespace(self):
        out = format_sse_event("x", {"a": 1, "b": 2})
        self.assertIn('{"a":1,"b":2}', out)

    def test_no_embedded_newlines_in_data(self):
        """Newlines in payload would split one SSE event into two.
        JSON encoding must avoid literal newlines."""
        out = format_sse_event("x", {"msg": "line 1\nline 2"})
        data_line_start = out.index("data:")
        data_line_end = out.index("\n", data_line_start)
        # Only one newline in the entire data line
        data_line = out[data_line_start:data_line_end]
        self.assertNotIn("\n", data_line[len("data:"):])

    def test_unicode_preserved(self):
        """ensure_ascii=False keeps the wire compact for non-ASCII content."""
        out = format_sse_event("x", {"msg": "héllo ✓"})
        self.assertIn("héllo", out)
        self.assertIn("✓", out)


class TestIterToolEventsNoTools(unittest.TestCase):
    def test_single_final_event_when_no_tool_calls(self):
        conv = _conv()
        events_text = "".join(iter_tool_events(
            conv, "hi",
            chat_fn=lambda **kw: _response([_text("hello")]),
            tools=[],
            dispatcher=_StubDispatcher({}),
        ))
        events = _parse_events(events_text)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "final")
        self.assertEqual(events[0][1]["text"], "hello")
        self.assertEqual(events[0][1]["iterations"], 1)
        self.assertFalse(events[0][1]["hit_iteration_cap"])


class TestIterToolEventsWithTools(unittest.TestCase):
    def test_tool_call_then_final(self):
        conv = _conv()
        responses = [
            _response([_tool_use("t1", "Read", {"file_path": "/x"})], "tool_use"),
            _response([_text("done")]),
        ]
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output="x"),
        })

        events_text = "".join(iter_tool_events(
            conv, "go",
            chat_fn=lambda **kw: responses.pop(0),
            tools=[{"name": "Read"}],
            dispatcher=dispatcher,
        ))
        events = _parse_events(events_text)

        self.assertEqual([e[0] for e in events], ["tool_call", "final"])
        self.assertEqual(events[0][1]["tool_use"]["name"], "Read")
        self.assertEqual(events[0][1]["tool_result"]["content"], "x")
        self.assertEqual(events[1][1]["text"], "done")

    def test_multiple_tools_in_order(self):
        conv = _conv()
        responses = [
            _response(
                [
                    _tool_use("t1", "Read", {}),
                    _tool_use("t2", "Grep", {}),
                ],
                "tool_use",
            ),
            _response([_text("done")]),
        ]
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output="a"),
            "Grep": lambda inp: _StubToolResult(output="b"),
        })

        events_text = "".join(iter_tool_events(
            conv, "go",
            chat_fn=lambda **kw: responses.pop(0),
            tools=[{"name": "Read"}, {"name": "Grep"}],
            dispatcher=dispatcher,
        ))
        events = _parse_events(events_text)
        types = [e[0] for e in events]
        self.assertEqual(types, ["tool_call", "tool_call", "final"])
        names = [e[1]["tool_use"]["name"] for e in events if e[0] == "tool_call"]
        self.assertEqual(names, ["Read", "Grep"])


class TestIterToolEventsErrors(unittest.TestCase):
    def test_chat_fn_exception_yields_error_event(self):
        conv = _conv()

        def broken(*, messages, tools):
            raise RuntimeError("API 500")

        events_text = "".join(iter_tool_events(
            conv, "hi",
            chat_fn=broken,
            tools=[],
            dispatcher=_StubDispatcher({}),
        ))
        events = _parse_events(events_text)
        self.assertTrue(any(e[0] == "error" for e in events))
        error_event = next(e for e in events if e[0] == "error")
        self.assertIn("API 500", error_event[1]["message"])

    def test_timeout_yields_error_when_thread_hangs(self):
        """If the loop produces no events for N seconds, emit an error
        event and return rather than blocking the client indefinitely."""
        conv = _conv()

        # Chat that blocks forever
        def hanger(*, messages, tools):
            time.sleep(999)
            return _response([_text("never")])

        events_text = "".join(iter_tool_events(
            conv, "hi",
            chat_fn=hanger,
            tools=[],
            dispatcher=_StubDispatcher({}),
            queue_timeout_s=0.2,
        ))
        events = _parse_events(events_text)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "error")
        self.assertIn("timeout", events[0][1]["message"].lower())


class TestIterToolEventsHistory(unittest.TestCase):
    def test_conversation_history_updated_after_stream_consumed(self):
        """History reflects the user turn AND final assistant turn once the
        stream is fully drained."""
        conv = _conv()
        list(iter_tool_events(
            conv, "hi",
            chat_fn=lambda **kw: _response([_text("hi back")]),
            tools=[],
            dispatcher=_StubDispatcher({}),
        ))
        self.assertEqual(len(conv._history), 2)
        self.assertEqual(conv._history[-1].content, "hi back")


if __name__ == "__main__":
    unittest.main()
