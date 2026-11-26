"""Tests for send_with_tools (W6 integration sprint).

Verifies the Conversation-level composition of W1-W5 machinery:

- Message history is updated correctly (user turn before, final
  assistant turn after) regardless of how many tool iterations fired
- ``api_messages`` passed to run_tool_loop reflects the full history
  so multi-turn tool sessions see prior context
- ``on_tool_call`` callback is forwarded and fires per dispatched tool
- The pipeline is BYPASSED intentionally — no MEDIA/INTENT/GUARD/ROUTE
  stages run (tool sessions have different semantics)
- Errors in chat_fn propagate to the caller (no silent swallowing)
  but the user message stays in history for retry
"""
import os
import sys
import unittest
from pathlib import Path

# Always resolve imports from the worktree, not from HAL_HOME, so edits
# to lib/ in this worktree are tested rather than the installed ~/.hal copy.
_WORKTREE = str(Path(__file__).resolve().parent.parent)
if _WORKTREE not in sys.path:
    sys.path.insert(0, _WORKTREE)

from lib.hal.conversation import Conversation
from lib.hal.tool_conversation import send_with_tools
from lib.hal.tool_loop import ToolLoopResult


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
        return _StubToolResult(error=f"unknown tool: {name}")


def _scripted(*responses):
    queue = list(responses)

    def chat_fn(*, messages, tools):
        chat_fn.last_messages = list(messages)
        return queue.pop(0)

    chat_fn.last_messages = []
    return chat_fn


def _tool_use(tid, name, inp):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def _text(text):
    return {"type": "text", "text": text}


def _response(content, stop_reason="end_turn"):
    return {"content": content, "stop_reason": stop_reason}


def _conv():
    """Construct a Conversation without real provider/registry coupling."""
    return Conversation(model="", registry=None)


class TestHappyPath(unittest.TestCase):
    def test_simple_response_no_tools_used(self):
        conv = _conv()
        chat = _scripted(_response([_text("hello world")]))
        result = send_with_tools(
            conv, "hi",
            chat_fn=chat,
            tools=[{"name": "Read"}],
            dispatcher=_StubDispatcher({}),
        )
        self.assertIsInstance(result, ToolLoopResult)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(conv._last_response_text, "hello world")

    def test_one_tool_call_round_trip(self):
        conv = _conv()
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output="file-data"),
        })
        chat = _scripted(
            _response([_tool_use("t1", "Read", {"file_path": "/x"})], "tool_use"),
            _response([_text("The file says 'file-data'.")]),
        )
        result = send_with_tools(
            conv, "read /x",
            chat_fn=chat,
            tools=[{"name": "Read"}],
            dispatcher=dispatcher,
        )
        self.assertEqual(result.iterations, 2)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(conv._last_response_text, "The file says 'file-data'.")


class TestHistoryManagement(unittest.TestCase):
    def test_user_message_appended_before_chat_fn_called(self):
        conv = _conv()
        chat = _scripted(_response([_text("ok")]))
        send_with_tools(
            conv, "hello",
            chat_fn=chat,
            tools=[],
            dispatcher=_StubDispatcher({}),
        )
        self.assertEqual(chat.last_messages[0]["role"], "user")
        self.assertEqual(chat.last_messages[0]["content"], "hello")

    def test_assistant_message_appended_after_final_response(self):
        conv = _conv()
        chat = _scripted(_response([_text("done")]))
        send_with_tools(
            conv, "hi",
            chat_fn=chat,
            tools=[],
            dispatcher=_StubDispatcher({}),
        )
        self.assertEqual(len(conv._history), 2)
        self.assertEqual(conv._history[0].role, "user")
        self.assertEqual(conv._history[0].content, "hi")
        self.assertEqual(conv._history[1].role, "assistant")
        self.assertEqual(conv._history[1].content, "done")

    def test_history_contains_only_user_and_final_assistant_after_tool_calls(self):
        """tool_use/tool_result turns stay inside run_tool_loop — history
        only records the user's ask and the model's final prose."""
        conv = _conv()
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output="x"),
        })
        chat = _scripted(
            _response([_tool_use("t1", "Read", {})], "tool_use"),
            _response([_tool_use("t2", "Read", {})], "tool_use"),
            _response([_text("final")]),
        )
        send_with_tools(
            conv, "go",
            chat_fn=chat,
            tools=[{"name": "Read"}],
            dispatcher=dispatcher,
        )
        self.assertEqual(len(conv._history), 2)
        self.assertEqual(conv._history[1].content, "final")

    def test_multi_turn_sees_prior_history(self):
        """Turn 2's chat_fn receives turn 1's user+assistant in messages."""
        conv = _conv()
        chat1 = _scripted(_response([_text("answer 1")]))
        send_with_tools(
            conv, "question 1", chat_fn=chat1, tools=[],
            dispatcher=_StubDispatcher({}),
        )
        chat2 = _scripted(_response([_text("answer 2")]))
        send_with_tools(
            conv, "question 2", chat_fn=chat2, tools=[],
            dispatcher=_StubDispatcher({}),
        )
        self.assertEqual(len(chat2.last_messages), 3)
        self.assertEqual(chat2.last_messages[0]["content"], "question 1")
        self.assertEqual(chat2.last_messages[1]["content"], "answer 1")
        self.assertEqual(chat2.last_messages[2]["content"], "question 2")


class TestCallbackForwarding(unittest.TestCase):
    def test_on_tool_call_fires_per_tool(self):
        conv = _conv()
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output="x"),
            "Grep": lambda inp: _StubToolResult(output="y"),
        })
        chat = _scripted(
            _response(
                [
                    _tool_use("t1", "Read", {}),
                    _tool_use("t2", "Grep", {}),
                ],
                "tool_use",
            ),
            _response([_text("done")]),
        )
        log = []
        send_with_tools(
            conv, "go",
            chat_fn=chat,
            tools=[{"name": "Read"}, {"name": "Grep"}],
            dispatcher=dispatcher,
            on_tool_call=lambda tu, tr: log.append(tu["name"]),
        )
        self.assertEqual(log, ["Read", "Grep"])


class TestErrorPropagation(unittest.TestCase):
    def test_chat_fn_error_propagates(self):
        conv = _conv()

        def broken(*, messages, tools):
            raise RuntimeError("API down")

        with self.assertRaises(RuntimeError) as ctx:
            send_with_tools(
                conv, "hi",
                chat_fn=broken,
                tools=[],
                dispatcher=_StubDispatcher({}),
            )
        self.assertIn("API down", str(ctx.exception))

    def test_user_message_still_in_history_after_chat_fn_error(self):
        conv = _conv()

        def broken(*, messages, tools):
            raise RuntimeError("rate limit")

        try:
            send_with_tools(
                conv, "hi",
                chat_fn=broken,
                tools=[],
                dispatcher=_StubDispatcher({}),
            )
        except RuntimeError:
            pass
        self.assertEqual(len(conv._history), 1)
        self.assertEqual(conv._history[0].role, "user")


class TestMaxIterations(unittest.TestCase):
    def test_forwards_max_iterations_to_run_tool_loop(self):
        conv = _conv()
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output="x"),
        })

        def chat_fn(*, messages, tools):
            return _response([_tool_use("t", "Read", {})], "tool_use")

        result = send_with_tools(
            conv, "go",
            chat_fn=chat_fn,
            tools=[{"name": "Read"}],
            dispatcher=dispatcher,
            max_iterations=2,
        )
        self.assertTrue(result.hit_iteration_cap)
        self.assertEqual(result.iterations, 2)


class TestReturnValue(unittest.TestCase):
    def test_returns_tool_loop_result_not_chatresponse(self):
        conv = _conv()
        chat = _scripted(_response([_text("ok")]))
        result = send_with_tools(
            conv, "hi", chat_fn=chat, tools=[],
            dispatcher=_StubDispatcher({}),
        )
        self.assertIsInstance(result, ToolLoopResult)
        self.assertEqual(conv._last_response_text, "ok")


class TestEmptyMessageGuard(unittest.TestCase):
    def test_send_with_tools_rejects_empty(self):
        """send_with_tools must not mutate history on empty message."""
        conv = _conv()
        starting = len(conv._history)
        with self.assertRaises(ValueError):
            send_with_tools(
                conv, "",
                chat_fn=lambda **kw: {},
                tools=[],
                dispatcher=_StubDispatcher({}),
            )
        self.assertEqual(len(conv._history), starting)

    def test_send_with_tools_rejects_whitespace(self):
        """Whitespace-only message is also empty — history must not grow."""
        conv = _conv()
        starting = len(conv._history)
        with self.assertRaises(ValueError):
            send_with_tools(
                conv, "   \n\t  ",
                chat_fn=lambda **kw: {},
                tools=[],
                dispatcher=_StubDispatcher({}),
            )
        self.assertEqual(len(conv._history), starting)


if __name__ == "__main__":
    unittest.main()
