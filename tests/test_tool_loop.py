"""Tests for the HAL LLM ↔ tool dispatch loop (W2).

Verifies :func:`run_tool_loop` drives the round-trip:

    user → LLM → tool_use → dispatch → tool_result → LLM → … → final text

Falsifiable claims (H057):

- H057a: single tool call completes and the final response returns
- H057b: multiple tools in one turn all dispatch before the next chat call
- H057c: guard blocks, dispatcher errors, and malformed tool_use blocks
  surface to the model as ``is_error=True`` tool_results without raising
  to the caller
- H057d: iteration cap is hit exactly when the model keeps emitting
  tool_use blocks past ``max_iterations``
- H057e: ``tool_use_id`` round-trips so Anthropic's server-side
  correlation check cannot mis-route a tool_result

The dispatcher is stubbed with a lightweight fake so tests stay hermetic:
they never load real handlers, real hooks, or touch the filesystem.
"""
import os
import sys
import unittest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.tool_loop import ToolLoopResult, run_tool_loop


class _StubToolResult:
    """Stand-in matching ToolResult's attribute contract."""

    def __init__(self, output="", error="", blocked=False, reason=""):
        self.output = output
        self.error = error
        self.blocked = blocked
        self.reason = reason


class _StubDispatcher:
    """Deterministic dispatcher that avoids loading real handlers."""

    def __init__(self, handlers):
        self._handlers = handlers
        self.calls = []

    def execute(self, name, tool_input):
        self.calls.append((name, tool_input))
        if name in self._handlers:
            return self._handlers[name](tool_input)
        return _StubToolResult(error=f"unknown tool: {name}")


def _tool_use(tid, name, inp):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def _text(text):
    return {"type": "text", "text": text}


def _response(content, stop_reason="end_turn"):
    return {"content": content, "stop_reason": stop_reason}


def _scripted(*responses):
    """Return a chat_fn that yields responses in order, one per call."""
    queue = list(responses)

    def chat_fn(*, messages, tools):
        return queue.pop(0)

    return chat_fn


class TestNoTools(unittest.TestCase):
    def test_single_chat_call_when_tools_none(self):
        calls = []

        def chat_fn(*, messages, tools):
            calls.append(tools)
            return _response([_text("hello")])

        result = run_tool_loop(messages=[], chat_fn=chat_fn)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.iterations, 1)
        self.assertFalse(result.hit_iteration_cap)
        self.assertEqual(result.tool_calls, [])

    def test_single_chat_call_when_tools_empty_list(self):
        result = run_tool_loop(
            messages=[],
            chat_fn=_scripted(_response([_text("hi")])),
            tools=[],
        )
        self.assertEqual(result.iterations, 1)
        self.assertFalse(result.hit_iteration_cap)


class TestDispatcherRequirement(unittest.TestCase):
    def test_raises_if_tools_given_but_no_dispatcher(self):
        with self.assertRaises(ValueError):
            run_tool_loop(
                messages=[],
                chat_fn=_scripted(_response([])),
                tools=[{"name": "Read"}],
                dispatcher=None,
            )


class TestSingleToolRoundTrip(unittest.TestCase):
    def test_one_tool_use_dispatched_and_final_text_returned(self):
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output="file contents: foo"),
        })
        chat = _scripted(
            _response([_tool_use("t1", "Read", {"file_path": "/tmp/x"})], "tool_use"),
            _response([_text("The file says foo.")]),
        )
        result = run_tool_loop(
            messages=[{"role": "user", "content": "read /tmp/x"}],
            chat_fn=chat,
            tools=[{"name": "Read"}],
            dispatcher=dispatcher,
        )
        self.assertEqual(result.iterations, 2)
        self.assertFalse(result.hit_iteration_cap)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["tool_use"]["name"], "Read")
        self.assertFalse(result.tool_calls[0]["tool_result"]["is_error"])
        self.assertEqual(
            result.tool_calls[0]["tool_result"]["content"],
            "file contents: foo",
        )

    def test_dispatcher_received_correct_input(self):
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output=inp.get("file_path", "")),
        })
        chat = _scripted(
            _response([_tool_use("t1", "Read", {"file_path": "/tmp/zzz"})], "tool_use"),
            _response([_text("done")]),
        )
        run_tool_loop(
            messages=[],
            chat_fn=chat,
            tools=[{"name": "Read"}],
            dispatcher=dispatcher,
        )
        self.assertEqual(dispatcher.calls, [("Read", {"file_path": "/tmp/zzz"})])


class TestMultipleToolsPerTurn(unittest.TestCase):
    def test_both_tools_dispatched_in_one_iteration(self):
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output="a"),
            "Grep": lambda inp: _StubToolResult(output="b"),
        })
        chat = _scripted(
            _response(
                [
                    _tool_use("t1", "Read", {"file_path": "x"}),
                    _tool_use("t2", "Grep", {"pattern": "foo"}),
                ],
                "tool_use",
            ),
            _response([_text("done")]),
        )
        result = run_tool_loop(
            messages=[],
            chat_fn=chat,
            tools=[{"name": "Read"}, {"name": "Grep"}],
            dispatcher=dispatcher,
        )
        self.assertEqual(result.iterations, 2)
        names = {c["tool_use"]["name"] for c in result.tool_calls}
        self.assertEqual(names, {"Read", "Grep"})
        # Both in iteration 0 — they happened in the same turn
        iters = {c["iteration"] for c in result.tool_calls}
        self.assertEqual(iters, {0})


class TestMultipleIterations(unittest.TestCase):
    def test_three_iterations_sequential_tools(self):
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output="a"),
            "Grep": lambda inp: _StubToolResult(output="b"),
        })
        chat = _scripted(
            _response([_tool_use("t1", "Read", {})], "tool_use"),
            _response([_tool_use("t2", "Grep", {})], "tool_use"),
            _response([_text("done")]),
        )
        result = run_tool_loop(
            messages=[],
            chat_fn=chat,
            tools=[{"name": "Read"}, {"name": "Grep"}],
            dispatcher=dispatcher,
        )
        self.assertEqual(result.iterations, 3)
        iters = {c["iteration"] for c in result.tool_calls}
        self.assertEqual(iters, {0, 1})


class TestErrorHandling(unittest.TestCase):
    def test_guard_block_becomes_is_error_tool_result(self):
        dispatcher = _StubDispatcher({
            "Bash": lambda inp: _StubToolResult(
                blocked=True, reason="rm -rf denied by guard"
            ),
        })
        chat = _scripted(
            _response([_tool_use("t1", "Bash", {"command": "rm -rf /"})], "tool_use"),
            _response([_text("I cannot do that.")]),
        )
        result = run_tool_loop(
            messages=[],
            chat_fn=chat,
            tools=[{"name": "Bash"}],
            dispatcher=dispatcher,
        )
        self.assertTrue(result.tool_calls[0]["tool_result"]["is_error"])
        self.assertIn("denied by guard", result.tool_calls[0]["tool_result"]["content"])

    def test_guard_block_without_reason_uses_default_message(self):
        dispatcher = _StubDispatcher({
            "Bash": lambda inp: _StubToolResult(blocked=True, reason=""),
        })
        chat = _scripted(
            _response([_tool_use("t1", "Bash", {})], "tool_use"),
            _response([_text("ok")]),
        )
        result = run_tool_loop(
            messages=[],
            chat_fn=chat,
            tools=[{"name": "Bash"}],
            dispatcher=dispatcher,
        )
        self.assertTrue(result.tool_calls[0]["tool_result"]["is_error"])
        self.assertEqual(
            result.tool_calls[0]["tool_result"]["content"], "blocked by guard"
        )

    def test_dispatcher_error_surfaces_as_is_error(self):
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(error="file not found"),
        })
        chat = _scripted(
            _response([_tool_use("t1", "Read", {})], "tool_use"),
            _response([_text("oh.")]),
        )
        result = run_tool_loop(
            messages=[],
            chat_fn=chat,
            tools=[{"name": "Read"}],
            dispatcher=dispatcher,
        )
        self.assertTrue(result.tool_calls[0]["tool_result"]["is_error"])
        self.assertIn("file not found", result.tool_calls[0]["tool_result"]["content"])

    def test_missing_tool_name_becomes_is_error_without_calling_dispatcher(self):
        dispatcher = _StubDispatcher({})
        malformed_response = {
            "content": [{"type": "tool_use", "id": "t1", "input": {}}],
            "stop_reason": "tool_use",
        }
        chat = _scripted(malformed_response, _response([_text("ok")]))
        result = run_tool_loop(
            messages=[],
            chat_fn=chat,
            tools=[{"name": "X"}],
            dispatcher=dispatcher,
        )
        self.assertTrue(result.tool_calls[0]["tool_result"]["is_error"])
        self.assertEqual(dispatcher.calls, [])  # never reached the dispatcher


class TestIterationCap(unittest.TestCase):
    def test_cap_hit_when_model_keeps_calling_tools(self):
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output="more"),
        })

        def chat_fn(*, messages, tools):
            return _response([_tool_use("t", "Read", {})], "tool_use")

        result = run_tool_loop(
            messages=[],
            chat_fn=chat_fn,
            tools=[{"name": "Read"}],
            dispatcher=dispatcher,
            max_iterations=3,
        )
        self.assertTrue(result.hit_iteration_cap)
        self.assertEqual(result.iterations, 3)
        self.assertEqual(len(result.tool_calls), 3)

    def test_cap_not_hit_when_loop_ends_on_final_iteration(self):
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output="x"),
        })
        chat = _scripted(
            _response([_tool_use("t1", "Read", {})], "tool_use"),
            _response([_text("done")]),
        )
        result = run_tool_loop(
            messages=[],
            chat_fn=chat,
            tools=[{"name": "Read"}],
            dispatcher=dispatcher,
            max_iterations=2,
        )
        self.assertFalse(result.hit_iteration_cap)
        self.assertEqual(result.iterations, 2)


class TestCorrelation(unittest.TestCase):
    def test_tool_use_id_preserved_in_tool_result(self):
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output="x"),
        })
        chat = _scripted(
            _response([_tool_use("toolu_abc123", "Read", {})], "tool_use"),
            _response([_text("done")]),
        )
        result = run_tool_loop(
            messages=[],
            chat_fn=chat,
            tools=[{"name": "Read"}],
            dispatcher=dispatcher,
        )
        self.assertEqual(
            result.tool_calls[0]["tool_result"]["tool_use_id"], "toolu_abc123"
        )


class TestMessageHistory(unittest.TestCase):
    def test_messages_mutated_in_place(self):
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(output="contents"),
        })
        chat = _scripted(
            _response([_tool_use("t1", "Read", {})], "tool_use"),
            _response([_text("done")]),
        )
        messages = [{"role": "user", "content": "go"}]
        result = run_tool_loop(
            messages=messages,
            chat_fn=chat,
            tools=[{"name": "Read"}],
            dispatcher=dispatcher,
        )
        self.assertIs(result.messages, messages)
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(messages[2]["content"][0]["type"], "tool_result")
        self.assertEqual(messages[2]["content"][0]["content"], "contents")

    def test_history_not_mutated_when_tools_empty(self):
        messages = [{"role": "user", "content": "hi"}]
        run_tool_loop(
            messages=messages,
            chat_fn=_scripted(_response([_text("hello")])),
            tools=[],
        )
        self.assertEqual(len(messages), 1)


class TestCallback(unittest.TestCase):
    def test_on_tool_call_fires_per_tool_with_both_args(self):
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

        def on_call(tool_use, tool_result):
            log.append((tool_use["name"], tool_result["is_error"]))

        run_tool_loop(
            messages=[],
            chat_fn=chat,
            tools=[{"name": "Read"}, {"name": "Grep"}],
            dispatcher=dispatcher,
            on_tool_call=on_call,
        )
        self.assertEqual(log, [("Read", False), ("Grep", False)])

    def test_on_tool_call_fires_for_errors_too(self):
        dispatcher = _StubDispatcher({
            "Read": lambda inp: _StubToolResult(blocked=True, reason="nope"),
        })
        chat = _scripted(
            _response([_tool_use("t1", "Read", {})], "tool_use"),
            _response([_text("ok")]),
        )
        log = []
        run_tool_loop(
            messages=[],
            chat_fn=chat,
            tools=[{"name": "Read"}],
            dispatcher=dispatcher,
            on_tool_call=lambda tu, tr: log.append(tr["is_error"]),
        )
        self.assertEqual(log, [True])


class TestReturnValue(unittest.TestCase):
    def test_result_is_tool_loop_result_instance(self):
        result = run_tool_loop(
            messages=[],
            chat_fn=_scripted(_response([_text("hi")])),
        )
        self.assertIsInstance(result, ToolLoopResult)

    def test_final_response_is_last_chat_return(self):
        final = _response([_text("final answer")])
        result = run_tool_loop(messages=[], chat_fn=_scripted(final))
        self.assertIs(result.final_response, final)


if __name__ == "__main__":
    unittest.main()
