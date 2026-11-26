"""Tests for :class:`SessionBudget` — per-session budget enforcement.

Verifies:
- record_response accrues tokens, iterations, and cost monotonically
- record_tool_call increments the tool_calls counter
- check_before_chat raises on iteration/cost caps, skips token caps
  (token caps only knowable post-call)
- check raises on every dimension that has a cap and is exceeded
- remaining() reports correct deltas and None for uncapped dimensions
- Malformed responses (non-dict, missing usage) do not crash the budget
- Integration with run_tool_loop enforces caps end-to-end
"""
import os
import sys
import unittest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.session_budget import (
    BudgetExceeded,
    BudgetLimits,
    SessionBudget,
)


def _usage_response(input_tokens=0, output_tokens=0, text="ok"):
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


class TestInitialState(unittest.TestCase):
    def test_all_counters_start_at_zero(self):
        b = SessionBudget()
        self.assertEqual(b.input_tokens, 0)
        self.assertEqual(b.output_tokens, 0)
        self.assertEqual(b.iterations, 0)
        self.assertEqual(b.cost_usd, 0.0)
        self.assertEqual(b.tool_calls, 0)

    def test_default_limits_all_none(self):
        b = SessionBudget()
        self.assertIsNone(b.limits.max_input_tokens)
        self.assertIsNone(b.limits.max_output_tokens)
        self.assertIsNone(b.limits.max_iterations)
        self.assertIsNone(b.limits.max_cost_usd)
        self.assertIsNone(b.limits.max_tool_calls)


class TestRecordResponse(unittest.TestCase):
    def test_accrues_tokens_from_usage_block(self):
        b = SessionBudget()
        b.record_response(_usage_response(input_tokens=50, output_tokens=20))
        self.assertEqual(b.input_tokens, 50)
        self.assertEqual(b.output_tokens, 20)

    def test_monotonically_increases_across_calls(self):
        b = SessionBudget()
        b.record_response(_usage_response(input_tokens=10, output_tokens=5))
        b.record_response(_usage_response(input_tokens=20, output_tokens=10))
        self.assertEqual(b.input_tokens, 30)
        self.assertEqual(b.output_tokens, 15)
        self.assertEqual(b.iterations, 2)

    def test_iterations_increment_per_call(self):
        b = SessionBudget()
        for _ in range(4):
            b.record_response(_usage_response())
        self.assertEqual(b.iterations, 4)

    def test_cost_accrues_from_arg(self):
        b = SessionBudget()
        b.record_response(_usage_response(), cost_usd=0.15)
        b.record_response(_usage_response(), cost_usd=0.35)
        self.assertAlmostEqual(b.cost_usd, 0.50)

    def test_missing_usage_block_treated_as_zero(self):
        b = SessionBudget()
        b.record_response({"content": [], "stop_reason": "end_turn"})
        self.assertEqual(b.input_tokens, 0)
        self.assertEqual(b.output_tokens, 0)
        self.assertEqual(b.iterations, 1)

    def test_non_dict_response_still_increments_iteration(self):
        """Provider returning garbage should not crash the budget."""
        b = SessionBudget()
        b.record_response("not a dict")
        b.record_response(None)
        b.record_response([])
        self.assertEqual(b.iterations, 3)
        self.assertEqual(b.input_tokens, 0)

    def test_non_dict_usage_tolerated(self):
        b = SessionBudget()
        b.record_response({"usage": "weird"})
        self.assertEqual(b.input_tokens, 0)


class TestRecordToolCall(unittest.TestCase):
    def test_increments_counter(self):
        b = SessionBudget()
        b.record_tool_call({"name": "Read"}, {"is_error": False})
        b.record_tool_call({"name": "Grep"}, {"is_error": True})
        self.assertEqual(b.tool_calls, 2)

    def test_signature_compatible_with_on_tool_call_callback(self):
        """Must accept exactly (tool_use, tool_result) — run_tool_loop's contract."""
        from lib.hal.tool_loop import run_tool_loop

        class _StubResult:
            def __init__(self):
                self.output = "x"
                self.error = ""
                self.blocked = False
                self.reason = ""

        class _StubDispatcher:
            def execute(self, name, inp):
                return _StubResult()

        budget = SessionBudget()
        responses = [
            {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
                    {"type": "tool_use", "id": "t2", "name": "Grep", "input": {}},
                ],
                "stop_reason": "tool_use",
            },
            {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
        ]
        run_tool_loop(
            messages=[],
            chat_fn=lambda **kw: responses.pop(0),
            tools=[{"name": "Read"}, {"name": "Grep"}],
            dispatcher=_StubDispatcher(),
            on_tool_call=budget.record_tool_call,
        )
        self.assertEqual(budget.tool_calls, 2)


class TestCheckBeforeChat(unittest.TestCase):
    def test_passes_when_under_iteration_cap(self):
        b = SessionBudget(limits=BudgetLimits(max_iterations=5))
        b.iterations = 4
        b.check_before_chat()  # no raise

    def test_raises_when_at_iteration_cap(self):
        b = SessionBudget(limits=BudgetLimits(max_iterations=5))
        b.iterations = 5
        with self.assertRaises(BudgetExceeded) as ctx:
            b.check_before_chat()
        self.assertEqual(ctx.exception.dimension, "iterations")
        self.assertEqual(ctx.exception.limit, 5)

    def test_raises_when_at_cost_cap(self):
        b = SessionBudget(limits=BudgetLimits(max_cost_usd=1.00))
        b.cost_usd = 1.00
        with self.assertRaises(BudgetExceeded) as ctx:
            b.check_before_chat()
        self.assertEqual(ctx.exception.dimension, "cost_usd")

    def test_does_not_check_token_caps(self):
        """Token caps are post-hoc only — before_chat can't know prompt size."""
        b = SessionBudget(limits=BudgetLimits(max_input_tokens=10))
        b.input_tokens = 100  # already blown, but check_before_chat shouldn't catch
        b.check_before_chat()  # no raise

    def test_no_caps_never_raises(self):
        b = SessionBudget()
        b.iterations = 9_999_999
        b.cost_usd = 999_999.0
        b.check_before_chat()  # all limits None → pass


class TestCheck(unittest.TestCase):
    def test_raises_on_input_token_cap(self):
        b = SessionBudget(limits=BudgetLimits(max_input_tokens=100))
        b.input_tokens = 101
        with self.assertRaises(BudgetExceeded) as ctx:
            b.check()
        self.assertEqual(ctx.exception.dimension, "input_tokens")

    def test_raises_on_output_token_cap(self):
        b = SessionBudget(limits=BudgetLimits(max_output_tokens=50))
        b.output_tokens = 51
        with self.assertRaises(BudgetExceeded) as ctx:
            b.check()
        self.assertEqual(ctx.exception.dimension, "output_tokens")

    def test_raises_on_tool_calls_cap(self):
        b = SessionBudget(limits=BudgetLimits(max_tool_calls=3))
        b.tool_calls = 4
        with self.assertRaises(BudgetExceeded) as ctx:
            b.check()
        self.assertEqual(ctx.exception.dimension, "tool_calls")

    def test_passes_when_all_under_caps(self):
        b = SessionBudget(limits=BudgetLimits(
            max_input_tokens=1000,
            max_output_tokens=1000,
            max_iterations=10,
            max_cost_usd=5.0,
            max_tool_calls=20,
        ))
        b.input_tokens = 500
        b.output_tokens = 300
        b.iterations = 5
        b.cost_usd = 2.0
        b.tool_calls = 8
        b.check()  # no raise

    def test_exception_carries_value_and_limit(self):
        b = SessionBudget(limits=BudgetLimits(max_tool_calls=5))
        b.tool_calls = 10
        with self.assertRaises(BudgetExceeded) as ctx:
            b.check()
        self.assertEqual(ctx.exception.value, 10)
        self.assertEqual(ctx.exception.limit, 5)


class TestRemaining(unittest.TestCase):
    def test_uncapped_dimensions_report_none(self):
        b = SessionBudget()
        rem = b.remaining()
        self.assertIsNone(rem["cost_usd"])
        self.assertIsNone(rem["iterations"])
        self.assertIsNone(rem["tool_calls"])

    def test_reports_positive_headroom(self):
        b = SessionBudget(limits=BudgetLimits(max_iterations=10, max_cost_usd=1.0))
        b.iterations = 3
        b.cost_usd = 0.25
        rem = b.remaining()
        self.assertEqual(rem["iterations"], 7)
        self.assertAlmostEqual(rem["cost_usd"], 0.75)

    def test_reports_zero_when_at_cap(self):
        b = SessionBudget(limits=BudgetLimits(max_iterations=10))
        b.iterations = 10
        self.assertEqual(b.remaining()["iterations"], 0)

    def test_never_reports_negative(self):
        """Overshoot (limit breached) reports zero, not negative."""
        b = SessionBudget(limits=BudgetLimits(max_iterations=10))
        b.iterations = 15
        self.assertEqual(b.remaining()["iterations"], 0)


class TestIntegrationWithToolLoop(unittest.TestCase):
    def test_iteration_cap_halts_loop(self):
        """Budget fires BudgetExceeded before the 4th iteration starts."""
        from lib.hal.tool_loop import run_tool_loop

        class _StubResult:
            output = "x"
            error = ""
            blocked = False
            reason = ""

        class _StubDispatcher:
            def execute(self, name, inp):
                return _StubResult()

        budget = SessionBudget(limits=BudgetLimits(max_iterations=3))

        def chat_fn(*, messages, tools):
            budget.check_before_chat()
            response = {
                "content": [{"type": "tool_use", "id": "t", "name": "R", "input": {}}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            budget.record_response(response)
            return response

        with self.assertRaises(BudgetExceeded) as ctx:
            run_tool_loop(
                messages=[],
                chat_fn=chat_fn,
                tools=[{"name": "R"}],
                dispatcher=_StubDispatcher(),
                on_tool_call=budget.record_tool_call,
                max_iterations=100,
            )
        self.assertEqual(ctx.exception.dimension, "iterations")
        self.assertEqual(budget.iterations, 3)


if __name__ == "__main__":
    unittest.main()
