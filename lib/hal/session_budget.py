"""Per-session budget enforcement for the HAL tool loop (prod hardening).

Production deployments (web, CI, multi-user) need a way to cap what a
single conversation can consume so a runaway loop or a malicious user
cannot burn unbounded credits. :class:`SessionBudget` is the stateful
counter + limit checker that wraps around
:func:`~lib.hal.tool_loop.run_tool_loop`.

Four dimensions are tracked and capped independently:

- **input_tokens / output_tokens** — cumulative Anthropic usage totals
- **cost_usd** — whatever the caller computed for each response (HAL's
  :mod:`~lib.hal.cost_tracker` has the canonical model-aware pricing)
- **iterations** — number of ``chat_fn`` round-trips
- **tool_calls** — total tools dispatched across all iterations

Every limit is optional — ``None`` means "no cap". Budgets compose with
:func:`run_tool_loop` by wrapping the provider's ``chat_fn`` once and
registering :meth:`record_tool_call` as the ``on_tool_call`` callback.

Usage:
    from lib.hal.session_budget import BudgetLimits, SessionBudget
    from lib.hal.tool_loop import run_tool_loop
    from lib.hal.tool_runtime import make_anthropic_chat_fn

    budget = SessionBudget(
        limits=BudgetLimits(max_cost_usd=1.00, max_iterations=20),
    )
    raw_chat_fn = make_anthropic_chat_fn(api_key=..., model="claude-opus-4-7")

    def chat_fn(**kw):
        budget.check_before_chat()       # fail before spending
        response = raw_chat_fn(**kw)
        budget.record_response(response, cost_usd=my_cost_estimator(response))
        return response

    result = run_tool_loop(
        messages=[...],
        chat_fn=chat_fn,
        tools=tools,
        dispatcher=dispatcher,
        on_tool_call=budget.record_tool_call,
    )
"""
from dataclasses import dataclass, field
from typing import Optional


class BudgetExceeded(RuntimeError):
    """Raised when a :class:`SessionBudget` limit is hit.

    The ``dimension`` attribute names which cap fired (``cost_usd``,
    ``input_tokens``, etc.) so the caller can tell the user which knob
    to raise.
    """

    def __init__(self, dimension: str, value: float, limit: float):
        super().__init__(
            f"budget exceeded: {dimension}={value} > limit={limit}",
        )
        self.dimension = dimension
        self.value = value
        self.limit = limit


@dataclass(frozen=True)
class BudgetLimits:
    """Immutable cap values. ``None`` means "no cap on this dimension"."""

    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_iterations: Optional[int] = None
    max_cost_usd: Optional[float] = None
    max_tool_calls: Optional[int] = None


@dataclass
class SessionBudget:
    """Cumulative counters + limit enforcement for one conversation.

    All totals start at zero and increase monotonically across calls
    to :meth:`record_response` and :meth:`record_tool_call`.
    :meth:`check_before_chat` is called BEFORE a chat_fn call to prevent
    starting one that would blow the iteration or cost cap;
    :meth:`check` is called AFTER a state mutation to verify post-call
    state has not breached a token cap the caller couldn't predict
    ahead of time.
    """

    limits: BudgetLimits = field(default_factory=BudgetLimits)
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0
    cost_usd: float = 0.0
    tool_calls: int = 0

    def record_response(self, response: dict, cost_usd: float = 0.0) -> None:
        """Update counters after a chat_fn call.

        Tolerates malformed responses (non-dict, missing usage) by
        treating missing fields as zero — the budget should not crash
        the loop even if a provider returns garbage.
        """
        usage: dict = {}
        if isinstance(response, dict):
            maybe_usage = response.get("usage")
            if isinstance(maybe_usage, dict):
                usage = maybe_usage
        self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)
        self.iterations += 1
        self.cost_usd += float(cost_usd)

    def record_tool_call(self, tool_use: dict, tool_result: dict) -> None:
        """Increment the tool-call counter.

        Signature matches :func:`run_tool_loop`'s ``on_tool_call``
        callback so it can be registered directly:
        ``run_tool_loop(..., on_tool_call=budget.record_tool_call)``.
        """
        del tool_use, tool_result  # we only need the count
        self.tool_calls += 1

    def check_before_chat(self) -> None:
        """Raise :class:`BudgetExceeded` if starting another chat_fn would breach a cap.

        Only iteration and cost caps are checked here — they're the two
        that a new chat_fn call would definitely touch (iteration +1,
        cost at least not-decrease). Token caps are checked post-hoc
        via :meth:`check` because the prompt's token count is unknown
        until the provider tokenises.
        """
        lim = self.limits
        if lim.max_iterations is not None and self.iterations >= lim.max_iterations:
            raise BudgetExceeded("iterations", self.iterations, lim.max_iterations)
        if lim.max_cost_usd is not None and self.cost_usd >= lim.max_cost_usd:
            raise BudgetExceeded("cost_usd", self.cost_usd, lim.max_cost_usd)

    def check(self) -> None:
        """Raise if any cap has been exceeded by the current totals.

        Checked AFTER state updates — for token caps that can only be
        known once the provider returns a usage block. Dimensions are
        evaluated in a deterministic order so the exception message is
        reproducible across runs.
        """
        lim = self.limits
        checks = (
            ("input_tokens", self.input_tokens, lim.max_input_tokens),
            ("output_tokens", self.output_tokens, lim.max_output_tokens),
            ("iterations", self.iterations, lim.max_iterations),
            ("cost_usd", self.cost_usd, lim.max_cost_usd),
            ("tool_calls", self.tool_calls, lim.max_tool_calls),
        )
        for dimension, value, limit in checks:
            if limit is not None and value > limit:
                raise BudgetExceeded(dimension, value, limit)

    def remaining(self) -> dict:
        """Return per-dimension remaining headroom.

        Useful for UI display ("3 / 20 iterations left") or structured
        logging. Dimensions with no cap report ``None`` so the caller
        can distinguish "uncapped" from "zero left".
        """
        lim = self.limits

        def _diff(limit: Optional[float], value: float):
            if limit is None:
                return None
            return max(limit - value, 0)

        return {
            "input_tokens": _diff(lim.max_input_tokens, self.input_tokens),
            "output_tokens": _diff(lim.max_output_tokens, self.output_tokens),
            "iterations": _diff(lim.max_iterations, self.iterations),
            "cost_usd": _diff(lim.max_cost_usd, self.cost_usd),
            "tool_calls": _diff(lim.max_tool_calls, self.tool_calls),
        }


__all__ = ["BudgetExceeded", "BudgetLimits", "SessionBudget"]
