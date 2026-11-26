"""HAL LLM ↔ tool dispatch loop.

Given a ``chat_fn`` that returns Anthropic-shaped responses, and a
``ToolDispatcher`` that executes tool calls through HAL's guard pipeline,
:func:`run_tool_loop` drives the round-trip:

    user message
        → chat_fn (with tools=...)
        → assistant emits tool_use blocks
        → dispatch each via ToolDispatcher
        → append tool_result blocks to messages
        → chat_fn again
        → repeat until stop_reason != "tool_use" (or iteration cap)

The loop is provider-agnostic: ``chat_fn`` can wrap ``AnthropicProvider``,
``CCHSubscription``, or a test fake. That indirection is what lets W2
ship with full coverage before W3 wires it into ``Conversation``.

Returns a :class:`ToolLoopResult` carrying the final response, the
mutated message history, the list of tool calls that fired, and a flag
for whether the iteration cap was hit — the CLI (W3) and server (W5)
will use the tool-call list for live display, and the cap flag for
clean degradation when the model loops on itself.

Usage:
    from lib.hal.tool_adapter import to_anthropic_tools
    from lib.hal.tool_dispatch import ToolDispatcher
    from lib.hal.tool_loop import run_tool_loop

    def chat_fn(*, messages, tools):
        # wrap your provider here; must return a dict with
        # "content" (list of blocks) and "stop_reason"
        return provider.call_raw(messages=messages, tools=tools)

    result = run_tool_loop(
        messages=[{"role": "user", "content": "read /tmp/x and summarise"}],
        chat_fn=chat_fn,
        tools=to_anthropic_tools(),
        dispatcher=ToolDispatcher(),
    )
    print(result.final_response["content"][0]["text"])
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from lib.hal.tool_adapter import parse_tool_use_blocks
from lib.hal.tool_dispatch import ToolDispatcher


@dataclass
class ToolLoopResult:
    """Outcome of a :func:`run_tool_loop` invocation.

    ``final_response`` is the raw Anthropic response dict from the LAST
    call to ``chat_fn`` — whether that call ended naturally (``stop_reason``
    of ``end_turn`` / ``stop_sequence`` / ``max_tokens``) or because the
    iteration cap was hit (``hit_iteration_cap=True``, ``stop_reason`` may
    still be ``tool_use``).

    ``messages`` is the same list the caller passed in — mutated in place
    so the conversation can continue across multiple calls without the
    caller rebuilding history.

    ``tool_calls`` is an ordered log of ``{iteration, tool_use, tool_result}``
    triples — the minimum payload for rendering the trace in a UI or
    persisting an audit record.
    """
    final_response: dict
    messages: list[dict]
    iterations: int
    tool_calls: list[dict] = field(default_factory=list)
    hit_iteration_cap: bool = False


def _build_tool_result(
    tool_use: dict,
    dispatcher: ToolDispatcher,
) -> dict:
    """Dispatch one ``tool_use`` block and return the matching ``tool_result``.

    ``tool_use`` is a dict from :func:`parse_tool_use_blocks`:
    ``{"id": "toolu_...", "name": "Read", "input": {...}}``.

    The returned block has the shape Anthropic expects in the next user
    turn's content array::

        {"type": "tool_result",
         "tool_use_id": tool_use["id"],
         "content": str,
         "is_error": bool}

    Guard blocks, dispatcher errors, and empty tool names all become
    ``is_error=True`` results — the loop continues so the model can
    react (apologise, try a different tool, give up gracefully) rather
    than exceptions bubbling up to the caller. ``tool_use_id`` is
    preserved so the model can correlate which call the error belongs to.
    """
    name = tool_use.get("name", "")
    tool_input = tool_use.get("input", {})
    tool_use_id = tool_use.get("id", "")

    if not name:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": "missing tool name in tool_use block",
            "is_error": True,
        }

    result = dispatcher.execute(name, tool_input)

    if getattr(result, "blocked", False):
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": getattr(result, "reason", "") or "blocked by guard",
            "is_error": True,
        }

    error = getattr(result, "error", "") or ""
    if error:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": str(error),
            "is_error": True,
        }

    output = getattr(result, "output", "") or ""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": str(output),
        "is_error": False,
    }


def _dispatch_tool_uses(
    tool_uses: list[dict],
    dispatcher: ToolDispatcher,
    iteration: int,
    on_tool_call: Optional[Callable[[dict, dict], None]],
) -> tuple[list[dict], list[dict]]:
    """Dispatch every tool_use in one turn; return (result_blocks, log_entries).

    Split out of :func:`run_tool_loop` so the top-level driver is a single
    linear scan over iterations — the inner per-turn fan-out lives here.
    """
    blocks: list[dict] = []
    log: list[dict] = []
    for tool_use in tool_uses:
        result_block = _build_tool_result(tool_use, dispatcher)
        blocks.append(result_block)
        log.append({
            "iteration": iteration,
            "tool_use": tool_use,
            "tool_result": result_block,
        })
        if on_tool_call is not None:
            on_tool_call(tool_use, result_block)
    return blocks, log


def run_tool_loop(
    messages: list[dict],
    chat_fn: Callable[..., dict],
    *,
    tools: Optional[list[dict]] = None,
    dispatcher: Optional[ToolDispatcher] = None,
    max_iterations: int = 10,
    on_tool_call: Optional[Callable[[dict, dict], None]] = None,
) -> ToolLoopResult:
    """Drive an LLM ↔ ToolDispatcher loop until the model stops calling tools.

    Loop invariant: after each ``chat_fn`` call, if the response contains
    ``tool_use`` blocks, append (assistant turn, user turn with
    tool_results) to ``messages`` and call ``chat_fn`` again. When the
    response has no ``tool_use`` blocks — or the iteration cap is hit —
    return.

    ``messages`` is mutated in place (the returned ``messages`` is the
    same list) so callers can continue the conversation by appending
    the next user turn and calling again without rebuilding history.

    ``tools`` None/empty short-circuits to a single ``chat_fn`` call
    with no tools, so the same helper works for tool-free conversations.

    ``dispatcher`` is required whenever ``tools`` is non-empty — fail
    fast on misuse rather than silently stripping tool_use blocks and
    confusing the model.

    ``on_tool_call(tool_use, tool_result)`` fires once per dispatched
    call, BEFORE the result is appended to messages — used by the CLI
    (W3) for live ANSI rendering and by the server (W5) for SSE streaming.
    """
    if tools and dispatcher is None:
        raise ValueError("dispatcher is required when tools are provided")

    tool_calls_log: list[dict] = []
    last_response: dict = {}

    for iteration in range(max_iterations):
        last_response = chat_fn(messages=messages, tools=tools or [])
        tool_uses = parse_tool_use_blocks(last_response) if tools else []

        if not tool_uses:
            return ToolLoopResult(
                final_response=last_response,
                messages=messages,
                iterations=iteration + 1,
                tool_calls=tool_calls_log,
                hit_iteration_cap=False,
            )

        # Assistant turn must carry the tool_use blocks verbatim so
        # tool_use_ids line up with the tool_result blocks we're about
        # to produce. Anthropic validates this correlation server-side.
        messages.append({
            "role": "assistant",
            "content": last_response.get("content", []),
        })

        result_blocks, log_entries = _dispatch_tool_uses(
            tool_uses, dispatcher, iteration, on_tool_call,
        )
        tool_calls_log.extend(log_entries)
        messages.append({"role": "user", "content": result_blocks})

    return ToolLoopResult(
        final_response=last_response,
        messages=messages,
        iterations=max_iterations,
        tool_calls=tool_calls_log,
        hit_iteration_cap=True,
    )


__all__ = ["ToolLoopResult", "run_tool_loop"]
