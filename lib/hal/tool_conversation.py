"""Conversation-level composition of the HAL tool-use loop.

W6 of the tool-loop integration sprint. Ships
:func:`send_with_tools` as a module-level function that binds a
:class:`~lib.hal.conversation.Conversation` to the W1-W5 machinery
(schema bridge, driver, factory, CCH tools, budget guard — all
already merged) without adding a public method to ``Conversation``
itself. Keeps Conversation's ISP surface small and makes
tool-augmented sessions a first-class composition, not an inheritance
branch.

Design choice: module-level function over method

Conversation already sits at 7 public methods (``send``,
``attribute_last``, ``export``, ``history``, ``model``, the
``contribute_*`` family, …). Adding ``send_with_tools`` as another
method would have pushed ISP past its threshold and coupled
tool-use semantics to the core chat engine. Instead, callers
pass the conversation instance explicitly:

    from lib.hal.conversation import Conversation
    from lib.hal.tool_adapter import to_anthropic_tools
    from lib.hal.tool_conversation import send_with_tools
    from lib.hal.tool_dispatch import ToolDispatcher
    from lib.hal.tool_runtime import make_anthropic_chat_fn

    conv = Conversation(model="claude-opus-4-7")
    chat_fn = make_anthropic_chat_fn(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-opus-4-7",
    )
    result = send_with_tools(
        conv, "read /tmp/x and summarise",
        chat_fn=chat_fn,
        tools=to_anthropic_tools(),
        dispatcher=ToolDispatcher(),
    )
    print(conv.last_response_text)

Pipeline bypass: :meth:`Conversation.send` runs a 7-stage pipeline
(MEDIA / INTENT / GUARD / ROUTE / CONTEXT / DISPATCH / RECORD).
Tool sessions deliberately skip it — the dispatcher's own
:class:`~lib.hal.guard.GuardEngine` enforces safety per tool call,
and cost routing / context assembly interferes with the multi-turn
round-trip that the tool loop requires. ``send`` stays the default
path for ordinary chat; tool sessions go through this function.
"""
from typing import Callable, Optional

from lib.hal.conversation import Conversation
from lib.hal.tool_dispatch import ToolDispatcher
from lib.hal.tool_loop import ToolLoopResult, run_tool_loop
from lib.hal.tool_runtime import extract_final_text
from lib.hal.types import ChatMessage


def send_with_tools(
    conversation: Conversation,
    message: str,
    *,
    chat_fn: Callable[..., dict],
    tools: list[dict],
    dispatcher: ToolDispatcher,
    max_iterations: int = 10,
    on_tool_call: Optional[Callable[[dict, dict], None]] = None,
) -> ToolLoopResult:
    """Send a message through an automatic LLM ↔ tool dispatch loop.

    Composes W1-W5 onto a Conversation:

    1. Append ``message`` to ``conversation._history`` as a user turn.
    2. Convert history to Anthropic-format messages.
    3. Call :func:`~lib.hal.tool_loop.run_tool_loop`, which
       orchestrates LLM → tool_use → dispatch → tool_result → LLM
       until the model stops calling tools (or ``max_iterations``
       is reached).
    4. Extract the final prose via
       :func:`~lib.hal.tool_runtime.extract_final_text` and append
       as an assistant turn so subsequent ``send`` / ``send_with_tools``
       calls see the full context.
    5. Return the raw :class:`ToolLoopResult` — caller gets the
       structured tool_calls log, iteration count, and cap flag
       without losing access to what happened mid-loop.

    On ``chat_fn`` error: the user message stays in history (already
    appended) but no assistant turn is added; the exception propagates.
    This mirrors how a REPL handles a failed send — the user's prompt
    is recorded, the caller decides whether to retry.

    ``on_tool_call(tool_use, tool_result)`` forwards directly to
    ``run_tool_loop`` — used by the CLI (W7) for ANSI rendering and
    by the server (W8) for SSE streaming.
    """
    if not message or not message.strip():
        raise ValueError("Cannot send empty message")
    conversation._history.append(ChatMessage(role="user", content=message))
    conversation._recorder.record("turn", message, role="user")
    conversation._emit_user_message(message)

    api_messages = [
        {"role": m.role, "content": m.content}
        for m in conversation._history
    ]

    result = run_tool_loop(
        messages=api_messages,
        chat_fn=chat_fn,
        tools=tools,
        dispatcher=dispatcher,
        max_iterations=max_iterations,
        on_tool_call=on_tool_call,
    )

    final_text = extract_final_text(result.final_response)
    if final_text:
        conversation._history.append(
            ChatMessage(role="assistant", content=final_text),
        )
        conversation._recorder.record("turn", final_text, role="assistant")
        conversation._last_response_text = final_text
    return result


__all__ = ["send_with_tools"]
