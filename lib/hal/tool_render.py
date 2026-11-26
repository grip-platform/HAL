"""ANSI rendering for HAL tool-call events (W7 integration sprint).

Makes a CLI/TUI operator able to SEE the tool-use loop happen: when
the model fires ``Read``, ``Grep``, ``Bash``, etc., a coloured line
like ``▸ Read file_path=/tmp/x``  or ``✖ Bash (blocked by guard)``
renders to stdout as the call completes.

Two public entry points:

* :func:`ansi_render_tool_call` — pure function ``(tool_use,
  tool_result) -> str``. Used by tests, logs, and custom sinks.
* :func:`send_with_tools_to_cli` — one-call wrapper that composes
  :func:`~lib.hal.tool_conversation.send_with_tools` with ANSI
  rendering on the chosen stream. The CLI (``hal``) calls this so
  tool use is visible by default without rewiring the REPL.

Rendering is intentionally TTY-friendly (short, one line per call)
rather than structured. The structured trace is still available via
``ToolLoopResult.tool_calls`` — callers who want a log can ignore
this module entirely.

Colour scheme (low-contrast-safe, no background fills):

* cyan  — tool name
* dim   — input arguments
* green — success
* red   — error / blocked
* gray  — separator

Disables colour automatically when the stream is not a TTY (piped
output, log files, CI) by checking ``stream.isatty()``. Callers can
force colour on/off with ``colour=True|False``.
"""
import json
import sys
from typing import Callable, Optional, TextIO

from lib.hal.conversation import Conversation
from lib.hal.tool_conversation import send_with_tools
from lib.hal.tool_dispatch import ToolDispatcher
from lib.hal.tool_loop import ToolLoopResult


# ANSI colour codes (safe subset — works in every modern terminal
# including Windows Terminal, iTerm2, kitty, alacritty).
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_SUCCESS_ARROW = "\u25B8"   # ▸
_ERROR_ARROW = "\u2716"     # ✖


def _format_input(tool_input: object) -> str:
    """Summarise tool input as a short ``k=v`` string.

    Long values (>80 chars) are truncated with an ellipsis so the
    call line stays readable on narrow terminals. Non-dict inputs
    fall back to ``repr`` — defensive against malformed blocks.
    """
    if not isinstance(tool_input, dict):
        return repr(tool_input)
    if not tool_input:
        return ""
    parts = []
    for key, value in tool_input.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, separators=(",", ":"))
        else:
            rendered = str(value)
        if len(rendered) > 80:
            rendered = rendered[:77] + "..."
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


def ansi_render_tool_call(
    tool_use: dict,
    tool_result: dict,
    *,
    colour: Optional[bool] = None,
) -> str:
    """Return a one-line ANSI-coloured render of a tool call + result.

    The rendered line has the shape::

        ▸ Read file_path=/tmp/x
        ✖ Bash command=rm -rf / (blocked by guard)

    ``colour=None`` (default) enables ANSI only if ``sys.stdout`` is
    a TTY. Pass ``colour=True`` to force colour (e.g. when piping to
    a colour-aware pager) or ``colour=False`` for plain text (log
    files, tests).

    ``tool_use`` and ``tool_result`` follow the shapes produced by
    :func:`~lib.hal.tool_adapter.parse_tool_use_blocks` and the
    tool-loop dispatcher — missing fields render as empty strings
    rather than raising, so partial payloads are renderable.
    """
    use_colour = colour if colour is not None else _stdout_is_tty()
    name = (tool_use.get("name") if isinstance(tool_use, dict) else "") or ""
    inp = tool_use.get("input") if isinstance(tool_use, dict) else None
    input_str = _format_input(inp)
    is_error = bool(
        tool_result.get("is_error")
        if isinstance(tool_result, dict)
        else False
    )

    if use_colour:
        arrow = f"{_RED}{_ERROR_ARROW}{_RESET}" if is_error else f"{_GREEN}{_SUCCESS_ARROW}{_RESET}"
        name_part = f"{_CYAN}{name}{_RESET}"
        input_part = f" {_DIM}{input_str}{_RESET}" if input_str else ""
    else:
        arrow = _ERROR_ARROW if is_error else _SUCCESS_ARROW
        name_part = name
        input_part = f" {input_str}" if input_str else ""

    line = f"{arrow} {name_part}{input_part}"

    if is_error:
        reason = tool_result.get("content", "") if isinstance(tool_result, dict) else ""
        reason_str = str(reason).splitlines()[0] if reason else ""
        if reason_str:
            if use_colour:
                line += f" {_DIM}({reason_str}){_RESET}"
            else:
                line += f" ({reason_str})"

    return line


def _stdout_is_tty() -> bool:
    """Return True if ``sys.stdout`` is a TTY.

    Isolated so tests can patch it without touching real stdout and
    so the ANSI-aware branch of :func:`ansi_render_tool_call` stays
    a single-argument function even without injection gymnastics.
    """
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def send_with_tools_to_cli(
    conversation: Conversation,
    message: str,
    *,
    chat_fn: Callable[..., dict],
    tools: list[dict],
    dispatcher: ToolDispatcher,
    max_iterations: int = 10,
    stream: Optional[TextIO] = None,
    colour: Optional[bool] = None,
) -> ToolLoopResult:
    """Drive a tool-augmented conversation turn and render calls to a stream.

    Thin composition of :func:`~lib.hal.tool_conversation.send_with_tools`
    with an ``on_tool_call`` callback that writes
    :func:`ansi_render_tool_call` output to ``stream`` (default: stdout).
    This is the one-call entry point for CLIs that want the tool loop
    visible by default — no callback wiring required.

    ``stream=None`` defaults to :data:`sys.stdout`. Pass a ``StringIO``
    in tests, or a log-rotated file handle for production services.
    """
    target = stream if stream is not None else sys.stdout

    def _on_tool_call(tool_use: dict, tool_result: dict) -> None:
        line = ansi_render_tool_call(tool_use, tool_result, colour=colour)
        target.write(line + "\n")
        flush = getattr(target, "flush", None)
        if callable(flush):
            flush()

    return send_with_tools(
        conversation,
        message,
        chat_fn=chat_fn,
        tools=tools,
        dispatcher=dispatcher,
        max_iterations=max_iterations,
        on_tool_call=_on_tool_call,
    )


__all__ = [
    "ansi_render_tool_call",
    "send_with_tools_to_cli",
]
