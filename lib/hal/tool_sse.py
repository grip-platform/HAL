"""Server-Sent Events stream for the HAL tool-use loop (W8 integration).

Produces an iterator of SSE-formatted event strings for a tool-augmented
conversation turn, so a web server (FastAPI, Flask, or the stdlib
http.server) can pipe them straight to a browser. The grip-web
``ToolBlock.tsx`` component renders them as the model calls tools.

Design: real streaming, not batch-then-yield

``run_tool_loop`` is synchronous — collecting tool events into a list
and yielding them after the loop returns would produce an SSE stream
that arrives all-at-once, defeating the "live trace" UX. Instead, the
loop runs on a background thread and the callback pushes events to a
:class:`queue.Queue`; the generator yields from the queue as events
arrive, interleaving naturally with the loop's progress.

The generator is synchronous (ordinary ``yield``) rather than async so
it works with any HTTP framework's streaming-response abstraction —
FastAPI's ``StreamingResponse(iter_tool_events(...))``, Flask's
``Response(iter_tool_events(...), mimetype="text/event-stream")``,
aiohttp's ``StreamResponse`` with adaptor, and so on.

Event taxonomy (three types):

* ``tool_call``      — one tool fired; carries ``{tool_use, tool_result}``
* ``final``          — loop done; carries ``{text, iterations, hit_cap}``
* ``error``          — chat_fn / dispatcher raised; carries ``{message}``

Usage:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from lib.hal.tool_sse import iter_tool_events

    app = FastAPI()

    @app.post("/api/tools/stream")
    async def stream(req: ChatRequest):
        return StreamingResponse(
            iter_tool_events(
                conv, req.message,
                chat_fn=chat_fn,
                tools=tools,
                dispatcher=dispatcher,
            ),
            media_type="text/event-stream",
        )
"""
import json
import queue
import threading
from typing import Callable, Iterator, Optional

from lib.hal.conversation import Conversation
from lib.hal.tool_conversation import send_with_tools
from lib.hal.tool_dispatch import ToolDispatcher


_SENTINEL = object()


def format_sse_event(event_type: str, data: dict) -> str:
    """Return an SSE-formatted event string.

    The SSE wire format is one or more ``field: value`` lines terminated
    by a blank line::

        event: tool_call
        data: {"tool_use": {...}, "tool_result": {...}}

        event: final
        data: {"text": "done", "iterations": 3}

    Data is JSON-encoded on a single line — SSE requires each ``data:``
    line be payload-terminated, and embedded newlines in JSON would
    split one event across two. Compact separators (no extra whitespace)
    keep bytes on the wire minimal.
    """
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


def iter_tool_events(
    conversation: Conversation,
    message: str,
    *,
    chat_fn: Callable[..., dict],
    tools: list[dict],
    dispatcher: ToolDispatcher,
    max_iterations: int = 10,
    queue_timeout_s: float = 300.0,
) -> Iterator[str]:
    """Yield SSE event strings as the tool loop runs.

    Runs :func:`~lib.hal.tool_conversation.send_with_tools` on a
    background thread; the callback pushes ``tool_call`` events to a
    queue as each tool fires, and the generator yields them in real
    time. A final ``final`` event carries the assistant's prose answer
    and iteration metadata; an ``error`` event fires if the loop
    crashes.

    ``queue_timeout_s`` guards against a stuck thread — if the loop
    produces no events for that long, an ``error`` event is emitted
    and the generator returns. Default 300s matches Anthropic's
    per-request ceiling.

    The thread is a daemon — if the server drops the client connection
    mid-stream, the thread continues to completion and its final
    result is lost (the queue is garbage-collected). That's the
    correct behaviour for SSE: the connection disappearing cannot
    rewind the in-flight tool calls.
    """
    q: queue.Queue = queue.Queue()
    error_box: list[Optional[Exception]] = [None]

    def _on_tool_call(tool_use: dict, tool_result: dict) -> None:
        q.put(("tool_call", {
            "tool_use": tool_use,
            "tool_result": tool_result,
        }))

    def _run() -> None:
        try:
            result = send_with_tools(
                conversation,
                message,
                chat_fn=chat_fn,
                tools=tools,
                dispatcher=dispatcher,
                max_iterations=max_iterations,
                on_tool_call=_on_tool_call,
            )
            q.put(("final", {
                "text": conversation._last_response_text,
                "iterations": result.iterations,
                "hit_iteration_cap": result.hit_iteration_cap,
            }))
        except Exception as exc:
            error_box[0] = exc
            q.put(("error", {"message": str(exc)}))
        finally:
            q.put(_SENTINEL)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    while True:
        try:
            item = q.get(timeout=queue_timeout_s)
        except queue.Empty:
            yield format_sse_event(
                "error",
                {"message": f"tool loop timeout after {queue_timeout_s}s"},
            )
            return
        if item is _SENTINEL:
            return
        event_type, data = item
        yield format_sse_event(event_type, data)


__all__ = ["format_sse_event", "iter_tool_events"]
