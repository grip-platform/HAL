"""HAL cross-wire STREAMING translation — OpenAI Chat SSE -> Anthropic Messages SSE.

The token-incremental half of cross-wire (Phase 0 wave 4c). Where
``gateway_crosswire.anthropic_message_to_sse_chunks`` re-emits a COMPLETE buffered
response as one Anthropic SSE burst, this consumes a provider's OpenAI Chat
Completions SSE stream delta-by-delta and translates each chunk into Anthropic
Messages SSE events as they arrive — so a Claude-Code client sees gpt/gemini/etc.
tokens stream progressively.

It is the streaming INVERSE of ``gateway_openai`` (which streams Anthropic ->
OpenAI). It mirrors the proven state-machine in
``gateway_responses.translate_responses_stream`` and reuses the shared SSE parser
``gateway_openai._iter_sse_events`` plus the frame/format helpers in
``gateway_crosswire`` (DRY).

The hard 20% is tool-call streaming: OpenAI streams ``tool_calls`` fragmented (the
id + function.name arrive in the first delta for a given tool-call index; the
``function.arguments`` JSON arrives as fragments across later deltas). Anthropic
represents each as a ``tool_use`` content block whose input streams via
``input_json_delta`` partials — so we accumulate per OpenAI tool-call index and map
each to its own Anthropic block. NEVER used for ``claude-*`` (verbatim, #3274); the
caller gates on ``model_route`` + ``HAL_GATEWAY_MULTIPROVIDER`` first.
"""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional

from lib.hal.gateway_crosswire import (
    _OPENAI_FINISH_TO_ANTHROPIC,
    _message_delta_event,
    _message_start_event,
    _sse_event,
)
from lib.hal.gateway import anthropic_stream_error
from lib.hal.gateway_openai import guarded_sse_events, upstream_error_text


def translate_chat_stream(byte_chunks: Iterable[bytes], *, model: str = "") -> Iterator[bytes]:
    """Translate an OpenAI Chat Completions SSE byte stream into Anthropic
    Messages SSE event bytes (message_start -> per-block start/delta/stop ->
    message_delta -> message_stop). Order + shapes match Anthropic's streaming
    contract so a Claude-Code client parses it natively.

    A provider's in-band error (an ``{"error": ...}`` chunk) or a lost
    connection ends the stream with one Anthropic ``error`` event and no
    ``message_stop``: the client sees an interruption, never a short answer
    passed off as a whole one (fleet finding f72)."""
    state = _ChatStreamState(model=model)
    for chunk in guarded_sse_events(byte_chunks):
        if isinstance(chunk.get("error"), dict):
            yield anthropic_stream_error(upstream_error_text(chunk)[0])
            return
        yield from state.on_chunk(chunk)
    yield from state.finalize()


class _ChatStreamState:
    """Stateful OpenAI-Chat-delta -> Anthropic-SSE-event machine.

    Anthropic streams exactly one content block open at a time, indices strictly
    increasing. We therefore close the open block before opening the next, map a
    text run to one block and each OpenAI tool-call index to its own block, and
    accumulate output_tokens / stop_reason for the closing ``message_delta``.
    """

    def __init__(self, *, model: str):
        self._model = model
        self._started = False
        self._next_index = 0
        self._open_index: Optional[int] = None  # currently-open Anthropic block
        self._text_index: Optional[int] = None
        self._tool_index_map: Dict[int, int] = {}  # openai tool-call index -> anthropic block index
        self._finish: Optional[str] = None
        self._output_tokens = 0
        #: Fresh input + cache split from the provider's final usage chunk.
        #: Held rather than emitted on sight: OpenAI sends usage LAST, by which
        #: time ``message_start`` -- where Anthropic puts input_tokens -- has
        #: already gone out. Reporting it on the closing ``message_delta`` is
        #: the earliest honest place, and beats the 0 it used to report.
        self._usage: Dict[str, int] = {}

    # --- chunk handling ---

    def on_chunk(self, chunk: dict) -> Iterator[bytes]:
        choices = chunk.get("choices") or []
        choice = choices[0] if choices else {}
        delta = choice.get("delta") or {} if isinstance(choice, dict) else {}

        if not self._started:
            yield self._start_event(chunk)
            self._started = True

        content = delta.get("content")
        if isinstance(content, str) and content:
            yield from self._text_delta(content)

        for call in delta.get("tool_calls") or []:
            if isinstance(call, dict):
                yield from self._tool_delta(call)

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            # Same normaliser the buffered wires use, so the cached-token
            # correction is applied in ONE place rather than reimplemented here
            # (reimplementing it is how a streamed turn ends up billed ~10x).
            from lib.hal.gateway_crosswire import normalise_provider_usage
            if usage.get("completion_tokens") is not None:
                self._output_tokens = usage["completion_tokens"]
            self._absorb_usage(normalise_provider_usage(chunk))

        finish = choice.get("finish_reason") if isinstance(choice, dict) else None
        if finish:
            self._finish = finish

    def finalize(self) -> Iterator[bytes]:
        # A stream that produced nothing still needs a well-formed envelope.
        if not self._started:
            yield self._start_event({})
            self._started = True
        yield from self._close_open()
        yield _message_delta_event(
            {"stop_reason": _OPENAI_FINISH_TO_ANTHROPIC.get(self._finish or "", "end_turn"),
             "stop_sequence": None},
            self._closing_usage(),
        )
        yield _sse_event("message_stop", {"type": "message_stop"})

    def _absorb_usage(self, counts: Optional[Dict[str, int]]) -> None:
        """Merge one chunk's counts into the running total, never replacing.

        Replacing would let a later partial chunk that reports only
        ``completion_tokens`` overwrite the ``prompt_tokens_details`` an
        earlier chunk already carried, silently returning the input count to
        zero. Merging only non-zero fields means whichever chunk actually
        carried a number wins, in any order the provider sends them.
        """
        if not counts:
            return
        merged = dict(self._usage)
        merged.update({k: v for k, v in counts.items() if v})
        self._usage = merged

    def _closing_usage(self) -> Dict[str, int]:
        """Usage for the final ``message_delta``.

        ``output_tokens`` is always present (Anthropic's contract). The input
        and cache counts ride along only when the provider actually sent a
        usage block -- omitted otherwise, because a zero there would report a
        free turn where the truth is that nothing was measured.
        """
        usage = {"output_tokens": self._output_tokens}
        if self._usage:
            usage["input_tokens"] = self._usage.get("input_tokens", 0)
            for key in ("cache_read_tokens", "cache_write_tokens", "reasoning_tokens"):
                if self._usage.get(key):
                    usage[key] = self._usage[key]
        return usage

    # --- block transitions ---

    def _text_delta(self, text: str) -> Iterator[bytes]:
        if self._text_index is None or self._open_index != self._text_index:
            yield from self._close_open()
            self._text_index = self._next_index
            self._next_index += 1
            self._open_index = self._text_index
            yield _sse_event("content_block_start", {
                "type": "content_block_start", "index": self._text_index,
                "content_block": {"type": "text", "text": ""}})
        yield _sse_event("content_block_delta", {
            "type": "content_block_delta", "index": self._text_index,
            "delta": {"type": "text_delta", "text": text}})

    def _tool_delta(self, call: dict) -> Iterator[bytes]:
        tc_index = call.get("index", 0)
        fn = call.get("function") or {}
        if tc_index not in self._tool_index_map:
            yield from self._close_open()
            block_index = self._next_index
            self._next_index += 1
            self._tool_index_map[tc_index] = block_index
            self._open_index = block_index
            yield _sse_event("content_block_start", {
                "type": "content_block_start", "index": block_index,
                "content_block": {"type": "tool_use", "id": call.get("id", ""),
                                  "name": fn.get("name", ""), "input": {}}})
        block_index = self._tool_index_map[tc_index]
        args = fn.get("arguments")
        if args:
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta", "index": block_index,
                "delta": {"type": "input_json_delta", "partial_json": args}})

    def _close_open(self) -> Iterator[bytes]:
        if self._open_index is not None:
            yield _sse_event("content_block_stop",
                             {"type": "content_block_stop", "index": self._open_index})
            self._open_index = None

    def _start_event(self, chunk: dict) -> bytes:
        return _message_start_event(
            {"id": chunk.get("id", ""), "role": "assistant",
             "model": chunk.get("model") or self._model},
            {},
        )
