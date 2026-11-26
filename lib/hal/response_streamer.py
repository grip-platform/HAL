"""HAL Response Streamer — token-by-token response delivery.

Wraps LLM responses for streaming delivery to clients via SSE,
WebSocket, or callback. Handles buffering and flush control.

Usage:
    from lib.hal.response_streamer import ResponseStreamer
    streamer = ResponseStreamer()
    for chunk in streamer.stream(response_text, chunk_size=10):
        send_to_client(chunk)
"""
import os
import sys
import time
from dataclasses import dataclass
from typing import TextIO


@dataclass(frozen=True)
class StreamChunk:
    text: str
    index: int
    is_final: bool
    elapsed_ms: float = 0.0


class ResponseStreamer:
    """Streams responses in chunks."""

    def stream(self, text: str, chunk_size: int = 50) -> list[StreamChunk]:
        """Split text into streaming chunks."""
        if not text:
            return [StreamChunk("", 0, True)]
        words = text.split()
        chunks: list[StreamChunk] = []
        start = time.monotonic()
        for i in range(0, len(words), chunk_size):
            batch = " ".join(words[i:i + chunk_size])
            is_final = (i + chunk_size) >= len(words)
            elapsed = (time.monotonic() - start) * 1000
            chunks.append(StreamChunk(batch, len(chunks), is_final, elapsed))
        return chunks

    def stream_chars(self, text: str, chars_per_chunk: int = 100) -> list[StreamChunk]:
        """Stream by character count."""
        if not text:
            return [StreamChunk("", 0, True)]
        chunks: list[StreamChunk] = []
        for i in range(0, len(text), chars_per_chunk):
            batch = text[i:i + chars_per_chunk]
            is_final = (i + chars_per_chunk) >= len(text)
            chunks.append(StreamChunk(batch, len(chunks), is_final))
        return chunks

    def render_to_stdout(
        self,
        text: str,
        *,
        chars_per_chunk: int = 20,
        delay_ms: int = 20,
        caret: str = "\033[2m\u258a\033[0m",
        stream: TextIO | None = None,
    ) -> int:
        """Incrementally render ``text`` to ``stream`` (default stdout).

        Returns the number of ``write`` calls issued (useful for tests).

        Behaviour:
          * TTY sink: chunk-by-chunk writes with flush, a dimmed caret
            painted at the cursor between chunks and erased before the
            next, and an optional inter-chunk sleep for visible cadence.
          * Non-TTY sink (pipe, file, tests, ``HAL_NO_STREAM=1``): a
            single atomic write — preserves existing block rendering.

        The caret is suppressed when a chunk ends in a newline (the
        cursor moves to column 0, where ``\\b`` is a no-op and would
        leave visible garbage behind).
        """
        out = stream if stream is not None else sys.stdout
        writes = 0
        if not text:
            return writes

        env_disabled = os.environ.get("HAL_NO_STREAM", "").lower() in ("1", "true", "yes")
        is_tty = getattr(out, "isatty", lambda: False)()

        if env_disabled or not is_tty:
            out.write(text)
            out.flush()
            return 1

        for chunk in self.stream_chars(text, chars_per_chunk):
            out.write(chunk.text)
            writes += 1
            if chunk.is_final or chunk.text.endswith("\n"):
                out.flush()
                if not chunk.is_final and delay_ms > 0:
                    time.sleep(delay_ms / 1000)
                continue
            if caret:
                out.write(caret)
                writes += 1
            out.flush()
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)
            if caret:
                out.write("\b \b")
                writes += 1
        return writes
