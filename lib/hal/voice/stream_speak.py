"""Real-time streaming TTS — HAPPI delta events piped to macOS `say` via stdin.

This module bridges the gap between HAPPI's streaming event vocabulary
(``delta`` events arriving as the LLM generates) and audible speech
playback. The existing ``voice.speak`` primitive is batch-oriented: it
accepts a complete text string, synthesises a full audio file, then
plays the file. Total latency is LLM-generation-time + TTS-synthesis-time
+ audio-playback-time, sequentially.

``stream_say`` pipelines those stages by writing each LLM delta chunk
directly to ``say``'s stdin. macOS ``say`` streams from stdin — it begins
vocalising as soon as it has a speakable segment (roughly sentence-
boundaried), so the user hears the first word while the LLM is still
generating the rest of the reply.

End-to-end latency of a streamed reply:

    dispatch → first-HAPPI-delta (~400-800ms) → first-bytes-to-say-stdin
    → say buffers to first sentence (~100-300ms of text) → speak

First audible word is typically well under 1.5s on a warm connection.

Contract:

    stream_say(envelope_source, *, voice="Jamie (Premium)", rate_wpm=None,
               spawn=subprocess.Popen) -> StreamSpeakResult

- ``envelope_source`` is either a HappiEnvelope (which stream_say will
  dispatch via ``runtime._dispatch``) OR an already-started Iterator[str]
  of NDJSON event lines (useful for tests + for embedders who want to
  compose with ``_expand_sub_requests`` or ``run_queue_lines``).
- ``spawn`` is the subprocess factory — tests inject a mock; default is
  ``subprocess.Popen`` bound to ``say``.

Returns a ``StreamSpeakResult`` with:
    spoken_text : str — full concatenated text that was passed to say
    delta_count : int — number of HAPPI delta events consumed
    error       : str | None — if the HAPPI stream yielded an error event

Stdlib-only. macOS-only for the default backend (say is the Darwin
stdlib); cross-platform implementations ship behind the same interface
in a follow-up (Piper streaming is already available as a batch voice
provider at tier 3 per task #1317 — streaming Piper is a future atom).
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from typing import Callable, Iterator, Iterable

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.runtime import _dispatch
from lib.hal.voice.types import PlatformUnsupported


DEFAULT_VOICE = "Jamie (Premium)"
DEFAULT_RATE_WPM: int | None = None  # say's default


@dataclass(frozen=True)
class StreamSpeakResult:
    """Observable outcome of a stream_say run.

    ``error`` is non-None iff the HAPPI stream yielded an error event;
    the text buffered before the error is still preserved in
    ``spoken_text`` so callers can debug what partial audio the user
    heard.
    """
    spoken_text: str
    delta_count: int
    error: str | None


def _require_darwin() -> None:
    if platform.system() != "Darwin":
        raise PlatformUnsupported(
            "stream_say default backend uses macOS `say`; "
            f"current platform is {platform.system()}"
        )


def _build_say_argv(voice: str, rate_wpm: int | None) -> list[str]:
    argv = ["say", "-v", voice]
    if rate_wpm is not None:
        argv.extend(["-r", str(rate_wpm)])
    return argv


def _iter_events(envelope_source) -> Iterator[str]:
    """Normalise input: envelope → _dispatch output, else iterate as-is."""
    if isinstance(envelope_source, HappiEnvelope):
        yield from _dispatch(envelope_source)
    elif hasattr(envelope_source, "__iter__"):
        yield from envelope_source
    else:
        raise TypeError(
            f"stream_say expected HappiEnvelope or Iterable[str], "
            f"got {type(envelope_source).__name__}"
        )


def _write_chunk(proc, text: str) -> None:
    """Push one delta's text to say's stdin + flush.

    Unicode encoding uses utf-8; say reads utf-8 cleanly on modern
    macOS. We do not filter control bytes (say ignores them), but null
    bytes terminate say's stdin stream — if an LLM ever emits one we
    simply skip that delta.
    """
    if "\x00" in text:
        text = text.replace("\x00", "")
    if not text:
        return
    proc.stdin.write(text.encode("utf-8"))
    proc.stdin.flush()


def stream_say(
    envelope_source,
    *,
    voice: str = DEFAULT_VOICE,
    rate_wpm: int | None = DEFAULT_RATE_WPM,
    spawn: Callable = subprocess.Popen,
) -> StreamSpeakResult:
    """Pipe HAPPI delta events to ``say`` stdin; block until playback ends.

    Args:
        envelope_source: HappiEnvelope to dispatch OR pre-existing Iterable
            of NDJSON event lines.
        voice: say voice name. Default "Jamie (Premium)" per the voice
            harness default voice landed in PR #241.
        rate_wpm: words-per-minute override. None uses say's default.
        spawn: subprocess factory (injected for tests). Must support
            ``stdin=subprocess.PIPE`` and the returned proc must expose
            ``stdin.write/flush/close`` + ``wait``.

    Returns:
        ``StreamSpeakResult`` with full spoken text, delta count, and
        any HAPPI error (None on success).

    Raises:
        PlatformUnsupported: if not running on macOS Darwin with the
            default spawn=subprocess.Popen. Tests injecting a mock
            ``spawn`` bypass this check by supplying their own factory
            which is signalled by ``spawn is not subprocess.Popen``.
    """
    if spawn is subprocess.Popen:
        _require_darwin()

    argv = _build_say_argv(voice, rate_wpm)
    proc = spawn(argv, stdin=subprocess.PIPE)
    spoken_parts: list[str] = []
    delta_count = 0
    error: str | None = None

    try:
        for line in _iter_events(envelope_source):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Malformed NDJSON from the runtime shouldn't crash playback;
                # surface it as a soft error and stop reading more events.
                error = f"runtime emitted non-JSON line: {line!r}"
                break

            etype = event.get("type")
            if etype == "delta":
                text = str(event.get("text", ""))
                _write_chunk(proc, text)
                spoken_parts.append(text)
                delta_count += 1
            elif etype == "error":
                error = (
                    f"HAPPI {event.get('code', 'unknown')}: "
                    f"{event.get('message', 'provider error')}"
                )
                break
    finally:
        try:
            proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        proc.wait()

    return StreamSpeakResult(
        spoken_text="".join(spoken_parts),
        delta_count=delta_count,
        error=error,
    )


def stream_say_envelope(
    envelope: HappiEnvelope,
    **kwargs,
) -> StreamSpeakResult:
    """Convenience: dispatch a HappiEnvelope and stream-speak its deltas."""
    return stream_say(envelope, **kwargs)


__all__ = [
    "DEFAULT_RATE_WPM",
    "DEFAULT_VOICE",
    "StreamSpeakResult",
    "stream_say",
    "stream_say_envelope",
]
