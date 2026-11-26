"""Tests for the real-time streaming TTS primitive (HAL welcome W1).

All tests inject a MockPopen factory so no real `say` process is spawned
— we only verify stdin writes, flush order, voice argv, and HAPPI event
interpretation.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.voice.stream_speak import (
    DEFAULT_RATE_WPM,
    DEFAULT_VOICE,
    StreamSpeakResult,
    _build_say_argv,
    stream_say,
    stream_say_envelope,
)
from lib.hal.voice.types import PlatformUnsupported
from tests._canonical_version import canonical_runtime_version


class _MockStdin:
    """Minimal file-like matching what subprocess.Popen gives when
    stdin=subprocess.PIPE. Records every write + flush call so tests
    can assert streaming order."""

    def __init__(self):
        self.writes: list[bytes] = []
        self.flushes: int = 0
        self.closed = False

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed = True


class _MockPopen:
    """Stand-in for subprocess.Popen. Records argv + stdin writes."""

    def __init__(self):
        self.invocations: list[list[str]] = []
        self.last_stdin: _MockStdin | None = None
        self.wait_called = False

    def __call__(self, argv: list[str], *, stdin=None):
        self.invocations.append(argv)
        self.last_stdin = _MockStdin()
        handle = MagicMock()
        handle.stdin = self.last_stdin
        handle.wait = self._wait
        return handle

    def _wait(self, timeout=None):
        self.wait_called = True
        return 0


def _make_ndjson_source(events: list[dict]) -> list[str]:
    return [json.dumps(e) for e in events]


# --- _build_say_argv --------------------------------------------------------

def test_build_argv_defaults_use_jamie_premium():
    assert _build_say_argv(DEFAULT_VOICE, None) == ["say", "-v", "Jamie (Premium)"]


def test_build_argv_adds_rate_flag_when_set():
    assert _build_say_argv("Samantha", 220) == [
        "say", "-v", "Samantha", "-r", "220",
    ]


# --- stream_say: envelope → iterator conversion ----------------------------

def test_stream_say_accepts_iterable_of_ndjson_lines_directly():
    popen = _MockPopen()
    events = _make_ndjson_source([
        {"v": "happi/1.0", "id": "r", "type": "started", "ts": 1},
        {"v": "happi/1.0", "id": "r", "type": "delta", "text": "hello ", "ts": 2},
        {"v": "happi/1.0", "id": "r", "type": "delta", "text": "world", "ts": 3},
        {"v": "happi/1.0", "id": "r", "type": "completed", "usage": {}, "ts": 4},
    ])
    result = stream_say(events, spawn=popen)
    assert result.spoken_text == "hello world"
    assert result.delta_count == 2
    assert result.error is None


def test_stream_say_raises_typeerror_on_non_iterable():
    popen = _MockPopen()
    with pytest.raises(TypeError):
        stream_say(42, spawn=popen)


def test_stream_say_accepts_happi_envelope_and_dispatches_it():
    """Envelope branch uses runtime._dispatch; verify end-to-end via echo."""
    popen = _MockPopen()
    env = HappiEnvelope.parse({
        "v": "happi/1.0", "id": "stream-echo-1",
        "cmd": "echo", "args": ["buffered", "tts"],
    })
    result = stream_say(env, spawn=popen)
    assert result.spoken_text == "buffered tts"
    assert result.delta_count == 1
    assert result.error is None


# --- Streaming behaviour ----------------------------------------------------

def test_each_delta_writes_to_stdin_in_order():
    popen = _MockPopen()
    events = _make_ndjson_source([
        {"v": "happi/1.0", "id": "r", "type": "started", "ts": 1},
        {"v": "happi/1.0", "id": "r", "type": "delta", "text": "Hel", "ts": 2},
        {"v": "happi/1.0", "id": "r", "type": "delta", "text": "lo, ", "ts": 3},
        {"v": "happi/1.0", "id": "r", "type": "delta", "text": "HAL.", "ts": 4},
        {"v": "happi/1.0", "id": "r", "type": "completed", "usage": {}, "ts": 5},
    ])
    stream_say(events, spawn=popen)
    writes = [w.decode("utf-8") for w in popen.last_stdin.writes]
    assert writes == ["Hel", "lo, ", "HAL."]


def test_each_write_is_flushed_for_streaming_latency():
    popen = _MockPopen()
    events = _make_ndjson_source([
        {"v": "happi/1.0", "id": "r", "type": "delta", "text": "A", "ts": 1},
        {"v": "happi/1.0", "id": "r", "type": "delta", "text": "B", "ts": 2},
        {"v": "happi/1.0", "id": "r", "type": "delta", "text": "C", "ts": 3},
    ])
    stream_say(events, spawn=popen)
    # One flush per delta write.
    assert popen.last_stdin.flushes == 3


def test_stdin_is_closed_after_last_event():
    popen = _MockPopen()
    events = _make_ndjson_source([
        {"v": "happi/1.0", "id": "r", "type": "delta", "text": "hi", "ts": 1},
    ])
    stream_say(events, spawn=popen)
    assert popen.last_stdin.closed is True
    assert popen.wait_called is True


def test_say_argv_built_with_voice_and_rate():
    popen = _MockPopen()
    stream_say([], spawn=popen, voice="Samantha", rate_wpm=200)
    assert popen.invocations == [["say", "-v", "Samantha", "-r", "200"]]


def test_default_voice_is_jamie_premium():
    popen = _MockPopen()
    stream_say([], spawn=popen)
    assert popen.invocations[0][:3] == ["say", "-v", "Jamie (Premium)"]


# --- Error + edge cases -----------------------------------------------------

def test_happi_error_event_stops_streaming_and_populates_result_error():
    popen = _MockPopen()
    events = _make_ndjson_source([
        {"v": "happi/1.0", "id": "r", "type": "delta", "text": "Pre-", "ts": 1},
        {"v": "happi/1.0", "id": "r", "type": "error",
         "code": "provider_http_429", "message": "rate limited", "ts": 2},
        {"v": "happi/1.0", "id": "r", "type": "delta", "text": "after", "ts": 3},
    ])
    result = stream_say(events, spawn=popen)
    assert result.spoken_text == "Pre-"
    assert result.delta_count == 1
    assert result.error is not None
    assert "provider_http_429" in result.error
    # "after" delta must NOT have been written — streaming stopped on error.
    writes = [w.decode("utf-8") for w in popen.last_stdin.writes]
    assert writes == ["Pre-"]


def test_malformed_ndjson_line_surfaces_as_soft_error():
    popen = _MockPopen()
    events = [
        json.dumps({"type": "delta", "text": "ok", "id": "r"}),
        "not-json-at-all",
        json.dumps({"type": "delta", "text": "after", "id": "r"}),
    ]
    result = stream_say(events, spawn=popen)
    assert result.spoken_text == "ok"
    assert result.error is not None
    assert "non-JSON" in result.error


def test_null_bytes_in_delta_text_are_stripped_not_propagated():
    popen = _MockPopen()
    events = _make_ndjson_source([
        {"v": "happi/1.0", "id": "r", "type": "delta", "text": "a\x00b", "ts": 1},
    ])
    stream_say(events, spawn=popen)
    writes = [w.decode("utf-8") for w in popen.last_stdin.writes]
    assert writes == ["ab"]
    assert "\x00" not in "".join(writes)


def test_empty_delta_text_is_a_noop_on_stdin():
    popen = _MockPopen()
    events = _make_ndjson_source([
        {"v": "happi/1.0", "id": "r", "type": "delta", "text": "", "ts": 1},
        {"v": "happi/1.0", "id": "r", "type": "delta", "text": "x", "ts": 2},
    ])
    stream_say(events, spawn=popen)
    writes = [w.decode("utf-8") for w in popen.last_stdin.writes]
    assert writes == ["x"]


def test_non_delta_events_do_not_write_to_stdin():
    popen = _MockPopen()
    events = _make_ndjson_source([
        {"v": "happi/1.0", "id": "r", "type": "started", "ts": 1},
        {"v": "happi/1.0", "id": "r", "type": "tool_call", "name": "x", "input": {},
         "call_id": "c1", "ts": 2},
        {"v": "happi/1.0", "id": "r", "type": "tool_result", "name": "x", "output": {},
         "call_id": "c1", "ts": 3},
        {"v": "happi/1.0", "id": "r", "type": "completed", "usage": {}, "ts": 4},
    ])
    result = stream_say(events, spawn=popen)
    assert popen.last_stdin.writes == []
    assert result.delta_count == 0
    assert result.spoken_text == ""


def test_platform_unsupported_raised_when_default_spawn_not_darwin(monkeypatch):
    """Using the real subprocess.Popen default on non-Darwin raises."""
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Linux")
    with pytest.raises(PlatformUnsupported):
        stream_say([])


def test_injected_spawn_bypasses_platform_check(monkeypatch):
    """Tests injecting their own spawn don't hit the darwin guard."""
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Linux")
    popen = _MockPopen()
    result = stream_say([], spawn=popen)
    assert result.error is None


def test_stream_say_envelope_convenience_wraps_stream_say():
    popen = _MockPopen()
    env = HappiEnvelope.parse({
        "v": "happi/1.0", "id": "conv-1", "cmd": "version",
    })
    result = stream_say_envelope(env, spawn=popen)
    # The version command speaks the runtime's own version, read from its
    # independent source (tests/_canonical_version.py), never a pinned literal.
    assert result.spoken_text == canonical_runtime_version()
    assert result.delta_count == 1


# --- Result type ------------------------------------------------------------

def test_result_is_frozen_and_carries_fields():
    result = StreamSpeakResult(spoken_text="hi", delta_count=1, error=None)
    assert result.spoken_text == "hi"
    assert result.delta_count == 1
    assert result.error is None
    with pytest.raises(Exception):
        result.spoken_text = "changed"  # type: ignore[misc]


def test_default_rate_is_none():
    assert DEFAULT_RATE_WPM is None
