"""Golden-envelope tests for HAPPI/1.0 α-1 runtime.

Each test feeds one JSON envelope via StringIO, captures stdout NDJSON,
and asserts the event sequence matches expectation. These are the seed
10 entries in the 50-envelope golden suite planned for α-4.

Anti-Goodhart discipline: every test asserts on output VALUE, not just
event count — a broken implementation that emits the right number of
events but wrong content will still fail.
"""
from __future__ import annotations

import io
import json

import pytest

from lib.hal.happi.envelope import EnvelopeError, HappiEnvelope
from lib.hal.happi.runtime import run
from tests._canonical_version import canonical_runtime_version


def _run_envelope(payload: dict) -> list[dict]:
    """Feed one envelope, return parsed NDJSON events in order."""
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    run(stream_in=stdin, stream_out=stdout)
    return [
        json.loads(line)
        for line in stdout.getvalue().splitlines()
        if line.strip()
    ]


def test_echo_command_emits_started_delta_completed():
    events = _run_envelope({
        "v": "happi/1.0",
        "id": "req-echo-1",
        "cmd": "echo",
        "args": ["hello", "world"],
    })
    assert [e["type"] for e in events] == ["started", "delta", "completed"]
    assert events[1]["text"] == "hello world"
    assert events[2]["usage"]["out_chars"] == len("hello world")


def test_echo_with_empty_args_emits_empty_delta():
    events = _run_envelope({
        "v": "happi/1.0",
        "id": "req-echo-empty",
        "cmd": "echo",
        "args": [],
    })
    assert [e["type"] for e in events] == ["started", "delta", "completed"]
    assert events[1]["text"] == ""


def test_noop_command_emits_started_completed_only():
    events = _run_envelope({
        "v": "happi/1.0",
        "id": "req-noop-1",
        "cmd": "noop",
    })
    assert [e["type"] for e in events] == ["started", "completed"]


def test_version_command_reports_runtime_protocol_version():
    """version cmd emits the runtime's PROTOCOL_VERSION as the delta text.

    Was the α-1 scaffold marker "happi/1.0 · α-1 scaffold"; post-P7 the runtime
    emits PROTOCOL_VERSION, aligning with polyglot canonical happi.md
    cmd_version which emits HAPPI_VERSION regardless of the envelope's v.

    The alignment target is DERIVED from the polyglot source rather than listed
    here. This assertion used to read `PROTOCOL_VERSION in {"happi/1.0",
    "happi/1.1"}` — a closed set that was complete when written and silently
    stopped being the alignment check it claimed to be once the canonical
    runtime moved past 1.1. Note that `delta["text"] == PROTOCOL_VERSION` alone
    proves only that the runtime uses its own constant; the independent witness
    is what makes this test able to fail for the reason it is named.
    """
    from lib.hal.happi.events import PROTOCOL_VERSION
    events = _run_envelope({
        "v": "happi/1.0",
        "id": "req-ver-1",
        "cmd": "version",
    })
    delta = next(e for e in events if e["type"] == "delta")
    assert delta["text"] == PROTOCOL_VERSION
    assert PROTOCOL_VERSION == canonical_runtime_version()


def test_unsupported_command_emits_error_then_completed():
    events = _run_envelope({
        "v": "happi/1.0",
        "id": "req-bad-cmd",
        "cmd": "brief",  # not in α-1 scaffold
    })
    types = [e["type"] for e in events]
    assert types == ["started", "error", "completed"]
    error = events[1]
    assert error["code"] == "unsupported_cmd"
    assert "brief" in error["message"]


def test_invalid_json_emits_parse_error_exit_1():
    stdin = io.StringIO("this is not json")
    stdout = io.StringIO()
    rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 1
    events = [
        json.loads(line)
        for line in stdout.getvalue().splitlines() if line.strip()
    ]
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "parse_error"
    assert events[0]["id"] == "req-invalid"


def test_wrong_protocol_version_rejected():
    stdin = io.StringIO(json.dumps({
        "v": "happi/0.9",
        "id": "req-old",
        "cmd": "echo",
    }))
    stdout = io.StringIO()
    rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 1
    events = [
        json.loads(line)
        for line in stdout.getvalue().splitlines() if line.strip()
    ]
    assert events[0]["type"] == "error"
    assert "happi/0.9" in events[0]["message"]


def test_missing_id_rejected():
    stdin = io.StringIO(json.dumps({
        "v": "happi/1.0",
        "cmd": "noop",
    }))
    stdout = io.StringIO()
    rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 1


def test_missing_cmd_rejected():
    stdin = io.StringIO(json.dumps({
        "v": "happi/1.0",
        "id": "req-no-cmd",
    }))
    stdout = io.StringIO()
    rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 1


def test_empty_stdin_rejected():
    stdin = io.StringIO("")
    stdout = io.StringIO()
    rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 1


def test_envelope_dataclass_roundtrip_preserves_fields():
    payload = {
        "v": "happi/1.0",
        "id": "req-rt",
        "cmd": "echo",
        "args": ["x", "y"],
        "flags": {"stream": True, "provider": "claude-opus-4-7"},
        "auth": {"scheme": "subscription", "token": "env:CLAUDE_CODE_OAUTH_TOKEN"},
        "ctx": {"cwd": "/tmp", "timezone": "Africa/Johannesburg"},
        "capabilities": ["read:stdin", "stream:stdout"],
    }
    env = HappiEnvelope.parse(payload)
    assert env.v == "happi/1.0"
    assert env.id == "req-rt"
    assert env.cmd == "echo"
    assert env.args == ("x", "y")
    assert env.flags == {"stream": True, "provider": "claude-opus-4-7"}
    assert env.auth["scheme"] == "subscription"
    assert env.capabilities == ("read:stdin", "stream:stdout")


def test_envelope_rejects_non_object():
    with pytest.raises(EnvelopeError):
        HappiEnvelope.parse("[1, 2, 3]")


def test_envelope_rejects_non_string_args():
    with pytest.raises(EnvelopeError):
        HappiEnvelope.parse({
            "v": "happi/1.0",
            "id": "req-x",
            "cmd": "echo",
            "args": "not-a-list",
        })


def test_all_events_carry_protocol_version_and_id_and_ts():
    """Emitted events stamp the runtime's PROTOCOL_VERSION, regardless of the
    envelope's declared client version.

    The envelope still echoes v="happi/1.0" — that is the client's
    declaration. The runtime's emission carries its own version string,
    matching the canonical polyglot runtime's behaviour (happi.md:60-68).

    That last clause is the actual claim, so it is what the assertion now
    checks: the expected stamp is read from the polyglot's own declaration
    instead of being written down as "happi/1.1", which was true when this was
    written and became a demand for the stale value three releases later.
    """
    events = _run_envelope({
        "v": "happi/1.0",
        "id": "req-v",
        "cmd": "echo",
        "args": ["test"],
    })
    for e in events:
        assert e["v"] == canonical_runtime_version()
        assert e["id"] == "req-v"
        assert isinstance(e["ts"], int)
        assert e["ts"] > 0


def test_events_emit_as_single_line_ndjson():
    """Each event line must be a single JSON object — no pretty-printing,
    no embedded newlines — so NDJSON readers can split on \\n safely."""
    stdin = io.StringIO(json.dumps({
        "v": "happi/1.0",
        "id": "req-nd",
        "cmd": "echo",
        "args": ["with\nembedded newline"],
    }))
    stdout = io.StringIO()
    run(stream_in=stdin, stream_out=stdout)
    for line in stdout.getvalue().splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        assert isinstance(parsed, dict)
