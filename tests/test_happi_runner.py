"""HAPPI γ-2 queue driver tests.

Exercises the queue driver at three layers:

  1. `extract_queue_lines(body)` — drops empty lines + `//` comments,
     preserves order.
  2. `run_envelope_line(raw)` — dispatches a single envelope or emits
     a parse_error event without raising.
  3. `run_queue_file(path)` — end-to-end: reads a polyglot plan,
     extracts the ndjson block, dispatches every envelope in order,
     returns NDJSON events.

Fixture `tests/happi_golden/queue_demo.md` supplies a realistic
polyglot with three envelopes (echo / noop / version) to verify
event-ordering guarantees.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from lib.hal.happi.runner import (
    QUEUE_LABEL,
    extract_queue_lines,
    main,
    run_envelope_line,
    run_queue_file,
    run_queue_lines,
)


FIXTURE_DIR = Path(__file__).parent / "happi_fixtures"
DEMO_PLAN = FIXTURE_DIR / "queue_demo.md"


def _parse_ndjson(lines_iter) -> list[dict]:
    return [json.loads(line) for line in lines_iter]


# --- extract_queue_lines ----------------------------------------------------

def test_extract_drops_empty_lines():
    body = '{"a":1}\n\n\n{"b":2}\n'
    assert extract_queue_lines(body) == ['{"a":1}', '{"b":2}']


def test_extract_drops_double_slash_comments():
    body = (
        "// This is a comment\n"
        '{"a":1}\n'
        "// Another comment\n"
        '{"b":2}\n'
    )
    assert extract_queue_lines(body) == ['{"a":1}', '{"b":2}']


def test_extract_preserves_order():
    body = '{"id":"x"}\n{"id":"y"}\n{"id":"z"}\n'
    assert [json.loads(line)["id"] for line in extract_queue_lines(body)] == [
        "x", "y", "z",
    ]


def test_extract_strips_whitespace():
    body = '   {"id":"x"}   \n  \t\n{"id":"y"}\n'
    assert extract_queue_lines(body) == ['{"id":"x"}', '{"id":"y"}']


# --- run_envelope_line ------------------------------------------------------

def test_run_envelope_line_dispatches_echo():
    raw = '{"v":"happi/1.0","id":"e1","cmd":"echo","args":["hi"]}'
    events = _parse_ndjson(run_envelope_line(raw))
    assert [e["type"] for e in events] == ["started", "delta", "completed"]
    assert events[0]["id"] == "e1"
    assert events[1]["text"] == "hi"


def test_run_envelope_line_dispatches_noop():
    raw = '{"v":"happi/1.0","id":"n1","cmd":"noop"}'
    events = _parse_ndjson(run_envelope_line(raw))
    assert [e["type"] for e in events] == ["started", "completed"]


def test_run_envelope_line_bad_json_yields_parse_error():
    events = _parse_ndjson(run_envelope_line("not-json"))
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "parse_error"
    assert events[0]["id"] == "queue-invalid"


def test_run_envelope_line_wrong_version_yields_parse_error():
    raw = '{"v":"happi/9.9","id":"x","cmd":"noop"}'
    events = _parse_ndjson(run_envelope_line(raw))
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "parse_error"


# --- run_queue_lines --------------------------------------------------------

def test_run_queue_dispatches_envelopes_in_order():
    lines = [
        '{"v":"happi/1.0","id":"q-1","cmd":"echo","args":["a"]}',
        '{"v":"happi/1.0","id":"q-2","cmd":"noop"}',
        '{"v":"happi/1.0","id":"q-3","cmd":"version"}',
    ]
    events = _parse_ndjson(run_queue_lines(lines))

    # Every envelope's events are a contiguous block; ids stay together.
    ids_in_order = [e["id"] for e in events]
    # Expected: q-1 (3 events), q-2 (2), q-3 (3)
    assert ids_in_order[0:3] == ["q-1"] * 3
    assert ids_in_order[3:5] == ["q-2"] * 2
    assert ids_in_order[5:8] == ["q-3"] * 3


def test_run_queue_isolates_per_envelope_errors():
    """A bad envelope yields a parse_error but does not halt the queue."""
    lines = [
        '{"v":"happi/1.0","id":"good-1","cmd":"noop"}',
        "not-json-at-all",
        '{"v":"happi/1.0","id":"good-2","cmd":"noop"}',
    ]
    events = _parse_ndjson(run_queue_lines(lines))

    types = [e["type"] for e in events]
    # good-1: started, completed (2)
    # bad:    error (1, id=queue-invalid)
    # good-2: started, completed (2)
    assert types == ["started", "completed", "error", "started", "completed"]
    assert events[2]["id"] == "queue-invalid"
    assert events[3]["id"] == "good-2"


def test_run_queue_empty_list_yields_no_events():
    events = list(run_queue_lines([]))
    assert events == []


# --- run_queue_file end-to-end ---------------------------------------------

def test_run_queue_file_dispatches_polyglot_demo():
    assert DEMO_PLAN.exists(), f"fixture missing: {DEMO_PLAN}"
    events = _parse_ndjson(run_queue_file(str(DEMO_PLAN)))

    # Three envelopes: echo/noop/version → 3 + 2 + 3 = 8 events total.
    ids = [e["id"] for e in events]
    assert ids == ["demo-1"] * 3 + ["demo-2"] * 2 + ["demo-3"] * 3

    types = [e["type"] for e in events]
    assert types == [
        "started", "delta", "completed",  # demo-1 echo
        "started", "completed",           # demo-2 noop
        "started", "delta", "completed",  # demo-3 version
    ]
    # Post-P7: demo-3's version delta carries PROTOCOL_VERSION
    # (was the α-1 scaffold marker pre-cutover).
    from lib.hal.happi.events import PROTOCOL_VERSION
    assert events[6]["text"] == PROTOCOL_VERSION


def test_run_queue_file_missing_block_yields_queue_missing():
    # Write a polyglot doc with NO ndjson fence.
    import tempfile
    text = "# just prose\nNo queue here.\n\n```python\nprint('hi')\n```\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(text)
        path = f.name

    try:
        events = _parse_ndjson(run_queue_file(path))
    finally:
        Path(path).unlink(missing_ok=True)

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "queue_missing"


# --- CLI main ---------------------------------------------------------------

def test_main_writes_ndjson_to_stdout_for_demo_plan():
    assert DEMO_PLAN.exists()
    stream = io.StringIO()
    rc = main([str(DEMO_PLAN)], stream_out=stream)
    assert rc == 0

    lines = [l for l in stream.getvalue().splitlines() if l.strip()]
    events = [json.loads(l) for l in lines]
    assert [e["type"] for e in events] == [
        "started", "delta", "completed",
        "started", "completed",
        "started", "delta", "completed",
    ]


def test_main_returns_1_on_missing_plan():
    stream = io.StringIO()
    rc = main(["/nonexistent/path/to/plan.md"], stream_out=stream)
    assert rc == 1
    events = [json.loads(l) for l in stream.getvalue().splitlines() if l.strip()]
    assert events[0]["code"] == "queue_missing"


def test_main_accepts_custom_label(tmp_path: Path):
    """--label flag should honour alternate fence labels."""
    custom = tmp_path / "custom_label.md"
    custom.write_text(
        '# custom\n\n```custom-queue\n'
        '{"v":"happi/1.0","id":"cl-1","cmd":"noop"}\n'
        '```\n',
        encoding="utf-8",
    )
    stream = io.StringIO()
    rc = main([str(custom), "--label", "custom-queue"], stream_out=stream)
    assert rc == 0
    events = [json.loads(l) for l in stream.getvalue().splitlines() if l.strip()]
    assert [e["type"] for e in events] == ["started", "completed"]
    assert events[0]["id"] == "cl-1"


# --- Default label constant ------------------------------------------------

def test_queue_label_default_is_ndjson():
    """Polyglot convention: the queue block is fenced as `ndjson`."""
    assert QUEUE_LABEL == "ndjson"
