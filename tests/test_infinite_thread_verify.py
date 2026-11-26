"""Verification uses a fixed head while a real writer advances the thread."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from lib.hal.infinite_thread import anchor_heads, canonical
from tests.test_infinite_thread import (
    _disabled_anchor,
    _fields,
    _lines,
    _store,
    _thread,
    _verify,
    _write_lines,
    backend,  # noqa: F401 -- shared parametrised signing fixture
)


@pytest.mark.parametrize("append_before_read", [False, True])
@pytest.mark.parametrize("publish_snapshot", [False, True])
def test_verify_during_concurrent_append(backend, monkeypatch, append_before_read,
                                         publish_snapshot):
    path = _thread(backend, rows=3)
    anchor_heads(backend.root, anchor=_disabled_anchor)
    store = _store(backend)
    ready, done = threading.Event(), threading.Event()
    errors = []
    reader = threading.current_thread()
    read_bytes = Path.read_bytes

    def append():
        try:
            assert ready.wait(10), "verification never reached the row read"
            result = store.append("served", _fields(4), session_id="sess-main")
            assert result.written, result.reason
            if publish_snapshot:
                anchor_heads(backend.root, anchor=_disabled_anchor)
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    def advance_writer():
        ready.set()
        assert done.wait(10), "writer did not finish"
        assert not errors, errors

    def read_with_append(file):
        if file != path or threading.current_thread() is not reader:
            return read_bytes(file)
        if append_before_read:
            advance_writer()
        data = read_bytes(file)
        if not append_before_read:
            advance_writer()
        return data

    worker = threading.Thread(target=append, daemon=True)
    worker.start()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", read_with_append)
            verdict = _verify(backend, path)
    finally:
        ready.set()
        worker.join(timeout=10)
    assert not worker.is_alive()
    assert not errors, errors
    assert verdict.ok, verdict
    assert verdict.rows == 3, verdict  # exactly the head captured before the read
    assert verdict.head == "matches row 3 of 3"
    # The append really reached disk and verifies on the next call.
    later = _verify(backend, path)
    assert later.ok and later.rows == 4, later


@pytest.mark.parametrize("keep", [1, 2])
def test_truncation_below_captured_head_is_rejected(backend, keep):
    path = _thread(backend, rows=3)
    _write_lines(path, _lines(path)[:keep])
    verdict = _verify(backend, path)
    assert not verdict.ok, verdict
    assert verdict.rows == keep and verdict.line == keep + 1
    assert verdict.head == "ahead" and "rows were cut" in verdict.reason


def test_an_unterminated_tail_past_the_head_is_outside_the_check(backend):
    # An append in progress: no checkpoint vouches for it yet.
    path = _thread(backend, rows=3)
    with path.open("ab") as stream:
        stream.write(b"unfinished")
    verdict = _verify(backend, path)
    assert verdict.ok and verdict.rows == 3, verdict


def test_a_complete_invalid_row_past_the_head_fails(backend):
    # Inverted from the first cut of this test, which let a complete invalid
    # row past the head verify: every complete row is checked now.
    path = _thread(backend, rows=3)
    with path.open("ab") as stream:
        stream.write(b"invalid row\n")
    verdict = _verify(backend, path)
    assert not verdict.ok and verdict.line == 4, verdict
    assert verdict.reason == "the row is not JSON", verdict


def test_an_unterminated_row_the_head_vouches_for_is_a_torn_write(backend):
    # The writer fsyncs a row before it moves the head, so a row the captured
    # head covers was complete on disk: an unterminated one is a torn write,
    # never an append in progress.
    path = _thread(backend, rows=3)
    path.write_bytes(path.read_bytes()[:-1])
    verdict = _verify(backend, path)
    assert not verdict.ok and verdict.line == 3, verdict
    assert "torn write" in verdict.reason, verdict


def _row_past_the_head(backend, fields: dict) -> Path:
    """Three vouched rows plus a fourth the head does not cover: written and
    fsynced, with the head left at row 3, as a crash between the two steps
    leaves it, or as anything other than the writer could leave it."""
    path = _thread(backend, rows=3)
    head_file = path.with_suffix(".head")
    head_at_three = head_file.read_bytes()
    result = _store(backend).append("served", fields, session_id="sess-main")
    assert result.written, result.reason
    head_file.write_bytes(head_at_three)
    return path


def test_a_valid_row_past_the_head_still_verifies(backend):
    verdict = _verify(backend, _row_past_the_head(backend, _fields(4)))
    assert verdict.ok, verdict
    assert verdict.rows == 3 and verdict.head == "matches row 3 of 3", verdict


def test_a_row_past_the_head_with_an_inflated_price_fails(backend):
    # The attack: infinite_cost.reconcile sums every row and tells its reader
    # to run verify_thread first, so a well-formed row past the head whose
    # price was raised after signing must fail verification, not be counted.
    path = _row_past_the_head(backend, {**_fields(4), "list_price_usd": "0.00012000"})
    lines = _lines(path)
    row = json.loads(lines[3])
    row["list_price_usd"] = "999.00000000"
    _write_lines(path, lines[:3] + [canonical(row)])
    verdict = _verify(backend, path)
    assert not verdict.ok, verdict
    assert verdict.line == 4 and "signature" in verdict.reason, verdict
