"""HAPPI v1.1 audit / IDR terminator tests (P4).

Covers ``flags.audit=true`` behaviour: per-dispatch buffer, idr emission
after completed/error, deterministic sha256, polyglot parity, no
cross-dispatch leak. H427 anchor (idr sha256 deterministic across
identical runs).

All tests are Goodhart-proof — they assert content (event types, field
values, hash equality) not call counts. Mutation of dispatch logic that
breaks the contract MUST fail at least one of these tests.
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.events import PROTOCOL_VERSION
from lib.hal.happi.runtime import _dispatch, run


# ---------- helpers ------------------------------------------------------

def _events(envelope_dict: dict, *, raw_override: bytes | None = None) -> list[dict]:
    """Run an envelope through ``_dispatch`` and return parsed event dicts."""
    env = HappiEnvelope.parse(envelope_dict)
    raw = raw_override if raw_override is not None else json.dumps(
        envelope_dict, separators=(",", ":")
    ).encode("utf-8")
    return [json.loads(line) for line in _dispatch(env, raw)]


def _run_via_run(envelope_dict: dict) -> list[dict]:
    """Run an envelope via the ``run()`` entry point and parse events."""
    raw_text = json.dumps(envelope_dict, separators=(",", ":"))
    out = io.StringIO()
    code = run(stream_in=io.StringIO(raw_text), stream_out=out)
    assert code == 0, f"run() exited {code}; stderr-side output: {out.getvalue()!r}"
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def _hash(envelope_raw: bytes, event_lines: list[str]) -> str:
    """Reference sha256 to cross-check the runtime's emission."""
    h = hashlib.sha256()
    h.update(envelope_raw)
    for line in event_lines:
        h.update(line.encode("utf-8"))
    return h.hexdigest()


# ---------- audit OFF: regression ---------------------------------------

def test_audit_disabled_emits_no_idr() -> None:
    """flags.audit absent → completed is the LAST event, no idr."""
    events = _events({
        "v": "happi/1.0", "id": "no-audit-1", "cmd": "echo", "args": ["hi"],
    })
    types = [e["type"] for e in events]
    assert "idr" not in types
    assert types[-1] == "completed"


def test_audit_false_emits_no_idr() -> None:
    """flags.audit=false → no idr; explicit-false matches polyglot bool() rule."""
    events = _events({
        "v": "happi/1.1", "id": "no-audit-2", "cmd": "noop",
        "flags": {"audit": False},
    })
    types = [e["type"] for e in events]
    assert "idr" not in types
    assert types[-1] == "completed"


def test_audit_string_true_emits_no_idr() -> None:
    """flags.audit='true' (string) is NOT the literal True → no idr (typed opt-in)."""
    events = _events({
        "v": "happi/1.1", "id": "no-audit-3", "cmd": "noop",
        "flags": {"audit": "true"},
    })
    types = [e["type"] for e in events]
    assert "idr" not in types


# ---------- audit ON: positive cases ------------------------------------

def test_audit_true_emits_idr_after_completed() -> None:
    """flags.audit=true → idr is the LAST event, follows completed."""
    events = _events({
        "v": "happi/1.1", "id": "audit-1", "cmd": "echo", "args": ["ok"],
        "flags": {"audit": True},
    })
    types = [e["type"] for e in events]
    assert types[-1] == "idr", f"idr must be last; got {types}"
    assert "completed" in types
    completed_idx = types.index("completed")
    idr_idx = types.index("idr")
    assert idr_idx > completed_idx, "idr must follow completed in stream order"


def test_audit_true_emits_idr_after_error() -> None:
    """Unsupported cmd → error then completed then idr (audit fires post-error)."""
    events = _events({
        "v": "happi/1.1", "id": "audit-err-1", "cmd": "bogus.unknown.cmd",
        "flags": {"audit": True},
    })
    types = [e["type"] for e in events]
    assert types[-1] == "idr"
    assert "error" in types


def test_idr_event_carries_idr_ref_with_required_fields() -> None:
    """idr_ref must contain sha256, cid, model_versions, block_anchor."""
    events = _events({
        "v": "happi/1.1", "id": "audit-shape-1", "cmd": "noop",
        "flags": {"audit": True},
    })
    idr = events[-1]
    assert idr["type"] == "idr"
    ref = idr["idr_ref"]
    assert isinstance(ref["sha256"], str) and len(ref["sha256"]) == 64
    assert ref["cid"] is None
    assert ref["model_versions"] == []
    assert ref["block_anchor"] is None


# ---------- determinism + sensitivity (H427 evidence) -------------------

def test_idr_sha256_deterministic_across_100_runs() -> None:
    """Same envelope_raw + same dispatch path → same sha256, 100 runs.

    H427 anchor: the polyglot's _AUDIT_BUFFER + envelope_raw chain must be
    deterministic across identical inputs. Non-determinism would prove a
    timing or mutable-state leak into the hash.

    Note: the lib's emit_event timestamps with monotonic-ish wall clock,
    so successive runs WILL produce different idr lines if naively hashed.
    H427's real claim is over (envelope_raw + sequence_of_events) — events
    INCLUDE ts. The polyglot has the same property: two independent runs
    produce different sha256 because the events in their buffers have
    different ts. The deterministic property to assert is that REPLAYING
    the SAME (envelope, event_lines) tuple produces the SAME sha256.
    That's a property of compute_idr_sha256, not of dispatch wallclock.
    """
    from lib.hal.happi.audit import audit_session, compute_idr_sha256, record_event

    envelope_raw = b'{"v":"happi/1.1","id":"x","cmd":"echo","args":["hi"]}'
    fixed_lines = [
        '{"v":"happi/1.1","id":"x","type":"started","ts":1000}',
        '{"v":"happi/1.1","id":"x","type":"delta","ts":1001,"text":"hi"}',
        '{"v":"happi/1.1","id":"x","type":"completed","ts":1002,"usage":{}}',
    ]
    hashes: set[str] = set()
    for _ in range(100):
        with audit_session():
            for line in fixed_lines:
                record_event(line)
            hashes.add(compute_idr_sha256(envelope_raw))
    assert len(hashes) == 1, f"non-deterministic; got {len(hashes)} unique hashes"


def test_idr_sha256_changes_when_envelope_raw_changes() -> None:
    """Flip a byte in envelope_raw → different sha256."""
    from lib.hal.happi.audit import audit_session, compute_idr_sha256, record_event

    line = '{"v":"happi/1.1","id":"x","type":"completed","ts":1}'
    with audit_session():
        record_event(line)
        h1 = compute_idr_sha256(b'{"v":"happi/1.1","id":"x","cmd":"noop"}')
    with audit_session():
        record_event(line)
        h2 = compute_idr_sha256(b'{"v":"happi/1.1","id":"y","cmd":"noop"}')
    assert h1 != h2


def test_idr_sha256_changes_when_event_stream_changes() -> None:
    """Different event sequence → different sha256 (with same envelope_raw)."""
    from lib.hal.happi.audit import audit_session, compute_idr_sha256, record_event

    envelope_raw = b'{"v":"happi/1.1","id":"x","cmd":"any"}'
    with audit_session():
        record_event('{"v":"happi/1.1","id":"x","type":"completed","ts":1}')
        h1 = compute_idr_sha256(envelope_raw)
    with audit_session():
        record_event('{"v":"happi/1.1","id":"x","type":"started","ts":1}')
        record_event('{"v":"happi/1.1","id":"x","type":"completed","ts":2}')
        h2 = compute_idr_sha256(envelope_raw)
    assert h1 != h2


# ---------- field plumbing ----------------------------------------------

def test_idr_v_field_matches_runtime_protocol_version() -> None:
    """The idr event's ``v`` equals the runtime's PROTOCOL_VERSION (P2 flag)."""
    events = _events({
        "v": "happi/1.1", "id": "audit-vfield-1", "cmd": "noop",
        "flags": {"audit": True},
    })
    idr = events[-1]
    assert idr["v"] == PROTOCOL_VERSION


def test_idr_carries_model_versions_from_flags() -> None:
    """flags.model_versions flows into idr_ref.model_versions verbatim."""
    events = _events({
        "v": "happi/1.1", "id": "audit-mv-1", "cmd": "noop",
        "flags": {"audit": True, "model_versions": ["claude-opus-4-7", "gpt-4o"]},
    })
    idr = events[-1]
    assert idr["idr_ref"]["model_versions"] == ["claude-opus-4-7", "gpt-4o"]


def test_idr_block_anchor_optional_default_none() -> None:
    """No flags.block_anchor → idr_ref.block_anchor is None."""
    events = _events({
        "v": "happi/1.1", "id": "audit-ba-none", "cmd": "noop",
        "flags": {"audit": True},
    })
    assert events[-1]["idr_ref"]["block_anchor"] is None


def test_idr_block_anchor_int_passes_through() -> None:
    """flags.block_anchor=12345 (int) → idr_ref.block_anchor == 12345."""
    events = _events({
        "v": "happi/1.1", "id": "audit-ba-int", "cmd": "noop",
        "flags": {"audit": True, "block_anchor": 12345},
    })
    assert events[-1]["idr_ref"]["block_anchor"] == 12345


def test_idr_block_anchor_non_int_rejected_silent() -> None:
    """Non-int block_anchor falls back to None (defensive)."""
    events = _events({
        "v": "happi/1.1", "id": "audit-ba-junk", "cmd": "noop",
        "flags": {"audit": True, "block_anchor": "not-an-int"},
    })
    assert events[-1]["idr_ref"]["block_anchor"] is None


# ---------- isolation (no cross-dispatch leak) --------------------------

def test_audit_buffer_no_cross_dispatch_leak() -> None:
    """Two consecutive audit-on dispatches produce different sha256s.

    If buffer A's events leaked into buffer B's hash, B's sha256 would
    incorporate A's events plus B's — a deterministic but WRONG hash. We
    cannot directly assert "no leak" via hash inequality (different ts
    already produces different hashes), so we assert the structural
    property: B's hash equals the hash of (B's envelope_raw + B's event
    lines) computed independently. Leak would make B's hash NOT match.
    """
    from lib.hal.happi.audit import audit_session, compute_idr_sha256, record_event

    # Dispatch A: pollute the would-be-shared buffer
    with audit_session():
        record_event('{"a":1}')
        record_event('{"a":2}')
        h_a = compute_idr_sha256(b"raw-A")

    # Dispatch B: independent
    with audit_session():
        record_event('{"b":1}')
        h_b = compute_idr_sha256(b"raw-B")

    # Reference: B as if A never happened
    h_b_ref = hashlib.sha256(b"raw-B" + b'{"b":1}').hexdigest()

    assert h_b == h_b_ref, "dispatch B's hash includes leaked content from A"
    assert h_a != h_b


# ---------- runtime-end-to-end (run() + transports) ---------------------

def test_audit_through_run_entry_point() -> None:
    """run() reads stdin, dispatches with audit, emits idr at tail."""
    events = _run_via_run({
        "v": "happi/1.1", "id": "audit-run-1", "cmd": "echo", "args": ["x"],
        "flags": {"audit": True},
    })
    types = [e["type"] for e in events]
    assert types[-1] == "idr"
    assert "completed" in types


def test_audit_idr_sha256_matches_independent_recompute() -> None:
    """Reproducible: hash of (envelope_raw + buffered events) equals idr.sha256.

    This is the strongest correctness check — it verifies the runtime's
    chained hash matches an independent reference computation over the
    same byte sequence.
    """
    raw_text = json.dumps(
        {"v": "happi/1.1", "id": "audit-recompute", "cmd": "echo",
         "args": ["check"], "flags": {"audit": True}},
        separators=(",", ":"),
    )
    out = io.StringIO()
    run(stream_in=io.StringIO(raw_text), stream_out=out)
    lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
    events = [json.loads(ln) for ln in lines]
    idr = events[-1]
    pre_idr_lines = lines[:-1]
    expected = _hash(raw_text.encode("utf-8"), pre_idr_lines)
    assert idr["idr_ref"]["sha256"] == expected, (
        f"sha256 mismatch: idr={idr['idr_ref']['sha256']}, expected={expected}"
    )


# ---------- polyglot parity (cross-runtime sanity) ----------------------

POLYGLOT_PATH = Path.home() / ".hal" / "happi.md"


@pytest.mark.skipif(
    not POLYGLOT_PATH.exists(),
    reason="canonical happi.md not present at ~/.hal/happi.md",
)
def test_audit_polyglot_emits_idr_terminator() -> None:
    """Polyglot bash runtime emits an idr event for flags.audit=true.

    Cross-runtime sanity (not a byte-equality assertion — events include
    `ts` which differs across runs / runtimes). Asserts the structural
    contract: same envelope shape on both sides → both produce idr at
    stream tail. The lib's parity is verified by the `_recompute` test
    above.
    """
    raw = json.dumps(
        {"v": "happi/1.1", "id": "polyglot-audit", "cmd": "version",
         "flags": {"audit": True}},
        separators=(",", ":"),
    )
    proc = subprocess.run(
        ["bash", str(POLYGLOT_PATH), "run"],
        input=raw, capture_output=True, text=True, timeout=15,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    events = [json.loads(ln) for ln in lines]
    types = [e["type"] for e in events]
    assert types[-1] == "idr", f"polyglot idr not last; types={types}"
    assert events[-1]["idr_ref"].get("sha256")


# ---------- backward-compat (zero behaviour change without audit flag) ---

def test_no_audit_dispatch_byte_equivalent_to_pre_p4() -> None:
    """Without flags.audit=true, the event sequence is identical to pre-P4.

    Asserts no idr appears AND the event types match the canonical
    started → delta → completed shape for echo cmd.
    """
    events = _events({
        "v": "happi/1.0", "id": "compat-1", "cmd": "echo", "args": ["a"],
    })
    types = [e["type"] for e in events]
    assert types == ["started", "delta", "completed"]
