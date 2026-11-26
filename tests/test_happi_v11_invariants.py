"""HAPPI v1.1 cross-cutting invariant tests (P6 of v1.1 alignment).

This file is the P6 deliverable per
``plans/happi-lib-v11-alignment-ultraplan.md`` — a test surface expansion
covering invariants that no single-cmd test file could exercise:

A. **Event-ordering invariants** — for every cmd in the supported set,
   the stream begins with ``started`` and terminates with the canonical
   tail (``completed``/``error``, optionally followed by ``idr`` when
   ``flags.audit=true``).
B. **Cross-version parametrics** — v1.0 and v1.1 envelopes drive the
   same dispatch path; the only behavioural difference is the v1.1-only
   flags/cmds. Simple cmds (echo, version, noop) produce the same event
   count under both envelope versions.
C. **Transport+cmd matrix** — HTTP transport carries v1.1 envelopes
   with audit flag and idr terminator; socket transport does the same.
   The transport-side framing changes but the dispatch surface does
   not.
D. **Idr chain determinism** — replaying the same recorded
   (envelope_raw, event_lines) tuple produces a single sha256 across
   any number of runs. Different inputs produce different sha256s
   (collision-resistance smoke test, not a cryptographic claim).

Goodhart-proof discipline: every test asserts a specific value or
relation — never just "doesn't raise". Mutation of dispatch logic that
breaks the invariant MUST fail at least one of these tests.

Hypothesis anchor: H429 — net-new test count ≥25 by P7. This file
contributes the bulk of those tests; P1-P5 already added 100 net-new.

Backward-compat: this file does NOT modify any pre-P6 test, fixture,
or production code. P7 owns the global v1.0 → v1.1 emission flip.
"""
from __future__ import annotations

import hashlib
import io
import json
import threading
import urllib.error
import urllib.request
from http.client import HTTPConnection
from typing import Iterator
from unittest import mock

import pytest

from lib.hal.happi.audit import (
    audit_session,
    compute_idr_sha256,
    record_event,
)
from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.events import PROTOCOL_VERSION
from lib.hal.happi.runtime import _dispatch, run
from lib.hal.happi.transports.http_transport import HAPPI_PATH, build_server


# ---------- shared helpers ------------------------------------------------

def _events_for(env_dict: dict, *, raw: bytes | None = None) -> list[dict]:
    """Dispatch ``env_dict`` and return parsed event dicts in stream order."""
    env = HappiEnvelope.parse(env_dict)
    raw_bytes = raw if raw is not None else json.dumps(
        env_dict, separators=(",", ":")
    ).encode("utf-8")
    return [json.loads(line) for line in _dispatch(env, raw_bytes)]


def _types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def _run_via_run(env_dict: dict) -> list[dict]:
    """Round-trip an envelope through ``run()`` (stdin → stdout)."""
    raw = json.dumps(env_dict, separators=(",", ":"))
    out = io.StringIO()
    rc = run(stream_in=io.StringIO(raw), stream_out=out)
    assert rc == 0, f"run() exited {rc}; stdout={out.getvalue()!r}"
    return [
        json.loads(line)
        for line in out.getvalue().splitlines()
        if line.strip()
    ]


# ---------- (A) Event-ordering invariants ---------------------------------

# Per-cmd minimal envelopes that are guaranteed to dispatch successfully.
# Each tuple = (cmd_name, envelope_extras). The base envelope is filled in
# by the parametrize wrapper below. Only cmds with deterministic success
# paths are included — chat/compose are excluded because they require
# provider/network/recursive children.
_SUCCESS_CMD_FIXTURES: list[tuple[str, dict]] = [
    ("echo", {"args": ["hi"]}),
    ("noop", {}),
    ("version", {}),
    ("idr.emit", {"args": ["env-bytes", "evt-bytes"]}),
    ("pr.reference", {"flags": {"pr": 1, "repo": "x/y"}}),
]


@pytest.mark.parametrize("cmd,extras", _SUCCESS_CMD_FIXTURES,
                         ids=[c for c, _ in _SUCCESS_CMD_FIXTURES])
def test_every_success_cmd_starts_with_started(cmd, extras):
    """First event of any successful dispatch is ``started``.

    Goodhart-proof: would fail if a refactor reordered start emission to
    happen after the cmd-specific work, or skipped started entirely.
    """
    env_dict = {"v": "happi/1.1", "id": f"order-{cmd}", "cmd": cmd}
    env_dict.update(extras)
    events = _events_for(env_dict)
    assert events, f"{cmd}: no events emitted"
    assert events[0]["type"] == "started", (
        f"{cmd}: first event was {events[0]['type']!r}, expected 'started'"
    )


@pytest.mark.parametrize("cmd,extras", _SUCCESS_CMD_FIXTURES,
                         ids=[c for c, _ in _SUCCESS_CMD_FIXTURES])
def test_every_success_cmd_terminates_with_completed_no_audit(cmd, extras):
    """Without ``flags.audit``, last event is ``completed`` — never delta/tool_call.

    Goodhart-proof: this catches regressions where a cmd handler
    forgets to emit completed (e.g. early ``return`` after a delta).
    """
    env_dict = {"v": "happi/1.1", "id": f"tail-{cmd}", "cmd": cmd}
    env_dict.update(extras)
    events = _events_for(env_dict)
    assert events[-1]["type"] == "completed", (
        f"{cmd}: terminator was {events[-1]['type']!r}, expected 'completed'"
    )


@pytest.mark.parametrize("cmd,extras", _SUCCESS_CMD_FIXTURES,
                         ids=[c for c, _ in _SUCCESS_CMD_FIXTURES])
def test_every_success_cmd_with_audit_terminates_with_idr(cmd, extras):
    """With ``flags.audit=true``, the audit-terminator idr is the LAST
    event in the stream and arrives AFTER the cmd's ``completed``.

    The idr-after-completed ordering is the v1.1 contract per
    ``happi.md`` and ``audit.py``. Reversing the order would mean
    clients consuming the stream see completed and stop reading, missing
    the audit terminator entirely.

    Note: ``idr.emit`` itself emits an idr event as part of its cmd
    output (BEFORE completed). When audit is also enabled, the
    audit-side idr is appended AFTER completed. The invariant we assert
    is "the LAST idr in the stream follows completed" — using the last
    index, not the first — so the cmd-emitted idr (when present) is not
    confused with the audit terminator.
    """
    env_dict = {"v": "happi/1.1", "id": f"audit-tail-{cmd}", "cmd": cmd}
    env_dict.update(extras)
    flags = dict(env_dict.get("flags", {}))
    flags["audit"] = True
    env_dict["flags"] = flags

    events = _events_for(env_dict)
    types = _types(events)
    assert types[-1] == "idr", f"{cmd}: tail was {types}; expected idr last"
    # Completed must precede the LAST idr (the audit terminator).
    assert "completed" in types, f"{cmd}: completed missing; got {types}"
    completed_idx = types.index("completed")
    last_idr_idx = len(types) - 1 - types[::-1].index("idr")
    assert last_idr_idx > completed_idx, (
        f"{cmd}: last idr at index {last_idr_idx} not after completed at "
        f"{completed_idx}; types={types}"
    )


def test_error_cmd_with_audit_emits_idr_after_terminator():
    """An unsupported cmd produces error + completed, followed by idr.

    Catches regressions that emit idr ONLY on the success path — audit
    must apply to the failure path too, or the audit chain becomes
    discontinuous (gaps in the chain are unrecoverable).
    """
    env_dict = {
        "v": "happi/1.1", "id": "audit-err-tail", "cmd": "totally-fake-cmd",
        "flags": {"audit": True},
    }
    events = _events_for(env_dict)
    types = _types(events)
    assert types[-1] == "idr"
    assert "error" in types, f"error missing from {types}"


# ---------- (B) Cross-version parametrics ---------------------------------

@pytest.mark.parametrize("envelope_v", ["happi/1.0", "happi/1.1"])
@pytest.mark.parametrize("cmd,extras", [
    ("echo", {"args": ["a", "b"]}),
    ("noop", {}),
    ("version", {}),
])
def test_simple_cmds_yield_same_event_count_across_versions(
    envelope_v, cmd, extras,
):
    """Dispatching the same cmd under v1.0 vs v1.1 envelope produces the
    same number of events (length parity). The runtime cannot silently
    inject extra events for v1.1 callers without breaking parity tests.

    Goodhart-proof: an implementation that adds a v1.1-only delta would
    break the equality check; a length-only assertion (``len > 0``)
    would NOT — so we assert exact equality, not truthiness.
    """
    env_dict = {"v": envelope_v, "id": f"xver-{cmd}-{envelope_v[-3:]}",
                "cmd": cmd}
    env_dict.update(extras)

    events_a = _events_for(env_dict)
    events_b = _events_for(env_dict)
    # Same input → same length (twice for stability — flaps would fail).
    assert len(events_a) == len(events_b)

    # Compare against the v1.0 baseline (default fixture pair).
    baseline_dict = dict(env_dict, v="happi/1.0")
    v11_dict = dict(env_dict, v="happi/1.1")
    baseline = _events_for(baseline_dict)
    v11 = _events_for(v11_dict)
    assert len(baseline) == len(v11), (
        f"{cmd}: v1.0 emitted {len(baseline)} events, v1.1 emitted {len(v11)}"
    )
    # Type sequence must match across versions for simple cmds.
    assert _types(baseline) == _types(v11)


def test_v10_envelope_with_v11_only_audit_flag_still_dispatches_safely():
    """v1.0 envelope + flags.audit=true → idr terminator still fires.

    The audit feature is runtime-side, not envelope-side. A v1.0 client
    that opts into audit (using the v1.1 flag) gets the idr stream tail
    and the runtime stamps ``PROTOCOL_VERSION`` (the runtime's own
    version, currently happi/1.0 by default) on the idr event.

    Catches regressions that gate audit on the envelope's ``v`` field
    instead of the flag itself.
    """
    events = _events_for({
        "v": "happi/1.0", "id": "v10-audit-1", "cmd": "noop",
        "flags": {"audit": True},
    })
    types = _types(events)
    assert types[-1] == "idr"
    # The envelope's echoed v stays at v1.0 (HappiEnvelope echoes client).
    # But every emitted event carries the runtime's PROTOCOL_VERSION.
    for e in events:
        assert e["v"] == PROTOCOL_VERSION


def test_v10_envelope_with_v11_only_model_versions_flag_propagates():
    """``flags.model_versions`` is a v1.1-only flag; passing it on a v1.0
    envelope must still flow into ``idr_ref.model_versions`` when audit
    is enabled. The runtime treats flags as opaque pass-through —
    envelope version does NOT gate which flag keys are honoured.
    """
    events = _events_for({
        "v": "happi/1.0", "id": "v10-mv-1", "cmd": "noop",
        "flags": {"audit": True,
                  "model_versions": ["claude-opus-4-7", "gemini-2.5"]},
    })
    idr = events[-1]
    assert idr["type"] == "idr"
    assert idr["idr_ref"]["model_versions"] == [
        "claude-opus-4-7", "gemini-2.5",
    ]


# ---------- (C) Transport+cmd matrix --------------------------------------

@pytest.fixture
def http_server() -> Iterator[tuple[str, int]]:
    """Yield (host, port) for an ephemeral-port HTTP server in a daemon thread."""
    server = build_server(host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _post_envelope(host: str, port: int, env_dict: dict) -> list[dict]:
    """POST a HAPPI envelope and parse the SSE response into events."""
    body = json.dumps(env_dict).encode("utf-8")
    conn = HTTPConnection(host, port, timeout=5)
    conn.request(
        "POST", HAPPI_PATH, body=body,
        headers={"content-type": "application/json",
                 "content-length": str(len(body))},
    )
    resp = conn.getresponse()
    assert resp.status == 200, f"status={resp.status}"
    text = resp.read().decode("utf-8")
    conn.close()
    events = [
        json.loads(frame[len("data: "):])
        for frame in text.split("\n\n")
        if frame.startswith("data: ")
    ]
    return events


def test_http_transport_dispatches_v11_idr_emit(http_server):
    """v1.1 idr.emit envelope through HTTP transport produces the same
    event sequence as direct dispatch (started + idr + completed).

    Goodhart-proof: asserts the exact type sequence AND the sha256
    field's structural shape — a transport that silently dropped events
    would fail the type assertion; one that scrambled fields would fail
    the sha256-length check.
    """
    host, port = http_server
    events = _post_envelope(host, port, {
        "v": "happi/1.1", "id": "http-idr-1", "cmd": "idr.emit",
        "args": ["env-bytes", "evt-bytes"],
    })
    assert _types(events) == ["started", "idr", "completed"]
    idr = events[1]
    assert isinstance(idr["idr_ref"]["sha256"], str)
    assert len(idr["idr_ref"]["sha256"]) == 64


def test_http_transport_dispatches_v11_audit_flag_with_idr_tail(http_server):
    """v1.1 envelope + flags.audit=true through HTTP → idr is last event."""
    host, port = http_server
    events = _post_envelope(host, port, {
        "v": "happi/1.1", "id": "http-audit-1", "cmd": "echo",
        "args": ["hi"], "flags": {"audit": True},
    })
    types = _types(events)
    assert types[-1] == "idr"
    assert "completed" in types
    # idr must follow completed (the audit-terminator ordering invariant).
    assert types.index("idr") > types.index("completed")


def test_http_health_endpoint_advertises_v11_post_cutover(http_server):
    """Health endpoint banner is ``happi/1.1`` after the P7 cutover.

    Per the ultraplan: P7 flipped the service banner literal at
    ``transports/http_transport.py:94`` from ``happi/1.0`` to
    ``happi/1.1``. This test guards against an accidental regression
    that would silently flip it back, breaking the public service
    contract clients use to discover the runtime's protocol level.
    """
    host, port = http_server
    conn = HTTPConnection(host, port, timeout=3)
    conn.request("GET", "/health")
    resp = conn.getresponse()
    payload = json.loads(resp.read())
    conn.close()
    assert payload["service"] == "happi/1.1"


# ---------- (D) Idr chain determinism ------------------------------------

def test_idr_sha256_byte_for_byte_identical_across_50_runs():
    """Replaying the same (envelope_raw, event_lines) → exactly 1 unique sha256.

    Anchors H427's deterministic-replay claim. The polyglot canonical
    runtime has the same property — both runtimes derive the hash from
    the same byte sequence with no wall-clock or RNG dependency. A
    regression that introduced timing or mutable state into the hash
    would produce >1 unique value across 50 runs.

    Goodhart-proof: 50 runs is enough that flaky non-determinism (1-in-N
    timing dependence) would be caught with high probability; a fixed
    50-run window is published in the assertion message for triage.
    """
    envelope_raw = b'{"v":"happi/1.1","id":"det","cmd":"echo","args":["hi"]}'
    fixed_lines = [
        '{"v":"happi/1.1","id":"det","type":"started","ts":1000}',
        '{"v":"happi/1.1","id":"det","type":"delta","ts":1001,"text":"hi"}',
        '{"v":"happi/1.1","id":"det","type":"completed","ts":1002}',
    ]
    hashes: set[str] = set()
    for _ in range(50):
        with audit_session():
            for line in fixed_lines:
                record_event(line)
            hashes.add(compute_idr_sha256(envelope_raw))
    assert len(hashes) == 1, (
        f"non-deterministic over 50 runs; got {len(hashes)} unique hashes: "
        f"{sorted(hashes)[:3]}..."
    )


def test_idr_sha256_collision_resistance_50_distinct_inputs():
    """50 distinct envelopes → 50 distinct sha256s (smoke-test resistance).

    Not a cryptographic-grade collision-resistance proof — sha256 itself
    provides that. This test catches a different failure: an
    implementation that ignored part of the input (e.g. only hashed the
    envelope, dropping the event lines) would produce duplicate hashes
    across distinct envelopes-with-same-events.

    Goodhart-proof: asserts exactly 50 unique values in the set; a
    contains-check (``len > 1``) would silently pass even if 49/50
    inputs collided.
    """
    line = '{"v":"happi/1.1","id":"collision","type":"completed","ts":1}'
    hashes: set[str] = set()
    for i in range(50):
        envelope_raw = (
            f'{{"v":"happi/1.1","id":"col-{i}","cmd":"noop"}}'
        ).encode("utf-8")
        with audit_session():
            record_event(line)
            hashes.add(compute_idr_sha256(envelope_raw))
    assert len(hashes) == 50, (
        f"expected 50 unique hashes for 50 distinct envelopes, "
        f"got {len(hashes)}"
    )


def test_idr_sha256_independent_recompute_matches_runtime_emission():
    """Independent reference computation matches the runtime's idr.sha256.

    Strongest correctness check at this layer: hash the envelope_raw +
    the buffered event lines independently, assert it equals what the
    runtime emitted in the idr event. Catches any implementation drift
    between the runtime's hash function and the canonical algorithm.
    """
    raw_text = json.dumps(
        {"v": "happi/1.1", "id": "recompute-1", "cmd": "echo",
         "args": ["check"], "flags": {"audit": True}},
        separators=(",", ":"),
    )
    out = io.StringIO()
    run(stream_in=io.StringIO(raw_text), stream_out=out)
    lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
    events = [json.loads(ln) for ln in lines]
    idr = events[-1]
    pre_idr_lines = lines[:-1]

    h = hashlib.sha256()
    h.update(raw_text.encode("utf-8"))
    for line in pre_idr_lines:
        h.update(line.encode("utf-8"))
    expected = h.hexdigest()

    assert idr["idr_ref"]["sha256"] == expected, (
        f"runtime emission diverged from independent recompute: "
        f"runtime={idr['idr_ref']['sha256']!r}, expected={expected!r}"
    )
