"""HAPPI v1.1 context event — lib-side tests (P2).

Covers ``lib/hal/happi/context.py`` + ``events.emit_context``: the content
address (byte-identical to the polyglot runtime AND GRIP ``content_addr`` — the
cross-runtime invariant), ``context_head`` resolution, the ``emit_context`` wire
shape, and the back-compat matrix (v1.0 stamp opt-back; ``ctx.context_head``
forward-compat round-trip).

Goodhart-proof — asserts content (sha256 equality, exclusion semantics, field
values), not call counts. Mirrors ``test_happi_audit_v11.py``.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import pytest

from lib.hal.happi import (
    HappiEnvelope,
    PROTOCOL_VERSION,
    build_context_ref,
    compute_content_addr,
    context_head,
    emit_context,
)
from lib.hal.happi import events as events_mod

_EXCLUDE = ("id", "ts", "audit")


def _ref_content_addr(body: dict) -> str:
    addressed = {k: v for k, v in body.items() if k not in _EXCLUDE}
    canonical = json.dumps(addressed, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------- content address ----------------------------------------------

def test_content_addr_matches_canonical_algorithm():
    body = {"kind": "context-delta", "decision": {"learned": "X"}, "predecessor_idr": "p-0"}
    assert compute_content_addr(body) == _ref_content_addr(body)


def test_content_addr_excludes_id_ts_audit():
    """Goodhart anchor: a body differing ONLY in id/ts/audit shares one address."""
    base = {"kind": "context-delta", "decision": {"learned": "X"}}
    a = compute_content_addr({**base, "id": "p-1", "ts": "2026-01-01T00:00:00Z", "audit": {"sig": "a"}})
    b = compute_content_addr({**base, "id": "p-9", "ts": "2099-12-31T23:59:59Z", "audit": {"sig": "b"}})
    assert a == b


def test_content_addr_sensitive_to_semantic_change():
    a = compute_content_addr({"kind": "context-delta", "decision": {"learned": "X"}})
    b = compute_content_addr({"kind": "context-delta", "decision": {"learned": "Y"}})
    assert a != b


def test_content_addr_unknown_key_is_addressed():
    """A key not in (id,ts,audit) is part of the address — fail-safe by construction."""
    a = compute_content_addr({"decision": {"a": 1}})
    b = compute_content_addr({"decision": {"a": 1}, "extra": "v"})
    assert a != b


def test_content_addr_non_ascii_deterministic():
    body = {"decision": {"learned": "café ☕ 北京"}}
    assert compute_content_addr(dict(body)) == compute_content_addr(dict(body))


# ---------- build_context_ref --------------------------------------------

def test_build_context_ref_shape():
    ref = build_context_ref({"decision": {"a": 1}}, predecessor_context="p-0",
                            snapshot_ref="sha256:snap", model_versions=["m1"])
    assert ref["sha256"] == compute_content_addr({"decision": {"a": 1}})
    assert ref["predecessor_context"] == "p-0"
    assert ref["snapshot_ref"] == "sha256:snap"
    assert ref["kind"] == "context-delta"
    assert ref["model_versions"] == ["m1"]


def test_build_context_ref_invalid_kind_falls_back():
    assert build_context_ref({"decision": {"a": 1}}, kind="bogus")["kind"] == "context-delta"


def test_build_context_ref_predecessor_not_in_address():
    """Address is over the body; the same body threaded to different predecessors
    keeps ONE sha256 (predecessor is chain-position metadata, not identity)."""
    body = {"decision": {"a": 1}}
    a = build_context_ref(body, predecessor_context="p-1")
    b = build_context_ref(body, predecessor_context="p-2")
    assert a["sha256"] == b["sha256"]
    assert a["predecessor_context"] != b["predecessor_context"]


# ---------- emit_context wire shape --------------------------------------

def test_emit_context_wire_shape():
    evt = json.loads(emit_context("ctx-1", sha256="sha256:abc", predecessor_context="p-0",
                                  kind="context-delta", model_versions=["m1"]))
    assert evt["type"] == "context"
    assert evt["id"] == "ctx-1"
    assert evt["v"] == PROTOCOL_VERSION
    ref = evt["context_ref"]
    assert ref["sha256"] == "sha256:abc"
    assert ref["predecessor_context"] == "p-0"
    assert ref["snapshot_ref"] is None
    assert ref["kind"] == "context-delta"
    assert ref["model_versions"] == ["m1"]


def test_emit_context_defaults():
    ref = json.loads(emit_context("ctx-2", sha256="sha256:x"))["context_ref"]
    assert ref["predecessor_context"] is None
    assert ref["snapshot_ref"] is None
    assert ref["kind"] == "context-delta"
    assert ref["model_versions"] == []


def test_emit_context_normalises_invalid_kind():
    """emit_context coerces an unrecognised kind to context-delta (HAL #433) —
    the shared normalise_kind keeps a malformed kind off the wire, matching
    build_context_ref. Mutation killed: pass kind through unchecked."""
    ref = json.loads(emit_context("ctx-k", sha256="sha256:x", kind="bogus-kind"))["context_ref"]
    assert ref["kind"] == "context-delta"
    ref2 = json.loads(emit_context("ctx-k2", sha256="sha256:x", kind="context-snapshot"))["context_ref"]
    assert ref2["kind"] == "context-snapshot"  # a valid kind passes through



# ---------- back-compat matrix -------------------------------------------

def test_emit_context_v_opt_back_to_v10(monkeypatch):
    """HAPPI_LIB_PROTOCOL_VERSION=happi/1.0 stamps v1.0 on `context` too (the
    same opt-back that governs every other event type)."""
    monkeypatch.setenv("HAPPI_LIB_PROTOCOL_VERSION", "happi/1.0")
    importlib.reload(events_mod)
    try:
        assert json.loads(events_mod.emit_context("ctx-v10", sha256="sha256:x"))["v"] == "happi/1.0"
    finally:
        monkeypatch.delenv("HAPPI_LIB_PROTOCOL_VERSION", raising=False)
        importlib.reload(events_mod)


def test_context_head_round_trips_through_envelope():
    """ctx.context_head survives parse → to_dict (forward-compat passthrough):
    the 'unknown event forwarded, not rejected' property at the pointer level."""
    env = HappiEnvelope.parse({"v": "happi/1.0", "id": "e1", "cmd": "echo",
                               "ctx": {"context_head": "precog-123"}})
    assert context_head(env) == "precog-123"
    assert env.to_dict()["ctx"]["context_head"] == "precog-123"


def test_context_head_from_dict_and_absent():
    assert context_head({"ctx": {"context_head": "precog-9"}}) == "precog-9"
    assert context_head({"ctx": {}}) is None
    assert context_head({}) is None
    assert context_head(HappiEnvelope.parse({"v": "happi/1.0", "id": "e2", "cmd": "echo"})) is None


def test_context_head_from_non_dict_mapping():
    """Speaker-C falsifier (HAL #433): a ctx that is a Mapping but NOT a dict
    subclass (e.g. MappingProxyType) must still resolve — context_head is
    duck-typed on .get, so any mapping works. Mutation killed: an
    isinstance(ctx, dict)-only gate, which MappingProxyType would fail."""
    from types import MappingProxyType
    env = HappiEnvelope.parse({"v": "happi/1.0", "id": "e3", "cmd": "echo"})
    env.ctx = MappingProxyType({"context_head": "precog-proxy"})
    assert context_head(env) == "precog-proxy"



# ---------- cross-runtime invariant --------------------------------------

_POLYGLOT = Path.home() / ".hal" / "happi.md"
_GRIP_IDR = Path.home() / ".claude" / "lib" / "precog" / "idr.py"


@pytest.mark.skipif(not _POLYGLOT.exists(), reason="happi.md absent")
def test_content_addr_matches_polyglot():
    """lib compute_content_addr == the polyglot runtime's context_ref.sha256."""
    body = {"kind": "context-delta", "decision": {"learned": "X"}, "predecessor_idr": "p-0"}
    proc = subprocess.run(
        ["bash", str(_POLYGLOT), "run"],
        input=json.dumps({"v": "happi/1.1", "id": "x", "cmd": "context.append",
                          "args": [json.dumps(body)]}),
        capture_output=True, text=True, timeout=20,
    )
    events = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    ctx = next(e for e in events if e["type"] == "context")
    assert ctx["context_ref"]["sha256"] == compute_content_addr(body)


@pytest.mark.skipif(not _GRIP_IDR.exists(), reason="GRIP idr.py absent")
def test_content_addr_matches_grip():
    """lib compute_content_addr byte-equals GRIP content_addr (loads idr.py by
    path to dodge the two-repo lib/ package-name collision)."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("grip_precog_idr_p2", _GRIP_IDR)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    body = {"kind": "context-delta", "decision": {"learned": "X"}, "predecessor_idr": "p-0"}
    assert compute_content_addr(body) == mod.content_addr(body)
