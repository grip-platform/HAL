"""HAPPI v1.3 native-Decision in the memory envelope — mutation-surviving tests.

Covers the ONE shared Decision schema carried natively by the ``context`` memory
event (the twin of the IDR side, GRIP ``lib/decision_record.py``):

  * round-trip: a context body carrying a valid Decision validates + addresses
  * tiered: a deterministic decision (confidence 1.0) stays MINIMAL (no anatomy);
    a delegated one (confidence < 1.0) MUST carry an externally-checkable falsifier
  * SMT gate: surface grafts (tone/persona/confidence-as-criterion) are rejected
  * back-compat: a decision-less body addresses byte-identically (invariant 5)
  * the opt-in is OFF by default — an invalid decision passes unvalidated unless asked
  * cross-runtime 1:1: HAL Decision == GRIP decision_record on every public fn
    (the H-ONE-SCHEMA precondition — divergence fails this test)
  * polyglot runtime: flags.validate_decision on context.append validates via subprocess

Goodhart-proof — asserts content (hashes, field values, error presence/absence,
exception types), never call counts. Each test fails under an obvious mutation
of the validator or the opt-in wiring.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lib.hal.happi import (
    Decision,
    DecisionError,
    build_context_ref,
    compute_content_addr,
    from_model_dict,
    validate_body_decision,
    validate_decision,
)
from lib.hal.happi.decision import is_predicate_shaped, reject_surface_fields


# A canonical valid delegated decision (full anatomy, externally-checkable falsifier)
_DELEGATED = {
    "what": "route this request to groq/gpt-oss-120b",
    "why": "cheapest provider that passed the latency budget",
    "confidence": 0.7,
    "uncertainties": ["provider may be rate-limited"],
    "assumptions": ["latency budget is 200ms"],
    "falsification_criteria": "wrong if p95 latency exceeds 200ms over the next 50 calls",
    "falsifier_ref": "H-ROUTE-LATENCY",
}
_DETERMINISTIC = {"what": "echo the input verbatim", "confidence": 1.0}


# ---------- round-trip: body carries the Decision shape -------------------

def test_valid_delegated_decision_round_trips():
    body = {"kind": "context-delta", "decision": dict(_DELEGATED)}
    d = validate_body_decision(body)
    assert isinstance(d, Decision)
    assert d.what == _DELEGATED["what"]
    assert d.confidence == 0.7
    # addressing the body still works and is over the whole body
    ref = build_context_ref(body, validate_decision=True)
    assert ref["sha256"] == compute_content_addr(body)


def test_deterministic_decision_stays_minimal():
    body = {"kind": "context-delta", "decision": dict(_DETERMINISTIC)}
    d = validate_body_decision(body)
    assert d.is_deterministic()
    assert d.uncertainties == ()
    assert d.assumptions == ()


def test_deterministic_with_uncertainties_rejected():
    """A certain decision (1.0) that lists doubt is fabricating it — rejected."""
    body = {"decision": {"what": "x", "confidence": 1.0, "uncertainties": ["a"]}}
    with pytest.raises(DecisionError):
        validate_body_decision(body)


def test_delegated_without_falsifier_rejected():
    body = {"decision": {"what": "x", "confidence": 0.5}}
    with pytest.raises(DecisionError):
        validate_body_decision(body)


def test_delegated_with_subjective_only_falsifier_rejected():
    """A 'falsifier' that is purely subjective is not externally checkable."""
    body = {"decision": {"what": "x", "confidence": 0.5,
                         "falsification_criteria": "it just feels off to me"}}
    with pytest.raises(DecisionError):
        validate_body_decision(body)


# ---------- SMT surface-graft gate ---------------------------------------

@pytest.mark.parametrize("graft", ["tone", "persona", "personality",
                                   "confidence_gate", "auto_merge", "should_merge"])
def test_surface_graft_field_rejected(graft):
    body = {"decision": {"what": "x", "confidence": 1.0, graft: "evil"}}
    with pytest.raises(DecisionError):
        validate_body_decision(body)


def test_unknown_field_rejected():
    body = {"decision": {"what": "x", "confidence": 1.0, "bogus_field": 1}}
    with pytest.raises(DecisionError):
        validate_body_decision(body)


# ---------- back-compat: decision-less + opt-in OFF ----------------------

def test_decisionless_body_returns_none_and_addresses_unchanged():
    """A body with NO decision slot is valid (returns None) and its address is
    identical with or without the validation gate (invariant 5: back-compat)."""
    body = {"kind": "context-delta", "learned": "the sky is blue"}
    assert validate_body_decision(body) is None
    addr_plain = compute_content_addr(body)
    ref_validated = build_context_ref(body, validate_decision=True)
    ref_default = build_context_ref(body)
    assert ref_validated["sha256"] == addr_plain
    assert ref_default["sha256"] == addr_plain


def test_opt_in_off_by_default_skips_validation():
    """Default build_context_ref does NOT validate — an invalid decision still
    addresses (the gate is opt-in; existing callers are byte-unchanged)."""
    bad_body = {"decision": {"what": "x", "confidence": 0.5}}  # missing falsifier
    # default path: no raise, addresses the body as-is
    ref = build_context_ref(bad_body)
    assert ref["sha256"] == compute_content_addr(bad_body)
    # opt-in path: raises
    with pytest.raises(DecisionError):
        build_context_ref(bad_body, validate_decision=True)


def test_decision_slot_not_object_rejected():
    with pytest.raises(DecisionError):
        validate_body_decision({"decision": "not-an-object"})


def test_already_a_decision_instance_revalidated():
    d = from_model_dict(dict(_DELEGATED))
    out = validate_body_decision({"decision": d})
    assert out is d


# ---------- cross-runtime 1:1 invariant (H-ONE-SCHEMA) -------------------

_GRIP_DECISION = Path.home() / ".claude" / "lib" / "decision_record.py"


def _load_grip_decision():
    spec = importlib.util.spec_from_file_location("grip_decision_record_p", _GRIP_DECISION)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not _GRIP_DECISION.exists(), reason="GRIP decision_record.py absent")
def test_admissible_fields_match_grip():
    grip = _load_grip_decision()
    from lib.hal.happi.decision import ADMISSIBLE_FIELDS
    assert ADMISSIBLE_FIELDS == grip.ADMISSIBLE_FIELDS


@pytest.mark.skipif(not _GRIP_DECISION.exists(), reason="GRIP decision_record.py absent")
def test_content_hash_byte_identical_to_grip():
    """The same Decision content hashes identically on both sides — the 1:1
    guarantee the v2 shim depends on. A field rename/drop on either side fails."""
    grip = _load_grip_decision()
    hal_d = from_model_dict(dict(_DELEGATED))
    grip_d = grip.from_model_dict(dict(_DELEGATED))
    assert hal_d.content_hash() == grip_d.content_hash()
    assert hal_d.to_dict() == grip_d.to_dict()


@pytest.mark.skipif(not _GRIP_DECISION.exists(), reason="GRIP decision_record.py absent")
def test_validation_verdicts_match_grip():
    """Both runtimes accept/reject the SAME decisions — behavioural 1:1."""
    grip = _load_grip_decision()
    cases = [
        dict(_DELEGATED),                                      # valid delegated
        dict(_DETERMINISTIC),                                  # valid deterministic
        {"what": "x", "confidence": 0.5},                     # bad: no falsifier
        {"what": "x", "confidence": 1.0, "assumptions": ["a"]},  # bad: det+assumptions
        {"what": "x", "confidence": 1.0, "tone": "snarky"},   # bad: surface graft
    ]
    for raw in cases:
        hal_ok = _accepts(from_model_dict, raw, DecisionError)
        grip_ok = _accepts(grip.from_model_dict, raw, grip.DecisionError)
        assert hal_ok == grip_ok, f"verdict diverged on {raw}"


def _accepts(fn, raw, exc):
    try:
        fn(dict(raw))
        return True
    except exc:
        return False


# ---------- polyglot runtime opt-in (subprocess) -------------------------

_POLYGLOT = Path(__file__).resolve().parent.parent / "happi.md"


def _run_context_append(body, flags=None):
    env = {"v": "happi/1.3", "id": "t-1", "cmd": "context.append",
           "args": [json.dumps(body)]}
    if flags:
        env["flags"] = flags
    proc = subprocess.run(["bash", str(_POLYGLOT), "run"],
                          input=json.dumps(env), capture_output=True,
                          text=True, timeout=20)
    return [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]


@pytest.mark.skipif(not _POLYGLOT.exists(), reason="happi.md absent")
def test_polyglot_validate_decision_accepts_valid():
    events = _run_context_append({"decision": dict(_DELEGATED)},
                                 flags={"validate_decision": True})
    types = [e["type"] for e in events]
    assert "context" in types and "completed" in types
    assert "error" not in types


@pytest.mark.skipif(not _POLYGLOT.exists(), reason="happi.md absent")
def test_polyglot_validate_decision_rejects_invalid():
    events = _run_context_append({"decision": {"what": "x", "confidence": 0.5}},
                                 flags={"validate_decision": True})
    err = next((e for e in events if e["type"] == "error"), None)
    assert err is not None
    assert err["code"] == "parse_error"
    assert "falsification_criteria" in err["message"]
    assert not any(e["type"] == "context" for e in events)


@pytest.mark.skipif(not _POLYGLOT.exists(), reason="happi.md absent")
def test_polyglot_opt_in_off_passes_invalid_unvalidated():
    """Without flags.validate_decision the invalid decision still emits context —
    byte-identical back-compat for every pre-v1.3 caller."""
    events = _run_context_append({"decision": {"what": "x", "confidence": 0.5}})
    assert any(e["type"] == "context" for e in events)
    assert not any(e["type"] == "error" for e in events)


@pytest.mark.skipif(not _POLYGLOT.exists(), reason="happi.md absent")
def test_polyglot_address_unchanged_by_validation_flag():
    """The content address is identical whether or not validation ran (gate
    before addressing, not a change to the address)."""
    body = {"decision": dict(_DELEGATED)}
    a = _run_context_append(body, flags={"validate_decision": True})
    b = _run_context_append(body)
    addr_a = next(e for e in a if e["type"] == "context")["context_ref"]["sha256"]
    addr_b = next(e for e in b if e["type"] == "context")["context_ref"]["sha256"]
    assert addr_a == addr_b == compute_content_addr(body)


@pytest.mark.skipif(not _POLYGLOT.exists(), reason="happi.md absent")
def test_polyglot_decision_address_matches_lib():
    """The polyglot runtime's content_ref.sha256 over a Decision-bearing body
    equals the lib compute_content_addr — cross-runtime address parity."""
    body = {"kind": "context-delta", "decision": dict(_DELEGATED)}
    events = _run_context_append(body, flags={"validate_decision": True})
    ctx = next(e for e in events if e["type"] == "context")
    assert ctx["context_ref"]["sha256"] == compute_content_addr(body)
