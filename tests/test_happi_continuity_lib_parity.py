"""HAPPI v1.4 continuity — lib/polyglot parity, and the deliberate fail-closed gap.

The lib had NO continuity support at all while the canonical polyglot runtime
had carried it since v1.4: ``lib/hal/happi/`` contained zero occurrences of the
word, ``envelope.SUPPORTED_VERSIONS`` stopped at ``happi/1.3``, and nothing in
the build could notice. ``build-happi``'s version-consistency gate reads only
the ``happi-sources/`` layers it assembles (``scripts/build-happi.py``
``_check_version_consistency``), so the lib is structurally outside its view.
That is HOW the drift happened, and it is why the parity test below drives the
REAL polyglot through bash rather than comparing two copies of our own code: a
test that only checked the lib against itself would have stayed green through
the entire divergence.

The asymmetry test is the one worth reading twice. ``context.normalise_kind``
COERCES an unknown kind to ``context-delta``; ``continuity.validate_kind``
RAISES. That looks like an inconsistency and is not — it is the difference
between a safe default that exists and one that does not. Coercing a malformed
premise into ``continuity-commit`` fabricates a commitment the caller never
made, and the whole value of the record is that the premise is the caller's.
``test_the_fail_closed_asymmetry_with_context_is_deliberate`` pins that
difference, so anyone "tidying up the inconsistency" by making continuity
coerce turns it red and reads why.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from lib.hal.happi import (
    SUPPORTED_VERSIONS,
    ContinuityError,
    build_continuity_ref,
    emit_continuity,
    validate_kind,
)
from lib.hal.happi.context import normalise_kind
from tests._canonical_version import canonical_runtime_version
from tests._bash import find_bash

REPO = Path(__file__).resolve().parents[1]
HAPPI = REPO / "happi.md"
_BASH = find_bash()
pytestmark = pytest.mark.skipif(
    _BASH is None,
    reason="no usable bash found on this machine (install Git Bash on Windows)",
)

_VOLATILE = ("id", "ts", "audit")
_BODY = {
    "claim": "the lib and the polyglot address a commitment identically",
    "falsifier": "the two sha256 values differ for one body",
    "deps": ["happi/1.4"],
    "severity": "reversible",
}


def _canonical_addr(body: dict) -> str:
    """The documented address algorithm, restated independently.

    Deliberately NOT imported from the implementation: a test that computes the
    expected value with the code under test asserts only that the code agrees
    with itself, which it cannot fail to do.
    """
    trimmed = {k: v for k, v in body.items() if k not in _VOLATILE}
    canonical = json.dumps(trimmed, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _polyglot_continuity_ref(body: dict, **flags) -> dict:
    """Dispatch a real continuity.append through the assembled happi.md."""
    envelope = {"v": "happi/1.4", "id": "parity", "cmd": "continuity.append",
                "args": [json.dumps(body)], "flags": flags}
    proc = subprocess.run(
        [_BASH, str(HAPPI), "run"],
        input=json.dumps(envelope), capture_output=True, text=True,
        cwd=str(REPO), timeout=60,
    )
    assert "bad substitution" not in proc.stderr, (
        "the assembled happi.md is not executable by bash: " + proc.stderr[:400]
    )
    events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    return next(e for e in events if e["type"] == "continuity")["continuity_ref"]


# --- parity with the canonical runtime --------------------------------------

def test_lib_address_is_byte_identical_to_the_polyglot():
    """The cross-runtime invariant the whole chain rests on.

    A content address is a dedup/Merkle coordinate. If the two runtimes compute
    it differently for one body, the same commitment occupies two coordinates
    and the chain silently forks. Drives the real polyglot, so this observes
    what a consumer would.
    """
    poly = _polyglot_continuity_ref(_BODY)
    lib = build_continuity_ref(_BODY)
    assert lib["sha256"] == poly["sha256"]
    assert lib["sha256"] == _canonical_addr(_BODY)


def test_lib_ref_carries_the_same_fields_as_the_polyglot():
    """Same keys, same defaults — a ref missing a field is a ref a reader of the
    other runtime's stream cannot parse."""
    poly = _polyglot_continuity_ref(_BODY)
    lib = build_continuity_ref(_BODY)
    assert set(lib) == set(poly)
    assert lib == poly


def test_verdict_ref_matches_the_polyglot():
    commit_addr = _canonical_addr(_BODY)
    verdict_body = {"verdict": "CONGRUENT", "axis": "reality",
                    "evidence": {"cmd": "bin/grip-tree-check", "exit": 0},
                    "verifier": "deterministic"}
    poly = _polyglot_continuity_ref(verdict_body, kind="continuity-verdict",
                                    settles=commit_addr)
    lib = build_continuity_ref(verdict_body, kind="continuity-verdict",
                               settles=commit_addr)
    assert lib == poly
    assert lib["settles"] == commit_addr


def test_address_ignores_metadata_and_key_order_but_not_content():
    """The address is over the BODY. Threading the same premise to a different
    predecessor must not move its dedup coordinate."""
    base = build_continuity_ref(_BODY)["sha256"]
    threaded = build_continuity_ref(_BODY, predecessor_continuity="sha256:" + "ab" * 32)
    assert threaded["sha256"] == base
    reordered = dict(reversed(list(_BODY.items())))
    assert build_continuity_ref(reordered)["sha256"] == base
    changed = dict(_BODY, claim="a different premise entirely")
    assert build_continuity_ref(changed)["sha256"] != base


# --- fail-closed refusals ----------------------------------------------------

@pytest.mark.parametrize("kind", ["not-a-real-kind", "context-delta", "", "CONTINUITY-COMMIT"])
def test_unknown_kind_is_refused_not_coerced(kind):
    """Mirrors the polyglot's rejection. ``context-delta`` is in the list on
    purpose: a valid kind for the OTHER record type is still invalid here."""
    with pytest.raises(ContinuityError) as excinfo:
        build_continuity_ref(_BODY, kind=kind)
    assert "kind must be one of" in str(excinfo.value)
    assert repr(kind) in str(excinfo.value)


def test_verdict_without_settles_is_refused():
    """A settlement that settles nothing can never be joined back to its premise,
    so it can never be checked — strictly worse than no record."""
    with pytest.raises(ContinuityError) as excinfo:
        build_continuity_ref({"verdict": "CONGRUENT"}, kind="continuity-verdict")
    assert "settles" in str(excinfo.value)


def test_refusal_happens_before_any_ref_is_built():
    """The polyglot emits its error and returns BEFORE emit_continuity. The lib's
    equivalent guarantee is that no ref object is produced at all — so a refused
    record cannot reach a caller that would go on to emit it."""
    for bad in ({"kind": "nonsense"}, {"kind": "continuity-verdict"}):
        with pytest.raises(ContinuityError):
            build_continuity_ref(_BODY, **bad)


def test_emit_continuity_refuses_an_unknown_kind_on_the_wire():
    """The encoder validates too, so bypassing build_continuity_ref does not
    bypass the gate."""
    with pytest.raises(ContinuityError):
        emit_continuity("x", sha256="sha256:" + "00" * 32, kind="nonsense")


def test_emit_continuity_wire_shape():
    line = emit_continuity("x", sha256=_canonical_addr(_BODY),
                           kind="continuity-commit", model_versions=["m1"])
    evt = json.loads(line)
    assert evt["type"] == "continuity"
    assert evt["id"] == "x"
    assert evt["continuity_ref"]["sha256"] == _canonical_addr(_BODY)
    assert evt["continuity_ref"]["model_versions"] == ["m1"]


def test_the_fail_closed_asymmetry_with_context_is_deliberate():
    """context COERCES, continuity RAISES — and that difference is the feature.

    A context delta has a safe default kind; a premise does not. If someone
    "fixes the inconsistency" by giving continuity a normalise_kind, this test
    goes red and the docstring above explains why it should not be fixed.
    """
    assert normalise_kind("not-a-real-kind") == "context-delta"
    with pytest.raises(ContinuityError):
        validate_kind("not-a-real-kind")


# --- version acceptance ------------------------------------------------------

def test_supported_versions_reaches_the_canonical_runtime_version():
    """The lib must ACCEPT envelopes the canonical runtime stamps.

    Asserts the property, never the membership list — enumerating the set
    guarantees a red test on every release by construction (the repair already
    applied in test_happi_envelope_v11.py). The target is DERIVED from the
    polyglot's own declaration, so this stays true across bumps and fails only
    when the lib genuinely falls behind.
    """
    canonical = canonical_runtime_version()
    assert canonical in SUPPORTED_VERSIONS, (
        f"the canonical runtime stamps {canonical} but the lib does not accept it — "
        "the lib has fallen behind the polyglot again"
    )
    assert "happi/1.0" in SUPPORTED_VERSIONS  # back-compat is a HAPPI guarantee
