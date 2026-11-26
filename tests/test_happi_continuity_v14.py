"""HAPPI/1.4 continuity event — behavioural tests against the assembled happi.md.

Every test drives the REAL polyglot file through bash, so it proves what a
consumer would actually observe rather than what the sources say.

The executability test is a forward anchor. `build-happi --check` verifies
round-trip identity and per-layer parser validity, but it never RUNS the
assembled file — so a change can be byte-perfect, compile as Python, and still
be unrunnable. That happened while writing this feature: one prose apostrophe
added to the Python layer flipped bash's quote pairing (bash tokenises the whole
polyglot file), which swallowed the closing paren of the command substitution
that carries the runtime, and every dispatch died with "bad substitution".
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

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
    "claim": "the shared tree is level with origin/main",
    "falsifier": "probe: bin/grip-tree-check",
    "deps": [],
    "severity": "reversible",
}


def _run(envelope: dict) -> list[dict]:
    """Dispatch one envelope through the assembled happi.md; return its events."""
    proc = subprocess.run(
        [_BASH, str(HAPPI), "run"],
        input=json.dumps(envelope), capture_output=True, text=True, cwd=str(REPO),
        timeout=60,
    )
    assert "bad substitution" not in proc.stderr, (
        "the assembled happi.md is not executable by bash: " + proc.stderr[:400]
    )
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def _canonical_addr(body: dict) -> str:
    """The documented content address: sha256 over canonical JSON (sorted keys),
    excluding the volatile id/ts/audit fields. Stated independently here so the
    test cannot pass merely by agreeing with the implementation's own helper."""
    trimmed = {k: v for k, v in body.items() if k not in _VOLATILE}
    canonical = json.dumps(trimmed, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _append(body: dict, **flags) -> list[dict]:
    return _run({"v": canonical_runtime_version(), "id": "t",
                 "cmd": "continuity.append",
                 "args": [json.dumps(body)], "flags": flags})


def test_assembled_happi_is_executable_by_bash():
    """The gap build-happi --check leaves open: does the file actually RUN?"""
    events = _run({"v": "happi/1.0", "id": "smoke", "cmd": "version"})
    assert [e["type"] for e in events] == ["started", "delta", "completed"]
    assert events[1]["text"] == canonical_runtime_version()


def test_commit_is_emitted_before_completed():
    """The pre-registration guarantee is the ORDERING, so it is what we assert:
    a commitment must be observable before the outcome it precedes. Inverting
    the two emits in cmd_continuity_append turns this red."""
    types = [e["type"] for e in _append(_BODY, model_versions=["m"])]
    assert types == ["started", "continuity", "completed"]
    assert types.index("continuity") < types.index("completed")


def test_content_address_matches_the_documented_algorithm():
    ref = next(e for e in _append(_BODY) if e["type"] == "continuity")["continuity_ref"]
    assert ref["sha256"] == _canonical_addr(_BODY)
    assert ref["kind"] == "continuity-commit"          # documented default
    assert ref["predecessor_continuity"] is None       # genesis


def test_address_is_insensitive_to_key_order_but_not_to_content():
    reordered = dict(reversed(list(_BODY.items())))
    same = next(e for e in _append(reordered) if e["type"] == "continuity")
    assert same["continuity_ref"]["sha256"] == _canonical_addr(_BODY)
    changed = dict(_BODY, claim="a different premise entirely")
    other = next(e for e in _append(changed) if e["type"] == "continuity")
    assert other["continuity_ref"]["sha256"] != _canonical_addr(_BODY)


def test_verdict_carries_the_commitment_it_settles():
    commit_addr = _canonical_addr(_BODY)
    verdict_body = {"verdict": "CONGRUENT", "axis": "reality",
                    "evidence": {"cmd": "bin/grip-tree-check", "exit": 0},
                    "verifier": "deterministic"}
    ref = next(e for e in _append(verdict_body, kind="continuity-verdict",
                                  settles=commit_addr)
               if e["type"] == "continuity")["continuity_ref"]
    assert ref["kind"] == "continuity-verdict"
    assert ref["settles"] == commit_addr


@pytest.mark.parametrize("flags,expected", [
    ({"kind": "continuity-verdict"}, "settles"),        # verdict naming nothing
    ({"kind": "not-a-real-kind"}, "kind must be one of"),
])
def test_fail_closed_rejections(flags, expected):
    events = _append({"verdict": "CONGRUENT"}, **flags)
    assert [e["type"] for e in events] == ["started", "error"]
    assert expected in events[-1]["message"]
    assert not any(e["type"] == "continuity" for e in events), (
        "a rejected record must not reach the stream"
    )


def test_back_compat_older_envelopes_still_served():
    for v in ("happi/1.0", "happi/1.1", "happi/1.2", "happi/1.3", "happi/1.4",
              "happi/1.5", "happi/1.6"):
        events = _run({"v": v, "id": "bc", "cmd": "version"})
        assert events[-1]["type"] == "completed", f"{v} envelope was rejected"
        assert all(e["v"] == canonical_runtime_version() for e in events), (
            "emitted events must carry the RUNTIME version, not the envelope's"
        )
