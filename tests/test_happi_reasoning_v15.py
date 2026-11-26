"""HAPPI/1.5 reasoning event — behavioural tests against the assembled happi.md.

Every test drives the REAL polyglot file through bash, so it proves what a
consumer would actually observe rather than what the sources say. Modelled on
``test_happi_continuity_v14.py``; read that file's docstring for why the
executability anchor exists.

The load-bearing tests here are the REFUSALS. A reasoning trace is admissible as
a diagnostic specimen and inadmissible as evidence of cause, and the whole
justification for adding the event type is that the fence sits on the wire
rather than in prose. So the tests assert the fence NEGATIVELY: a record
claiming authority, or claiming to settle something, must never reach the
stream. A test that only checked the happy path would stay green through the
exact failure the design exists to prevent.

One constraint discovered the hard way while writing this feature, recorded so
nobody re-pays for it: the Python layer is embedded inside
``HAPPI_PY="$(cat <<'PYTHON_EOF'``, so bash pairs quotes across the whole
polyglot while scanning for the closing paren. An ODD number of prose
apostrophes in that layer leaves one unpaired, swallows the paren, and the
assembled file stops being executable — reported as a syntax error hundreds of
lines away in the bash dispatcher. ``build-happi --check`` cannot see it,
because round-trip identity still holds. ``test_polyglot_still_executes`` below
is what actually catches it.
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
    "trace": "weighed the shared-tree hypothesis against the probe output",
    "model": "claude-opus-5",
}


def _run(envelope: dict) -> list[dict]:
    """Dispatch one envelope through the assembled happi.md; return its events."""
    proc = subprocess.run(
        [_BASH, str(HAPPI), "run"],
        input=json.dumps(envelope), capture_output=True, text=True, cwd=str(REPO),
        timeout=60,
    )
    assert "syntax error" not in proc.stderr and "bad substitution" not in proc.stderr, (
        "the assembled happi.md is not executable by bash: " + proc.stderr[:400]
    )
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def _canonical_addr(body: dict) -> str:
    """The documented content address, restated independently.

    Deliberately NOT imported from the implementation: a test that computes its
    expected value with the code under test asserts only that the code agrees
    with itself, which it cannot fail to do.
    """
    trimmed = {k: v for k, v in body.items() if k not in _VOLATILE}
    canonical = json.dumps(trimmed, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _append(body: dict, **flags) -> list[dict]:
    return _run({"v": canonical_runtime_version(), "id": "t",
                 "cmd": "reasoning.append",
                 "args": [json.dumps(body)], "flags": flags})


def test_polyglot_still_executes():
    """The apostrophe-parity anchor: does the assembled file actually RUN?"""
    events = _run({"v": "happi/1.0", "id": "smoke", "cmd": "version"})
    assert [e["type"] for e in events] == ["started", "delta", "completed"]
    assert events[1]["text"] == canonical_runtime_version()


def test_trace_is_stamped_non_authoritative_and_diagnostic():
    """The two fields that make the record safe are STAMPED, not supplied."""
    ref = next(e for e in _append(_BODY) if e["type"] == "reasoning")["reasoning_ref"]
    assert ref["authoritative"] is False
    assert ref["role"] == "diagnostic"
    assert ref["sha256"] == _canonical_addr(_BODY)


def test_reasoning_is_not_a_terminator():
    """Like continuity and unlike idr/context, the record precedes the outcome."""
    types = [e["type"] for e in _append(_BODY, model_versions=["m"])]
    assert types == ["started", "reasoning", "completed"]
    assert types.index("reasoning") < types.index("completed")


def test_address_is_insensitive_to_key_order_but_not_to_content():
    reordered = dict(reversed(list(_BODY.items())))
    same = next(e for e in _append(reordered) if e["type"] == "reasoning")
    assert same["reasoning_ref"]["sha256"] == _canonical_addr(_BODY)
    changed = dict(_BODY, trace="an entirely different line of thought")
    other = next(e for e in _append(changed) if e["type"] == "reasoning")
    assert other["reasoning_ref"]["sha256"] != _canonical_addr(_BODY)


@pytest.mark.parametrize("flags,expected", [
    ({"authoritative": True}, "must be false"),
    ({"settles": "sha256:deadbeef"}, "cannot settle anything"),
])
def test_the_fence_refuses_before_anything_reaches_the_stream(flags, expected):
    """THE load-bearing test. A trace that could assert its own authority, or
    pose as a settlement, is the precise failure this design fences against.
    Deleting either guard in cmd_reasoning_append turns this red."""
    events = _append(_BODY, **flags)
    assert [e["type"] for e in events] == ["started", "error"]
    assert events[-1]["code"] == "parse_error"
    assert expected in events[-1]["message"]
    assert not any(e["type"] == "reasoning" for e in events), (
        "a refused record must never reach the stream"
    )


def test_authoritative_cannot_be_smuggled_through_the_body():
    """The stamp wins over body content — the body is addressed, not trusted."""
    ref = next(e for e in _append(dict(_BODY, authoritative=True))
               if e["type"] == "reasoning")["reasoning_ref"]
    assert ref["authoritative"] is False
    assert ref["role"] == "diagnostic"


def test_back_compat_every_prior_envelope_version_still_served():
    for v in ("happi/1.0", "happi/1.1", "happi/1.2", "happi/1.3", "happi/1.4",
              "happi/1.5", "happi/1.6"):
        events = _run({"v": v, "id": "bc", "cmd": "version"})
        assert events[-1]["type"] == "completed", f"{v} envelope was rejected"
        assert all(e["v"] == canonical_runtime_version() for e in events), (
            "emitted events must carry the RUNTIME version, not the envelope's"
        )
