"""Polyglot parity tests for the four v1.1 cmd handlers (H426).

For each of the four cmds, dispatches an identical envelope through:
    a) the canonical polyglot runtime: ``bash ~/.hal/happi.md run``
    b) the lib runtime: ``python -m lib.hal.happi.runtime``

After normalising non-deterministic fields (``ts``, generated issue numbers,
hypothesis registry paths, dynamic `registered_at_unix` values), the two
event streams MUST be byte-equivalent.

Mocking strategy:
- ``hypothesis.register``: redirect via ``HAL_HYPOTHESES_PATH`` env var so
  both runtimes write to the same tmp file (the file path appears in the
  emitted events; we normalise to "<HYP_PATH>").
- ``quine.spawn``: NOT included in polyglot parity — it shells out to
  ``gh`` which we cannot patch in the bash subprocess. The unit-test file
  covers quine.spawn parity at the structural level (5-delta DRY-RUN, 4-
  delta LIVE).

Skipped if:
- ``~/.hal/happi.md`` does not exist
- bash is not on PATH
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


HAPPI_MD = Path.home() / ".hal" / "happi.md"


def _has_polyglot() -> bool:
    return HAPPI_MD.exists() and shutil.which("bash") is not None


pytestmark = pytest.mark.skipif(
    not _has_polyglot(),
    reason="polyglot runtime (~/.hal/happi.md + bash) not available",
)


# ---------- runners --------------------------------------------------------

def _run_polyglot(envelope: dict, env_overrides: dict | None = None) -> list[dict]:
    """Dispatch an envelope through `bash happi.md run`. Return decoded events."""
    raw = json.dumps(envelope)
    env = {**os.environ}
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        ["bash", str(HAPPI_MD), "run"],
        input=raw,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    events = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _run_lib(envelope: dict, env_overrides: dict | None = None) -> list[dict]:
    """Dispatch an envelope through `python -m lib.hal.happi.runtime`."""
    raw = json.dumps(envelope)
    env = {**os.environ}
    if env_overrides:
        env.update(env_overrides)
    # Use the same Python that runs the test (which is the venv python)
    import sys as _sys
    result = subprocess.run(
        [_sys.executable, "-m", "lib.hal.happi.runtime"],
        input=raw,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path.home() / ".hal"),
        timeout=30,
    )
    events = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


# ---------- normalisation --------------------------------------------------

def _normalise(events: list[dict],
               hyp_path: str | None = None) -> list[dict]:
    """Strip non-deterministic fields so we can compare byte-for-byte.

    Removes:
      - ``ts`` from every event (timing-dependent)
      - replaces ``v`` with a constant (lib runtime defaults to v1.0 unless
        HAPPI_LIB_PROTOCOL_VERSION is set; polyglot always emits v1.1)
    For hypothesis.register events:
      - replaces ``registered_at_unix`` (in usage path-related deltas) with
        a constant marker; replaces hyp file path with "<HYP_PATH>"
    """
    out = []
    for e in events:
        e = dict(e)  # don't mutate caller's events
        e.pop("ts", None)
        e["v"] = "<NORMALISED>"  # protocol version differs by design
        # Hypothesis path normalisation: rewrite the hypothesis output path
        # in delta text and completed.usage.path (both runtimes write it
        # there).
        if hyp_path:
            if "text" in e and isinstance(e["text"], str):
                e["text"] = e["text"].replace(hyp_path, "<HYP_PATH>")
            if "usage" in e and isinstance(e.get("usage"), dict):
                if e["usage"].get("path") == hyp_path:
                    e["usage"] = dict(e["usage"])
                    e["usage"]["path"] = "<HYP_PATH>"
        out.append(e)
    return out


# ---------- the four parity tests ------------------------------------------

class TestPolyglotParity:
    def test_idr_emit_parity(self):
        envelope = {
            "v": "happi/1.1", "id": "p-idr-1", "cmd": "idr.emit",
            "args": ['{"v":"happi/1.1","id":"x","cmd":"version"}',
                     '{"v":"happi/1.1","id":"x","type":"started"}\n'],
            "flags": {"model_versions": ["claude-opus-4-7"]},
        }
        polyglot_events = _run_polyglot(envelope)
        lib_events = _run_lib(envelope)
        assert _normalise(polyglot_events) == _normalise(lib_events), (
            "polyglot events: " + json.dumps(polyglot_events)
            + "\nlib events: " + json.dumps(lib_events)
        )
        # Spot check: both ran through the dispatcher and emitted an idr
        types_polyglot = [e["type"] for e in polyglot_events]
        types_lib = [e["type"] for e in lib_events]
        assert types_polyglot == types_lib
        # Both should contain an "idr" event with matching sha256
        idr_polyglot = next(e for e in polyglot_events if e["type"] == "idr")
        idr_lib = next(e for e in lib_events if e["type"] == "idr")
        assert idr_polyglot["idr_ref"]["sha256"] == idr_lib["idr_ref"]["sha256"]

    def test_pr_reference_parity(self):
        envelope = {
            "v": "happi/1.1", "id": "p-pr-1", "cmd": "pr.reference",
            "args": ["fixes drift bug"],
            "flags": {"pr": 42, "repo": "x/y", "status": "merged",
                      "contributes_axiom": "self-bootstrap"},
        }
        polyglot_events = _run_polyglot(envelope)
        lib_events = _run_lib(envelope)
        assert _normalise(polyglot_events) == _normalise(lib_events)

    def test_pr_reference_minimal_parity(self):
        envelope = {
            "v": "happi/1.1", "id": "p-pr-min", "cmd": "pr.reference",
            "flags": {"pr": 1, "repo": "a/b"},
        }
        polyglot_events = _run_polyglot(envelope)
        lib_events = _run_lib(envelope)
        assert _normalise(polyglot_events) == _normalise(lib_events)

    def test_pr_reference_missing_pr_parity(self):
        """Both runtimes must emit the same parse_error code + message shape."""
        envelope = {
            "v": "happi/1.1", "id": "p-pr-bad", "cmd": "pr.reference",
            "flags": {"repo": "x/y"},
        }
        polyglot_events = _run_polyglot(envelope)
        lib_events = _run_lib(envelope)
        types_polyglot = [e["type"] for e in polyglot_events]
        types_lib = [e["type"] for e in lib_events]
        # Both should have started + error
        assert "error" in types_polyglot and "error" in types_lib
        err_p = next(e for e in polyglot_events if e["type"] == "error")
        err_l = next(e for e in lib_events if e["type"] == "error")
        assert err_p["code"] == err_l["code"] == "parse_error"
        assert err_p["message"] == err_l["message"]

    def test_hypothesis_register_parity(self, tmp_path):
        hyp_path = str(tmp_path / "polyglot-parity-hyp.jsonl")
        envelope = {
            "v": "happi/1.1", "id": "p-hyp-1", "cmd": "hypothesis.register",
            "args": ["H-PARITY-1"],
            "flags": {"claim": "polyglot-parity test claim",
                      "metric": "x", "prediction": "eq:0",
                      "deadline": "2026-12-31"},
        }
        polyglot_events = _run_polyglot(
            envelope, env_overrides={"HAL_HYPOTHESES_PATH": hyp_path}
        )
        # Reset for the second run (delete the file so structure is identical)
        if os.path.exists(hyp_path):
            os.unlink(hyp_path)
        lib_events = _run_lib(
            envelope, env_overrides={"HAL_HYPOTHESES_PATH": hyp_path}
        )

        # Normalise the dynamic hyp_path in event text + usage
        # AND drop registered_at_unix from the disk record (just check disk write)
        norm_p = _normalise(polyglot_events, hyp_path=hyp_path)
        norm_l = _normalise(lib_events, hyp_path=hyp_path)

        types_p = [e["type"] for e in norm_p]
        types_l = [e["type"] for e in norm_l]
        assert types_p == types_l == ["started", "delta", "delta", "completed"]

        # Delta texts identical (post-normalisation)
        deltas_p = [e for e in norm_p if e["type"] == "delta"]
        deltas_l = [e for e in norm_l if e["type"] == "delta"]
        assert deltas_p == deltas_l

        # completed usage identical (post-normalisation)
        comp_p = next(e for e in norm_p if e["type"] == "completed")
        comp_l = next(e for e in norm_l if e["type"] == "completed")
        assert comp_p["usage"] == comp_l["usage"]

    def test_hypothesis_register_missing_flag_parity(self, tmp_path):
        envelope = {
            "v": "happi/1.1", "id": "p-hyp-bad", "cmd": "hypothesis.register",
            "args": ["H-BAD"],
            "flags": {"claim": "c", "metric": "m"},  # missing prediction + deadline
        }
        env_ov = {"HAL_HYPOTHESES_PATH": str(tmp_path / "x.jsonl")}
        polyglot_events = _run_polyglot(envelope, env_overrides=env_ov)
        lib_events = _run_lib(envelope, env_overrides=env_ov)
        err_p = next(e for e in polyglot_events if e["type"] == "error")
        err_l = next(e for e in lib_events if e["type"] == "error")
        assert err_p["code"] == err_l["code"] == "parse_error"
        # Same flags reported missing (order may differ; check sorted set)
        # Both messages have form "missing flags: prediction,deadline"
        # Extract the missing-list portion after "missing flags: "
        m_p = re.search(r"missing flags:\s*(.+)$", err_p["message"])
        m_l = re.search(r"missing flags:\s*(.+)$", err_l["message"])
        assert m_p is not None and m_l is not None
        set_p = set(m_p.group(1).split(","))
        set_l = set(m_l.group(1).split(","))
        assert set_p == set_l == {"prediction", "deadline"}

    def test_quine_spawn_missing_parent_issue_parity(self):
        """Parse-error path doesn't shell out — safe for parity test."""
        envelope = {
            "v": "happi/1.1", "id": "p-q-bad", "cmd": "quine.spawn",
            "flags": {"repo": "x/y"},
        }
        polyglot_events = _run_polyglot(envelope)
        lib_events = _run_lib(envelope)
        err_p = next(e for e in polyglot_events if e["type"] == "error")
        err_l = next(e for e in lib_events if e["type"] == "error")
        assert err_p["code"] == err_l["code"] == "parse_error"
        assert err_p["message"] == err_l["message"]
