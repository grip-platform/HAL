"""Goodhart-proof P3 W3b — `hal llm` is a pure front-door to GRIP's
canonical active-llm CLI: one store, two doors (syscall-doctrine).

Byte-identical symmetry with GRIP `/llm` is guaranteed BY CONSTRUCTION
— the block `exec`s the SAME `python3 -m lib.active_llm` process, so a
2-process diff would only re-test what `exec` already guarantees. The
real, mutation-killable invariants are: it forwards verbatim, it adds
NO second store / no marker logic, and it fails safe when GRIP is
absent. Hermetic: absent-path uses a guaranteed-missing dir; present
-path uses read-only `show` (never mutates the real marker).

Mutation-survival:
  - drop the forward / change the module      -> test_forwards_to_canonical_module FAILS
  - add a 2nd store / marker logic in bin/hal -> test_no_second_store_no_marker_logic FAILS
  - GRIP absent not fail-safe (exit 0 / write) -> test_absent_grip_failsafe FAILS
  - block doesn't reach the real module        -> test_present_reaches_canonical_module FAILS
  - bespoke (not the canonical dispatch idiom) -> test_block_mirrors_dispatch_idiom FAILS
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_HAL = Path(__file__).resolve().parents[1] / "bin" / "hal"


def _llm_block() -> str:
    """The bytes between the `== "llm"` guard and its closing `fi`."""
    src = _HAL.read_text()
    i = src.index('"${1:-}" == "llm"')
    fi = src.index("\nfi\n", i)
    return src[i:fi]


def test_forwards_to_canonical_module():
    blk = _llm_block()
    assert "shift" in blk                                  # consume `llm`
    assert "python3 -m lib.active_llm" in blk               # canonical module
    assert '"$@"' in blk                                    # verbatim forward
    assert 'exec ' in blk                                   # replace process


def test_no_second_store_no_marker_logic():
    """The one-store invariant: bin/hal must NOT write/own a marker —
    no 2nd active-llm.json, no json serialisation, no set_active."""
    blk = _llm_block()
    # Precise marker-WRITE surfaces only. NOT a bare ">" — the block
    # legitimately uses ">&2" for the fail-safe stderr message, which is
    # not a store write (over-broad substring matching was Goodhart-weak,
    # cf. the W2 "closeout" / W3 "Agent(" precision fixes). Any real
    # shell 2nd-store write necessarily touches the marker path or a
    # set_active call — all caught semantically below.
    for forbidden in ("active-llm.json", "active_llm.json", "json.dump",
                      "set_active", "set_council", "state/session"):
        assert forbidden not in blk, f"2nd-store/marker surface: {forbidden!r}"
    # The only redirect in the block must be the stderr fd dup (>&2),
    # never a redirect to a file path (which would be a 2nd-store write).
    assert all(r == ">&2" for r in re.findall(r">\S*", blk)), \
        "unexpected redirect — possible 2nd-store write"


def test_absent_grip_failsafe(tmp_path):
    missing = str(tmp_path / "no-grip-here")               # guaranteed absent
    r = subprocess.run(["bash", str(_HAL), "llm", "show"],
                        capture_output=True, text=True, timeout=20,
                        env={**os.environ, "CLAUDE_DIR": missing})
    assert r.returncode != 0                                # fail-safe non-zero
    assert "GRIP home not found" in r.stderr
    # never created a second store under the bogus dir
    assert not Path(missing).exists()


def test_present_reaches_canonical_module():
    grip = os.environ.get("CLAUDE_DIR", os.path.expanduser("~/.claude"))
    if not Path(grip, "lib", "active_llm.py").is_file():
        import pytest
        pytest.skip("GRIP not installed in this env")
    # `show` is READ-ONLY — never mutates the marker.
    r = subprocess.run(["bash", str(_HAL), "llm", "show"],
                        capture_output=True, text=True, timeout=30,
                        env={**os.environ, "CLAUDE_DIR": grip})
    assert r.returncode == 0
    out = r.stdout.strip()
    # canonical active_llm `show` prints either marker JSON or the
    # explicit no-marker line — proves the door reached the real module.
    assert out.startswith("{") or "No active LLM" in out


def test_block_mirrors_dispatch_idiom():
    src = _HAL.read_text()
    # same shape as the canonical save/welcome/voice blocks
    assert re.search(r'if \[\[ "\$\{1:-\}" == "llm" \]\]; then', src)
    assert re.search(r'if \[\[ "\$\{1:-\}" == "save" \]\]; then', src)  # sibling exists
