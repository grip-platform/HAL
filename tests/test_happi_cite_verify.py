"""Goodhart-resistant tests for the HAPPI cite.verify cmd (v1.3).

The deterministic citation-provenance floor, exercised through the embedded
runtime exactly as any harness would — `bash happi.md run` with a JSON envelope.
THE falsifier: a quote absent from its source MUST resolve not_found. Mutate the
verifier to always-pass and test_fabricated_quote_is_not_found fails. A
hallucinated citation cannot be proven — on any AI, on any harness.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests._bash import find_bash

REPO = Path(__file__).resolve().parent.parent
HAPPI = REPO / "happi.md"
_BASH = find_bash()
pytestmark = pytest.mark.skipif(
    _BASH is None,
    reason="no usable bash found on this machine (install Git Bash on Windows)",
)

SRC = "The Limitation Act 1980 bars the claim after six years."


def run_cite(flags):
    """Dispatch a cite.verify envelope through happi.md; return (rc, events)."""
    env = {"v": "happi/1.3", "id": "t", "cmd": "cite.verify", "flags": flags}
    proc = subprocess.run(
        [_BASH, str(HAPPI), "run"],
        input=json.dumps(env), capture_output=True, text=True,
    )
    events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    return proc.returncode, events


def _completed(events):
    return next((e for e in events if e.get("type") == "completed"), None)


def test_verified_quote_carries_real_offsets():
    rc, events = run_cite({
        "sources": [{"id": "s1", "text": SRC}],
        "citations": [{"id": "c1", "source_id": "s1", "quote": "bars the claim after six years"}],
    })
    assert rc == 0
    c1 = _completed(events)["usage"]["citations"][0]
    assert c1["status"] == "verified"
    assert SRC[c1["start"]:c1["end"]] == "bars the claim after six years"


def test_fabricated_quote_is_not_found():
    # THE falsifier — mutate the verifier to always-pass and this fails.
    rc, events = run_cite({
        "sources": [{"id": "s1", "text": SRC}],
        "citations": [{"id": "c1", "source_id": "s1", "quote": "bars the claim after three years"}],
    })
    rec = _completed(events)["usage"]
    assert rec["citations"][0]["status"] == "not_found"
    assert rec["tally"]["not_found"] == 1
    assert rec["grounding_rate"] == 0.0


def test_unknown_source_is_not_found():
    rc, events = run_cite({
        "sources": [{"id": "s1", "text": SRC}],
        "citations": [{"id": "c1", "source_id": "nope", "quote": "anything"}],
    })
    assert _completed(events)["usage"]["citations"][0]["status"] == "not_found"


def test_whitespace_variant_is_fuzzy():
    rc, events = run_cite({
        "sources": [{"id": "s1", "text": "bars the\nclaim after six years"}],
        "citations": [{"id": "c1", "source_id": "s1", "quote": "bars the claim after"}],
    })
    assert _completed(events)["usage"]["citations"][0]["status"] == "fuzzy"


def test_per_source_sha256_and_chars():
    rc, events = run_cite({
        "sources": [{"id": "s1", "text": SRC}],
        "citations": [{"id": "c1", "source_id": "s1", "quote": "Limitation Act"}],
    })
    src = _completed(events)["usage"]["sources"]["s1"]
    assert src["sha256"] == hashlib.sha256(SRC.encode("utf-8")).hexdigest()
    assert src["chars"] == len(SRC)


def test_strict_mode_errors_on_any_not_found():
    rc, events = run_cite({
        "strict": True,
        "sources": [{"id": "s1", "text": "alpha beta"}],
        "citations": [{"id": "c1", "source_id": "s1", "quote": "gamma delta"}],
    })
    assert rc != 0
    assert any(e.get("type") == "error" for e in events)
    assert not any(e.get("type") == "completed" for e in events)


def test_malformed_flags_emit_parse_error():
    rc, events = run_cite({"sources": "notalist", "citations": []})
    assert rc != 0
    assert any(e.get("type") == "error" and e.get("code") == "parse_error" for e in events)


def test_cross_runtime_agreement_with_grip_prove_it():
    # The any-AI/any-harness guarantee: HAPPI and GRIP prove_it MUST agree on the
    # same (quote, source) — same verdict AND same offsets. Skipped where the GRIP
    # lib is not on the path (e.g. HAL CI), where this is not exercised.
    grip_lib = Path.home() / ".claude" / "lib"
    if not (grip_lib / "prove_it.py").is_file():
        pytest.skip("GRIP prove_it not present in this environment")
    sys.path.insert(0, str(grip_lib))
    import prove_it as pi
    for quote in ("bars the claim after six years", "bars the claim after three years"):
        _, events = run_cite({
            "sources": [{"id": "s1", "text": SRC}],
            "citations": [{"id": "c1", "source_id": "s1", "quote": quote}],
        })
        happi_c = _completed(events)["usage"]["citations"][0]
        grip_status, grip_start, grip_end = pi.verify_quote(quote, SRC)
        assert happi_c["status"] == grip_status
        assert (happi_c["start"], happi_c["end"]) == (grip_start, grip_end)
