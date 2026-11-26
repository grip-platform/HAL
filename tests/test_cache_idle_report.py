"""Tests for lib/hal/cache_telemetry.py and lib/hal/cache_idle_report.py.

All tests are:
- CBC (correct-by-construction): assert on values, not just call counts.
- Goodhart-resistant: mutations to the classifying logic would flip assertions.
- Network-free: no I/O beyond tmp file writes.

Test IDs:
    TCI-01  cold-after-idle: cache_creation 600 s after prior call => classified
    TCI-02  within-TTL:      120 s gap => NOT cold-after-idle (still warm)
    TCI-03  beyond-window:   7200 s gap => NOT cold-after-idle (too stale)
    TCI-04  determinism:     same input => same output every run
    TCI-05  mutation guard:  removing the prior same-prefix call flips classification
    TCI-06  empty JSONL:     no records => zeros, no exception
    TCI-07  single call:     one record => total_calls=1, cold_after_idle=0
    TCI-08  logger appends:  log_cache_call appends a correct line
    TCI-09  logger fail-soft: unwritable dir => no exception raised
    TCI-10  opt-out env:     HAL_DISABLE_CACHE_TELEMETRY=1 => nothing written
"""
from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module-level import of the modules under test
# ---------------------------------------------------------------------------
import sys

_HAL_ROOT = Path(__file__).resolve().parent.parent
if str(_HAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_HAL_ROOT))

from lib.hal.cache_telemetry import log_cache_call, compute_prefix_hash  # noqa: E402
from lib.hal.cache_idle_report import analyse, ReportData, _TTL_S, _WINDOW_S  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _make_record(
    prefix_hash: str,
    ts: float,
    cache_creation: int = 0,
    cache_read: int = 0,
    input_tokens: int = 100,
    output_tokens: int = 50,
    model: str = "claude-opus-4-7",
    provider: str = "anthropic",
) -> dict:
    return {
        "ts": ts,
        "model": model,
        "provider": provider,
        "prefix_hash": prefix_hash,
        "cache_creation": cache_creation,
        "cache_read": cache_read,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# ---------------------------------------------------------------------------
# TCI-01: cold-after-idle (gap in open interval (300, 3600))
# ---------------------------------------------------------------------------

def test_tci_01_cold_after_idle_classified(tmp_path):
    """A cache_creation call 600 s after a prior same-prefix call is cold-after-idle."""
    ph = "abc123def456abcd"
    t0 = 1_000_000.0
    records = [
        _make_record(ph, t0, cache_creation=4096),   # first call — creates cache
        _make_record(ph, t0 + 600, cache_creation=4096),  # 600 s later — cold resume
    ]
    p = tmp_path / "cache_calls.jsonl"
    _write_jsonl(p, records)

    data = analyse(p)

    assert data.total_calls == 2
    assert data.total_resumes == 2
    assert data.cold_after_idle == 1
    assert data.idle_gap_hist["5-15m"] == 1
    assert data.would_have_saved_tok_equiv > 0


# ---------------------------------------------------------------------------
# TCI-02: within TTL — should NOT be cold-after-idle
# ---------------------------------------------------------------------------

def test_tci_02_within_ttl_not_cold(tmp_path):
    """A 120 s gap is within the ~5 min Anthropic TTL — not a cold-after-idle resume."""
    ph = "1234567890abcdef"
    t0 = 2_000_000.0
    records = [
        _make_record(ph, t0, cache_creation=2048),
        _make_record(ph, t0 + 120, cache_creation=2048),  # 2 min — still warm
    ]
    p = tmp_path / "cache_calls.jsonl"
    _write_jsonl(p, records)

    data = analyse(p)

    assert data.cold_after_idle == 0
    assert data.idle_gap_hist["<5m"] == 1
    assert data.would_have_saved_tok_equiv == 0.0


# ---------------------------------------------------------------------------
# TCI-03: beyond addressable window (>3600 s) — NOT classified
# ---------------------------------------------------------------------------

def test_tci_03_beyond_window_not_classified(tmp_path):
    """A 7200 s gap is beyond the 1-hour addressable window — a keepalive wouldn't help."""
    ph = "feedcafe12345678"
    t0 = 3_000_000.0
    records = [
        _make_record(ph, t0, cache_creation=1024),
        _make_record(ph, t0 + 7200, cache_creation=1024),  # 2 h later — beyond window
    ]
    p = tmp_path / "cache_calls.jsonl"
    _write_jsonl(p, records)

    data = analyse(p)

    assert data.cold_after_idle == 0
    assert data.idle_gap_hist[">60m"] == 1
    assert data.would_have_saved_tok_equiv == 0.0


# ---------------------------------------------------------------------------
# TCI-04: determinism
# ---------------------------------------------------------------------------

def test_tci_04_determinism(tmp_path):
    """Same input produces identical output across two runs."""
    ph = "deaddead12341234"
    t0 = 4_000_000.0
    records = [
        _make_record(ph, t0, cache_creation=512),
        _make_record(ph, t0 + 400, cache_creation=512),
    ]
    p = tmp_path / "cache_calls.jsonl"
    _write_jsonl(p, records)

    result_a = analyse(p)
    result_b = analyse(p)

    assert result_a.cold_after_idle == result_b.cold_after_idle
    assert result_a.total_resumes == result_b.total_resumes
    assert result_a.would_have_saved_tok_equiv == result_b.would_have_saved_tok_equiv


# ---------------------------------------------------------------------------
# TCI-05: mutation guard — removing prior call flips classification
# ---------------------------------------------------------------------------

def test_tci_05_mutation_guard(tmp_path):
    """Removing the prior same-prefix call means there is no gap to classify."""
    ph = "cafebabe87654321"
    t0 = 5_000_000.0

    # With prior call: cold-after-idle expected
    records_with_prior = [
        _make_record(ph, t0, cache_creation=2000),
        _make_record(ph, t0 + 500, cache_creation=2000),
    ]
    p = tmp_path / "with_prior.jsonl"
    _write_jsonl(p, records_with_prior)
    data_with = analyse(p)
    assert data_with.cold_after_idle == 1

    # Without prior call: no gap => not classified
    records_without = [
        _make_record(ph, t0 + 500, cache_creation=2000),  # only the second call
    ]
    p2 = tmp_path / "without_prior.jsonl"
    _write_jsonl(p2, records_without)
    data_without = analyse(p2)
    assert data_without.cold_after_idle == 0


# ---------------------------------------------------------------------------
# TCI-06: empty JSONL
# ---------------------------------------------------------------------------

def test_tci_06_empty_jsonl(tmp_path):
    """Empty file yields all zeros without raising."""
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")

    data = analyse(p)

    assert data.total_calls == 0
    assert data.total_resumes == 0
    assert data.cold_after_idle == 0
    assert data.would_have_saved_tok_equiv == 0.0


# ---------------------------------------------------------------------------
# TCI-07: single call — no prior, so never cold-after-idle
# ---------------------------------------------------------------------------

def test_tci_07_single_call(tmp_path):
    """One call with cache_creation > 0 registers as one resume but never cold-after-idle."""
    ph = "0000000000000001"
    p = tmp_path / "single.jsonl"
    _write_jsonl(p, [_make_record(ph, 1_000.0, cache_creation=100)])

    data = analyse(p)

    assert data.total_calls == 1
    assert data.total_resumes == 1
    assert data.cold_after_idle == 0


# ---------------------------------------------------------------------------
# TCI-08: logger appends a correct line
# ---------------------------------------------------------------------------

def test_tci_08_logger_appends(tmp_path, monkeypatch):
    """log_cache_call writes a valid JSON record with all expected fields."""
    log_path = tmp_path / "state" / "hal" / "cache_calls.jsonl"

    # Patch the module-level path constant so writes go to tmp_path
    import lib.hal.cache_telemetry as ct
    monkeypatch.setattr(ct, "_TELEMETRY_DIR", log_path.parent)
    monkeypatch.setattr(ct, "_TELEMETRY_FILE", log_path)

    ts_before = time.time()
    log_cache_call(
        model="claude-opus-4-7",
        provider="anthropic",
        prefix_hash="aabbccdd11223344",
        cache_creation=4096,
        cache_read=0,
        input_tokens=5000,
        output_tokens=200,
    )
    ts_after = time.time()

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    rec = json.loads(lines[0])
    assert rec["model"] == "claude-opus-4-7"
    assert rec["provider"] == "anthropic"
    assert rec["prefix_hash"] == "aabbccdd11223344"
    assert rec["cache_creation"] == 4096
    assert rec["cache_read"] == 0
    assert rec["input_tokens"] == 5000
    assert rec["output_tokens"] == 200
    assert ts_before <= rec["ts"] <= ts_after


# ---------------------------------------------------------------------------
# TCI-09: logger fail-soft on unwritable directory
# ---------------------------------------------------------------------------

def test_tci_09_logger_fail_soft(tmp_path, monkeypatch):
    """log_cache_call raises no exception when the target dir is unwritable."""
    import lib.hal.cache_telemetry as ct

    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    locked_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # read+exec, no write

    fake_file = locked_dir / "cache_calls.jsonl"
    monkeypatch.setattr(ct, "_TELEMETRY_DIR", locked_dir)
    monkeypatch.setattr(ct, "_TELEMETRY_FILE", fake_file)

    # Must not raise
    log_cache_call(
        model="test",
        provider="test",
        prefix_hash="0000000000000000",
        cache_creation=1,
        cache_read=0,
        input_tokens=1,
        output_tokens=1,
    )

    # Restore permissions so tmp_path cleanup doesn't fail
    locked_dir.chmod(stat.S_IRWXU)


# ---------------------------------------------------------------------------
# TCI-10: opt-out env var suppresses all writes
# ---------------------------------------------------------------------------

def test_tci_10_opt_out_env(tmp_path, monkeypatch):
    """HAL_DISABLE_CACHE_TELEMETRY=1 means no file is created."""
    monkeypatch.setenv("HAL_DISABLE_CACHE_TELEMETRY", "1")

    log_path = tmp_path / "state" / "hal" / "cache_calls.jsonl"
    import lib.hal.cache_telemetry as ct
    monkeypatch.setattr(ct, "_TELEMETRY_DIR", log_path.parent)
    monkeypatch.setattr(ct, "_TELEMETRY_FILE", log_path)

    log_cache_call(
        model="claude-sonnet-4-6",
        provider="anthropic",
        prefix_hash="ffffffffffffffff",
        cache_creation=100,
        cache_read=0,
        input_tokens=200,
        output_tokens=50,
    )

    assert not log_path.exists()


# ---------------------------------------------------------------------------
# TCI-11: compute_prefix_hash is stable and hex-16
# ---------------------------------------------------------------------------

def test_tci_11_prefix_hash_stable():
    """compute_prefix_hash produces a 16-char hex string, stable across calls."""
    ph1 = compute_prefix_hash("system prompt text", "tools json")
    ph2 = compute_prefix_hash("system prompt text", "tools json")
    assert ph1 == ph2
    assert len(ph1) == 16
    assert all(c in "0123456789abcdef" for c in ph1)

    # Different inputs => different hash (falsifier)
    ph3 = compute_prefix_hash("different prompt", "tools json")
    assert ph3 != ph1
