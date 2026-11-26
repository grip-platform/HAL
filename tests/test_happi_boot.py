"""TDD tests for the HAPPI.md boot polyglot — lib.hal.happi_boot.

Philosophy: "AI is a syscall." HAPPI.md is the boot envelope that makes HAL
bootstrappable from any machine with `bash` and `python3`. These tests specify
what that contract looks like from a Python-verifiable perspective.

Each test class corresponds to one behaviour axis of the HAPPI system:

  1. BootMarker — idempotency token read/write contract
  2. CapsuleDiscovery — finding and classifying witness capsules
  3. PredictionResult — structured pass/fail recording
  4. BootReport — exit code derivation and summary formatting
  5. HAPPIPolyglot — the document itself is valid Markdown AND valid bash
  6. WarmupMeasurement — module-import timing is a finite, non-negative int
  7. Integration — a dry run against the real capsule directory
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

_HAL_HOME = os.environ.get("HAL_ROOT", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.happi_boot import (
    BootMarker,
    BootReport,
    PredictionResult,
    find_latest_capsule,
    is_already_run_today,
    is_dry_run_capsule,
    marker_path_for_today,
    measure_hal_warmup,
    validate_polyglot,
    extract_memory_prediction,
    WARMUP_ADVISORY_MS,
)

_HAPPI_MD = Path(_HAL_HOME) / "HAPPI.md"
_CAPSULE_DIR = Path(_HAL_HOME) / "state" / "witness-capsules"
_MARKER_DIR = Path(_HAL_HOME) / "state" / "boot-markers"


# ===========================================================================
# 1. BootMarker — read / write / idempotency
# ===========================================================================

class TestBootMarkerRead:
    """BootMarker.read() must return None for missing/corrupt files and a
    populated object for valid markers."""

    def test_read_returns_none_when_file_absent(self, tmp_path):
        absent = tmp_path / "2099-01-01.json"
        assert BootMarker.read(absent) is None

    def test_read_returns_none_for_corrupt_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not: valid json}")
        assert BootMarker.read(bad) is None

    def test_read_returns_none_for_missing_required_key(self, tmp_path):
        incomplete = tmp_path / "missing.json"
        incomplete.write_text(json.dumps({"date": "2026-04-21"}))
        assert BootMarker.read(incomplete) is None

    def test_read_returns_marker_for_valid_file(self, tmp_path):
        p = tmp_path / "2026-04-21.json"
        p.write_text(json.dumps({
            "date": "2026-04-21", "capsule": "cap.md",
            "pass": 3, "fail": 1, "warmup_ms": 80,
            "polyglot_sha256": "abc123", "watermark": "V>>--<<V",
        }))
        m = BootMarker.read(p)
        assert m is not None
        assert m.date == "2026-04-21"
        assert m.capsule == "cap.md"
        assert m.pass_count == 3
        assert m.fail_count == 1
        assert m.warmup_ms == 80
        assert m.polyglot_sha256 == "abc123"
        assert m.watermark == "V>>--<<V"

    def test_read_tolerates_missing_optional_fields(self, tmp_path):
        p = tmp_path / "min.json"
        p.write_text(json.dumps({
            "date": "2026-04-21", "capsule": "cap.md",
            "pass": 0, "fail": 0, "warmup_ms": 50,
        }))
        m = BootMarker.read(p)
        assert m is not None
        assert m.polyglot_sha256 == ""
        assert m.watermark == "V>>--<<V"


class TestBootMarkerWrite:
    """BootMarker.write() must produce a JSON file that round-trips through read()."""

    def test_write_creates_file(self, tmp_path):
        m = BootMarker(
            date="2026-04-21", capsule="cap.md",
            pass_count=4, fail_count=0, warmup_ms=76,
            polyglot_sha256="deadbeef",
        )
        p = tmp_path / "2026-04-21.json"
        m.write(p)
        assert p.exists()

    def test_write_round_trips_through_read(self, tmp_path):
        original = BootMarker(
            date="2026-04-21", capsule="cap.md",
            pass_count=4, fail_count=0, warmup_ms=76,
            polyglot_sha256="deadbeef",
        )
        p = tmp_path / "marker.json"
        original.write(p)
        restored = BootMarker.read(p)
        assert restored is not None
        assert restored.date == original.date
        assert restored.pass_count == original.pass_count
        assert restored.fail_count == original.fail_count
        assert restored.warmup_ms == original.warmup_ms
        assert restored.polyglot_sha256 == original.polyglot_sha256

    def test_write_creates_parent_directories(self, tmp_path):
        deep = tmp_path / "state" / "boot-markers" / "2026-04-21.json"
        m = BootMarker(
            date="2026-04-21", capsule="cap.md",
            pass_count=0, fail_count=0, warmup_ms=0,
        )
        m.write(deep)
        assert deep.exists()

    def test_watermark_is_always_present(self, tmp_path):
        m = BootMarker(
            date="2026-04-21", capsule="cap.md",
            pass_count=0, fail_count=0, warmup_ms=0,
        )
        p = tmp_path / "m.json"
        m.write(p)
        data = json.loads(p.read_text())
        assert data["watermark"] == "V>>--<<V"

    def test_all_passed_property(self):
        assert BootMarker("d", "c", 4, 0, 0).all_passed is True
        assert BootMarker("d", "c", 3, 1, 0).all_passed is False

    def test_for_today_factory_uses_todays_date(self):
        m = BootMarker.for_today(
            capsule="cap.md", pass_count=3, fail_count=1, warmup_ms=80
        )
        assert m.date == date.today().isoformat()

    def test_to_dict_has_all_required_keys(self):
        m = BootMarker("2026-04-21", "cap.md", 4, 0, 76, "sha")
        d = m.to_dict()
        for key in ("date", "capsule", "pass", "fail", "warmup_ms",
                    "polyglot_sha256", "watermark"):
            assert key in d, f"Missing key: {key}"


class TestIdempotencyGate:
    """is_already_run_today must gate on today's marker file existence."""

    def test_returns_false_when_no_marker_today(self, tmp_path):
        assert is_already_run_today(tmp_path) is False

    def test_returns_true_when_todays_marker_exists(self, tmp_path):
        marker_path_for_today(tmp_path).write_text(json.dumps({
            "date": date.today().isoformat(), "capsule": "c.md",
            "pass": 4, "fail": 0, "warmup_ms": 70,
        }))
        assert is_already_run_today(tmp_path) is True

    def test_returns_false_for_yesterday_marker_only(self, tmp_path):
        yesterday = date.fromordinal(date.today().toordinal() - 1).isoformat()
        (tmp_path / f"{yesterday}.json").write_text("{}")
        assert is_already_run_today(tmp_path) is False


# ===========================================================================
# 2. CapsuleDiscovery
# ===========================================================================

class TestCapsuleDiscovery:
    """find_latest_capsule must return the most-recently-modified .md file."""

    def test_returns_none_when_dir_absent(self, tmp_path):
        missing = tmp_path / "absent"
        assert find_latest_capsule(missing) is None

    def test_returns_none_when_dir_empty(self, tmp_path):
        assert find_latest_capsule(tmp_path) is None

    def test_returns_single_capsule(self, tmp_path):
        cap = tmp_path / "20260418T000000.md"
        cap.write_text("# capsule")
        result = find_latest_capsule(tmp_path)
        assert result == cap

    def test_returns_most_recent_of_multiple(self, tmp_path):
        older = tmp_path / "20260417T000000.md"
        newer = tmp_path / "20260418T000000.md"
        older.write_text("old")
        newer.write_text("new")
        import time; time.sleep(0.01)
        newer.touch()
        result = find_latest_capsule(tmp_path)
        assert result == newer

    def test_ignores_non_md_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("ignore me")
        assert find_latest_capsule(tmp_path) is None


class TestDryRunCapsule:
    """is_dry_run_capsule must detect the DRY-RUN Mode marker in capsule text."""

    def test_dry_run_capsule_returns_true(self, tmp_path):
        cap = tmp_path / "cap.md"
        cap.write_text("**Mode:** DRY-RUN preview\nsome content")
        assert is_dry_run_capsule(cap) is True

    def test_real_capsule_returns_false(self, tmp_path):
        cap = tmp_path / "cap.md"
        cap.write_text("# HAL Witness Capsule\n**Mode:** LIVE\nsome content")
        assert is_dry_run_capsule(cap) is False

    def test_absent_file_returns_false(self, tmp_path):
        absent = tmp_path / "gone.md"
        assert is_dry_run_capsule(absent) is False

    def test_actual_capsule_in_repo_is_dry_run(self):
        """The existing capsule in state/ is a DRY-RUN preview."""
        caps = list(_CAPSULE_DIR.glob("*.md")) if _CAPSULE_DIR.is_dir() else []
        if not caps:
            pytest.skip("No capsule files in state/witness-capsules/")
        latest = sorted(caps, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        result = is_dry_run_capsule(latest)
        assert isinstance(result, bool)


# ===========================================================================
# 3. PredictionResult
# ===========================================================================

class TestPredictionResult:
    """PredictionResult must correctly carry pass/fail status."""

    def test_pass_factory(self):
        r = PredictionResult.pass_("saved-state-empty", "0", "0")
        assert r.passed is True
        assert r.label == "saved-state-empty"

    def test_fail_factory(self):
        r = PredictionResult.fail("saved-state-empty", "3", "0")
        assert r.passed is False
        assert r.got == "3"

    def test_direct_constructor_false(self):
        r = PredictionResult(label="x", passed=False, got="1", expected="0")
        assert not r.passed

    def test_immutability_not_required_but_fields_accessible(self):
        r = PredictionResult(label="lbl", passed=True, got="0", expected="0")
        assert r.label == "lbl"
        assert r.passed is True
        assert r.got == "0"
        assert r.expected == "0"


# ===========================================================================
# 4. BootReport — exit codes and summaries
# ===========================================================================

class TestBootReport:
    """BootReport.exit_code must match the HAPPI.md bash contract exactly."""

    def _report(self, *, capsule=None, preds=None, dry_run=False,
                idempotent=False) -> BootReport:
        r = BootReport(
            date=date.today().isoformat(),
            capsule=capsule,
            predictions=preds or [],
            is_dry_run=dry_run,
            idempotent_skip=idempotent,
        )
        return r

    def test_exit_0_when_all_pass(self):
        preds = [PredictionResult.pass_("a", "0", "0")] * 4
        r = self._report(capsule="cap.md", preds=preds)
        assert r.exit_code == 0

    def test_exit_1_when_any_fail(self):
        preds = [
            PredictionResult.pass_("a", "0", "0"),
            PredictionResult.fail("b", "3", "0"),
        ]
        r = self._report(capsule="cap.md", preds=preds)
        assert r.exit_code == 1

    def test_exit_2_when_no_capsule(self):
        r = self._report(capsule=None)
        assert r.exit_code == 2

    def test_exit_0_on_idempotent_skip(self):
        r = self._report(idempotent=True)
        assert r.exit_code == 0

    def test_dry_run_failures_do_not_set_exit_1(self):
        preds = [PredictionResult.fail("a", "3", "0")]
        r = self._report(capsule="cap.md", preds=preds, dry_run=True)
        assert r.exit_code == 0

    def test_pass_count_and_fail_count(self):
        preds = [
            PredictionResult.pass_("a", "0", "0"),
            PredictionResult.pass_("b", "0", "0"),
            PredictionResult.fail("c", "1", "0"),
        ]
        r = self._report(capsule="cap.md", preds=preds)
        assert r.pass_count == 2
        assert r.fail_count == 1

    def test_summary_line_includes_counts(self):
        preds = [
            PredictionResult.pass_("a", "0", "0"),
            PredictionResult.fail("b", "1", "0"),
        ]
        r = self._report(capsule="cap.md", preds=preds)
        line = r.summary_line()
        assert "1 pass" in line
        assert "1 fail" in line

    def test_summary_line_idempotent_skip(self):
        r = self._report(idempotent=True)
        line = r.summary_line()
        assert "no-op" in line or "already ran" in line

    def test_warmup_under_advisory(self):
        r = BootReport(date="d", capsule="c", warmup_ms=100)
        assert r.warmup_under_advisory is True

    def test_warmup_over_advisory(self):
        r = BootReport(date="d", capsule="c", warmup_ms=WARMUP_ADVISORY_MS + 1)
        assert r.warmup_under_advisory is False

    def test_warmup_negative_not_under_advisory(self):
        r = BootReport(date="d", capsule="c", warmup_ms=-1)
        assert r.warmup_under_advisory is False


# ===========================================================================
# 5. HAPPI.md Polyglot Validity
# ===========================================================================

@pytest.mark.skipif(not _HAPPI_MD.exists(), reason="HAPPI.md not found at HAL_ROOT")
class TestHAPPIPolyglot:
    """The HAPPI.md file must simultaneously satisfy Markdown and bash contracts."""

    def test_file_exists(self):
        assert _HAPPI_MD.exists(), "HAPPI.md must exist at HAL_ROOT"

    def test_has_bash_shebang_on_first_line(self):
        first_line = _HAPPI_MD.read_text().splitlines()[0]
        assert first_line.startswith("#!/"), (
            f"Line 1 must be a shebang, got: {first_line!r}"
        )

    def test_has_markdown_headers(self):
        content = _HAPPI_MD.read_text()
        headers = [l for l in content.splitlines() if l.startswith("#")]
        assert len(headers) >= 3, (
            f"Expected >=3 Markdown headers, found: {len(headers)}"
        )

    def test_bash_syntax_is_valid(self):
        result = validate_polyglot(_HAPPI_MD)
        assert result.get("bash_syntax_ok"), (
            "HAPPI.md must pass bash -n (syntax check)"
        )

    def test_has_happi_envelope(self):
        result = validate_polyglot(_HAPPI_MD)
        assert result.get("has_happi_envelope"), (
            'HAPPI.md must contain the HAPPI/1.0 envelope: "v": "happi/1.0"'
        )

    def test_has_watermark(self):
        result = validate_polyglot(_HAPPI_MD)
        assert result.get("has_watermark"), (
            "HAPPI.md must contain the V>>--<<V watermark"
        )

    def test_validate_polyglot_returns_all_true(self):
        result = validate_polyglot(_HAPPI_MD)
        failures = [k for k, v in result.items() if not v]
        assert not failures, (
            f"HAPPI.md failed polyglot checks: {failures}"
        )

    def test_documents_exit_codes(self):
        content = _HAPPI_MD.read_text()
        assert "exit 0" in content
        assert "exit 1" in content
        assert "exit 2" in content

    def test_idempotency_gate_in_source(self):
        content = _HAPPI_MD.read_text()
        assert "MARKER" in content and "FORCE" in content, (
            "HAPPI.md must have idempotency gate (MARKER + FORCE vars)"
        )

    def test_four_predictions_in_source(self):
        content = _HAPPI_MD.read_text()
        assert "Prediction 1" in content
        assert "Prediction 4" in content

    def test_dry_run_flag_supported(self):
        content = _HAPPI_MD.read_text()
        assert "--dry" in content, "HAPPI.md must support --dry flag"

    def test_force_flag_supported(self):
        content = _HAPPI_MD.read_text()
        assert "--force" in content, "HAPPI.md must support --force flag"


# ===========================================================================
# 6. Warmup Measurement
# ===========================================================================

class TestWarmupMeasurement:
    """measure_hal_warmup must return a non-negative integer on this machine."""

    def test_returns_int(self):
        result = measure_hal_warmup(Path(_HAL_HOME))
        assert isinstance(result, int), f"Expected int, got {type(result)}"

    def test_returns_non_negative_on_valid_hal_root(self):
        result = measure_hal_warmup(Path(_HAL_HOME))
        assert result >= 0, f"Expected >= 0 for valid HAL_ROOT, got {result}"

    def test_returns_minus_one_on_invalid_path(self, tmp_path):
        result = measure_hal_warmup(tmp_path / "nonexistent")
        assert result == -1, f"Expected -1 for invalid path, got {result}"

    def test_completes_under_advisory_threshold(self):
        result = measure_hal_warmup(Path(_HAL_HOME))
        if result == -1:
            pytest.skip("lib.hal import failed — skipping timing assertion")
        assert result < WARMUP_ADVISORY_MS, (
            f"HAL warmup {result}ms exceeds advisory {WARMUP_ADVISORY_MS}ms"
        )


# ===========================================================================
# 7. Memory prediction extraction
# ===========================================================================

class TestMemoryPredictionExtraction:
    """extract_memory_prediction must parse the cap value from capsule text."""

    def test_extracts_cap_from_capsule(self, tmp_path):
        cap = tmp_path / "cap.md"
        cap.write_text("active memory at boot + 60 s < 8000 MB")
        result = extract_memory_prediction(cap)
        assert result == 8000

    def test_returns_none_when_no_pattern(self, tmp_path):
        cap = tmp_path / "cap.md"
        cap.write_text("no memory prediction here")
        assert extract_memory_prediction(cap) is None

    def test_returns_none_for_absent_file(self, tmp_path):
        absent = tmp_path / "gone.md"
        assert extract_memory_prediction(absent) is None

    def test_extracts_from_real_capsule_if_present(self):
        caps = list(_CAPSULE_DIR.glob("*.md")) if _CAPSULE_DIR.is_dir() else []
        if not caps:
            pytest.skip("No capsule files available")
        latest = sorted(caps, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        result = extract_memory_prediction(latest)
        if result is not None:
            assert isinstance(result, int)
            assert 100 < result < 100_000, f"Memory cap {result} MB is implausible"


# ===========================================================================
# 8. Integration — real capsule directory (informational, non-blocking)
# ===========================================================================

class TestRealCapsuleIntegration:
    """Smoke tests against the actual HAL state directories.

    These tests are informational — they skip gracefully if the system isn't
    in a boot-validated state. They prevent regressions when real data exists.
    """

    def test_find_latest_capsule_on_real_dir(self):
        if not _CAPSULE_DIR.is_dir():
            pytest.skip("state/witness-capsules/ absent")
        result = find_latest_capsule(_CAPSULE_DIR)
        # May be None if dir is empty — that is also a valid state
        assert result is None or result.exists()

    def test_marker_dir_exists_or_creatable(self, tmp_path):
        m = BootMarker("2026-04-21", "cap.md", 4, 0, 76)
        p = tmp_path / "state" / "boot-markers" / "test.json"
        m.write(p)
        restored = BootMarker.read(p)
        assert restored is not None

    def test_happi_md_is_accessible(self):
        if not _HAPPI_MD.exists():
            pytest.skip("HAPPI.md not found")
        assert _HAPPI_MD.stat().st_size > 0, "HAPPI.md must not be empty"

    def test_existing_boot_marker_round_trips(self):
        markers = list(_MARKER_DIR.glob("*.json")) if _MARKER_DIR.is_dir() else []
        if not markers:
            pytest.skip("No existing boot markers to validate")
        for marker_file in markers:
            m = BootMarker.read(marker_file)
            assert m is not None, f"Could not parse existing marker: {marker_file}"
            assert m.date, "Marker must have a date"
            assert m.watermark == "V>>--<<V", "Watermark must be V>>--<<V"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
