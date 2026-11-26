"""Tests for lib/hal/happi_boot.py — bootstrap_hal() and prerequisite checks.

All tests are hermetic — no real git clone, no pip install, no network calls.
subprocess.run is patched throughout.

Covers:
  1. check_prerequisites — Python version, pip, git presence
  2. check_api_keys — env var detection
  3. BootstrapResult — .ok, .missing_prereqs, .summary()
  4. bootstrap_hal — already installed, dry_run, missing prereqs, clone+install
  5. validate_polyglot — HAPPI.md is simultaneously valid bash and Markdown
"""
from __future__ import annotations

import os
import sys
from collections import namedtuple
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_HAL_HOME = os.environ.get("HAL_ROOT", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

# A bare (major, minor) tuple compares and slices like sys.version_info, but
# has no .major/.minor attributes -- fine for happi_boot.check_prerequisites
# (which only slices [:2]), an AttributeError trap for a check that reads
# .major/.minor instead (AGORA, qwen4-max, PR #946 round 2). This stub shapes
# like the real structseq so the same patching pattern is safe everywhere.
_FakeVersion = namedtuple("_FakeVersion", "major minor micro releaselevel serial")


def _fake_version(major: int, minor: int) -> _FakeVersion:
    return _FakeVersion(major, minor, 0, "final", 0)

from lib.hal.happi_boot import (
    BootstrapResult,
    PrerequisiteCheck,
    bootstrap_hal,
    check_api_keys,
    check_prerequisites,
    validate_polyglot,
)


# ---------------------------------------------------------------------------
# 1. check_prerequisites
# ---------------------------------------------------------------------------

class TestCheckPrerequisites:
    def test_returns_three_checks(self, tmp_path):
        checks = check_prerequisites(tmp_path)
        assert len(checks) == 3

    def test_check_names(self, tmp_path):
        names = {c.name for c in check_prerequisites(tmp_path)}
        assert names == {"python", "pip", "git"}

    def test_python_check_passes_current_version(self, tmp_path):
        checks = {c.name: c for c in check_prerequisites(tmp_path)}
        assert checks["python"].ok, "Current Python should satisfy >= 3.11"

    def test_python_check_reflects_version_string(self, tmp_path):
        checks = {c.name: c for c in check_prerequisites(tmp_path)}
        major, minor = sys.version_info[:2]
        assert f"{major}.{minor}" in checks["python"].found

    def test_pip_check_ok_when_importable(self, tmp_path):
        checks = {c.name: c for c in check_prerequisites(tmp_path)}
        import importlib.util
        expected = importlib.util.find_spec("pip") is not None
        assert checks["pip"].ok == expected

    def test_git_check_ok_when_git_on_path(self, tmp_path):
        import shutil
        checks = {c.name: c for c in check_prerequisites(tmp_path)}
        expected = shutil.which("git") is not None
        assert checks["git"].ok == expected

    def test_git_check_fails_when_git_absent(self, tmp_path):
        with patch("shutil.which", return_value=None):
            checks = {c.name: c for c in check_prerequisites(tmp_path)}
        assert not checks["git"].ok

    def test_git_check_found_shows_path(self, tmp_path):
        with patch("shutil.which", return_value="/usr/bin/git"):
            checks = {c.name: c for c in check_prerequisites(tmp_path)}
        assert "/usr/bin/git" in checks["git"].found

    def test_prerequisite_check_dataclass_fields(self, tmp_path):
        check = check_prerequisites(tmp_path)[0]
        assert hasattr(check, "name")
        assert hasattr(check, "required")
        assert hasattr(check, "found")
        assert hasattr(check, "ok")

    @pytest.mark.parametrize(
        "version, expected_ok",
        [
            (_fake_version(3, 9), False),
            (_fake_version(3, 10), False),
            (_fake_version(3, 11), True),
            (_fake_version(3, 12), True),
        ],
    )
    def test_python_floor_is_3_11(self, tmp_path, version, expected_ok):
        """V>> raised HAL's floor to 3.11 on 2026-11-26 (HAL#929): HAPPI's
        own bootstrap checker must refuse 3.10, not just 3.9, because
        lib/hal/utils/cch.py imports tomllib (3.11+, no fallback). Mutation:
        put _MIN_PYTHON back to (3, 10) -> the (3, 10) case flips to True ->
        RED. The stub is namedtuple-shaped (not a bare tuple) so the same
        sys.version_info patching pattern is also safe for a check that reads
        .major/.minor, such as lib.hal.diagnostic's (AGORA, qwen4-max,
        PR #946 round 2).
        """
        with patch("sys.version_info", version):
            checks = {c.name: c for c in check_prerequisites(tmp_path)}
        assert checks["python"].ok is expected_ok
        assert checks["python"].required == ">= 3.11"


# ---------------------------------------------------------------------------
# 2. check_api_keys
# ---------------------------------------------------------------------------

class TestCheckApiKeys:
    def test_returns_list(self):
        assert isinstance(check_api_keys(), list)

    def test_detects_anthropic_key(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-123"}):
            keys = check_api_keys()
        assert "ANTHROPIC_API_KEY" in keys

    def test_ignores_empty_env_var(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            keys = check_api_keys()
        assert "ANTHROPIC_API_KEY" not in keys

    def test_ignores_whitespace_only_value(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "   "}):
            keys = check_api_keys()
        assert "GROQ_API_KEY" not in keys

    def test_multiple_keys_detected(self):
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "sk-a",
            "OPENAI_API_KEY": "sk-b",
        }):
            keys = check_api_keys()
        assert "ANTHROPIC_API_KEY" in keys
        assert "OPENAI_API_KEY" in keys

    def test_unknown_env_var_not_included(self):
        with patch.dict(os.environ, {"TOTALLY_UNKNOWN_KEY": "value"}):
            keys = check_api_keys()
        assert "TOTALLY_UNKNOWN_KEY" not in keys


# ---------------------------------------------------------------------------
# 3. BootstrapResult
# ---------------------------------------------------------------------------

class TestBootstrapResult:
    def _ok_prereqs(self):
        return [
            PrerequisiteCheck("python", ">= 3.11", "3.11", True),
            PrerequisiteCheck("pip", "available", "present", True),
            PrerequisiteCheck("git", "available", "/usr/bin/git", True),
        ]

    def _bad_prereqs(self):
        return [
            PrerequisiteCheck("python", ">= 3.11", "3.9", False),
            PrerequisiteCheck("pip", "available", "present", True),
            PrerequisiteCheck("git", "available", "not found", False),
        ]

    def test_ok_true_when_all_pass_and_no_errors(self):
        r = BootstrapResult(Path("/tmp"), True, self._ok_prereqs(), [])
        assert r.ok is True

    def test_ok_false_when_prereq_fails(self):
        r = BootstrapResult(Path("/tmp"), False, self._bad_prereqs(), [])
        assert r.ok is False

    def test_ok_false_when_errors_present(self):
        r = BootstrapResult(Path("/tmp"), True, self._ok_prereqs(), [], ["git clone failed"])
        assert r.ok is False

    def test_missing_prereqs_filters_failed(self):
        r = BootstrapResult(Path("/tmp"), False, self._bad_prereqs(), [])
        missing_names = {p.name for p in r.missing_prereqs}
        assert "python" in missing_names
        assert "git" in missing_names
        assert "pip" not in missing_names

    def test_missing_prereqs_empty_when_all_pass(self):
        r = BootstrapResult(Path("/tmp"), True, self._ok_prereqs(), [])
        assert r.missing_prereqs == []

    def test_summary_contains_hal_root(self):
        r = BootstrapResult(Path("/custom/hal"), True, self._ok_prereqs(), [])
        assert "/custom/hal" in r.summary()

    def test_summary_shows_installed_status(self):
        r = BootstrapResult(Path("/tmp"), True, self._ok_prereqs(), [])
        assert "installed" in r.summary()

    def test_summary_shows_api_keys(self):
        r = BootstrapResult(Path("/tmp"), True, self._ok_prereqs(), ["ANTHROPIC_API_KEY"])
        assert "ANTHROPIC_API_KEY" in r.summary()

    def test_summary_warns_no_keys(self):
        r = BootstrapResult(Path("/tmp"), True, self._ok_prereqs(), [])
        assert "none configured" in r.summary()

    def test_summary_includes_errors(self):
        r = BootstrapResult(Path("/tmp"), False, self._ok_prereqs(), [], ["git clone failed: timeout"])
        assert "git clone failed" in r.summary()


# ---------------------------------------------------------------------------
# 4. bootstrap_hal
# ---------------------------------------------------------------------------

class TestBootstrapHal:
    def test_already_installed_returns_immediately(self, tmp_path):
        # tmp_path exists — simulates an installed HAL_ROOT
        result = bootstrap_hal(tmp_path)
        assert result.already_installed is True
        assert result.ok is True

    def test_already_installed_no_clone_attempted(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            bootstrap_hal(tmp_path)
        mock_run.assert_not_called()

    def test_dry_run_no_subprocess_calls(self, tmp_path):
        non_existent = tmp_path / "fresh_hal"
        with patch("subprocess.run") as mock_run:
            result = bootstrap_hal(non_existent, dry_run=True)
        mock_run.assert_not_called()
        assert result.errors == []

    def test_dry_run_fresh_machine_not_ok_when_prereqs_fail(self, tmp_path):
        non_existent = tmp_path / "fresh_hal"
        bad_checks = [
            PrerequisiteCheck("python", ">= 3.11", "3.9", False),
            PrerequisiteCheck("pip", "available", "present", True),
            PrerequisiteCheck("git", "available", "not found", False),
        ]
        with patch("lib.hal.happi_boot.check_prerequisites", return_value=bad_checks):
            result = bootstrap_hal(non_existent, dry_run=True)
        # dry_run returns early before missing-prereqs check — no errors added
        assert result.already_installed is False

    def test_missing_prereqs_adds_error(self, tmp_path):
        non_existent = tmp_path / "fresh_hal"
        bad_checks = [
            PrerequisiteCheck("python", ">= 3.11", "3.9", False),
            PrerequisiteCheck("pip", "available", "present", True),
            PrerequisiteCheck("git", "available", "not found", False),
        ]
        with patch("lib.hal.happi_boot.check_prerequisites", return_value=bad_checks):
            result = bootstrap_hal(non_existent)
        assert result.errors
        assert "Prerequisites not met" in result.errors[0]
        assert "python" in result.errors[0]

    def test_successful_fresh_install_calls_git_and_pip(self, tmp_path):
        non_existent = tmp_path / "fresh_hal"
        ok_checks = [
            PrerequisiteCheck("python", ">= 3.11", "3.11", True),
            PrerequisiteCheck("pip", "available", "present", True),
            PrerequisiteCheck("git", "available", "/usr/bin/git", True),
        ]
        with patch("lib.hal.happi_boot.check_prerequisites", return_value=ok_checks), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = bootstrap_hal(non_existent, repo_url="https://example.com/hal.git")

        assert mock_run.call_count == 2
        first_call_args = mock_run.call_args_list[0][0][0]
        assert "git" in first_call_args and "clone" in first_call_args

    def test_git_clone_failure_recorded_in_errors(self, tmp_path):
        non_existent = tmp_path / "fresh_hal"
        ok_checks = [
            PrerequisiteCheck("python", ">= 3.11", "3.11", True),
            PrerequisiteCheck("pip", "available", "present", True),
            PrerequisiteCheck("git", "available", "/usr/bin/git", True),
        ]
        import subprocess
        with patch("lib.hal.happi_boot.check_prerequisites", return_value=ok_checks), \
             patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "git")):
            result = bootstrap_hal(non_existent)
        assert any("git clone failed" in e for e in result.errors)

    def test_pip_failure_recorded_in_errors(self, tmp_path):
        non_existent = tmp_path / "fresh_hal"
        ok_checks = [
            PrerequisiteCheck("python", ">= 3.11", "3.11", True),
            PrerequisiteCheck("pip", "available", "present", True),
            PrerequisiteCheck("git", "available", "/usr/bin/git", True),
        ]
        import subprocess
        call_count = [0]

        def _side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(returncode=0)  # git clone ok
            raise subprocess.CalledProcessError(1, "pip")  # pip fails

        with patch("lib.hal.happi_boot.check_prerequisites", return_value=ok_checks), \
             patch("subprocess.run", side_effect=_side_effect):
            result = bootstrap_hal(non_existent)
        assert any("pip install failed" in e for e in result.errors)

    def test_api_keys_detected_in_result(self, tmp_path):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            result = bootstrap_hal(tmp_path)
        assert "ANTHROPIC_API_KEY" in result.api_keys_found

    def test_result_hal_root_matches_input(self, tmp_path):
        result = bootstrap_hal(tmp_path)
        assert result.hal_root == tmp_path


# ---------------------------------------------------------------------------
# 5. validate_polyglot — HAPPI.md stays valid bash + Markdown
# ---------------------------------------------------------------------------

# The tracked name, exactly. These tests looked for "HAPPI.md" while git tracks
# "happi.md", so on a case-sensitive file system (Linux CI) every one of them
# skipped (HAL#950).
_HAPPI_NAME = "happi.md"


class TestValidatePolyglot:
    def _happi_path(self) -> Path:
        candidates = [
            Path(os.environ.get("HAL_ROOT", "~/.hal")).expanduser() / _HAPPI_NAME,
            Path(__file__).parent.parent / _HAPPI_NAME,
        ]
        for p in candidates:
            if p.exists():
                return p
        pytest.skip(f"{_HAPPI_NAME} not found")

    def test_the_spec_name_is_the_tracked_name(self):
        """HAL#950: the name these tests use must be the file git tracks, with its
        exact case, or they skip on Linux. git ls-files matches a pathspec by case.
        Mutation: set _HAPPI_NAME back to "HAPPI.md" -> RED."""
        import subprocess
        tracked = subprocess.run(
            ["git", "ls-files", "--", _HAPPI_NAME], cwd=Path(__file__).parent.parent,
            capture_output=True, text=True, check=True,
        ).stdout.split()
        assert tracked == [_HAPPI_NAME], tracked

    def test_has_bash_shebang(self):
        result = validate_polyglot(self._happi_path())
        assert result.get("has_bash_shebang") is True

    def test_has_markdown_headers(self):
        result = validate_polyglot(self._happi_path())
        assert result.get("has_markdown_headers") is True

    def test_bash_syntax_valid(self):
        result = validate_polyglot(self._happi_path())
        assert result.get("bash_syntax_ok") is True

    def test_has_happi_envelope(self):
        result = validate_polyglot(self._happi_path())
        assert result.get("has_happi_envelope") is True

    def test_has_watermark(self):
        result = validate_polyglot(self._happi_path())
        assert result.get("has_watermark") is True

    def test_missing_file_returns_empty_dict(self, tmp_path):
        result = validate_polyglot(tmp_path / "nonexistent.md")
        assert result == {}

    def test_all_five_checks_present(self):
        result = validate_polyglot(self._happi_path())
        expected = {
            "has_bash_shebang", "has_markdown_headers",
            "bash_syntax_ok", "has_happi_envelope", "has_watermark",
        }
        assert expected <= set(result.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
