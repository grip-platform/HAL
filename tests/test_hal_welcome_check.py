"""Tests for hal welcome --check pre-flight diagnostics (W5)."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lib.hal.welcome import (
    CHECK_ANTHROPIC_ENV,
    CHECK_ANTHROPIC_KEYCHAIN,
    CheckResult,
    LAUNCH_AGENT_LABEL,
    main,
    render_check_report,
    run_checks,
    _check_active_identity,
    _check_anthropic_token,
    _check_git_accessible,
    _check_hal_bin,
    _check_launch_agent,
    _check_say_binary,
    _check_say_voice,
)


# ---------------- Individual checks ----------------


def test_hal_bin_missing_is_critical_fail(tmp_path):
    result = _check_hal_bin(tmp_path / "nope")
    assert result.ok is False
    assert result.critical is True
    assert "missing" in result.detail


def test_hal_bin_present_and_executable_is_ok(tmp_path):
    hal_bin = tmp_path / "hal"
    hal_bin.write_text("#!/bin/bash\n", encoding="utf-8")
    hal_bin.chmod(0o755)
    result = _check_hal_bin(hal_bin)
    assert result.ok is True
    assert result.critical is True
    assert "executable" in result.detail


def test_hal_bin_present_but_not_executable_fails(tmp_path):
    hal_bin = tmp_path / "hal"
    hal_bin.write_text("#!/bin/bash\n", encoding="utf-8")
    hal_bin.chmod(0o644)
    result = _check_hal_bin(hal_bin)
    assert result.ok is False


def test_git_accessible_when_binary_present():
    result = _check_git_accessible()
    assert result.ok is True
    assert result.critical is True


def test_git_failure_when_mocked_missing():
    with patch("lib.hal.welcome.subprocess.run", side_effect=FileNotFoundError):
        result = _check_git_accessible()
    assert result.ok is False
    assert "not found" in result.detail


def test_say_binary_present_on_macos():
    import platform
    if platform.system() == "Darwin":
        result = _check_say_binary()
        assert result.ok is True


def test_say_binary_absent_when_mocked_missing():
    with patch("lib.hal.welcome._which", return_value=None):
        result = _check_say_binary()
    assert result.ok is False
    assert result.critical is True


def test_say_voice_available_when_listed():
    fake_output = MagicMock(stdout="Jamie (Premium) en_GB  # Hello\n", returncode=0)
    with patch("lib.hal.welcome._which", return_value="/usr/bin/say"):
        with patch("lib.hal.welcome.subprocess.run", return_value=fake_output):
            result = _check_say_voice("Jamie (Premium)")
    assert result.ok is True
    assert "available" in result.detail


def test_say_voice_missing_is_soft_warning():
    fake_output = MagicMock(stdout="Samantha en_US  # Hello\n", returncode=0)
    with patch("lib.hal.welcome._which", return_value="/usr/bin/say"):
        with patch("lib.hal.welcome.subprocess.run", return_value=fake_output):
            result = _check_say_voice("Jamie (Premium)")
    assert result.ok is False
    assert result.critical is False


def test_say_voice_reports_missing_say_binary_cleanly():
    with patch("lib.hal.welcome._which", return_value=None):
        result = _check_say_voice()
    assert result.ok is False
    assert "say binary missing" in result.detail


def test_active_identity_ok_when_file_exists(tmp_path):
    path = tmp_path / "active-identity"
    path.write_text("V>>\n", encoding="utf-8")
    result = _check_active_identity(path)
    assert result.ok is True
    assert "readable" in result.detail


def test_active_identity_warn_when_file_missing(tmp_path):
    result = _check_active_identity(tmp_path / "absent")
    assert result.ok is False
    assert result.critical is False


def test_anthropic_token_ok_via_env(monkeypatch):
    monkeypatch.setenv(CHECK_ANTHROPIC_ENV, "fake-token")
    with patch("lib.hal.welcome._keychain_has", return_value=False):
        result = _check_anthropic_token()
    assert result.ok is True
    assert CHECK_ANTHROPIC_ENV in result.detail


def test_anthropic_token_ok_via_keychain(monkeypatch):
    monkeypatch.delenv(CHECK_ANTHROPIC_ENV, raising=False)
    with patch("lib.hal.welcome._keychain_has", return_value=True):
        result = _check_anthropic_token()
    assert result.ok is True
    assert CHECK_ANTHROPIC_KEYCHAIN in result.detail


def test_anthropic_token_missing_is_soft_warning(monkeypatch):
    monkeypatch.delenv(CHECK_ANTHROPIC_ENV, raising=False)
    with patch("lib.hal.welcome._keychain_has", return_value=False):
        result = _check_anthropic_token()
    assert result.ok is False
    assert result.critical is False
    assert "no token source" in result.detail


def test_launch_agent_absent_is_ok_informational(tmp_path):
    result = _check_launch_agent(tmp_path / "absent.plist")
    assert result.ok is True
    assert "not installed" in result.detail
    assert result.critical is False


def test_launch_agent_loaded_reports_true(tmp_path):
    plist = tmp_path / "agent.plist"
    plist.write_text("<plist/>", encoding="utf-8")
    fake_output = MagicMock(
        stdout=f"-\t0\t{LAUNCH_AGENT_LABEL}\nother.service\n",
        returncode=0,
    )
    with patch("lib.hal.welcome.subprocess.run", return_value=fake_output):
        result = _check_launch_agent(plist)
    assert result.ok is True
    assert "loaded" in result.detail


def test_launch_agent_not_loaded_is_soft_warning(tmp_path):
    plist = tmp_path / "agent.plist"
    plist.write_text("<plist/>", encoding="utf-8")
    fake_output = MagicMock(stdout="other.service\n", returncode=0)
    with patch("lib.hal.welcome.subprocess.run", return_value=fake_output):
        result = _check_launch_agent(plist)
    assert result.ok is False
    assert result.critical is False
    assert "NOT loaded" in result.detail


# ---------------- run_checks aggregation ----------------


def test_run_checks_returns_seven_checks_in_stable_order(tmp_path):
    hal_bin = tmp_path / "hal"
    hal_bin.write_text("#!/bin/bash\n", encoding="utf-8")
    hal_bin.chmod(0o755)
    results = run_checks(hal_bin=hal_bin)
    names = [r.name for r in results]
    assert names == [
        "hal binary",
        "git accessible",
        "say binary (macOS TTS)",
        "say voice 'Jamie (Premium)'",
        "active identity",
        "anthropic token",
        "LaunchAgent",
    ]


# ---------------- render_check_report ----------------


def test_render_all_ok_reports_clean_pass():
    checks = [
        CheckResult("a", True, "ok", "", True),
        CheckResult("b", True, "ok", "", False),
    ]
    out = render_check_report(checks)
    assert "All checks passed" in out
    assert "[ok]" in out


def test_render_soft_warnings_only_calls_out_non_blocking():
    checks = [
        CheckResult("critical", True, "ok", "", True),
        CheckResult("soft", False, "fail", "do X", False),
    ]
    out = render_check_report(checks)
    assert "[warn]" in out
    assert "1 soft warning" in out
    assert "no critical failures" in out
    assert "remedy: do X" in out


def test_render_critical_failure_flags_blocking():
    checks = [
        CheckResult("hal binary", False, "missing", "reinstall", True),
    ]
    out = render_check_report(checks)
    assert "[FAIL]" in out
    assert "CRITICAL" in out
    assert "remedy: reinstall" in out


def test_render_table_aligns_check_names():
    checks = [
        CheckResult("short", True, "ok", "", False),
        CheckResult("a much longer name", True, "ok", "", False),
    ]
    out = render_check_report(checks)
    lines = [line for line in out.splitlines() if line.startswith("[")]
    assert all("ok" in line for line in lines)


# ---------------- main() --check CLI ----------------


def test_main_check_runs_and_exits_0_when_all_ok(monkeypatch):
    def _fake_run(**_):
        return [
            CheckResult("a", True, "ok", "", True),
            CheckResult("b", True, "ok", "", False),
        ]
    monkeypatch.setattr("lib.hal.welcome.run_checks", _fake_run)
    stream = io.StringIO()
    rc = main(["--check"], stream_out=stream)
    assert rc == 0
    assert "All checks passed" in stream.getvalue()


def test_main_check_exits_1_on_critical_failure(monkeypatch):
    def _fake_run(**_):
        return [
            CheckResult("hal", False, "missing", "reinstall", True),
        ]
    monkeypatch.setattr("lib.hal.welcome.run_checks", _fake_run)
    stream = io.StringIO()
    rc = main(["--check"], stream_out=stream)
    assert rc == 1
    assert "CRITICAL" in stream.getvalue()


def test_main_check_exits_0_on_soft_warnings_only(monkeypatch):
    def _fake_run(**_):
        return [
            CheckResult("hal", True, "ok", "", True),
            CheckResult("voice", False, "missing", "download", False),
        ]
    monkeypatch.setattr("lib.hal.welcome.run_checks", _fake_run)
    stream = io.StringIO()
    rc = main(["--check"], stream_out=stream)
    assert rc == 0
    assert "no critical failures" in stream.getvalue()


def test_main_check_forwards_voice_flag_to_run_checks(monkeypatch):
    captured = {}

    def _fake_run(*, voice, **_):
        captured["voice"] = voice
        return [CheckResult("x", True, "ok", "", False)]

    monkeypatch.setattr("lib.hal.welcome.run_checks", _fake_run)
    main(["--check", "--voice", "Samantha"], stream_out=io.StringIO())
    assert captured["voice"] == "Samantha"
