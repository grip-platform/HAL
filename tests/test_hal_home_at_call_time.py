"""HAL paths that must follow HAL_HOME read it, instead of writing to ~/.hal (HAL#909).

With ~/.hal hard-coded, a local test run appended test lines to the live
install's tool audit log and cache telemetry: tests/conftest.py gives each test
its own HAL_HOME, but a path that never reads HAL_HOME cannot follow it.
"""
import os
import subprocess
import sys
from pathlib import Path

from lib.hal import cache_idle_report, file_sandbox
from lib.hal.home import hal_home

REPO = Path(__file__).resolve().parent.parent


def test_hal_home_reads_the_variable_each_time(monkeypatch, tmp_path):
    monkeypatch.setenv("HAL_HOME", str(tmp_path / "a"))
    assert hal_home() == tmp_path / "a"
    monkeypatch.setenv("HAL_HOME", str(tmp_path / "b"))
    assert hal_home() == tmp_path / "b"


def test_an_unset_or_empty_hal_home_means_the_home_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows Python reads this, not HOME
    monkeypatch.delenv("HAL_HOME", raising=False)
    assert hal_home() == tmp_path / ".hal"
    monkeypatch.setenv("HAL_HOME", "")
    assert hal_home() == tmp_path / ".hal"


def test_the_constant_and_the_function_agree_on_an_empty_hal_home(tmp_path):
    """bin/hal treats an empty HAL_HOME as unset; both Python readers now do too."""
    # Compared as paths: on Windows the constant keeps the "/" of "~/.hal" while
    # hal_home() returns a normalised Path. One value per line, as a path may hold a space.
    probe = ("from pathlib import Path; from lib.hal.home import HAL_HOME, hal_home; "
             "print(Path(HAL_HOME) == hal_home()); print(Path(HAL_HOME))")
    env = {**os.environ, "HAL_HOME": "", "HOME": str(tmp_path), "USERPROFILE": str(tmp_path),
           "PYTHONPATH": str(REPO)}
    run = subprocess.run([sys.executable, "-c", probe], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip().splitlines()[-2:] == ["True", str(tmp_path / ".hal")]


def test_the_tool_audit_log_follows_hal_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HAL_HOME", str(tmp_path))
    assert file_sandbox._audit_path() == tmp_path / "logs" / "infer-with-tools.jsonl"


def test_the_idle_report_reads_under_hal_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HAL_HOME", str(tmp_path))
    log = tmp_path / "state" / "hal" / "cache_calls.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text('{"ts": 1.0, "prefix_hash": "example", "cache_creation": 0}\n')
    assert cache_idle_report.analyse().total_calls == 1


def test_cache_telemetry_writes_under_the_hal_home_it_was_imported_with(tmp_path):
    probe = "from lib.hal import cache_telemetry as ct; print(ct._TELEMETRY_FILE)"
    env = {**os.environ, "HAL_HOME": str(tmp_path), "PYTHONPATH": str(REPO)}
    run = subprocess.run([sys.executable, "-c", probe], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip().splitlines()[-1] == str(tmp_path / "state" / "hal" / "cache_calls.jsonl")
