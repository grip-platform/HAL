"""Exercise the actual dispatcher with only controlled programs on PATH."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from tests._bash import find_bash

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def shell(tmp_path):
    bash = find_bash()
    bash = shutil.which(bash) if bash else None
    if not bash:
        pytest.skip("Bash is unavailable")
    programs = tmp_path / "bin"
    programs.mkdir()
    log = tmp_path / "calls"
    env = dict(os.environ, PATH=str(programs), CALLS=str(log),
               TMPDIR=str(tmp_path), HOME=str(tmp_path),
               HAPPI_INSTALL_DIR=str(tmp_path / "installed"))
    # Wrappers keep real shell utilities available without exposing host Python.
    for name in ("mkdir", "cat", "wc", "tr", "cut", "sed", "awk", "ln", "dirname", "basename"):
        found = subprocess.run([bash, "-c", f"command -v {name}"], capture_output=True, text=True).stdout.strip()
        assert found, name
        program = programs / name
        program.write_text(f'#!/bin/bash\nexec {shlex.quote(found)} "$@"\n', encoding="utf-8")
        program.chmod(0o755)
    return bash, programs, log, env


def invoke(shell, *args):
    bash, _, _, env = shell
    return subprocess.run([bash, str(REPO / "happi.md"), *args], env=env,
                          input="", text=True, capture_output=True, timeout=15)


@pytest.mark.parametrize("args", [("identity",), ("help",), ("install",),
    *(('extract', layer) for layer in ("markdown", "bash", "python", "envelope", "openapi"))])
def test_python_free_commands(shell, args):
    result = invoke(shell, *args)
    assert result.returncode == 0, result.stderr
    assert result.stdout
    assert "Python 3 interpreter not found" not in result.stderr
    if args == ("install",):
        assert (Path(shell[3]["HAPPI_INSTALL_DIR"]) / "happi").exists()


@pytest.mark.parametrize("command", ["run", "spec.describe", "morning"])
def test_missing_python_reported_only_when_needed(shell, command):
    result = invoke(shell, command)
    assert result.returncode == 1
    assert "Python 3 interpreter not found (tried python3, python, py -3)" in result.stderr


@pytest.mark.parametrize("winner,expected", [
    ("python3", ["python3"]), ("python", ["python3", "python"]),
    ("py", ["python3", "python", "py"]),
])
@pytest.mark.parametrize("command", ["run", "spec.describe"])
def test_resolution_order_and_single_probe_per_candidate(shell, winner, expected, command):
    bash, programs, log, env = shell
    for name in ("python3", "python", "py"):
        program = programs / name
        program.write_text(
            f'#!/bin/bash\nprintf "%s\\n" "{name} $*" >> "$CALLS"\n'
            + ('[[ "$1" == "-3" ]] || exit 9\nshift\n' if name == "py" else '')
            + f'if [[ "$1" == "-c" ]]; then exit {0 if name == winner else 1}; fi\n'
            + 'printf "runtime invoked\\n"\n', encoding="utf-8")
        program.chmod(0o755)
    result = invoke(shell, command)
    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert [line.split()[0] for line in calls] == expected + [winner]
    assert all("-c import sys" in line for line in calls[:-1])
    assert "runtime invoked" in result.stdout
