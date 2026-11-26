"""install-starter.sh's KONO-deps pip installs now pass --prefer-binary
(GRIP#8628 decision brief, V>> via /hitl 2026-11-26 18:15 UTC). The flag
tells pip to take a package's ready-made wheel over compiling it from
source whenever both exist -- the convention GRIP's own
scripts/lib/venv-utils.sh already applies to every pip line it runs, for
the reason the decision brief names for the gift kit's *cryptography*
install elsewhere in the same issue: a platform missing a wheel for the
newest release otherwise falls back to a source build, which needs a
compiler the recipient may not have. Neither KONO-dep package here is
cryptography, but apsw and sqlite-vec both ship C extensions, so a missing
wheel is a real (not hypothetical) risk for this pair too.

Two of install-starter.sh's three KONO-deps pip3 calls were missing the
flag: the CORE_DEPS install and the OPTIONAL_DEPS install. The third,
upgrading pip itself a few lines earlier, is left untouched -- out of
scope for this fix, and deliberately not asserted against below, so a
later parity pass that extends the flag there too does not turn this test
red for doing the right thing.

Extracted the same way tests/test_install_python_floor.py extracts
ensure_venv(): sed pulls the whole "# --- 5." section (ensure_venv's
definition and call, then the two KONO-dep installs) out of the real script
and evals it, with a stub pip3 that only logs its argv rather than
installing anything.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tests._bash import find_bash

REPO = Path(__file__).resolve().parent.parent
STARTER_SH = REPO / "scripts" / "install-starter.sh"
_BASH = find_bash()
pytestmark = pytest.mark.skipif(
    _BASH is None,
    reason="no usable bash found on this machine (install Git Bash on Windows)",
)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _run_kono_deps_section(tmp_path: Path) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run the real "# --- 5." section (ensure_venv, then the two KONO pip
    installs) against a venv that already exists on a floor-passing Python,
    with a pip3 stub that logs each call's argv instead of installing
    anything."""
    venv_dir = tmp_path / "venv"
    (venv_dir / "bin").mkdir(parents=True)
    _write_executable(venv_dir / "bin" / "python3", '#!/usr/bin/env bash\nprintf \'%s\' "3.12"\n')
    calls_log = tmp_path / "calls.log"
    _write_executable(
        venv_dir / "bin" / "pip3",
        f'#!/usr/bin/env bash\necho "$*" >> "{calls_log}"\nexit 0\n',
    )

    script = (
        "say() { printf '%s\\n' \"$1\" >&2; }\n"
        f'VENV_DIR="{venv_dir}"\n'
        f'HAL_HOME="{tmp_path}"\n'
        + f"eval \"$(sed -n '/^# --- 5\\./,/^# --- 6\\./p' '{STARTER_SH}')\"\n"
    )
    result = subprocess.run(
        [_BASH, "-euo", "pipefail", "-c", script],
        capture_output=True, text=True,
    )
    calls = calls_log.read_text().splitlines() if calls_log.exists() else []
    return result, calls


def test_kono_deps_pip_installs_pass_prefer_binary(tmp_path):
    """Mutation: drop --prefer-binary from either the CORE_DEPS or the
    OPTIONAL_DEPS pip3 call in the "# --- 5." section -> its logged argv
    stops containing the flag -> RED."""
    result, calls = _run_kono_deps_section(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    # pip self-upgrade, CORE_DEPS, OPTIONAL_DEPS, in that order; the first is
    # out of scope for this fix and checked only by position here, never by
    # content, so a later parity pass that extends --prefer-binary to it too
    # does not turn this test red for doing the right thing.
    assert len(calls) == 3, calls
    _upgrade_call, core_call, optional_call = calls
    assert "--prefer-binary" in core_call, core_call
    assert "--prefer-binary" in optional_call, optional_call
    assert "xxhash apsw sqlite-vec" in core_call, core_call
    assert "sqlite-lembed" in optional_call, optional_call
