"""install.sh's prerequisite check requires Python >= 3.11 (CodeTonight-SA/HAL#922,
raised from 3.10 to 3.11 within #924).

HAL's library already uses Python 3.10 annotation syntax (``str | None`` with
no ``from __future__ import annotations``), so a 3.9 interpreter cannot load
it -- the original reason for #922. V>> decided on 2026-11-26 that the real
floor is 3.11, not 3.10: ``lib/hal/utils/cch.py`` line 68 does a bare
``import tomllib`` (added to the standard library in 3.11, no fallback) to
load HAL's own model-capability registry (``configs/models.toml``) as soon
as the module is first loaded, so on 3.10 that statement raises
``ModuleNotFoundError`` and the subscription signer silently drops out.
This test pins the installer's own gate at 3.11, so a 3.10 install fails
fast with a clear message instead of cloning the whole repo and failing
later, deep inside a library load.

check_prereqs() is extracted from install.sh with sed rather than sourcing
the whole script, because the script unconditionally runs main() (clone,
venv, install) at its bottom. A fake python3 is planted first on PATH so no
branch needs a real sub-3.11 interpreter on the test machine.

The starter kit ships without install.sh, so scripts/install-starter.sh carries
the same floor in check_python_floor(), extracted and tested the same way.
Both scripts' ensure_venv() also rebuild an existing venv whose OWN
interpreter predates the floor (see the tests below) -- a venv built before
the floor was raised would otherwise keep its stale Python forever, since
neither installer previously looked at what was already there.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from tests._bash import find_bash

REPO = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO / "install.sh"
STARTER_SH = REPO / "scripts" / "install-starter.sh"
_BASH = find_bash()
pytestmark = pytest.mark.skipif(
    _BASH is None,
    reason="no usable bash found on this machine (install Git Bash on Windows)",
)

_STUB_LOGGERS = (
    'log_step() { :; }\n'
    'log_ok()   { :; }\n'
    'log_warn() { printf \'%s\\n\' "$1" >&2; }\n'
    'log_fail() { printf \'%s\\n\' "$1" >&2; }\n'
)


def _run_check_prereqs(tmp_path: Path, fake_version: str) -> subprocess.CompletedProcess:
    """Run install.sh's real check_prereqs against a fake python3 reporting fake_version."""
    fake_python = tmp_path / "python3"
    fake_python.write_text(f'#!/usr/bin/env bash\nprintf \'%s\' "{fake_version}"\n')
    fake_python.chmod(0o755)
    script = (
        _STUB_LOGGERS
        + f"eval \"$(sed -n '/^check_prereqs() {{/,/^}}/p' '{INSTALL_SH}')\"\n"
        + "check_prereqs\n"
    )
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    return subprocess.run(
        [_BASH, "-euo", "pipefail", "-c", script],
        env=env, capture_output=True, text=True,
    )


@pytest.mark.parametrize(
    "fake_version, accepted",
    [("3.9", False), ("3.10", False), ("3.11", True), ("3.12", True)],
)
def test_check_prereqs_enforces_the_3_11_floor(tmp_path, fake_version, accepted):
    """V>> raised the floor from 3.10 to 3.11 on 2026-11-26 (HAL#924): cch.py's
    tomllib import (3.11+, no fallback) silently drops the subscription signer
    on 3.10. Mutation: lower the threshold back to 10 -> the 3.10 case stops
    failing -> RED. 3.9 guards the numeric (not lexical) comparison: "9" > "11"
    lexically (first char '9' > '1'), so a naive string compare would wrongly
    accept it; 3.12 is the next minor, accepted either way it is compared."""
    result = _run_check_prereqs(tmp_path, fake_version)
    assert (result.returncode == 0) is accepted, result.stdout + result.stderr
    if not accepted:
        assert "3.11" in result.stderr, result.stderr


def _run_starter_floor(tmp_path: Path, fake_version: str) -> subprocess.CompletedProcess:
    """Run install-starter.sh's real check_python_floor against a fake interpreter."""
    fake_python = tmp_path / "python3"
    fake_python.write_text(f'#!/usr/bin/env bash\nprintf \'%s\' "{fake_version}"\n')
    fake_python.chmod(0o755)
    script = (
        "say() { printf '%s\\n' \"$1\" >&2; }\n"
        + f"eval \"$(sed -n '/^check_python_floor() {{/,/^}}/p' '{STARTER_SH}')\"\n"
        + f"check_python_floor '{fake_python}'\n"
    )
    return subprocess.run(
        [_BASH, "-euo", "pipefail", "-c", script],
        capture_output=True, text=True,
    )


@pytest.mark.parametrize(
    "fake_version, accepted",
    [("3.9", False), ("3.10", False), ("3.11", True), ("3.12", True)],
)
def test_the_starter_kit_installer_checks_the_same_floor(tmp_path, fake_version, accepted):
    """Codex on HAL#924: a starter-kit recipient never runs install.sh, so the
    kit's own installer checks the floor. Mutation: lower its threshold to 10
    -> the 3.10 case stops failing -> RED."""
    result = _run_starter_floor(tmp_path, fake_version)
    assert (result.returncode == 0) is accepted, result.stdout + result.stderr
    if not accepted:
        assert "3.11" in result.stderr, result.stderr


# ---------------------------------------------------------------------------
# ensure_venv(): an existing venv is checked against the floor too, not just
# kept because something is there. Codex review on HAL#924 (thread
# PRRT_kwDOR3PFSc68abkl): a venv built before the floor was raised would
# otherwise stay on its stale interpreter forever.
# ---------------------------------------------------------------------------

_FAKE_VENV_CREATOR = """#!/usr/bin/env bash
# Stands in for a real "python3 -m venv DIR": writes just enough of a venv
# (a bin/python3 that reports FRESH) to prove a rebuild happened, with no
# real interpreter or network access needed.
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
    mkdir -p "$3/bin"
    cat > "$3/bin/python3" <<'INNEREOF'
#!/usr/bin/env bash
printf '%s' FRESH
INNEREOF
    chmod +x "$3/bin/python3"
    exit 0
fi
exit 1
"""


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _write_fake_version_python(path: Path, version: str) -> None:
    """Stands in for an existing venv's own bin/python3: reports a fixed
    version regardless of how it is invoked, the same style _run_check_prereqs
    already uses for the system python3."""
    _write_executable(path, f'#!/usr/bin/env bash\nprintf \'%s\' "{version}"\n')


def _run_install_sh_ensure_venv(tmp_path: Path, old_version: str) -> subprocess.CompletedProcess:
    """Run install.sh's real ensure_venv() against a pre-existing fake venv at
    VENV_DIR reporting old_version. install.sh's ensure_venv calls a literal
    `python3 -m venv`, so the fake creator goes on PATH."""
    venv_dir = tmp_path / "venv"
    (venv_dir / "bin").mkdir(parents=True)
    _write_fake_version_python(venv_dir / "bin" / "python3", old_version)

    sysbin = tmp_path / "sysbin"
    sysbin.mkdir()
    _write_executable(sysbin / "python3", _FAKE_VENV_CREATOR)

    script = (
        _STUB_LOGGERS
        + f'VENV_DIR="{venv_dir}"\n'
        + f"eval \"$(sed -n '/^ensure_venv() {{/,/^}}/p' '{INSTALL_SH}')\"\n"
        + "ensure_venv\n"
    )
    env = dict(os.environ, PATH=f"{sysbin}:{os.environ['PATH']}")
    return subprocess.run(
        [_BASH, "-euo", "pipefail", "-c", script],
        env=env, capture_output=True, text=True,
    )


def _run_starter_ensure_venv(tmp_path: Path, old_version: str) -> subprocess.CompletedProcess:
    """Run install-starter.sh's real ensure_venv() the same way. It calls
    "$PY" -m venv rather than a literal python3, so the fake creator is
    passed through the PY variable instead of PATH."""
    venv_dir = tmp_path / "venv"
    (venv_dir / "bin").mkdir(parents=True)
    _write_fake_version_python(venv_dir / "bin" / "python3", old_version)

    creator = tmp_path / "py-creator"
    _write_executable(creator, _FAKE_VENV_CREATOR)

    script = (
        "say() { printf '%s\\n' \"$1\" >&2; }\n"
        + f'VENV_DIR="{venv_dir}"\n'
        + f'PY="{creator}"\n'
        + f"eval \"$(sed -n '/^ensure_venv() {{/,/^}}/p' '{STARTER_SH}')\"\n"
        + "ensure_venv\n"
    )
    return subprocess.run(
        [_BASH, "-euo", "pipefail", "-c", script],
        capture_output=True, text=True,
    )


def _assert_rebuild_decision(
    tmp_path: Path, venv_dir: Path, result: subprocess.CompletedProcess, rebuilt: bool
) -> None:
    assert result.returncode == 0, result.stdout + result.stderr
    backups = list(tmp_path.glob("venv.pre-3.11-*"))
    new_python = venv_dir / "bin" / "python3"
    if rebuilt:
        assert len(backups) == 1, result.stdout + result.stderr
        assert "FRESH" in new_python.read_text()
    else:
        assert backups == [], result.stdout + result.stderr
        assert "FRESH" not in new_python.read_text()


@pytest.mark.parametrize("old_version, rebuilt", [("3.10", True), ("3.11", False)])
def test_install_sh_ensure_venv_rebuilds_pre_3_11_venvs(tmp_path, old_version, rebuilt):
    """Mutation: drop the version check -> the 3.10 case stops moving the
    venv aside and stops rebuilding it -> RED."""
    result = _run_install_sh_ensure_venv(tmp_path, old_version)
    _assert_rebuild_decision(tmp_path, tmp_path / "venv", result, rebuilt)


@pytest.mark.parametrize("old_version, rebuilt", [("3.10", True), ("3.11", False)])
def test_starter_kit_ensure_venv_rebuilds_pre_3_11_venvs(tmp_path, old_version, rebuilt):
    """Same Codex thread, the starter-kit installer's copy of the gap.
    Mutation: drop the version check -> the 3.10 case stops rebuilding -> RED."""
    result = _run_starter_ensure_venv(tmp_path, old_version)
    _assert_rebuild_decision(tmp_path, tmp_path / "venv", result, rebuilt)


def _run_install_sh_update(venv_fresh: int) -> subprocess.CompletedProcess:
    """Run install.sh's real main() in --update mode, every step it calls stubbed."""
    stubs = (
        "C_CYAN=; C_RESET=; C_DIM=; HAL_HOME=/nowhere; HAL_BRANCH=main\n"
        "UPDATE_ONLY=1; VENV_FRESH=0\n"
        "log_ok() { :; }\nprint_summary() { :; }\n"
        "check_prereqs() { :; }\nclone_or_update_repo() { :; }\n"
        f"ensure_venv() {{ VENV_FRESH={venv_fresh}; }}\n"
        "install_kono_deps() { echo KONO_DEPS_INSTALLED; }\n"
        "bootstrap_dirs() { echo FULL_INSTALL_RAN; }\n"
    )
    script = stubs + f"eval \"$(sed -n '/^main() {{/,/^}}/p' '{INSTALL_SH}')\"\nmain\n"
    return subprocess.run([_BASH, "-euo", "pipefail", "-c", script],
                          capture_output=True, text=True)


@pytest.mark.parametrize("venv_fresh, deps_installed", [(1, True), (0, False)])
def test_update_installs_deps_into_a_venv_it_just_built(venv_fresh, deps_installed):
    """AGORA on HAL#924: --update returned straight after ensure_venv, so a venv
    it had just rebuilt (older than the floor, moved aside) was left empty.
    Mutation: drop the VENV_FRESH branch from main() -> the fresh case installs
    nothing -> RED."""
    result = _run_install_sh_update(venv_fresh)
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("KONO_DEPS_INSTALLED" in result.stdout) is deps_installed, result.stdout
    assert "FULL_INSTALL_RAN" not in result.stdout, result.stdout
