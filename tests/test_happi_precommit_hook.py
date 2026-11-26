"""Tests for happi.md edit-discipline pre-commit hook (issue #306 Phase 2).

Goodhart-proof tests verify the hook actually changes behaviour based on
staged file shape — not just that it runs and returns 0.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests._public_tree import folder_absent, reason

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / "scripts" / "happi-precommit-hook.sh"
# Public HAL leaves scripts/ out, so on a public checkout every test here is
# skipped with that reason (HAL#859); in the full repository they all run.
pytestmark = pytest.mark.skipif(folder_absent("scripts"), reason=reason("scripts"))
INSTALLER = REPO_ROOT / "scripts" / "install-happi-precommit-hook.sh"


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Initialise a temporary git repo with stub happi.md and happi-sources/."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.local"], cwd=tmp_path, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True
    )

    (tmp_path / "happi.md").write_text("# initial happi\n")
    sources = tmp_path / "happi-sources"
    sources.mkdir()
    (sources / "01-shebang.bash").write_text("#!/usr/bin/env bash\n")

    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "initial"], cwd=tmp_path, check=True
    )
    return tmp_path


def _run_hook(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(HOOK)],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def test_warns_when_happi_md_staged_without_sources(temp_repo: Path) -> None:
    """Goodhart-proof: warning MUST fire on the offending stage shape."""
    (temp_repo / "happi.md").write_text("# changed happi\n")
    subprocess.run(["git", "add", "happi.md"], cwd=temp_repo, check=True)

    result = _run_hook(temp_repo)

    assert result.returncode == 0, "hook must fail-soft (exit 0)"
    assert "WARNING" in result.stderr, "warning text missing from stderr"
    assert "happi.md staged without happi-sources/" in result.stderr
    assert "Phase 2 of issue #306" in result.stderr


def test_no_warning_when_both_happi_and_sources_staged(temp_repo: Path) -> None:
    """Goodhart-proof: warning MUST NOT fire when sources accompany happi.md."""
    (temp_repo / "happi.md").write_text("# changed happi\n")
    (temp_repo / "happi-sources" / "01-shebang.bash").write_text("#!/bin/bash\n")
    subprocess.run(
        ["git", "add", "happi.md", "happi-sources/01-shebang.bash"],
        cwd=temp_repo,
        check=True,
    )

    result = _run_hook(temp_repo)

    assert result.returncode == 0
    assert "WARNING" not in result.stderr, (
        f"unexpected warning when both staged: {result.stderr!r}"
    )


def test_no_warning_when_only_sources_staged(temp_repo: Path) -> None:
    """Editing sources only is the canonical Phase 2 workflow."""
    (temp_repo / "happi-sources" / "01-shebang.bash").write_text("#!/bin/bash\n")
    subprocess.run(
        ["git", "add", "happi-sources/01-shebang.bash"],
        cwd=temp_repo,
        check=True,
    )

    result = _run_hook(temp_repo)

    assert result.returncode == 0
    assert "WARNING" not in result.stderr


def test_no_warning_when_unrelated_files_staged(temp_repo: Path) -> None:
    """Hook must be quiet on stages that don't touch the polyglot."""
    (temp_repo / "other.txt").write_text("hello\n")
    subprocess.run(["git", "add", "other.txt"], cwd=temp_repo, check=True)

    result = _run_hook(temp_repo)

    assert result.returncode == 0
    assert "WARNING" not in result.stderr


def test_no_warning_when_nothing_staged(temp_repo: Path) -> None:
    """Empty stage shape: silent no-op."""
    result = _run_hook(temp_repo)

    assert result.returncode == 0
    assert result.stderr == ""


def test_hook_script_is_executable() -> None:
    """Installer relies on chmod +x — verify the source has it set."""
    assert os.access(HOOK, os.X_OK), f"{HOOK} must be executable (chmod +x)"


def test_installer_is_executable() -> None:
    assert os.access(INSTALLER, os.X_OK), (
        f"{INSTALLER} must be executable (chmod +x)"
    )


def test_installer_idempotent(temp_repo: Path) -> None:
    """Goodhart-proof: re-running installer must not duplicate the hook line."""
    repo_scripts = temp_repo / "scripts"
    repo_scripts.mkdir()
    repo_scripts.joinpath("happi-precommit-hook.sh").write_text(
        HOOK.read_text()
    )
    repo_scripts.joinpath("happi-precommit-hook.sh").chmod(0o755)
    (repo_scripts / "install-happi-precommit-hook.sh").write_text(
        INSTALLER.read_text()
    )
    (repo_scripts / "install-happi-precommit-hook.sh").chmod(0o755)

    first = subprocess.run(
        ["bash", "scripts/install-happi-precommit-hook.sh"],
        cwd=temp_repo,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0
    assert "installed" in first.stdout

    second = subprocess.run(
        ["bash", "scripts/install-happi-precommit-hook.sh"],
        cwd=temp_repo,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0
    assert "already installed" in second.stdout
