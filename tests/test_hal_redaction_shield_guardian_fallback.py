"""HAL's shield finds the operator's private list through GRIP's guardian config
when HAL has none of its own (HAL#892).

The operator names his private guardian repository once, in GRIP's gitignored
``config/guardian.local.json``. Before HAL#892 HAL looked only in its own
checkout, so every HAL worktree on his machine redacted with less than GRIP did.
Every value here is invented: each guardian repository is a temporary folder,
and "Mirabel Quenchworth" is nobody. The name differs on purpose from the fake
private list in tests/test_build_oss_candidate.py: this file ships in the
public candidate, and that list's name in a shipped file blocks the release.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_LIB = Path(__file__).resolve().parent.parent / "lib"
sys.path.insert(0, str(_LIB))

import hal_redaction_shield as shield  # noqa: E402

FAKE_NAME = "Mirabel Quenchworth"
FAKE_PSEUDONYM = "[PERSON-902]"


def _guardian_repo(root: Path) -> Path:
    """A stand-in guardian repository whose private list holds one fake entry."""
    listing = root / "config" / "redaction-denylist.json"
    listing.parent.mkdir(parents=True)
    listing.write_text(json.dumps({"entities": [{
        "canonical": FAKE_NAME, "aliases": [FAKE_NAME.lower()],
        "pseudonym": FAKE_PSEUDONYM, "class": "person"}]}), encoding="utf-8")
    return root


def _guardian_config(home: Path, repo: Path | None) -> Path:
    """A config/guardian.local.json under ``home`` that names ``repo``."""
    config = home / "config" / "guardian.local.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    body = {"guardian_repo": str(repo)} if repo is not None else {}
    config.write_text(json.dumps(body), encoding="utf-8")
    return config


@pytest.fixture
def hal_config(tmp_path, monkeypatch) -> Path:
    """HAL's own guardian config path, absent until a test writes it, with no
    overlay named through the environment."""
    path = tmp_path / "hal" / "config" / "guardian.local.json"
    monkeypatch.setattr(shield, "GUARDIAN_LOCAL", path)
    monkeypatch.delenv(shield.OVERLAY_ENV, raising=False)
    return path


@pytest.fixture
def grip_dir(tmp_path, monkeypatch) -> Path:
    """GRIP's directory, named through CLAUDE_DIR as GRIP's own shield reads it."""
    grip = tmp_path / "grip"
    grip.mkdir()
    monkeypatch.setenv("CLAUDE_DIR", str(grip))
    return grip


def test_the_suite_runs_with_the_private_list_turned_off():
    """tests/conftest.py turns the overlay off for the whole run, so no test reads
    the operator's real list through GRIP's guardian config."""
    assert os.environ.get(shield.OVERLAY_ENV) == "none"
    assert shield.overlay_state() == "disabled"


def test_grips_guardian_config_supplies_the_private_list(tmp_path, hal_config, grip_dir):
    repo = _guardian_repo(tmp_path / "guardian")
    _guardian_config(grip_dir, repo)
    assert shield.private_overlay_path() == repo / "config" / "redaction-denylist.json"
    assert shield.overlay_state() == "ok"
    assert [e.canonical for e in shield.load_overlay().entries] == [FAKE_NAME]


def test_the_default_list_then_redacts_the_private_name(tmp_path, hal_config, grip_dir,
                                                        monkeypatch):
    """End to end: load_denylist() merges the private list, and the shield
    rewrites the name. With no private list the text comes back unchanged."""
    base = tmp_path / "base-denylist.json"
    base.write_text('{"entities": []}', encoding="utf-8")
    monkeypatch.setattr(shield, "DEFAULT_CONFIG", base)
    _guardian_config(grip_dir, _guardian_repo(tmp_path / "guardian"))
    result = shield.Shield(shield.load_denylist()).redact(f"Met {FAKE_NAME} today.")
    assert result.changed
    assert FAKE_NAME not in result.text


def test_hals_own_guardian_config_wins(tmp_path, hal_config, grip_dir):
    hal_repo = _guardian_repo(tmp_path / "hal-guardian")
    _guardian_config(hal_config.parent.parent, hal_repo)
    _guardian_config(grip_dir, _guardian_repo(tmp_path / "grip-guardian"))
    assert shield.private_overlay_path() == hal_repo / "config" / "redaction-denylist.json"


def test_neither_config_reads_unconfigured(hal_config, grip_dir):
    assert shield.private_overlay_path() is None
    assert shield.overlay_state() == "unconfigured"
    assert shield.load_overlay().entries == ()


def test_grips_config_naming_no_repository_fails_closed(hal_config, grip_dir):
    """The operator meant to have a private list, so a GRIP config that names no
    repository reads 'invalid', and loading raises with GRIP's file named."""
    config = _guardian_config(grip_dir, None)
    assert shield.overlay_state() == "invalid"
    with pytest.raises(shield.RedactionConfigError) as raised:
        shield.load_overlay()
    assert str(config) in str(raised.value)


def test_grips_config_naming_a_missing_list_fails_closed(tmp_path, hal_config, grip_dir):
    _guardian_config(grip_dir, tmp_path / "no-such-guardian")
    assert shield.overlay_state() == "missing"
    with pytest.raises(shield.RedactionConfigError):
        shield.load_overlay()


def test_the_environment_still_outranks_both_configs(tmp_path, hal_config, grip_dir,
                                                     monkeypatch):
    _guardian_config(grip_dir, _guardian_repo(tmp_path / "guardian"))
    named = tmp_path / "named.json"
    monkeypatch.setenv(shield.OVERLAY_ENV, str(named))
    assert shield.private_overlay_path() == named
    monkeypatch.setenv(shield.OVERLAY_ENV, "none")
    assert shield.overlay_state() == "disabled"
    assert shield.load_overlay().entries == ()


def test_without_claude_dir_grips_config_is_read_from_home(tmp_path, hal_config,
                                                          monkeypatch):
    monkeypatch.delenv("CLAUDE_DIR", raising=False)
    home = tmp_path / "home"
    # Path.home() reads HOME on POSIX and USERPROFILE on Windows.
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    repo = _guardian_repo(tmp_path / "guardian")
    _guardian_config(home / ".claude", repo)
    assert shield.private_overlay_path() == repo / "config" / "redaction-denylist.json"


def test_a_list_deleted_after_a_lookup_fails_closed(tmp_path, hal_config, grip_dir):
    """A falsifier from the HAL#994 review: load_overlay() checks the list again
    rather than trusting an earlier lookup, so a list deleted after the lookup
    found it raises instead of loading nothing."""
    _guardian_config(grip_dir, _guardian_repo(tmp_path / "guardian"))
    listing = shield.private_overlay_path()
    assert shield.overlay_state() == "ok"
    listing.unlink()
    with pytest.raises(shield.RedactionConfigError):
        shield.load_overlay()
