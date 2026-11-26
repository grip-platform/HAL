"""The redaction shield's denylist loader has two paths, both tested here.

A standalone install (no operator denylist configured at all) must never
crash -- it degrades to the generic, no-real-names default. An install that
DOES configure a private file (outside this repo; see ``private_overlay_path``
and ``$GRIP_REDACTION_OVERLAY``) must still load that file's real entries.
"""
from __future__ import annotations

import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parent.parent / "lib"
sys.path.insert(0, str(_LIB))

import hal_redaction_shield as shield  # noqa: E402


def test_a_missing_file_degrades_to_the_generic_default(tmp_path: Path) -> None:
    """No file at all (the standalone-install case) never raises, and carries
    no named entries -- just the structural PII floor, which is unaffected."""
    missing = tmp_path / "does-not-exist.json"
    result = shield.load_denylist_or_default(missing)
    assert result == shield.GENERIC_DEFAULT_DENYLIST
    assert result.entries == ()


def test_an_explicit_private_file_still_loads_its_real_entries(tmp_path: Path) -> None:
    """The 'private file outside the repo' path: an explicit path that DOES
    exist and IS valid loads normally through the same safe entry point --
    degrading to the default is only for the MISSING case, never for a file
    that is actually there."""
    private_file = tmp_path / "private-overlay.json"
    private_file.write_text(
        '{"entities": [{"canonical": "Example Org", '
        '"aliases": ["example org"], "pseudonym": "[ORG-1]", "class": "org"}]}'
    )
    result = shield.load_denylist_or_default(private_file)
    assert result != shield.GENERIC_DEFAULT_DENYLIST
    assert result.entries[0].canonical == "Example Org"


def test_a_ciphertext_file_still_raises_not_degrades(tmp_path: Path) -> None:
    """A file that IS configured but unusable (git-crypt-locked, invalid JSON)
    is a real misconfiguration -- it must still fail loud, exactly as
    ``load_denylist`` does, never silently narrow to the generic default."""
    locked = tmp_path / "locked.json"
    locked.write_bytes(shield._GITCRYPT_MAGIC + b"\x00\x01\x02")
    import pytest
    with pytest.raises(shield.RedactionConfigError):
        shield.load_denylist_or_default(locked)
