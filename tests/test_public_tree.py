"""tests/_public_tree.py skips only when a whole folder is absent, never for one missing file.

HAL#867: a skip that fired for a single missing file would hide a real deletion in the
full repository behind a passing run. These tests pin the line between the two cases.
"""
from __future__ import annotations

import pytest

from tests import _public_tree as pt


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "REPO", tmp_path)
    return tmp_path


def test_a_missing_folder_skips_with_the_folder_named(checkout):
    with pytest.raises(pytest.skip.Exception, match=r"^happi-sources/ is not in the public tree$"):
        pt.require("happi-sources/04-python-runtime.py")


def test_a_missing_file_in_a_present_folder_does_not_skip_and_the_read_fails(checkout):
    (checkout / "deploy").mkdir()
    path = pt.require("deploy/com.hal.gateway.plist")
    assert path == checkout / "deploy" / "com.hal.gateway.plist"
    with pytest.raises(FileNotFoundError):
        path.read_text()


def test_a_dotted_top_folder_is_one_folder(checkout):
    assert pt.folder_absent(".github/scripts/council_converge.py")
    assert pt.reason(".github/workflows/agora-pr-review.yml") == ".github/ is not in the public tree"
    (checkout / ".github").mkdir()
    assert not pt.folder_absent(".github/scripts/council_converge.py")


def test_a_file_withheld_from_a_partly_public_folder_skips_only_on_a_public_checkout(checkout):
    # The public build ships .github/workflows/stranger.yml alone (HAL#881), so on a
    # public checkout .github/ exists and its other files are withheld, not deleted.
    (checkout / ".github" / "workflows").mkdir(parents=True)
    (checkout / ".github" / "workflows" / "stranger.yml").write_text("name: Stranger check\n")
    with pytest.raises(pytest.skip.Exception, match=r"ships only some files"):
        pt.require(".github/workflows/agora-pr-review.yml")
    assert pt.require(".github/workflows/stranger.yml").exists()


def test_the_same_missing_file_fails_in_the_full_repository(checkout):
    # Mutation: drop the full-repository check, and a real deletion of a private CI
    # file hides behind a skip.
    (checkout / ".github" / "workflows").mkdir(parents=True)
    (checkout / "scripts").mkdir()
    (checkout / pt.FULL_REPO_MARKER).write_text("# the builder\n")
    # A skip is not a failure, so assert the predicate rather than only calling require.
    assert not pt.withheld(".github/workflows/agora-pr-review.yml")
    path = pt.require(".github/workflows/agora-pr-review.yml")
    with pytest.raises(FileNotFoundError):
        path.read_text()
