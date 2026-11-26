"""Skip, with a stated reason, a test whose input lives in a folder public HAL leaves out.

Public HAL ships lib/, bin/, tests/, docs/, configs/ and LICENSES/, but not scripts/,
deploy/, tools/, happi-sources/ or .github/ (ALLOWLIST_PATHSPEC in
scripts/build_oss_candidate.py). A test that reads one of those folders cannot run on
a public checkout, so there it is skipped with the folder named (HAL#859).

The skip fires only when the WHOLE top-level folder is absent. A single file missing
from a folder that is present still fails, so a real deletion in the full repository
is never hidden behind a skip.

One exception, because the public build ships single files from it: .github/ carries
the stranger-check caller and nothing else. A file missing from such a partly public
folder skips only on a public checkout, which has no scripts/ (FULL_REPO_MARKER); in
the full repository the same miss is a real deletion and still fails.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

REPO = Path(__file__).resolve().parents[1]
# Folders the public build ships single files from (DATA_KEEP in the builder).
PARTLY_PUBLIC = frozenset({".github"})
# Public HAL never ships scripts/, so this file marks the full repository.
FULL_REPO_MARKER = "scripts/build_oss_candidate.py"


def _top(rel: str) -> str:
    return PurePosixPath(rel).parts[0]


def folder_absent(rel: str) -> bool:
    """True when the top-level folder of ``rel`` is not in this checkout."""
    return not (REPO / _top(rel)).exists()


def withheld(rel: str) -> bool:
    """True when ``rel`` is missing because public HAL leaves it out."""
    if folder_absent(rel):
        return True
    public = not (REPO / FULL_REPO_MARKER).exists()
    return public and _top(rel) in PARTLY_PUBLIC and not (REPO / rel).exists()


def reason(rel: str) -> str:
    if folder_absent(rel):
        return f"{_top(rel)}/ is not in the public tree"
    return f"{rel} is not in the public tree ({_top(rel)}/ ships only some files)"


def require(rel: str) -> Path:
    """Return the path to ``rel``, or skip the calling test on a public checkout.

    Call it inside a test, a fixture or a helper a test calls, never at module level
    or in a parametrize list: pytest refuses a skip during collection, and the whole
    file then errors instead of skipping (HAL#867). For a whole-module skip, use
    ``pytestmark = pytest.mark.skipif(folder_absent(...), reason=reason(...))``.
    """
    if withheld(rel):
        pytest.skip(reason(rel))
    return REPO / rel
