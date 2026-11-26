"""Shell-rc scan for unsafe osa environment overrides.

The council RETHINK verdict flagged that a malicious or mistaken entry in
the user's shell rc files — ``export GRIP_OSA_UNSAFE=1`` — would silently
disable the tier-2+ capability-token check for every future interactive
session. α-3b surfaces that state at dispatch time so the operator sees
a warning the first time they invoke a script in a process where an rc
file would enable the bypass.

We scan files, not the live environment, because the live environment is
inherited from the parent shell and we *already* know whether the current
process sees the variable. The question we answer is: *would a future
session silently inherit this?*

Files scanned (best-effort, no hard failures on unreadable paths):

    ~/.zshrc
    ~/.bashrc
    ~/.profile
    ~/.bash_profile

The scan is case-insensitive and matches any line that sets the variable,
including ``export GRIP_OSA_UNSAFE``, ``GRIP_OSA_UNSAFE=1``, and
``alias ... GRIP_OSA_UNSAFE``. False positives are preferable to false
negatives — the operator can silence a false warning by moving the
variable out of the rc file into a safer location.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

__all__ = [
    "UNSAFE_VAR_NAME",
    "RC_FILES",
    "detect_unsafe_env_in_shell_rc",
]


UNSAFE_VAR_NAME = "GRIP_OSA_UNSAFE"

RC_FILES = (
    "~/.zshrc",
    "~/.bashrc",
    "~/.profile",
    "~/.bash_profile",
)


def detect_unsafe_env_in_shell_rc(
    rc_files: List[str] = None,
) -> List[Path]:
    """Return the list of rc files that appear to export ``GRIP_OSA_UNSAFE``.

    Args:
        rc_files: Override for the default rc-file list. Used by tests.

    Returns:
        Existing rc files whose contents mention ``GRIP_OSA_UNSAFE`` as a
        variable set or alias. Empty when none match.
    """
    patterns = re.compile(
        rf"\b{re.escape(UNSAFE_VAR_NAME)}\b\s*=|"
        rf"export\s+{re.escape(UNSAFE_VAR_NAME)}\b",
        re.IGNORECASE,
    )
    sources = rc_files if rc_files is not None else RC_FILES
    hits: List[Path] = []
    for entry in sources:
        path = Path(entry).expanduser()
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if patterns.search(text):
            hits.append(path)
    return hits
