"""osa γ-2c — file-backed allowlist loader.

Closes Gap 2 from the γ-2 broly MESH RETHINK (2026-04-19): regex is
not an allowlist. A regex that accepts ``+27...`` cannot tell "my
sister" from "a stranger" — it expresses *syntax*, not *identity*.

This module replaces regex-only param validation for kinds where
identity matters (iMessage recipients, Shortcut names, HomeKit scene
names, AirPlay speaker names). Each kind is backed by one file under
``$HAL_HOME/config/osa-allowlists/<kind>.txt``. Format:

* One entry per line
* Leading/trailing whitespace stripped
* Blank lines and ``#`` comments ignored
* Everything after the first whitespace on the line is treated as an
  inline annotation (ignored for matching). Example::

      +27123456789      Grandma
      +27987654321      Alex

  Both rows are allowed; ``Grandma`` / ``Alex`` are just operator notes.

On macOS the allowlist file SHOULD have ``chflags uappnd`` applied so
edits require explicit unlock — tamper attempts surface loudly. The
:func:`apply_immutable` helper is provided for operator bootstrap; it
is NOT applied automatically because a fresh HAL install wouldn't
have a file to lock.

Caching: within one process the result of :func:`load_allowlist` is
cached per ``(kind, file-mtime)``. Re-reading is free when the file
hasn't changed and forced when it has.

Hypothesis H162 (2026-05-19): zero dispatches succeed for values
absent from the declared allowlist file. Falsification: if any γ-2
registry entry lands that allows an unlisted recipient/shortcut
name.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet

__all__ = [
    "AllowlistNotFound",
    "AllowlistRejection",
    "allowlist_path",
    "load_allowlist",
    "check_allowed",
    "apply_immutable",
    "_reset_cache_for_tests",
]


class AllowlistNotFound(FileNotFoundError):
    """The declared allowlist kind has no backing file."""


class AllowlistRejection(PermissionError):
    """The supplied value is not present in the allowlist."""


def _hal_home() -> Path:
    return Path(os.environ.get("HAL_HOME", str(Path.home() / ".hal")))


def allowlist_path(kind: str) -> Path:
    """Return ``$HAL_HOME/config/osa-allowlists/<kind>.txt``."""
    if not kind or "/" in kind or ".." in kind:
        raise ValueError(f"invalid allowlist kind: {kind!r}")
    return _hal_home() / "config" / "osa-allowlists" / f"{kind}.txt"


@dataclass
class _CacheEntry:
    mtime_ns: int
    entries: FrozenSet[str]


_CACHE: Dict[str, _CacheEntry] = {}


def _parse(content: str) -> FrozenSet[str]:
    """Strip comments/blank lines, take the first whitespace-token per line."""
    entries = set()
    for raw in content.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        first_token = stripped.split(None, 1)[0]
        entries.add(first_token)
    return frozenset(entries)


def load_allowlist(kind: str) -> FrozenSet[str]:
    """Return the set of allowed values for ``kind``.

    Cached per ``(kind, file mtime)``; a post-load edit invalidates
    on next call because the mtime changed.

    Raises :class:`AllowlistNotFound` if the backing file is missing.
    """
    path = allowlist_path(kind)
    if not path.exists():
        raise AllowlistNotFound(
            f"allowlist {kind!r} missing: create {path}"
        )
    mtime_ns = path.stat().st_mtime_ns
    cached = _CACHE.get(kind)
    if cached is not None and cached.mtime_ns == mtime_ns:
        return cached.entries
    entries = _parse(path.read_text(encoding="utf-8"))
    _CACHE[kind] = _CacheEntry(mtime_ns=mtime_ns, entries=entries)
    return entries


def check_allowed(kind: str, value: str) -> None:
    """Raise :class:`AllowlistRejection` if ``value`` is not allowed."""
    allowed = load_allowlist(kind)
    if value not in allowed:
        raise AllowlistRejection(
            f"value {value!r} not in allowlist {kind!r} "
            f"({len(allowed)} entries)"
        )


def apply_immutable(kind: str) -> bool:
    """Set BSD ``uappnd`` on the allowlist file (macOS only, best-effort).

    Returns True if the flag was applied, False otherwise (non-Darwin,
    missing file, subprocess failure). Operator workflow::

        echo "+27123456789  Grandma" >> ~/.hal/config/osa-allowlists/imessage.txt
        python3 -c "from lib.hal.osa.allowlist import apply_immutable; apply_immutable('imessage')"
    """
    if sys.platform != "darwin":
        return False
    path = allowlist_path(kind)
    if not path.exists():
        return False
    try:
        result = subprocess.run(
            ["chflags", "uappnd", str(path)],
            capture_output=True, text=True, timeout=2, check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _reset_cache_for_tests() -> None:
    """Clear the per-(kind, mtime) cache."""
    _CACHE.clear()
