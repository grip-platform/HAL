"""osa γ-2d — Shortcut dispatch backend.

Closes Gap 1 from the γ-2 broly MESH RETHINK (2026-04-19): the naive
seed (``shortcut_run_by_name`` via ``do shell script "shortcuts run "
& quoted form of {name}``) stacks two parsers — the AppleScript layer
unwraps the literal into ``do shell script``, which hands the inner
text to ``/bin/sh``. Any character that survives the registry pattern
AND has shell meaning (backtick, ``$(...)``, ``;``, ``|``) becomes
shell injection.

This module bypasses that vector entirely by invoking
``/usr/bin/shortcuts run <name>`` as a subprocess with an argv list —
no shell wrapping, no AppleScript parsing. Input data is piped over
stdin via ``--input-path -`` and remains uninterpreted.

Public API::

    from lib.hal.osa.shortcuts import run_shortcut, ShortcutResult

    result = run_shortcut(
        "Home: Goodnight",
        input_data=None,
        timeout_s=10,
        allowlist_kind="shortcuts",   # optional
        rate_limit=RateLimit(window_s=60, max_calls=10),  # optional
    )

Hypothesis H168 (2026-05-19): zero shell-escape injections over 30
days of dispatch. Falsification: any input that contains backtick /
dollar-paren / pipe / semicolon reaches a shell interpreter and
expands. The injection-safety test in tests/test_osa_shortcuts.py is
the executable expression of that claim.

Unavailability: non-Darwin platforms and macOS versions without the
Shortcuts app (pre-Monterey) raise :class:`ShortcutUnavailable`. This
is intentional — there is no graceful fallback because the capability
genuinely does not exist on those platforms.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

from lib.hal.osa.allowlist import (
    AllowlistNotFound,
    AllowlistRejection,
    check_allowed,
)
from lib.hal.osa.audit import make_entry, write_audit
from lib.hal.osa.provenance import ProvenanceType
from lib.hal.osa.rate_limit import (
    RateLimit,
    RateLimitExceeded,
    check_and_record,
)

__all__ = [
    "ShortcutError",
    "ShortcutUnavailable",
    "ShortcutResult",
    "run_shortcut",
]


_SHORTCUTS_BIN = "/usr/bin/shortcuts"


class ShortcutError(RuntimeError):
    """A shortcut dispatch failed at the subprocess or CLI level."""


class ShortcutUnavailable(ShortcutError):
    """The /usr/bin/shortcuts CLI is not present (wrong OS or too old)."""


@dataclass(frozen=True)
class ShortcutResult:
    """Outcome of a ``shortcuts run`` invocation."""
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    name: str


def _shortcuts_binary_available() -> bool:
    """Return True if /usr/bin/shortcuts exists and is executable."""
    if sys.platform != "darwin":
        return False
    return shutil.which(_SHORTCUTS_BIN) is not None or os.access(
        _SHORTCUTS_BIN, os.X_OK,
    )


def _build_argv(name: str, input_data: Optional[str]) -> list:
    """Compose the argv for ``shortcuts run <name> [--input-path -]``."""
    argv = [_SHORTCUTS_BIN, "run", name]
    if input_data is not None:
        argv.extend(["--input-path", "-"])
    return argv


def _enforce_allowlist(name: str, allowlist_kind: Optional[str]) -> None:
    """Raise ShortcutError if the allowlist is configured and rejects ``name``."""
    if allowlist_kind is None:
        return
    try:
        check_allowed(allowlist_kind, name)
    except AllowlistNotFound as exc:
        raise ShortcutError(
            f"allowlist {allowlist_kind!r} missing backing file: {exc}"
        ) from exc
    except AllowlistRejection as exc:
        raise ShortcutError(
            f"shortcut {name!r} rejected by allowlist {allowlist_kind!r}: {exc}"
        ) from exc


def _enforce_rate_limit(
    name: str, rate_limit: Optional[RateLimit],
) -> None:
    """Raise RateLimitExceeded if configured and the cap is hit."""
    if rate_limit is None:
        return
    check_and_record(f"shortcut:{name}", rate_limit)


def _audit_rejection(
    name: str, error: str, caller: str, provenance: ProvenanceType,
) -> None:
    """Write a failure audit entry for a pre-dispatch reject."""
    write_audit(make_entry(
        script=f"shortcut:{name}",
        tier=2,
        ok=False,
        duration_s=0.0,
        params={"name": name},
        caller=caller,
        provenance=provenance.value,
        host="local",
        error=error,
    ))


def run_shortcut(
    name: str,
    *,
    input_data: Optional[str] = None,
    timeout_s: int = 15,
    allowlist_kind: Optional[str] = None,
    rate_limit: Optional[RateLimit] = None,
    caller: str = "local",
    provenance: ProvenanceType = ProvenanceType.TOOL_CHAIN,
    audit: bool = True,
) -> ShortcutResult:
    """Dispatch a macOS Shortcut by name via the native CLI.

    ``name`` must match a Shortcut in the user's Shortcuts library. If
    ``allowlist_kind`` is set, the name must also appear in
    ``$HAL_HOME/config/osa-allowlists/<kind>.txt``. If ``rate_limit`` is
    supplied, the shortcut is capped per rolling window (counter keyed
    by ``shortcut:<name>``).

    ``input_data`` is piped over stdin via ``--input-path -``. The
    shortcuts runtime reads it as raw bytes — shell metacharacters in
    the body never reach an interpreter.

    Raises :class:`ShortcutUnavailable` on non-Darwin platforms.
    Raises :class:`ShortcutError` on allowlist rejection, timeout, or
    non-zero exit. Raises :class:`RateLimitExceeded` on cap hit.
    """
    if not _shortcuts_binary_available():
        raise ShortcutUnavailable(
            f"{_SHORTCUTS_BIN} not available on this platform"
        )

    try:
        _enforce_rate_limit(name, rate_limit)
    except RateLimitExceeded:
        if audit:
            _audit_rejection(name, "ratelimit_exceeded", caller, provenance)
        raise

    try:
        _enforce_allowlist(name, allowlist_kind)
    except ShortcutError as exc:
        if audit:
            _audit_rejection(name, str(exc), caller, provenance)
        raise

    argv = _build_argv(name, input_data)
    started = time.monotonic()
    try:
        result = subprocess.run(
            argv,
            input=input_data if input_data is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        if audit:
            _audit_rejection(
                name, f"timeout after {timeout_s}s", caller, provenance,
            )
        raise ShortcutError(
            f"shortcut {name!r} timed out after {timeout_s}s"
        ) from exc
    except OSError as exc:
        if audit:
            _audit_rejection(name, str(exc), caller, provenance)
        raise ShortcutError(
            f"shortcut {name!r} dispatch failed: {exc}"
        ) from exc

    duration = time.monotonic() - started
    outcome = ShortcutResult(
        ok=(result.returncode == 0),
        returncode=result.returncode,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
        duration_s=duration,
        name=name,
    )

    if audit:
        write_audit(make_entry(
            script=f"shortcut:{name}",
            tier=2,
            ok=outcome.ok,
            duration_s=duration,
            params={"name": name},
            caller=caller,
            provenance=provenance.value,
            host="local",
            returncode=result.returncode,
            error=None if outcome.ok else (
                (result.stderr or "").strip() or "non-zero exit"
            ),
        ))

    if not outcome.ok:
        raise ShortcutError(
            f"shortcut {name!r} exited {result.returncode}: "
            f"{(result.stderr or '').strip() or '(no stderr)'}"
        )

    return outcome
