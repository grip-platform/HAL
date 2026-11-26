"""Kernel execution primitive for osascript.

Phase α-2 of the osascript control layer plan. Ships the bare subprocess
wrapper and typed exceptions — no registry, no tier gate, no allowlist,
no audit log, no host dispatch. Those arrive in α-3, built on top of
this kernel without modifying it.

Public API::

    from lib.hal.osa import osa_raw, OsaResult, OsaError, OsaTimeout

    result = osa_raw('return "hello"')
    assert result.ok and result.stdout.strip() == "hello"

Callers are responsible for escaping any interpolated values — use
:func:`lib.hal.osa.applescript_escape` for AppleScript string literals.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "OsaResult",
    "OsaError",
    "OsaTimeout",
    "PlatformUnsupported",
    "osa_raw",
]


Language = Literal["AppleScript", "JavaScript"]


class OsaError(RuntimeError):
    """osa primitive failed (non-zero exit from osascript)."""


class OsaTimeout(OsaError):
    """osa primitive exceeded its ``timeout_s`` wall-clock budget."""


class PlatformUnsupported(OsaError):
    """Host is not darwin or ``osascript`` is not on PATH."""


@dataclass(frozen=True)
class OsaResult:
    """Outcome of a single :func:`osa_raw` call.

    ``ok`` is True iff ``returncode == 0``. ``duration_s`` captures
    wall-clock time from just-before subprocess spawn to just-after-exit;
    used by the α-3 audit layer for telemetry.
    """
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    language: Language


def _probe_platform() -> None:
    """Raise :class:`PlatformUnsupported` if this host cannot run osascript."""
    if sys.platform != "darwin":
        raise PlatformUnsupported(
            f"osa_raw requires darwin; running on {sys.platform}"
        )
    if shutil.which("osascript") is None:
        raise PlatformUnsupported("osascript not found on PATH")


def osa_raw(
    script_body: str,
    *,
    language: Language = "AppleScript",
    timeout_s: int = 10,
    check: bool = True,
) -> OsaResult:
    """Execute ``script_body`` via osascript and return a structured result.

    Args:
        script_body: Complete AppleScript or JavaScript OSA source. Callers
            must pre-escape any interpolated values.
        language: ``"AppleScript"`` (default, safer) or ``"JavaScript"``.
            JavaScript OSA escapes the AppleScript sandbox entirely via the
            ObjC bridge; α-3 will gate it behind a per-script declaration.
        timeout_s: Wall-clock cap. Raises :class:`OsaTimeout` on overrun.
        check: When True (default), raise :class:`OsaError` on non-zero
            exit. When False, return the failed :class:`OsaResult` so the
            caller can inspect stderr/returncode.

    Returns:
        :class:`OsaResult` with stdout, stderr, returncode, and duration.

    Raises:
        PlatformUnsupported: Not on darwin or osascript missing.
        OsaTimeout: Subprocess exceeded ``timeout_s``.
        OsaError: Subprocess returned non-zero and ``check`` is True.
    """
    _probe_platform()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            ["osascript", "-l", language, "-e", script_body],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - started
        raise OsaTimeout(
            f"osa_raw({language}) exceeded {timeout_s}s "
            f"(elapsed={elapsed:.2f}s)"
        ) from e
    elapsed = time.monotonic() - started
    result = OsaResult(
        ok=completed.returncode == 0,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_s=elapsed,
        language=language,
    )
    if check and not result.ok:
        raise OsaError(
            f"osa_raw({language}) exited {completed.returncode}: "
            f"{completed.stderr.strip() or '(no stderr)'}"
        )
    return result
