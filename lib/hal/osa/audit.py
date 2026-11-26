"""Append-only audit log for osa dispatch.

Every :func:`lib.hal.osa.dispatch.osa` call — success or failure — emits
one JSONL entry to ``$CLAUDE_DIR/logs/osa.log``. The log is shared with
GRIP (GRIP and HAL both use the same user-level audit surface) so that
security review can reconstruct what the system did on this host
irrespective of which harness invoked it.

α-3b closes the council-flagged tamper window on macOS by calling
``chflags uappnd`` on the log the first time a write happens in a given
process. ``uappnd`` is the BSD user-level append-only flag: once set,
even the owning user cannot rewrite log history without first clearing
the flag (``chflags nouappnd``, which itself requires explicit intent).
On non-Darwin hosts the call is skipped silently. Tests opt out via
``HAL_OSA_SKIP_IMMUTABLE=1``.

Rotation at 10 MB keeps growth bounded. When we rotate, the append-only
flag on the backup is left in place — the backup is still immutable from
the operator's perspective; only the **active** file becomes append-only
again after rotation.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

__all__ = ["AuditEntry", "write_audit", "AUDIT_LOG_PATH", "make_entry"]


_CLAUDE_DIR = Path(os.environ.get("CLAUDE_DIR", Path.home() / ".claude"))
AUDIT_LOG_PATH = _CLAUDE_DIR / "logs" / "osa.log"
_AUDIT_ROLL_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
_IMMUTABLE_APPLIED: Dict[str, bool] = {}


@dataclass(frozen=True)
class AuditEntry:
    """One dispatch outcome captured for the audit stream.

    The ``provenance`` field is the canonical origin label added in α-3b.
    ``caller`` is kept for α-3a back-compat as a free-form annotation.
    """
    ts: str
    script: str
    tier: int
    ok: bool
    duration_s: float
    params: Dict[str, object] = field(default_factory=dict)
    host: str = "local"
    caller: str = "local"
    provenance: str = "local"
    returncode: Optional[int] = None
    error: Optional[str] = None


def _now_utc() -> str:
    """ISO-8601 UTC timestamp matching the wider HAL/GRIP convention."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _maybe_roll(log_path: Path) -> None:
    """Rename the current log to ``.log.1`` if it exceeds the roll size.

    Rotation also clears the append-only-applied marker for this path so
    the fresh file gets ``chflags uappnd`` re-applied on its first write.
    """
    if not log_path.exists():
        return
    if log_path.stat().st_size <= _AUDIT_ROLL_SIZE_BYTES:
        return
    backup = log_path.with_suffix(log_path.suffix + ".1")
    if backup.exists():
        _clear_append_only(backup)
        backup.unlink()
    log_path.rename(backup)
    _IMMUTABLE_APPLIED.pop(str(log_path), None)


def _apply_append_only(log_path: Path) -> None:
    """Set the BSD ``uappnd`` flag on ``log_path`` (macOS only, once per path).

    Silently no-op when:
      * ``HAL_OSA_SKIP_IMMUTABLE=1`` (test/CI environments)
      * platform is not Darwin
      * ``chflags`` is missing, errors, or times out
      * the flag was already applied to this path in this process

    A failure here never blocks audit writes — audit availability beats
    audit immutability if we have to pick.
    """
    key = str(log_path)
    if _IMMUTABLE_APPLIED.get(key):
        return
    _IMMUTABLE_APPLIED[key] = True  # mark first so a failure doesn't retry every write

    if os.environ.get("HAL_OSA_SKIP_IMMUTABLE") == "1":
        return
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(
            ["chflags", "uappnd", str(log_path)],
            check=False,
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        pass  # loud failure would make every write noisy


def _clear_append_only(log_path: Path) -> None:
    """Best-effort removal of the ``uappnd`` flag so rotation can unlink."""
    if os.environ.get("HAL_OSA_SKIP_IMMUTABLE") == "1":
        return
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(
            ["chflags", "nouappnd", str(log_path)],
            check=False,
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def write_audit(
    entry: AuditEntry, log_path: Optional[Path] = None
) -> None:
    """Append one :class:`AuditEntry` as a single JSONL line.

    The log path defaults to :data:`AUDIT_LOG_PATH` but can be overridden
    for tests. Ensures the parent directory exists. On overrun, rotates to
    ``osa.log.1`` (destructive replace) then starts a fresh log. No partial
    writes — ``json.dumps`` runs before ``open`` so a serialisation error
    surfaces before the file is touched.
    """
    target = log_path or AUDIT_LOG_PATH
    payload = json.dumps(asdict(entry), sort_keys=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    _maybe_roll(target)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(payload + "\n")
    _apply_append_only(target)


def make_entry(
    *,
    script: str,
    tier: int,
    ok: bool,
    duration_s: float,
    params: Optional[Dict[str, object]] = None,
    host: str = "local",
    caller: str = "local",
    provenance: str = "local",
    returncode: Optional[int] = None,
    error: Optional[str] = None,
) -> AuditEntry:
    """Build an :class:`AuditEntry` with the timestamp filled in."""
    return AuditEntry(
        ts=_now_utc(),
        script=script,
        tier=tier,
        ok=ok,
        duration_s=duration_s,
        params=dict(params or {}),
        host=host,
        caller=caller,
        provenance=provenance,
        returncode=returncode,
        error=error,
    )
