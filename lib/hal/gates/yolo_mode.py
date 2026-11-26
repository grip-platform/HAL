"""HAL yolo-mode gate — autonomous-loop bypass marker, HAL_HOME-rooted.

Standalone port of GRIP's ``lib/gates/yolo_mode.py`` for Phase B of the
GRIP-independence rollout (H155). All paths resolve via ``$HAL_HOME``;
no dependency on ``$CLAUDE_DIR`` or a GRIP install.

Contract:

* :func:`activate_yolo(context, ttl_seconds, reason)` — write marker
  atomically (tempfile + rename). Clamps TTL to ``MAX_TTL_SECONDS``.
* :func:`deactivate_yolo(context)` — remove marker. Audit-logs the
  deactivation. Missing marker is a no-op (idempotent).
* :func:`is_yolo_active()` — returns ``(active, context)``. Expired
  markers return ``(False, "")``. Malformed JSON fails closed — a typo
  cannot accidentally bypass halting gates.
* :func:`yolo_marker_path()` / :func:`yolo_log_path()` — overridable
  via ``HAL_HOME`` env var; primarily used by tests.

Design invariants inherited from the GRIP original:

1. TTL auto-invalidation is mechanical — a marker past its deadline is
   inactive without any cleanup step.
2. Atomic write: ``tempfile + os.replace`` so a crashed activate never
   leaves half-JSON.
3. Fail-closed on parse error (returns False, not True).
4. Log rotation: append-only JSONL; caller manages size externally.

The bypass SET (which gates yolo actually relaxes) is DELIBERATELY not
encoded here. HAL's own halting gates ship their bypass IDs in the gate
code itself. This module only provides the MARKER that those gates
consult — it does not declare the gate taxonomy on behalf of callers.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Tuple

__all__ = [
    "activate_yolo",
    "deactivate_yolo",
    "is_yolo_active",
    "yolo_marker_path",
    "yolo_log_path",
    "MAX_TTL_SECONDS",
    "DEFAULT_TTL_SECONDS",
]


MAX_TTL_SECONDS = 12 * 3600  # 12 hours — prevents an infinite yolo
DEFAULT_TTL_SECONDS = 14400  # 4 hours — common /ultrado cadence


def _hal_home() -> Path:
    return Path(os.environ.get("HAL_HOME", str(Path.home() / ".hal")))


def yolo_marker_path() -> Path:
    """Return the marker path (``$HAL_HOME/state/session/yolo-mode.json``)."""
    return _hal_home() / "state" / "session" / "yolo-mode.json"


def yolo_log_path() -> Path:
    """Return the audit log path (``$HAL_HOME/logs/yolo-mode.log``)."""
    return _hal_home() / "logs" / "yolo-mode.log"


def _now_epoch() -> float:
    return time.time()


def _iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write(target: Path, payload: str) -> None:
    """Write ``payload`` to ``target`` via tempfile + os.replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=target.name + ".tmp-", dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, target)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _append_log(line: str) -> None:
    """Append one audit line to the yolo log (best-effort)."""
    log = yolo_log_path()
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass  # log failures never block the gate state change


def activate_yolo(
    context: str,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    reason: str = "",
) -> Path:
    """Write the yolo marker with a clamped TTL and return its path.

    ``context`` labels the activator (e.g. ``/ultrado``, ``/rsi-optimal``).
    ``ttl_seconds`` is clamped to ``[60, MAX_TTL_SECONDS]`` — below-60 is
    treated as a typo, above-12h is treated as adversarial.
    """
    effective_ttl = max(60, min(int(ttl_seconds), MAX_TTL_SECONDS))
    now = _now_epoch()
    payload = {
        "context": context,
        "activated_at": _iso_utc(),
        "activated_at_epoch": now,
        "ttl_seconds": effective_ttl,
        "expires_at_epoch": now + effective_ttl,
        "reason": reason,
    }
    path = yolo_marker_path()
    _atomic_write(path, json.dumps(payload, sort_keys=True))
    _append_log(
        f"{_iso_utc()} activate context={context} ttl={effective_ttl} reason={reason}"
    )
    return path


def deactivate_yolo(context: str = "") -> None:
    """Remove the yolo marker (idempotent). Audit-logs the deactivation.

    ``context`` is recorded in the audit line so multi-activator flows
    can be reconstructed; it does not gate the removal.
    """
    path = yolo_marker_path()
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
    _append_log(f"{_iso_utc()} deactivate context={context}")


def is_yolo_active() -> Tuple[bool, str]:
    """Return ``(active, context)`` — fail-closed on parse failure.

    An expired marker (now ≥ expires_at_epoch) returns (False, "") without
    deleting the marker; cleanup is the caller's job (e.g. session-start
    hook). A malformed JSON file returns (False, "") — never True.
    """
    path = yolo_marker_path()
    if not path.exists():
        return False, ""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return False, ""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False, ""
    expires = payload.get("expires_at_epoch")
    context = str(payload.get("context", ""))
    if not isinstance(expires, (int, float)):
        return False, ""
    if _now_epoch() >= expires:
        return False, ""
    return True, context
