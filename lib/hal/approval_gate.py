"""HAL Approval Gate — graduated autonomy for self-build operations.

Safety layer between code generation and deployment. Supports four
autonomy levels from full human control to overnight autonomous mode.
Even at FULL_AUTO, critical operations (delete, merge) require approval.

Usage:
    from lib.hal.approval_gate import ApprovalGate, ApprovalLevel

    gate = ApprovalGate()
    gate.set_level(ApprovalLevel.APPROVE_PR)
    approved = gate.check("write_module", "Writing new module foo.py")
"""
import json
import os
import time
from dataclasses import dataclass
from enum import Enum

from lib.hal.home import HAL_HOME as _HAL_HOME

_STATE_PATH = os.path.join(_HAL_HOME, "state", "hal", "approval-level.json")
_PENDING_PATH = os.path.join(_HAL_HOME, "state", "hal", "approval-pending.json")


class ApprovalLevel(str, Enum):
    """Graduated autonomy levels for self-build."""

    FULL_MANUAL = "full_manual"
    APPROVE_PR = "approve_pr"
    APPROVE_MERGE = "approve_merge"
    FULL_AUTO = "full_auto"


@dataclass(frozen=True)
class ApprovalRequest:
    """A request for human approval."""

    request_id: str
    action: str
    description: str
    risk_level: str
    timestamp: float = 0.0
    auto_approved: bool = False


@dataclass
class ApprovalDecision:
    """A human decision on an approval request."""

    request_id: str
    approved: bool
    reason: str = ""
    decided_at: float = 0.0


# Dispatch table: action -> risk level (not if/else)
_RISK_MAP: dict[str, str] = {
    "write_module": "low",
    "write_test": "low",
    "edit_module": "medium",
    "commit": "low",
    "create_pr": "medium",
    "push": "medium",
    "merge": "high",
    "delete_file": "critical",
    "edit_config": "high",
    "code_review": "low",   # auto-approved at APPROVE_PR and above
    "merge_pr": "high",     # auto-approved only at FULL_AUTO
}

# Dispatch table: (level, risk) -> auto-approve decision
_AUTO_APPROVE: dict[tuple[str, str], bool] = {
    (ApprovalLevel.FULL_MANUAL, "low"): False,
    (ApprovalLevel.FULL_MANUAL, "medium"): False,
    (ApprovalLevel.FULL_MANUAL, "high"): False,
    (ApprovalLevel.FULL_MANUAL, "critical"): False,
    (ApprovalLevel.APPROVE_PR, "low"): True,
    (ApprovalLevel.APPROVE_PR, "medium"): False,
    (ApprovalLevel.APPROVE_PR, "high"): False,
    (ApprovalLevel.APPROVE_PR, "critical"): False,
    (ApprovalLevel.APPROVE_MERGE, "low"): True,
    (ApprovalLevel.APPROVE_MERGE, "medium"): True,
    (ApprovalLevel.APPROVE_MERGE, "high"): False,
    (ApprovalLevel.APPROVE_MERGE, "critical"): False,
    (ApprovalLevel.FULL_AUTO, "low"): True,
    (ApprovalLevel.FULL_AUTO, "medium"): True,
    (ApprovalLevel.FULL_AUTO, "high"): True,
    (ApprovalLevel.FULL_AUTO, "critical"): False,
}


class ApprovalGate:
    """Graduated autonomy gate for self-build operations."""

    def __init__(self, state_path: str | None = None, pending_path: str | None = None):
        self._state_path = state_path or _STATE_PATH
        self._pending_path = pending_path or _PENDING_PATH
        self._level = ApprovalLevel.APPROVE_PR
        self._pending: dict[str, dict] = {}
        self._decisions: dict[str, dict] = {}
        self._load_state()

    def check(self, action: str, description: str = "") -> bool:
        """Check if an action is approved under current autonomy level.

        Returns True if auto-approved, False if requires human approval.
        When False, the request is queued in pending.
        """
        risk = _RISK_MAP.get(action, "high")
        auto = _AUTO_APPROVE.get((self._level, risk), False)

        request_id = f"{action}-{int(time.time() * 1000)}"
        request = ApprovalRequest(
            request_id=request_id,
            action=action,
            description=description,
            risk_level=risk,
            timestamp=time.time(),
            auto_approved=auto,
        )

        if auto:
            return True

        self._pending[request_id] = {
            "request_id": request_id,
            "action": action,
            "description": description,
            "risk_level": risk,
            "timestamp": request.timestamp,
        }
        self._save_pending()
        return False

    def set_level(self, level: ApprovalLevel) -> None:
        """Change the autonomy level. Persisted to disk."""
        self._level = level
        self._save_state()

    def get_level(self) -> ApprovalLevel:
        """Get current autonomy level."""
        return self._level

    def get_pending(self) -> list[ApprovalRequest]:
        """List pending requests awaiting human approval."""
        return [
            ApprovalRequest(
                request_id=p["request_id"],
                action=p["action"],
                description=p["description"],
                risk_level=p["risk_level"],
                timestamp=p.get("timestamp", 0.0),
            )
            for p in self._pending.values()
        ]

    def approve(self, request_id: str) -> bool:
        """Human approves a pending request. Returns False if not found."""
        if request_id not in self._pending:
            return False
        self._decisions[request_id] = {
            "approved": True, "decided_at": time.time(),
        }
        del self._pending[request_id]
        self._save_pending()
        return True

    def deny(self, request_id: str, reason: str = "") -> bool:
        """Human denies a pending request. Returns False if not found."""
        if request_id not in self._pending:
            return False
        self._decisions[request_id] = {
            "approved": False, "reason": reason, "decided_at": time.time(),
        }
        del self._pending[request_id]
        self._save_pending()
        return True

    def classify_risk(self, action: str) -> str:
        """Get the risk level for an action."""
        return _RISK_MAP.get(action, "high")

    def _load_state(self) -> None:
        """Load autonomy level from disk."""
        try:
            with open(self._state_path, encoding="utf-8") as f:
                data = json.load(f)
            self._level = ApprovalLevel(data.get("level", "approve_pr"))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            self._level = ApprovalLevel.APPROVE_PR

        try:
            with open(self._pending_path, encoding="utf-8") as f:
                self._pending = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._pending = {}

    def _save_state(self) -> None:
        """Persist autonomy level to disk (owner-only permissions)."""
        _write_secure_json(self._state_path, {"level": self._level.value})

    def _save_pending(self) -> None:
        """Persist pending requests to disk (owner-only permissions)."""
        _write_secure_json(self._pending_path, self._pending)


def _write_secure_json(path: str, data: dict) -> None:
    """Write JSON atomically with owner-only permissions (no TOCTOU race)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        handle = os.fdopen(fd, "w", encoding="utf-8")
    except BaseException:
        # Only a failed fdopen leaves the descriptor ours to close. Once it
        # succeeds the file object owns it, and a second close would hide the
        # real error behind EBADF.
        os.close(fd)
        raise
    with handle as f:
        json.dump(data, f, indent=2)
