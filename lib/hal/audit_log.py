"""HAL Audit Log — compliance-ready operation tracking.

Records every significant operation (LLM calls, tool executions, config
changes, security events) with timestamps, actors, and outcomes. Designed
for regulated environments (legal, healthcare, finance) where audit trails
are mandatory.

Usage:
    from lib.hal.audit_log import AuditLog

    log = AuditLog()
    log.record("llm_call", actor="user", detail="groq/llama: 'How do I...'")
    log.record("file_write", actor="agent", detail="Wrote auth.py")
    entries = log.query(actor="agent", limit=10)
"""
import json
import os
import time
from dataclasses import dataclass, field

from lib.hal.home import HAL_HOME as _HAL_HOME

_AUDIT_DIR = os.path.join(_HAL_HOME, "state", "hal", "audit")


@dataclass(frozen=True)
class AuditEntry:
    """A single audit log entry."""
    timestamp: float
    operation: str
    actor: str  # "user", "agent", "system", "hook"
    detail: str
    outcome: str = "success"  # "success", "failure", "blocked"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)),
            "operation": self.operation,
            "actor": self.actor,
            "detail": self.detail[:500],
            "outcome": self.outcome,
            "metadata": self.metadata,
        }


class AuditLog:
    """Append-only audit log with query support."""

    def __init__(self, max_memory: int = 1000):
        self._entries: list[AuditEntry] = []
        self._max_memory = max_memory

    def record(
        self, operation: str, *, actor: str = "system",
        detail: str = "", outcome: str = "success", **metadata,
    ) -> AuditEntry:
        """Record an operation."""
        entry = AuditEntry(
            timestamp=time.time(), operation=operation,
            actor=actor, detail=detail, outcome=outcome,
            metadata=metadata,
        )
        self._entries.append(entry)
        if len(self._entries) > self._max_memory:
            self._flush_to_disk()
        return entry

    def query(
        self, operation: str = "", actor: str = "",
        outcome: str = "", limit: int = 50,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filters."""
        results = self._entries
        if operation:
            results = [e for e in results if e.operation == operation]
        if actor:
            results = [e for e in results if e.actor == actor]
        if outcome:
            results = [e for e in results if e.outcome == outcome]
        return results[-limit:]

    def count(self, operation: str = "") -> int:
        """Count entries, optionally filtered by operation."""
        if operation:
            return sum(1 for e in self._entries if e.operation == operation)
        return len(self._entries)

    def flush(self) -> str:
        """Write all entries to disk and clear memory. Returns file path."""
        return self._flush_to_disk()

    def _flush_to_disk(self) -> str:
        """Append entries to JSONL file on disk."""
        os.makedirs(_AUDIT_DIR, exist_ok=True)
        date_str = time.strftime("%Y-%m-%d")
        path = os.path.join(_AUDIT_DIR, f"audit-{date_str}.jsonl")
        with open(path, "a") as f:
            for entry in self._entries:
                f.write(json.dumps(entry.to_dict()) + "\n")
        count = len(self._entries)
        self._entries.clear()
        return path
