"""HAL Context Snapshot — save and restore conversation state.

Captures the full state of a conversation (history, model, mode, cost,
context usage) as a portable snapshot. Enables session handoff, backup,
and restore across devices or sessions.

Usage:
    from lib.hal.context_snapshot import ContextSnapshot

    snap = ContextSnapshot.capture(conversation)
    snap.save("checkpoint-1.json")
    # Later...
    restored = ContextSnapshot.load("checkpoint-1.json")
"""
import json
import os
import time
from dataclasses import dataclass, field

from lib.hal.home import HAL_HOME as _HAL_HOME
from lib.hal.types import ChatMessage

_SNAPSHOTS_DIR = os.path.join(_HAL_HOME, "state", "hal", "snapshots")


@dataclass
class ContextSnapshot:
    """Portable conversation state snapshot."""
    session_id: str = ""
    model: str = ""
    mode: str = ""
    history: list[ChatMessage] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.history if m.role == "user")

    def save(self, path: str = "") -> str:
        """Save snapshot to JSON. Returns file path."""
        if not path:
            os.makedirs(_SNAPSHOTS_DIR, exist_ok=True)
            path = os.path.join(_SNAPSHOTS_DIR, f"snap-{int(self.created_at)}.json")
        data = {
            "session_id": self.session_id, "model": self.model,
            "mode": self.mode, "created_at": self.created_at,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "turn_count": self.turn_count,
            "history": [{"role": m.role, "content": m.content} for m in self.history],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "ContextSnapshot":
        """Load snapshot from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return ContextSnapshot(
            session_id=data.get("session_id", ""),
            model=data.get("model", ""),
            mode=data.get("mode", ""),
            history=[ChatMessage(role=m["role"], content=m["content"]) for m in data.get("history", [])],
            total_cost_usd=data.get("total_cost_usd", 0),
            total_tokens=data.get("total_tokens", 0),
            created_at=data.get("created_at", 0),
        )

    @staticmethod
    def list_snapshots() -> list[str]:
        """List saved snapshot files."""
        if not os.path.isdir(_SNAPSHOTS_DIR):
            return []
        return sorted(
            os.path.join(_SNAPSHOTS_DIR, f)
            for f in os.listdir(_SNAPSHOTS_DIR) if f.endswith(".json")
        )
