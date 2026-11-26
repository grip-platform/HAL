"""HAL Session Handoff — transfer conversations between instances.

Packages conversation state, context, and metadata into a portable
format that can be transferred to another HAL instance (different
device, different session, different user).

Usage:
    from lib.hal.session_handoff import SessionHandoff
    handoff = SessionHandoff()
    package = handoff.package(conversation)
    handoff.restore(package, new_conversation)
"""
import json
import os
import time
from dataclasses import dataclass, field

from lib.hal.home import HAL_HOME as _HAL_HOME
from lib.hal.types import ChatMessage

_HANDOFF_DIR = os.path.join(_HAL_HOME, "state", "hal", "handoffs")


@dataclass
class HandoffPackage:
    """Portable conversation state for transfer."""
    session_id: str
    model: str
    history: list[ChatMessage] = field(default_factory=list)
    context_summary: str = ""
    cost_usd: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.history if m.role == "user")

    def save(self, path: str = "") -> str:
        if not path:
            os.makedirs(_HANDOFF_DIR, exist_ok=True)
            path = os.path.join(_HANDOFF_DIR, f"handoff-{int(self.created_at)}.json")
        data = {
            "session_id": self.session_id, "model": self.model,
            "context_summary": self.context_summary, "cost_usd": self.cost_usd,
            "created_at": self.created_at, "turn_count": self.turn_count,
            "history": [{"role": m.role, "content": m.content} for m in self.history],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "HandoffPackage":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return HandoffPackage(
            session_id=data.get("session_id", ""),
            model=data.get("model", ""),
            history=[ChatMessage(role=m["role"], content=m["content"]) for m in data.get("history", [])],
            context_summary=data.get("context_summary", ""),
            cost_usd=data.get("cost_usd", 0),
            created_at=data.get("created_at", 0),
        )
