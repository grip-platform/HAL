"""HAL Conversation Branching — fork conversations to explore alternatives.

Allows users to create branches at any point in a conversation, explore
a different direction, and optionally merge insights back. Enables
"what if" exploration without losing the original context.

Usage:
    from lib.hal.conversation_branch import BranchManager

    manager = BranchManager()
    branch_id = manager.create_branch(history, branch_point=4)
    # Explore the branch independently...
    branches = manager.list_branches()
"""
import json
import os
import time
from dataclasses import dataclass, field

from lib.hal.home import HAL_HOME as _HAL_HOME
from lib.hal.types import ChatMessage

_BRANCHES_DIR = os.path.join(_HAL_HOME, "state", "hal", "branches")


@dataclass
class Branch:
    """A conversation branch from a specific point."""
    branch_id: str
    parent_session: str
    branch_point: int  # Turn number where branch diverges
    created_at: float
    history: list[ChatMessage] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "branch_id": self.branch_id,
            "parent_session": self.parent_session,
            "branch_point": self.branch_point,
            "created_at": self.created_at,
            "description": self.description,
            "turn_count": len([m for m in self.history if m.role == "user"]),
            "message_count": len(self.history),
        }


class BranchManager:
    """Manages conversation branches."""

    def __init__(self, session_id: str = "default"):
        self._session = session_id
        self._branches: dict[str, Branch] = {}

    def create_branch(
        self, history: list[ChatMessage],
        branch_point: int = -1,
        description: str = "",
    ) -> str:
        """Create a branch from the current conversation.

        Args:
            history: Full conversation history
            branch_point: Message index to branch from (-1 = current point)
            description: Human description of what this branch explores
        """
        point = branch_point if branch_point >= 0 else len(history)
        branch_id = f"branch-{int(time.time())}-{len(self._branches)}"

        branch = Branch(
            branch_id=branch_id,
            parent_session=self._session,
            branch_point=point,
            created_at=time.time(),
            history=list(history[:point]),
            description=description,
        )
        self._branches[branch_id] = branch
        return branch_id

    def get_branch(self, branch_id: str) -> Branch | None:
        """Get a branch by ID."""
        return self._branches.get(branch_id)

    def add_to_branch(self, branch_id: str, message: ChatMessage) -> bool:
        """Add a message to a branch. Returns True if successful."""
        branch = self._branches.get(branch_id)
        if not branch:
            return False
        branch.history.append(message)
        return True

    def list_branches(self) -> list[dict]:
        """List all branches for the current session."""
        return [b.to_dict() for b in self._branches.values()]

    def delete_branch(self, branch_id: str) -> bool:
        """Delete a branch."""
        return self._branches.pop(branch_id, None) is not None

    def save(self) -> None:
        """Persist all branches to disk."""
        os.makedirs(_BRANCHES_DIR, exist_ok=True)
        for branch_id, branch in self._branches.items():
            path = os.path.join(_BRANCHES_DIR, f"{branch_id}.json")
            data = branch.to_dict()
            data["history"] = [{"role": m.role, "content": m.content} for m in branch.history]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
