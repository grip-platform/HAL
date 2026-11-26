"""HAL Session Registry — multi-session conversation management.

Manages multiple concurrent conversations, each with its own session ID,
history, model, and cost tracking. Disk-backed for persistence across
server restarts.

Matches GRIP-GUI's chat-storage model: each chat tab maps to a session.

Usage:
    from lib.hal.session_registry import SessionRegistry

    registry = SessionRegistry()
    session_id = registry.create(model="groq/llama-3.3-70b")
    session = registry.get(session_id)
    response = session.conversation.send("Hello")
    sessions = registry.list_sessions()
"""
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from lib.hal.conversation import Conversation
from lib.hal.home import HAL_HOME as _HAL_HOME
from lib.hal.registry import ProviderRegistry, get_registry

_SESSIONS_DIR = os.path.join(_HAL_HOME, "state", "sessions")
_MAX_SESSIONS = 50


@dataclass
class SessionSummary:
    """Lightweight session metadata for list responses."""
    id: str
    title: str
    model: str
    message_count: int
    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "model": self.model,
            "message_count": self.message_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ConversationSession:
    """A single chat session with its conversation engine."""
    id: str
    title: str
    model: str
    conversation: Conversation
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    message_count: int = 0
    # Serialises concurrent turns on THIS session under the multi-threaded
    # server: two requests sharing one session_id must not interleave writes to
    # the one Conversation (history/cost corruption). compare/repr excluded — a
    # Lock has no meaningful equality or repr. Not serialised (persist builds its
    # dict explicitly), so disk round-trips get a fresh lock per load.
    lock: "threading.Lock" = field(
        default_factory=threading.Lock, compare=False, repr=False,
    )

    def summary(self) -> SessionSummary:
        return SessionSummary(
            id=self.id,
            title=self.title,
            model=self.model,
            message_count=self.message_count,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def touch(self) -> None:
        self.updated_at = time.time()


class SessionRegistry:
    """Thread-safe registry of conversation sessions with disk persistence."""

    def __init__(self, registry: Optional[ProviderRegistry] = None):
        self._registry = registry or get_registry()
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = threading.Lock()
        os.makedirs(_SESSIONS_DIR, exist_ok=True)
        self._load_from_disk()

    def create(self, model: str = "", title: str = "New conversation") -> str:
        """Create a new session. Returns the session ID."""
        session_id = str(uuid.uuid4())[:8]
        conversation = Conversation(model=model, registry=self._registry)

        session = ConversationSession(
            id=session_id,
            title=title,
            model=model or conversation.model,
            conversation=conversation,
        )

        with self._lock:
            self._sessions[session_id] = session
            self._evict_if_needed()
            self._persist(session)

        return session_id

    def get(self, session_id: str) -> Optional[ConversationSession]:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create(self, session_id: Optional[str], model: str = "") -> ConversationSession:
        """Get existing session or create a new one. Returns the session."""
        if session_id:
            session = self.get(session_id)
            if session:
                return session

        new_id = self.create(model=model)
        return self._sessions[new_id]

    def list_sessions(self) -> list[SessionSummary]:
        """List all sessions, most recently updated first."""
        with self._lock:
            summaries = [s.summary() for s in self._sessions.values()]
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries

    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if found and deleted."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return False
            self._remove_from_disk(session_id)
            return True

    def update_title(self, session_id: str, title: str) -> None:
        """Update session title (e.g. from first user message)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.title = title
                self._persist(session)

    def record_turn(self, session_id: str) -> None:
        """Record that a conversation turn happened."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.message_count = len(session.conversation.history)
                session.touch()
                self._persist(session)

    def _evict_if_needed(self) -> None:
        """Remove oldest sessions if over capacity. Must hold lock. O(n log n) sort + O(k) slice."""
        overflow = len(self._sessions) - _MAX_SESSIONS
        if overflow <= 0:
            return
        by_age = sorted(self._sessions.values(), key=lambda s: s.updated_at)
        for victim in by_age[:overflow]:
            del self._sessions[victim.id]
            self._remove_from_disk(victim.id)

    def _persist(self, session: ConversationSession) -> None:
        """Save session metadata to disk. Must hold lock."""
        path = os.path.join(_SESSIONS_DIR, f"{session.id}.json")
        data = {
            "id": session.id,
            "title": session.title,
            "model": session.model,
            "message_count": session.message_count,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "history": [
                {"role": m.role, "content": m.content}
                for m in session.conversation.history
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _remove_from_disk(self, session_id: str) -> None:
        """Remove session file. Must hold lock."""
        path = os.path.join(_SESSIONS_DIR, f"{session_id}.json")
        if os.path.exists(path):
            os.remove(path)

    def _load_from_disk(self) -> None:
        """Rehydrate sessions from disk on startup. O(S) over session files."""
        if not os.path.isdir(_SESSIONS_DIR):
            return
        for fname in os.listdir(_SESSIONS_DIR):
            if not fname.endswith(".json"):
                continue
            session = self._load_session_file(os.path.join(_SESSIONS_DIR, fname))
            if session:
                self._sessions[session.id] = session

    def _load_session_file(self, path: str) -> Optional[ConversationSession]:
        """Load a single session from a JSON file. O(M) over messages."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            session_id = data["id"]
            model = data.get("model", "")
            conversation = Conversation(model=model, registry=self._registry)
            conversation._history = _restore_history(data.get("history", []))
            return ConversationSession(
                id=session_id,
                title=data.get("title", "Restored session"),
                model=model,
                conversation=conversation,
                created_at=data.get("created_at", time.time()),
                updated_at=data.get("updated_at", time.time()),
                message_count=data.get("message_count", 0),
            )
        except (json.JSONDecodeError, KeyError, OSError):
            return None


def _restore_history(messages: list[dict]) -> list:
    """Restore ChatMessage history from serialised dicts. O(M)."""
    from lib.hal.types import ChatMessage
    return [ChatMessage(role=m["role"], content=m["content"]) for m in messages]
