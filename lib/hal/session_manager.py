"""HAL Session Manager — session lifecycle abstraction.

Manages session ID, state persistence, and cross-session continuity
without depending on CC's specific environment variables.

Usage:
    from lib.hal.session_manager import SessionManager

    mgr = SessionManager()
    print(mgr.session_id)
    mgr.save_state({"next_up": "Phase 4", "key_files": [...]})
    restored = mgr.load_state()
"""
import json
import os
import time
import uuid
from typing import Optional

from lib.hal.home import HAL_HOME as _HAL_HOME
_STATE_DIR = os.path.join(_HAL_HOME, "state", "hal")
_SESSION_FILE = os.path.join(_STATE_DIR, "session.json")


class SessionManager:
    """Manages session lifecycle and state persistence."""

    def __init__(self, session_id: Optional[str] = None):
        self._session_id = session_id or _detect_session_id()
        self._start_time = time.time()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self._start_time

    def is_cc_session(self) -> bool:
        """Check if running inside Claude Code."""
        return bool(os.environ.get("CLAUDE_CODE_SESSION_ID"))

    def save_state(self, state: dict) -> None:
        """Persist session state to disk."""
        os.makedirs(_STATE_DIR, exist_ok=True)
        payload = {
            "session_id": self._session_id,
            "saved_at": time.time(),
            "elapsed_s": self.elapsed_seconds,
            **state,
        }
        with open(_SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_state(self) -> Optional[dict]:
        """Load the most recent session state."""
        if not os.path.exists(_SESSION_FILE):
            return None
        try:
            with open(_SESSION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def is_same_session(self) -> bool:
        """Check if loaded state is from the current session."""
        state = self.load_state()
        if not state:
            return False
        return state.get("session_id") == self._session_id


def _detect_session_id() -> str:
    """Detect session ID from environment or generate a new one."""
    cc_id = os.environ.get("CLAUDE_CODE_SESSION_ID")
    if cc_id:
        return cc_id

    grip_id = os.environ.get("GRIP_SESSION_ID")
    if grip_id:
        return grip_id

    return str(uuid.uuid4())
