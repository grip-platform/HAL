"""HAL Fleet Gateway — cross-transport thread store.

Holds per-canonical-user thread state so the same human reaching HAL
from Telegram and Slack sees a single coherent conversation. The store
is additive — HAL's existing session_manager.py handles runtime
session state (the Claude Code process), and session_registry.py holds
per-agent deployment state; this module is strictly for the gateway's
conversational threads.

Design decisions:

- Key is the canonical user id (e.g. "M>>"), not the platform-native
  thread_id. This is the whole point of cross-transport continuity.
- For users who DO need multiple parallel conversations (group chats,
  topical bots), a `thread_name` arg partitions within a canonical
  user. Default is "default".
- Message history is bounded (ring buffer of 128 entries by default).
  Callers who need durable long-term history should persist separately.
- Thread-safe via a single lock. File-based persistence at
  `state/hal/gateway_threads.json`, flushed on each mutation.

Usage:
    from lib.hal.gateway_threads import GatewayThreadStore
    from lib.hal.channels import MessageEnvelope

    store = GatewayThreadStore()
    env = MessageEnvelope(transport="telegram", external_id="123",
                          thread_id="chat99", body="hello")
    store.append("M>>", env)
    history = store.messages("M>>")          # latest first, default thread
    all_threads = store.thread_names("M>>")  # e.g. ("default",)
"""
import json
import os
import threading
from typing import Optional

from lib.hal.channels import MessageEnvelope
from lib.hal.home import HAL_HOME as _HAL_HOME

_STATE_DIR = os.path.join(_HAL_HOME, "state", "hal")
_STORE_PATH = os.path.join(_STATE_DIR, "gateway_threads.json")
_DEFAULT_THREAD = "default"
_MAX_HISTORY = 128


class GatewayThreadStore:
    """Per-canonical-user Fleet Gateway thread state.

    One store instance per HAL process. Multiple canonical users, each
    with one or more named threads, each thread holding a bounded
    message history and arbitrary meta.
    """

    def __init__(
        self,
        store_path: Optional[str] = None,
        *,
        max_history: int = _MAX_HISTORY,
    ):
        self._path = store_path or _STORE_PATH
        self._max_history = max_history
        self._lock = threading.Lock()
        self._threads: dict[str, dict[str, dict]] = self._load()

    def _load(self) -> dict:
        if not os.path.isfile(self._path):
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _flush_locked(self) -> None:
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{self._path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._threads, f, indent=2, sort_keys=True)
        os.replace(tmp, self._path)

    def _get_thread_locked(self, canonical_user: str, thread_name: str) -> dict:
        user_threads = self._threads.setdefault(canonical_user, {})
        return user_threads.setdefault(
            thread_name, {"messages": [], "meta": {}}
        )

    def append(
        self,
        canonical_user: str,
        envelope: MessageEnvelope,
        thread_name: str = _DEFAULT_THREAD,
    ) -> None:
        """Append a MessageEnvelope to the named thread for the canonical user.

        History is trimmed to max_history most recent entries.
        """
        if not canonical_user:
            raise ValueError("canonical_user must be non-empty")
        with self._lock:
            thread = self._get_thread_locked(canonical_user, thread_name)
            thread["messages"].append(envelope.to_dict())
            excess = len(thread["messages"]) - self._max_history
            if excess > 0:
                thread["messages"] = thread["messages"][excess:]
            self._flush_locked()

    def messages(
        self,
        canonical_user: str,
        thread_name: str = _DEFAULT_THREAD,
        *,
        limit: Optional[int] = None,
    ) -> tuple:
        """Return the recent messages in the thread, oldest first."""
        with self._lock:
            user_threads = self._threads.get(canonical_user, {})
            thread = user_threads.get(thread_name)
            if thread is None:
                return ()
            history = list(thread.get("messages", []))
        if limit is not None and limit >= 0:
            history = history[-limit:]
        return tuple(history)

    def set_meta(
        self,
        canonical_user: str,
        meta: dict,
        thread_name: str = _DEFAULT_THREAD,
    ) -> None:
        """Replace the thread meta dict with the given mapping."""
        with self._lock:
            thread = self._get_thread_locked(canonical_user, thread_name)
            thread["meta"] = dict(meta or {})
            self._flush_locked()

    def get_meta(
        self,
        canonical_user: str,
        thread_name: str = _DEFAULT_THREAD,
    ) -> dict:
        """Return a copy of the thread meta dict, or empty dict if absent."""
        with self._lock:
            user_threads = self._threads.get(canonical_user, {})
            thread = user_threads.get(thread_name, {})
            return dict(thread.get("meta", {}))

    def thread_names(self, canonical_user: str) -> tuple:
        """Return the sorted names of threads for a canonical user."""
        with self._lock:
            user_threads = self._threads.get(canonical_user, {})
            return tuple(sorted(user_threads))

    def delete(
        self,
        canonical_user: str,
        thread_name: Optional[str] = None,
    ) -> None:
        """Remove a single thread (thread_name given) or all threads (None).

        Idempotent — deleting an absent thread or user is a no-op.
        """
        with self._lock:
            user_threads = self._threads.get(canonical_user)
            if not user_threads:
                return
            if thread_name is None:
                del self._threads[canonical_user]
                self._flush_locked()
                return
            if thread_name in user_threads:
                del user_threads[thread_name]
                if not user_threads:
                    del self._threads[canonical_user]
                self._flush_locked()

    def canonical_users(self) -> tuple:
        """Return the sorted canonical user ids with at least one thread."""
        with self._lock:
            return tuple(sorted(self._threads))
