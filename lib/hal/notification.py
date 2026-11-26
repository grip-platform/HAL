"""HAL Notification System — event-driven alerts for background operations.

Notifies users when background tasks complete, errors occur, or important
events happen. Supports multiple channels: in-process callbacks, file-based
signals, and optional webhook delivery.

Usage:
    from lib.hal.notification import NotificationHub

    hub = NotificationHub()
    hub.subscribe("task_complete", lambda n: print(n.message))
    hub.notify("task_complete", "PR #33 merged successfully")
"""
import json
import os
import time
from dataclasses import dataclass, field

from lib.hal.home import hal_home


def notifications_dir() -> str:
    """``$HAL_HOME/state/hal/notifications``, resolved when called, never at
    import: a value fixed at import keeps the HAL_HOME the process started
    with, so a test that set its own would still write into that one."""
    return os.path.join(hal_home(), "state", "hal", "notifications")


@dataclass(frozen=True)
class Notification:
    """A single notification event."""
    event: str
    message: str
    severity: str = "info"  # "info", "success", "warning", "error"
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event": self.event, "message": self.message,
            "severity": self.severity, "timestamp": self.timestamp or time.time(),
            "metadata": self.metadata,
        }


class NotificationHub:
    """Central notification dispatch with pub/sub."""

    def __init__(self):
        self._subscribers: dict[str, list[callable]] = {}
        self._history: list[Notification] = []
        self._max_history = 100

    def subscribe(self, event: str, callback: callable) -> None:
        """Subscribe to an event type."""
        if event not in self._subscribers:
            self._subscribers[event] = []
        self._subscribers[event].append(callback)

    def notify(self, event: str, message: str, severity: str = "info", **metadata) -> Notification:
        """Emit a notification. Calls subscribers and persists to history."""
        notification = Notification(
            event=event, message=message, severity=severity,
            timestamp=time.time(), metadata=metadata,
        )

        # Dispatch to subscribers
        for callback in self._subscribers.get(event, []):
            try:
                callback(notification)
            except Exception:
                pass  # Never let subscriber errors break the notifier

        # Also dispatch to wildcard subscribers
        for callback in self._subscribers.get("*", []):
            try:
                callback(notification)
            except Exception:
                pass

        self._history.append(notification)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return notification

    def recent(self, limit: int = 10, event: str = "") -> list[Notification]:
        """Get recent notifications, optionally filtered by event."""
        if event:
            filtered = [n for n in self._history if n.event == event]
            return filtered[-limit:]
        return self._history[-limit:]

    def save_history(self) -> None:
        """Persist notification history to disk."""
        directory = notifications_dir()
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "history.json")
        data = [n.to_dict() for n in self._history]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def clear(self) -> int:
        """Clear notification history. Returns count cleared."""
        count = len(self._history)
        self._history.clear()
        return count
