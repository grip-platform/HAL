"""HAL Event Bus — cross-module publish/subscribe communication.

Enables loose coupling between HAL modules. Any module can emit events
and any other module can subscribe without direct imports.

Usage:
    from lib.hal.event_bus import EventBus

    bus = EventBus()
    bus.on("query_complete", lambda data: print(data))
    bus.emit("query_complete", {"model": "groq", "tokens": 500})
"""
import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    name: str
    data: dict = field(default_factory=dict)


class EventBus:
    """Thread-safe publish/subscribe event bus."""

    def __init__(self):
        self._handlers: dict[str, list[callable]] = {}
        self._lock = threading.Lock()
        self._history: list[Event] = []
        self._max_history = 200

    def on(self, event_name: str, handler: callable) -> None:
        """Subscribe to an event."""
        with self._lock:
            if event_name not in self._handlers:
                self._handlers[event_name] = []
            self._handlers[event_name].append(handler)

    def off(self, event_name: str, handler: callable) -> bool:
        """Unsubscribe from an event. Returns True if removed."""
        with self._lock:
            handlers = self._handlers.get(event_name, [])
            if handler in handlers:
                handlers.remove(handler)
                return True
            return False

    def emit(self, event_name: str, data: dict | None = None) -> int:
        """Emit an event. Returns number of handlers called."""
        event = Event(name=event_name, data=data or {})
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            handlers = list(self._handlers.get(event_name, []))

        called = 0
        for handler in handlers:
            try:
                handler(event.data)
                called += 1
            except Exception:
                pass
        return called

    def recent(self, limit: int = 10) -> list[Event]:
        """Get recent events."""
        with self._lock:
            return list(self._history[-limit:])

    def clear(self) -> None:
        """Remove all handlers and history."""
        with self._lock:
            self._handlers.clear()
            self._history.clear()
