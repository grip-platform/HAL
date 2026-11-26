"""HAL Scheduler — lightweight task scheduling without system cron.

Uses threading.Timer for periodic task execution. Persists schedules
to config/schedules.json. No external dependencies.

Usage:
    from lib.hal.scheduler import Scheduler, Schedule

    sched = Scheduler()
    sched.add(Schedule(name="warmup", interval_s=300, command="warmup_models"))
    sched.start()
"""
import json
import os
import threading
import time

from lib.hal.home import HAL_HOME as _HAL_HOME

_SCHEDULES_PATH = os.path.join(_HAL_HOME, "config", "schedules.json")


class Schedule:
    """A scheduled task definition."""

    def __init__(self, name: str, interval_s: int, command: str,
                 enabled: bool = True, last_run: float = 0.0):
        self.name = name
        self.interval_s = interval_s
        self.command = command
        self.enabled = enabled
        self.last_run = last_run


class Scheduler:
    """Lightweight task scheduler using threading.Timer."""

    def __init__(self):
        self._schedules: dict[str, Schedule] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._handlers: dict[str, callable] = {}
        self._running = False
        self._lock = threading.Lock()
        self._load()

    def add(self, schedule: Schedule) -> None:
        """Add or update a schedule."""
        with self._lock:
            self._schedules[schedule.name] = schedule
        self._save()
        if self._running and schedule.enabled:
            self._schedule_next(schedule.name)

    def remove(self, name: str) -> bool:
        """Remove a schedule. Returns True if found."""
        with self._lock:
            if name in self._timers:
                self._timers[name].cancel()
                del self._timers[name]
            removed = self._schedules.pop(name, None) is not None
        if removed:
            self._save()
        return removed

    def list_schedules(self) -> list[Schedule]:
        """List all schedules."""
        return list(self._schedules.values())

    def register_handler(self, command: str, handler: callable) -> None:
        """Register a handler function for a command name."""
        self._handlers[command] = handler

    def start(self) -> None:
        """Start all enabled schedules."""
        self._running = True
        for name, sched in self._schedules.items():
            if sched.enabled:
                self._schedule_next(name)

    def stop(self) -> None:
        """Stop all timers."""
        self._running = False
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()

    def _schedule_next(self, name: str) -> None:
        """Schedule the next execution of a task."""
        sched = self._schedules.get(name)
        if not sched or not sched.enabled or not self._running:
            return

        def _run():
            handler = self._handlers.get(sched.command)
            if handler:
                try:
                    handler()
                except Exception:
                    pass  # Fail silently, reschedule
            sched.last_run = time.time()
            self._save()
            if self._running:
                self._schedule_next(name)

        timer = threading.Timer(sched.interval_s, _run)
        timer.daemon = True
        timer.name = f"hal-sched-{name}"
        with self._lock:
            if name in self._timers:
                self._timers[name].cancel()
            self._timers[name] = timer
        timer.start()

    def _load(self) -> None:
        """Load schedules from disk."""
        if not os.path.exists(_SCHEDULES_PATH):
            return
        try:
            with open(_SCHEDULES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                self._schedules[item["name"]] = Schedule(
                    name=item["name"],
                    interval_s=item.get("interval_s", 300),
                    command=item.get("command", ""),
                    enabled=item.get("enabled", True),
                    last_run=item.get("last_run", 0),
                )
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    def _save(self) -> None:
        """Persist schedules to disk."""
        os.makedirs(os.path.dirname(_SCHEDULES_PATH), exist_ok=True)
        data = [
            {
                "name": s.name,
                "interval_s": s.interval_s,
                "command": s.command,
                "enabled": s.enabled,
                "last_run": s.last_run,
            }
            for s in self._schedules.values()
        ]
        with open(_SCHEDULES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
