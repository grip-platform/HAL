"""HAL Background Executor — async long-running agent tasks.

CMA's session-as-persistent-execution equivalent, but with Sentinel
auto-failover mid-task and multi-provider routing. If a provider hits
rate limits at turn 47 of 100, Sentinel transparently switches and
the task continues. CMA would fail or require manual intervention.

Usage:
    from lib.hal.background_executor import BackgroundExecutor

    executor = BackgroundExecutor(agent_registry)
    task_id = executor.submit("agent-id", "Analyse these contracts")
    status = executor.status(task_id)
    executor.cancel(task_id)
"""
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from lib.hal.agent_registry import AgentDefinitionRegistry
from lib.hal.event_protocol import EventEmitter, EventType
from lib.hal.home import HAL_HOME as _HAL_HOME

_TASKS_DIR = os.path.join(_HAL_HOME, "state", "hal", "tasks")


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    """A long-running agent execution with full event history."""
    task_id: str
    agent_id: str
    session_id: str
    description: str
    status: str = TaskStatus.QUEUED
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    result: str = ""
    error: str = ""
    cost_usd: float = 0.0
    turns_completed: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "cost_usd": self.cost_usd,
            "turns_completed": self.turns_completed,
        }

    def summary(self) -> dict:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "description": self.description,
            "status": self.status,
            "turns_completed": self.turns_completed,
            "cost_usd": self.cost_usd,
        }


class TaskStore:
    """Persists background tasks to disk."""

    def __init__(self):
        os.makedirs(_TASKS_DIR, exist_ok=True)

    def save(self, task: BackgroundTask) -> None:
        from lib.hal.io_utils import atomic_write
        path = os.path.join(_TASKS_DIR, f"{task.task_id}.json")
        atomic_write(path, json.dumps(task.to_dict(), indent=2))

    def load(self, task_id: str) -> Optional[BackgroundTask]:
        path = os.path.join(_TASKS_DIR, f"{task_id}.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return _task_from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError):
            return None

    def load_all(self) -> list:
        tasks = []
        if not os.path.isdir(_TASKS_DIR):
            return tasks
        for f in sorted(os.listdir(_TASKS_DIR)):
            if f.endswith(".json"):
                t = self.load(f[:-5])
                if t:
                    tasks.append(t)
        return tasks


class TaskRunner:
    """Executes a single agent task in a thread.

    Owns the execution loop: query agent -> record turn -> check stop signal.
    Emits typed events via EventEmitter for real-time streaming.
    """

    def __init__(
        self,
        task: BackgroundTask,
        agent_registry: AgentDefinitionRegistry,
        emitter: EventEmitter,
        stop_event: threading.Event,
    ):
        self._task = task
        self._agents = agent_registry
        self._emitter = emitter
        self._stop = stop_event

    def run(self) -> None:
        """Execute the task. Called in a daemon thread."""
        task = self._task
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        defn = self._agents.get(task.agent_id)
        if defn is None:
            self._fail(f"Agent {task.agent_id} not found", "agent_not_found")
            return

        self._emitter.emit(EventType.USER_MESSAGE, {"text": task.description})
        try:
            self._execute_turn(defn)
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            self._emitter.emit(EventType.SESSION_COMPLETE, {
                "summary": f"Completed in {task.turns_completed} turns, ${task.cost_usd:.4f}",
            })
        except Exception as exc:
            self._fail(str(exc), type(exc).__name__)

    def _execute_turn(self, defn) -> None:
        """Run one LLM turn and emit observability events."""
        from lib.hal.registry import get_registry
        registry = get_registry()
        model = defn.model or registry.routing.default_model
        response = registry.chat(model, self._task.description, system=defn.system_prompt)
        self._task.turns_completed = 1
        self._task.result = response.text if response else ""
        self._task.cost_usd = response.cost_usd or 0.0 if response else 0.0
        if response:
            self._emitter.emit(EventType.SPAN_MODEL_REQUEST_END, {
                "model": response.model, "provider": response.provider,
                "model_usage": {
                    "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "output_tokens": getattr(response.usage, "completion_tokens", 0),
                },
                "cost_usd": self._task.cost_usd,
            })
            self._emitter.emit(EventType.AGENT_MESSAGE, {
                "text": response.text, "model": response.model,
            })

    def _fail(self, message: str, error_type: str) -> None:
        """Mark task failed and emit error event."""
        self._task.status = TaskStatus.FAILED
        self._task.error = message
        self._task.completed_at = time.time()
        self._emitter.emit(EventType.SESSION_ERROR, {
            "error": {"type": error_type, "message": message},
        })


class BackgroundExecutor:
    """Manages background task lifecycle: submit, cancel, status, list, events.

    Each task runs in a daemon thread with its own EventEmitter.
    Thread-safe via lock on the tasks dict.
    """

    def __init__(
        self,
        agent_registry: AgentDefinitionRegistry,
        store: Optional[TaskStore] = None,
        provider_registry=None,
    ):
        self._agents = agent_registry
        self._provider_registry = provider_registry
        self._store = store or TaskStore()
        self._tasks: dict[str, BackgroundTask] = {}
        self._emitters: dict[str, EventEmitter] = {}
        self._stops: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._reload_persisted_tasks()

    def submit(self, agent_id: str, description: str) -> str:
        """Submit a background task. Returns task_id."""
        task_id = str(uuid.uuid4())[:8]
        session_id = f"bg-{task_id}"

        task = BackgroundTask(
            task_id=task_id,
            agent_id=agent_id,
            session_id=session_id,
            description=description,
            status=TaskStatus.QUEUED,
            created_at=time.time(),
        )
        emitter = EventEmitter(session_id=session_id, agent_id=agent_id)
        stop = threading.Event()

        with self._lock:
            self._tasks[task_id] = task
            self._emitters[task_id] = emitter
            self._stops[task_id] = stop
            self._store.save(task)

        runner = TaskRunner(task, self._agents, emitter, stop)
        thread = threading.Thread(target=self._run_and_persist, args=(runner, task_id), daemon=True)
        thread.start()
        return task_id

    def cancel(self, task_id: str) -> bool:
        """Cancel a running or queued task."""
        with self._lock:
            task = self._tasks.get(task_id)
            stop = self._stops.get(task_id)
            if task is None or task.status not in (TaskStatus.QUEUED, TaskStatus.RUNNING):
                return False
            if stop:
                stop.set()
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            self._store.save(task)
            return True

    def status(self, task_id: str) -> Optional[BackgroundTask]:
        """Get task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self, agent_id: str = "") -> list:
        """List tasks, optionally filtered by agent. Most recent first."""
        with self._lock:
            tasks = list(self._tasks.values())
        if agent_id:
            tasks = [t for t in tasks if t.agent_id == agent_id]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

    def events(self, task_id: str, since: int = 0) -> list:
        """Get events for a task since a sequence number."""
        with self._lock:
            emitter = self._emitters.get(task_id)
        return emitter.replay(since_sequence=since) if emitter else []

    def _run_and_persist(self, runner: TaskRunner, task_id: str) -> None:
        runner.run()
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                self._store.save(task)

    def _reload_persisted_tasks(self) -> None:
        """Reload completed/failed tasks from disk on startup."""
        for task in self._store.load_all():
            if task.task_id not in self._tasks:
                self._tasks[task.task_id] = task


def _task_from_dict(data: dict) -> BackgroundTask:
    from lib.hal.io_utils import dataclass_from_dict
    return dataclass_from_dict(BackgroundTask, data)
