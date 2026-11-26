"""HAL Event Protocol — typed, structured event streaming.

CMA-compatible event types PLUS HAL-exclusive events that single-vendor
platforms cannot produce: council votes, sentinel failovers, mesh results.

Every event is a frozen dataclass with monotonic sequencing per session.
Events flow through an EventEmitter to multiple sinks: SSE streams,
EventBus, SessionRecorder, disk persistence.

Usage:
    from lib.hal.event_protocol import EventType, HALEvent, EventEmitter, EventHelpers

    emitter = EventEmitter(session_id="abc123", agent_id="def456")
    emitter.add_sink(lambda e: print(e.to_dict()))
    emitter.emit(EventType.USER_MESSAGE, {"text": "Hello"})

    helpers = EventHelpers(emitter)
    helpers.agent_message("Hi there", model="groq/llama-3.3-70b")
    helpers.council_vote("groq/llama-3.3-70b", "groq", "I agree.", vote_index=0)
"""
import json
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    """Typed event categories — superset of CMA events."""
    # --- CMA-compatible events ---
    USER_MESSAGE = "user.message"
    AGENT_MESSAGE = "agent.message"
    AGENT_TOOL_USE = "agent.tool_use"
    AGENT_TOOL_RESULT = "agent.tool_result"
    SESSION_ERROR = "session.error"
    SESSION_COMPLETE = "session.complete"
    SESSION_PAUSED = "session.paused"
    SPAN_MODEL_REQUEST_START = "span.model_request_start"
    SPAN_MODEL_REQUEST_END = "span.model_request_end"

    # --- HAL-exclusive events (CMA cannot produce these) ---
    COUNCIL_VOTE = "council.vote"
    COUNCIL_SYNTHESIS = "council.synthesis"
    SENTINEL_FAILOVER = "sentinel.failover"
    AGENT_SPAWN = "agent.spawn"
    AGENT_SPAWN_RESULT = "agent.spawn_result"
    AGENT_MESH_RESULT = "agent.mesh_result"
    COST_UPDATE = "cost.update"
    BUDGET_WARNING = "budget.warning"
    HOOK_BLOCKED = "hook.blocked"
    PROVIDER_HEALTH = "provider.health"
    ENVIRONMENT_ENFORCED = "environment.enforced"


@dataclass(frozen=True)
class HALEvent:
    """A single typed event in the streaming protocol.

    Frozen for thread safety. Monotonic sequence per session ensures ordering.
    Uses tuple of (key, value) pairs for data to remain hashable.
    """
    event_type: str
    timestamp: float
    session_id: str
    agent_id: str = ""
    sequence: int = 0
    data: tuple = ()

    def to_dict(self) -> dict:
        return {
            "type": self.event_type,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "sequence": self.sequence,
            **dict(self.data),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def from_dict(data: dict) -> "HALEvent":
        reserved = {"type", "timestamp", "session_id", "agent_id", "sequence"}
        extra = tuple((k, v) for k, v in data.items() if k not in reserved)
        return HALEvent(
            event_type=data.get("type", ""),
            timestamp=data.get("timestamp", 0.0),
            session_id=data.get("session_id", ""),
            agent_id=data.get("agent_id", ""),
            sequence=data.get("sequence", 0),
            data=extra,
        )


class EventEmitter:
    """Emits typed events to multiple sinks with monotonic sequencing.

    Thread-safe. Sinks receive events synchronously in registration order.
    Failed sinks do not block other sinks.
    """

    def __init__(self, session_id: str, agent_id: str = ""):
        self._session_id = session_id
        self._agent_id = agent_id
        self._sequence = 0
        self._lock = threading.Lock()
        self._sinks: list = []
        self._history: list = []
        self._max_history = 5000

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def emit(self, event_type, data: Optional[dict] = None) -> HALEvent:
        """Emit an event to all sinks. Returns the created event."""
        with self._lock:
            self._sequence += 1
            seq = self._sequence

        etype = event_type if isinstance(event_type, str) else event_type.value
        extra = data or {}
        event = HALEvent(
            event_type=etype,
            timestamp=time.time(),
            session_id=self._session_id,
            agent_id=self._agent_id,
            sequence=seq,
            data=tuple(extra.items()),
        )

        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            sinks = list(self._sinks)

        for sink in sinks:
            try:
                sink(event)
            except Exception:
                pass

        return event

    def add_sink(self, sink) -> None:
        """Register an event sink."""
        with self._lock:
            self._sinks.append(sink)

    def remove_sink(self, sink) -> bool:
        """Remove a sink. Returns True if found."""
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)
                return True
            return False

    def replay(self, since_sequence: int = 0) -> list:
        """Get event history, optionally from a sequence number."""
        with self._lock:
            if since_sequence <= 0:
                return list(self._history)
            return [e for e in self._history if e.sequence > since_sequence]

    def count(self) -> int:
        """Number of events emitted."""
        with self._lock:
            return self._sequence


class SessionEvents:
    """Session lifecycle event helpers: messages, tools, errors, completion."""

    def __init__(self, emitter: EventEmitter):
        self._e = emitter

    def user_message(self, text: str) -> HALEvent:
        return self._e.emit(EventType.USER_MESSAGE, {"text": text})

    def agent_message(self, text: str, model: str = "", provider: str = "") -> HALEvent:
        return self._e.emit(EventType.AGENT_MESSAGE, {
            "text": text, "model": model, "provider": provider,
        })

    def tool_use(self, tool_name: str, args: Optional[dict] = None) -> HALEvent:
        return self._e.emit(EventType.AGENT_TOOL_USE, {
            "name": tool_name, "args": args or {},
        })

    def tool_result(self, tool_name: str, result: str, success: bool = True) -> HALEvent:
        return self._e.emit(EventType.AGENT_TOOL_RESULT, {
            "name": tool_name, "result": result, "success": success,
        })

    def error(self, error_type: str, message: str) -> HALEvent:
        return self._e.emit(EventType.SESSION_ERROR, {
            "error": {"type": error_type, "message": message},
        })

    def complete(self, summary: str = "") -> HALEvent:
        return self._e.emit(EventType.SESSION_COMPLETE, {"summary": summary})


class ObservabilityEvents:
    """Observability event helpers: model spans, cost, budget."""

    def __init__(self, emitter: EventEmitter):
        self._e = emitter

    def model_request_end(
        self, model: str, provider: str,
        input_tokens: int = 0, output_tokens: int = 0,
        latency_ms: float = 0, cost_usd: float = 0,
    ) -> HALEvent:
        return self._e.emit(EventType.SPAN_MODEL_REQUEST_END, {
            "model": model, "provider": provider,
            "model_usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "latency_ms": latency_ms, "cost_usd": cost_usd,
        })

    def cost_update(self, total_usd: float, budget_usd: float = 0) -> HALEvent:
        return self._e.emit(EventType.COST_UPDATE, {
            "total_usd": total_usd, "session_budget_usd": budget_usd,
        })

    def budget_warning(self, used_usd: float, budget_usd: float) -> HALEvent:
        pct = (used_usd / budget_usd * 100) if budget_usd > 0 else 0
        return self._e.emit(EventType.BUDGET_WARNING, {
            "used_usd": used_usd, "budget_usd": budget_usd, "utilisation_pct": pct,
        })


class MultiProviderEvents:
    """HAL-exclusive event helpers: council, sentinel, hooks.

    These events are structurally impossible in single-vendor platforms like CMA.
    """

    def __init__(self, emitter: EventEmitter):
        self._e = emitter

    def council_vote(self, model: str, provider: str, text: str, vote_index: int = 0) -> HALEvent:
        return self._e.emit(EventType.COUNCIL_VOTE, {
            "model": model, "provider": provider, "text": text, "vote_index": vote_index,
        })

    def council_synthesis(self, text: str, agreement_score: float, models: Optional[list] = None) -> HALEvent:
        return self._e.emit(EventType.COUNCIL_SYNTHESIS, {
            "text": text, "agreement_score": agreement_score, "models": models or [],
        })

    def sentinel_failover(self, from_provider: str, to_provider: str, reason: str) -> HALEvent:
        return self._e.emit(EventType.SENTINEL_FAILOVER, {
            "from_provider": from_provider, "to_provider": to_provider, "reason": reason,
        })

    def hook_blocked(self, hook_name: str, tool_name: str, reason: str) -> HALEvent:
        return self._e.emit(EventType.HOOK_BLOCKED, {
            "hook": hook_name, "tool": tool_name, "reason": reason,
        })


class EventHelpers:
    """Composite facade providing all event helpers via sub-interfaces.

    Usage:
        h = EventHelpers(emitter)
        h.session.user_message("Hello")
        h.observe.model_request_end("groq/llama", "groq", ...)
        h.multi.council_vote("groq/llama", "groq", "I agree.")
    """

    def __init__(self, emitter: EventEmitter):
        self.session = SessionEvents(emitter)
        self.observe = ObservabilityEvents(emitter)
        self.multi = MultiProviderEvents(emitter)
