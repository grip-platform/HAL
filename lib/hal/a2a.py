"""HAL A2A Protocol — Agent-to-Agent communication bus.

In-process message bus for inter-agent communication. Thread-safe.
Agents at any depth can send messages to any other agent.

Trust boundary
--------------
The bus lives inside a single Python process — all callers share memory
and the GIL is the only trust boundary. There is no cryptographic
authentication because there is no meaningful peer to authenticate
against when every sender has direct access to ``self._queues``. What
this module protects against is **resource exhaustion**: a runaway
sender (stuck agent, infinite-loop broadcast, flooding attack from a
compromised subagent) cannot grow a queue without bound. Each per-agent
queue and the global broadcast queue are bounded at
``max_queue_depth`` (default 10000), and when the cap is exceeded the
oldest message is dropped to make room. The cap is deliberately high
— a normal interactive workflow stays well under 100 messages per
agent — so the drop policy should only fire under adversarial or
buggy conditions.

Usage:
    from lib.hal.a2a import A2ABus, A2AMessage

    bus = A2ABus()
    bus.send(A2AMessage(from_agent="agent-1", to_agent="agent-2", content="result"))
    messages = bus.receive("agent-2")
"""
import collections
import threading
import time
from dataclasses import dataclass

DEFAULT_QUEUE_DEPTH = 10_000


@dataclass(frozen=True)
class A2AMessage:
    """A message between agents."""
    from_agent: str
    to_agent: str  # "broadcast" for all agents
    content: str
    depth: int = 0
    timestamp: float = 0.0
    metadata: tuple = ()  # Immutable pairs: (("key", "val"), ...)

    def __post_init__(self):
        if self.timestamp == 0.0:
            object.__setattr__(self, "timestamp", time.time())


class A2ABus:
    """Thread-safe message bus with bounded per-queue depth.

    Per-agent queues and the global broadcast queue use ``collections.deque``
    with a fixed ``maxlen`` so an ``append`` past the cap atomically drops
    the oldest message. This is the eviction policy: adversarial flooding
    pushes older legitimate messages out of the queue, but it can never
    grow memory past ``max_queue_depth`` per queue.
    """

    def __init__(self, max_queue_depth: int = DEFAULT_QUEUE_DEPTH):
        if max_queue_depth <= 0:
            raise ValueError(f"max_queue_depth must be > 0, got {max_queue_depth}")
        self._max_queue_depth = max_queue_depth
        self._queues: dict[str, collections.deque[A2AMessage]] = {}
        self._broadcast: collections.deque[A2AMessage] = collections.deque(
            maxlen=max_queue_depth
        )
        self._lock = threading.Lock()

    @property
    def max_queue_depth(self) -> int:
        return self._max_queue_depth

    def send(self, msg: A2AMessage) -> None:
        """Send a message to a specific agent or broadcast.

        If the target queue is already at ``max_queue_depth``, the oldest
        message is atomically dropped to make room for the new one. This
        is a DoS mitigation, not a reliability guarantee — callers that
        care about delivery must drain their queue faster than senders
        produce, or raise the cap at construction time.
        """
        with self._lock:
            if msg.to_agent == "broadcast":
                self._broadcast.append(msg)
                return
            queue = self._queues.get(msg.to_agent)
            if queue is None:
                queue = collections.deque(maxlen=self._max_queue_depth)
                self._queues[msg.to_agent] = queue
            queue.append(msg)

    def receive(self, agent_id: str) -> list[A2AMessage]:
        """Receive all pending messages for an agent. Clears the queue."""
        with self._lock:
            queue = self._queues.pop(agent_id, None)
            direct = list(queue) if queue is not None else []
            broadcasts = [m for m in self._broadcast if m.from_agent != agent_id]
            return direct + broadcasts

    def broadcast(self, from_agent: str, content: str, depth: int = 0) -> None:
        """Convenience: broadcast a message to all agents."""
        self.send(A2AMessage(
            from_agent=from_agent, to_agent="broadcast",
            content=content, depth=depth,
        ))

    def pending_count(self, agent_id: str) -> int:
        """Count pending messages for an agent (without consuming them)."""
        with self._lock:
            queue = self._queues.get(agent_id)
            direct = len(queue) if queue is not None else 0
            broadcasts = sum(
                1 for m in self._broadcast if m.from_agent != agent_id
            )
            return direct + broadcasts

    def clear(self) -> None:
        """Clear all queues and broadcasts."""
        with self._lock:
            self._queues.clear()
            self._broadcast.clear()
