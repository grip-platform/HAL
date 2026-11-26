"""HAL Fleet Manager — fleet-level aggregation and statistics.

CMA's dashboard equivalent, but with multi-provider cost analytics that
single-vendor platforms cannot offer: zero-cost tracking, provider diversity,
Sentinel intervention counting, council deliberation stats.

Usage:
    from lib.hal.fleet import FleetManager

    fleet = FleetManager(agent_registry, session_registry, cost_tracker)
    snap = fleet.snapshot()
    print(f"Agents: {snap.total_agents}, Cost: ${snap.total_cost_usd:.4f}")
    print(f"Zero-cost ratio: {snap.zero_cost_ratio:.0%}")
"""
import threading
import time
from dataclasses import dataclass
from typing import Optional

from lib.hal.agent_registry import AgentDefinitionRegistry
from lib.hal.cost_tracker import CostTracker
from lib.hal.session_registry import SessionRegistry


@dataclass(frozen=True)
class AgentStats:
    """Per-agent operational statistics."""
    agent_id: str
    agent_name: str
    total_sessions: int = 0
    active_sessions: int = 0
    completed_sessions: int = 0
    failed_sessions: int = 0
    total_cost_usd: float = 0.0
    models_used: tuple = ()
    providers_used: tuple = ()


@dataclass(frozen=True)
class FleetSnapshot:
    """Point-in-time fleet overview with multi-provider analytics."""
    timestamp: float
    total_agents: int
    total_sessions: int
    active_sessions: int
    completed_tasks: int
    total_cost_usd: float
    cost_by_provider: tuple = ()
    cost_by_model: tuple = ()
    agent_stats: tuple = ()
    zero_cost_ratio: float = 0.0
    provider_diversity: int = 0
    council_usage: int = 0
    sentinel_interventions: int = 0
    # Fleet Gateway (#116) — per-transport inbound message counts.
    # Tuple of (transport, count) pairs, sorted by transport name for
    # deterministic snapshot hashing.
    gateway_messages_by_transport: tuple = ()

    @property
    def total_gateway_messages(self) -> int:
        """Total inbound messages across all Fleet Gateway transports."""
        return sum(count for _, count in self.gateway_messages_by_transport)


class FleetAggregator:
    """Computes fleet-wide statistics from registries.

    Stateless — queries registries on demand. Thread-safe via
    source registry locks.
    """

    def __init__(
        self,
        agent_registry: AgentDefinitionRegistry,
        session_registry: SessionRegistry,
        cost_tracker: CostTracker,
    ):
        self._agents = agent_registry
        self._sessions = session_registry
        self._costs = cost_tracker

    def agent_stats(self, agent_id: str) -> Optional[AgentStats]:
        """Compute statistics for a single agent."""
        defn = self._agents.get(agent_id)
        if defn is None:
            return None
        sessions = self._sessions.list_sessions()
        agent_sessions = [s for s in sessions if getattr(s, "agent_id", "") == agent_id]
        return AgentStats(
            agent_id=agent_id,
            agent_name=defn.name,
            total_sessions=len(agent_sessions),
        )

    def cost_report(self, period_hours: int = 24) -> dict:
        """Cost breakdown by provider and model."""
        return {
            "total_usd": self._costs.total_usd,
            "by_provider": dict(self._costs.per_provider()),
            "by_model": dict(self._costs.per_model()),
            "period_hours": period_hours,
        }


class FleetCounters:
    """Tracks HAL-exclusive fleet metrics: sentinel, council, zero-cost.

    These counters have no CMA equivalent — they measure multi-provider
    behaviour that single-vendor platforms cannot exhibit.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._sentinel_interventions = 0
        self._council_deliberations = 0
        self._zero_cost_sessions = 0
        self._total_tasks_completed = 0
        self._gateway_messages: dict[str, int] = {}

    def record_sentinel(self) -> None:
        with self._lock:
            self._sentinel_interventions += 1

    def record_council(self) -> None:
        with self._lock:
            self._council_deliberations += 1

    def record_zero_cost_session(self) -> None:
        with self._lock:
            self._zero_cost_sessions += 1

    def record_task_complete(self) -> None:
        with self._lock:
            self._total_tasks_completed += 1

    def record_gateway_message(self, transport: str) -> None:
        """Increment the inbound-message counter for a Fleet Gateway transport.

        Called from the ChannelRouter's deliver() path (or equivalent). Counts
        per transport so the snapshot can attribute multi-platform traffic
        without collapsing everything into one number.
        """
        if not transport:
            return
        with self._lock:
            self._gateway_messages[transport] = self._gateway_messages.get(transport, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "sentinel_interventions": self._sentinel_interventions,
                "council_deliberations": self._council_deliberations,
                "zero_cost_sessions": self._zero_cost_sessions,
                "total_tasks_completed": self._total_tasks_completed,
                "gateway_messages_by_transport": tuple(sorted(self._gateway_messages.items())),
            }


class FleetManager:
    """Fleet-level management combining aggregation and HAL-exclusive counters.

    Composes FleetAggregator (stateless queries) and FleetCounters
    (stateful HAL-exclusive metrics).
    """

    def __init__(
        self,
        agent_registry: AgentDefinitionRegistry,
        session_registry: SessionRegistry,
        cost_tracker: CostTracker,
    ):
        self._aggregator = FleetAggregator(agent_registry, session_registry, cost_tracker)
        self._counters = FleetCounters()
        self._agents = agent_registry
        self._sessions = session_registry
        self._costs = cost_tracker

    @property
    def counters(self) -> FleetCounters:
        return self._counters

    @property
    def aggregator(self) -> FleetAggregator:
        return self._aggregator

    def snapshot(self) -> FleetSnapshot:
        """Build a complete fleet snapshot."""
        agents = self._agents.list_agents()
        sessions = self._sessions.list_sessions()
        cost_data = self._costs.per_provider()
        model_data = self._costs.per_model()
        counters = self._counters.snapshot()

        providers_seen = {p for p, _ in cost_data} if cost_data else set()
        total_sessions = len(sessions)
        zero_ratio = (
            counters["zero_cost_sessions"] / total_sessions
            if total_sessions > 0 else 0.0
        )

        return FleetSnapshot(
            timestamp=time.time(),
            total_agents=len(agents),
            total_sessions=total_sessions,
            active_sessions=sum(1 for s in sessions if getattr(s, "status", "active") == "active"),
            completed_tasks=counters["total_tasks_completed"],
            total_cost_usd=self._costs.total_usd,
            cost_by_provider=tuple(cost_data) if cost_data else (),
            cost_by_model=tuple(model_data) if model_data else (),
            zero_cost_ratio=zero_ratio,
            provider_diversity=len(providers_seen),
            council_usage=counters["council_deliberations"],
            sentinel_interventions=counters["sentinel_interventions"],
            gateway_messages_by_transport=counters["gateway_messages_by_transport"],
        )
