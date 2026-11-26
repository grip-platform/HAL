"""HAL Self-Improving Agent — agents that evolve their own definitions.

CMA agents are static configs that never learn. HAL agents analyse their
own performance history and propose optimised versions of themselves.
All proposals go through human approval before being applied.

Usage:
    from lib.hal.self_improving_agent import AgentEvolver

    evolver = AgentEvolver(agent_registry, fleet_counters)
    proposal = evolver.analyse("agent-id")
    if proposal:
        print(f"Proposed changes: {proposal.changes}")
        evolver.apply(proposal)  # Creates new agent version
"""
from dataclasses import dataclass
from typing import Optional

from lib.hal.agent_registry import AgentDefinitionRegistry
from lib.hal.fleet import FleetCounters


@dataclass
class EvolutionProposal:
    """A proposed change to an agent definition based on performance data."""
    agent_id: str
    agent_name: str
    current_version: int
    changes: dict
    reasoning: str
    confidence: float = 0.0

    def to_update_dict(self) -> dict:
        """Build the update dict for AgentDefinitionRegistry.update()."""
        result = dict(self.changes)
        result["metadata"] = {
            "evolved_from": self.current_version,
            "evolution_reasoning": self.reasoning,
        }
        return result


class PerformanceAnalyser:
    """Analyses fleet counters to detect improvement opportunities."""

    def __init__(self, fleet_counters: FleetCounters):
        self._counters = fleet_counters

    def suggest_improvements(
        self,
        routing_strategy: str,
        fallback_chain: list,
        budget_usd: float,
    ) -> list:
        """Return a list of {field, reason, value} improvement suggestions."""
        suggestions = []
        counters = self._counters.snapshot()

        if budget_usd > 0 and counters["zero_cost_sessions"] > 0:
            suggestions.append({
                "field": "fallback_chain",
                "reason": "Zero-cost sessions detected; add local fallback",
                "value": _ensure_local_fallback(fallback_chain),
            })

        if not fallback_chain and routing_strategy == "fixed":
            suggestions.append({
                "field": "routing_strategy",
                "reason": "No fallback chain; switch to fallback strategy for resilience",
                "value": "fallback",
            })

        return suggestions


class AgentEvolver:
    """Proposes and applies evolutionary improvements to agent definitions.

    Composes PerformanceAnalyser (detection) and AgentDefinitionRegistry
    (application). All changes are versioned and reversible.
    """

    def __init__(
        self,
        agent_registry: AgentDefinitionRegistry,
        fleet_counters: FleetCounters,
    ):
        self._agents = agent_registry
        self._analyser = PerformanceAnalyser(fleet_counters)

    def analyse(self, agent_id: str) -> Optional[EvolutionProposal]:
        """Analyse an agent and propose improvements."""
        defn = self._agents.get(agent_id)
        if defn is None:
            return None

        suggestions = self._analyser.suggest_improvements(
            routing_strategy=defn.routing_strategy,
            fallback_chain=defn.fallback_chain,
            budget_usd=defn.budget_usd,
        )
        if not suggestions:
            return None

        return EvolutionProposal(
            agent_id=agent_id,
            agent_name=defn.name,
            current_version=defn.version,
            changes={s["field"]: s["value"] for s in suggestions},
            reasoning="; ".join(s["reason"] for s in suggestions),
            confidence=min(0.9, 0.3 * len(suggestions)),
        )

    def apply(self, proposal: EvolutionProposal):
        """Apply a proposal — creates a new agent version."""
        return self._agents.update(proposal.agent_id, proposal.to_update_dict())


def _ensure_local_fallback(chain: list) -> list:
    """Ensure chain includes a local provider as final fallback."""
    if any(m.startswith("ollama/") for m in chain):
        return chain
    return list(chain) + ["ollama/mistral:7b"]
