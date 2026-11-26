"""HAL Recursive Agent Spawning — infinite depth, cross-LLM mesh.

Any agent can spawn sub-agents to any depth. Each agent can use a
different LLM provider. The coordinator-convergence duality applies:
spawning is breadth (multiple agents), each agent converges at depth.

Usage:
    from lib.hal.agent_spawn import AgentSpawner

    spawner = AgentSpawner(registry)
    result = spawner.spawn("Analyse this contract", model="groq/llama-3.3-70b", depth=0)

    # Cross-LLM mesh: orchestrator on Opus, workers on Groq/Ollama
    result = spawner.spawn_mesh(
        task="Review and improve this code",
        agents=[
            ("groq/llama-3.3-70b", "Review for bugs"),
            ("ollama/mistral:7b", "Check for security issues"),
            ("groq/openai/gpt-oss-120b", "Suggest improvements"),
        ],
    )
"""
import time
import uuid
from typing import Optional

pass  # Types used via registry.chat() return values


class AgentContext:
    """Execution context for a spawned agent."""

    def __init__(
        self,
        agent_id: str = "",
        parent_id: str = "",
        depth: int = 0,
        max_depth: int = 21,
        model: str = "",
    ):
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.parent_id = parent_id
        self.depth = depth
        self.max_depth = max_depth
        self.model = model


class AgentResult:
    """Result from a spawned agent."""

    def __init__(
        self,
        text: str = "",
        model: str = "",
        provider: str = "",
        context: Optional[AgentContext] = None,
        latency_ms: float = 0,
        cost_usd: float = 0,
    ):
        self.text = text
        self.model = model
        self.provider = provider
        self.context = context
        self.latency_ms = latency_ms
        self.cost_usd = cost_usd


class AgentSpawner:
    """Spawns agents recursively with cross-LLM support."""

    def __init__(self, registry: "ProviderRegistry"):
        self._registry = registry

    def spawn(
        self,
        task: str,
        *,
        model: str = "",
        system: str = "",
        parent_context: Optional[AgentContext] = None,
        max_depth: int = 21,
    ) -> AgentResult:
        """Spawn a single agent. Can recurse to max_depth."""
        depth = (parent_context.depth + 1) if parent_context else 0
        if depth > max_depth:
            return AgentResult(text=f"[Recursion limit: depth {depth} > max {max_depth}]")

        ctx = AgentContext(
            parent_id=parent_context.agent_id if parent_context else "",
            depth=depth,
            max_depth=max_depth,
            model=model or self._registry.routing.default_model,
        )

        target = model or self._registry.routing.default_model
        start = time.monotonic()
        response = self._registry.chat(target, task, system=system)
        ms = (time.monotonic() - start) * 1000

        from lib.hal.cost_tracker import calculate_cost_from_usage
        cost = calculate_cost_from_usage(response.model, response.usage)

        return AgentResult(
            text=response.text,
            model=response.model,
            provider=response.provider,
            context=ctx,
            latency_ms=ms,
            cost_usd=cost,
        )

    def spawn_mesh(
        self,
        task: str,
        agents: list[tuple[str, str]],
        system: str = "",
        max_depth: int = 21,
    ) -> list[AgentResult]:
        """Spawn multiple agents in parallel (breadth), each on its own LLM.

        Args:
            task: The shared task context
            agents: List of (model_address, sub_task) tuples
            system: Shared system prompt
        """
        results = []
        for model, sub_task in agents:
            prompt = f"{task}\n\nFocus: {sub_task}"
            result = self.spawn(prompt, model=model, system=system, max_depth=max_depth)
            results.append(result)
        return results
