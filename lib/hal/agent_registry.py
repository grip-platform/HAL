"""HAL Agent Definition Registry — versioned, provider-agnostic agent configs.

CMA's /v1/agents equivalent, but provider-agnostic. Every agent can use any
provider, any routing strategy (fixed/fallback/quality_cost/council), and any
combination of tools/skills/MCP servers.

Unique HAL capabilities that CMA cannot offer:
- fallback_chain: automatic provider failover per agent
- council_models: multi-LLM deliberation per agent
- Limit-Sentinel failover: rate-limit auto-avoidance wrapping every dispatch
  (always on -- lib/hal/sentinel.py; not a selectable routing_strategy)
- Any model address: "groq/llama-3.3-70b", "ollama/mistral:7b", etc.

Usage:
    from lib.hal.agent_registry import AgentDefinitionRegistry, get_agent_registry

    registry = get_agent_registry()
    defn = registry.create(name="legal-reviewer", model="groq/llama-3.3-70b",
                           system_prompt="You are a legal document reviewer.")
    defn = registry.update(defn.agent_id, {"budget_usd": 1.0})  # Creates v2
    registry.rollback(defn.agent_id, 1)  # Restore v1
"""
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

from lib.hal.home import HAL_HOME as _HAL_HOME

_AGENTS_STATE_DIR = os.path.join(_HAL_HOME, "state", "agents")
# Singleton instance.
_instance: Optional["AgentDefinitionRegistry"] = None
_instance_lock = threading.Lock()


@dataclass
class AgentDefinition:
    """A versioned agent configuration.

    Immutable once created — updates produce new versions.
    """
    agent_id: str
    name: str
    version: int = 1
    model: str = ""
    system_prompt: str = ""
    tools: list = field(default_factory=list)
    skills: list = field(default_factory=list)
    mcp_servers: list = field(default_factory=list)
    max_turns: int = 100
    budget_usd: float = 5.0
    routing_strategy: str = "fallback"
    fallback_chain: list = field(default_factory=list)
    council_models: list = field(default_factory=list)
    environment_id: str = ""
    metadata: dict = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "AgentDefinition":
        from lib.hal.io_utils import dataclass_from_dict
        return dataclass_from_dict(AgentDefinition, data)

    def summary(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "model": self.model,
            "routing_strategy": self.routing_strategy,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class AgentVersionStore:
    """Reads and writes versioned agent definitions to disk.

    Separated from the registry (ISP) — version history operations
    are independent of CRUD lifecycle.

    State directory layout:
        state/agents/{agent_id}/v1.json
        state/agents/{agent_id}/v2.json
        state/agents/{agent_id}/latest.json  (copy of highest version)
    """

    def persist(self, defn: AgentDefinition) -> None:
        """Save a version to disk."""
        agent_dir = os.path.join(_AGENTS_STATE_DIR, defn.agent_id)
        os.makedirs(agent_dir, exist_ok=True)
        data = json.dumps(defn.to_dict(), indent=2)
        _atomic_write(os.path.join(agent_dir, f"v{defn.version}.json"), data)
        _atomic_write(os.path.join(agent_dir, "latest.json"), data)

    def load_latest(self, agent_id: str) -> Optional[AgentDefinition]:
        """Load the latest version from disk."""
        return self._read(os.path.join(_AGENTS_STATE_DIR, agent_id, "latest.json"))

    def load_version(self, agent_id: str, version: int) -> Optional[AgentDefinition]:
        """Load a specific version from disk."""
        return self._read(os.path.join(_AGENTS_STATE_DIR, agent_id, f"v{version}.json"))

    def list_versions(self, agent_id: str) -> list[int]:
        """List all version numbers for an agent."""
        agent_dir = os.path.join(_AGENTS_STATE_DIR, agent_id)
        if not os.path.isdir(agent_dir):
            return []
        versions = []
        for f in os.listdir(agent_dir):
            if f.startswith("v") and f.endswith(".json") and f != "latest.json":
                try:
                    versions.append(int(f[1:-5]))
                except ValueError:
                    pass
        return sorted(versions)

    def remove_all(self, agent_id: str) -> None:
        """Delete all version files for an agent."""
        agent_dir = os.path.join(_AGENTS_STATE_DIR, agent_id)
        if os.path.isdir(agent_dir):
            for f in os.listdir(agent_dir):
                os.remove(os.path.join(agent_dir, f))
            os.rmdir(agent_dir)

    def load_all_latest(self) -> dict[str, AgentDefinition]:
        """Load all latest agent definitions from disk."""
        agents: dict[str, AgentDefinition] = {}
        if not os.path.isdir(_AGENTS_STATE_DIR):
            return agents
        for agent_id in os.listdir(_AGENTS_STATE_DIR):
            defn = self.load_latest(agent_id)
            if defn:
                agents[agent_id] = defn
        return agents

    @staticmethod
    def _read(path: str) -> Optional[AgentDefinition]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return AgentDefinition.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


class AgentDefinitionRegistry:
    """Thread-safe agent definition CRUD with versioned persistence.

    Version history operations are delegated to AgentVersionStore (ISP).
    """

    def __init__(self, store: Optional[AgentVersionStore] = None):
        self._store = store or AgentVersionStore()
        self._agents: dict[str, AgentDefinition] = {}
        self._lock = threading.Lock()
        os.makedirs(_AGENTS_STATE_DIR, exist_ok=True)
        self._agents = self._store.load_all_latest()

    @property
    def store(self) -> AgentVersionStore:
        """Access the version store for history operations."""
        return self._store

    def create(
        self,
        *,
        name: str,
        model: str = "",
        system_prompt: str = "",
        tools: Optional[list] = None,
        skills: Optional[list] = None,
        mcp_servers: Optional[list] = None,
        max_turns: int = 100,
        budget_usd: float = 5.0,
        routing_strategy: str = "fallback",
        fallback_chain: Optional[list] = None,
        council_models: Optional[list] = None,
        environment_id: str = "",
        metadata: Optional[dict] = None,
    ) -> AgentDefinition:
        """Create a new agent definition. Returns the definition."""
        now = time.time()
        agent_id = str(uuid.uuid4())[:8]
        defn = AgentDefinition(
            agent_id=agent_id,
            name=name,
            version=1,
            model=model,
            system_prompt=system_prompt,
            tools=tools or [],
            skills=skills or [],
            mcp_servers=mcp_servers or [],
            max_turns=max_turns,
            budget_usd=budget_usd,
            routing_strategy=routing_strategy,
            fallback_chain=fallback_chain or [],
            council_models=council_models or [],
            environment_id=environment_id,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._agents[agent_id] = defn
            self._store.persist(defn)
        return defn

    def get(self, agent_id: str, version: int = 0) -> Optional[AgentDefinition]:
        """Get an agent definition. version=0 means latest."""
        if version > 0:
            return self._store.load_version(agent_id, version)
        with self._lock:
            return self._agents.get(agent_id)

    def update(self, agent_id: str, changes: dict) -> Optional[AgentDefinition]:
        """Update an agent definition — creates a new version. Returns new version."""
        with self._lock:
            current = self._agents.get(agent_id)
            if current is None:
                return None
            data = current.to_dict()
            data.update(changes)
            data["version"] = current.version + 1
            data["updated_at"] = time.time()
            data["agent_id"] = agent_id
            new_defn = AgentDefinition.from_dict(data)
            self._agents[agent_id] = new_defn
            self._store.persist(new_defn)
            return new_defn

    def delete(self, agent_id: str) -> bool:
        """Delete an agent definition and all versions. Returns True if found."""
        with self._lock:
            if agent_id not in self._agents:
                return False
            del self._agents[agent_id]
            self._store.remove_all(agent_id)
            return True

    def list_agents(self) -> list[AgentDefinition]:
        """List all agent definitions (latest versions), sorted by name."""
        with self._lock:
            agents = list(self._agents.values())
        agents.sort(key=lambda a: a.name)
        return agents

    def rollback(self, agent_id: str, version: int) -> Optional[AgentDefinition]:
        """Rollback to a specific version. Creates a new version with old content."""
        old = self._store.load_version(agent_id, version)
        if old is None:
            return None
        with self._lock:
            current = self._agents.get(agent_id)
            new_version = (current.version + 1) if current else version + 1
            data = old.to_dict()
            data["version"] = new_version
            data["updated_at"] = time.time()
            rolled = AgentDefinition.from_dict(data)
            self._agents[agent_id] = rolled
            self._store.persist(rolled)
            return rolled


def import_agent_from_md(
    registry: AgentDefinitionRegistry, name: str, agents_dir: str = "",
) -> Optional[AgentDefinition]:
    """Import an existing .md agent definition into the versioned registry.

    Standalone bridge function (not a registry method) — keeps the registry
    focused on CRUD + versioning while this handles format translation.
    """
    from lib.hal.agent_runner import AgentRunner
    runner = AgentRunner(agents_dir or None)
    spec = runner.parse_agent(name)
    if spec is None:
        return None
    for defn in registry.list_agents():
        if defn.name == name:
            return defn
    return registry.create(
        name=spec.name,
        model=spec.model,
        system_prompt=spec.prompt,
        tools=spec.tools,
        skills=spec.skills,
        metadata={"imported_from": "md", "description": spec.description},
    )


from lib.hal.io_utils import atomic_write as _atomic_write


def get_agent_registry() -> AgentDefinitionRegistry:
    """Get or create the singleton AgentDefinitionRegistry."""
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None:
            _instance = AgentDefinitionRegistry()
        return _instance
