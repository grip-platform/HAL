"""HAL Environment Abstraction — lightweight execution profiles.

CMA uses heavyweight Docker containers. HAL uses config-driven profiles
enforced through the existing hook dispatch system. No Docker required.

Unique HAL capabilities:
- require_local_only: true air-gap mode (no data leaves machine)
- allowed_providers: restrict to EU-hosted providers for GDPR
- Hook-enforced: dynamic PreToolUse hooks check environment constraints

Usage:
    from lib.hal.environment import EnvironmentRegistry, Environment

    registry = EnvironmentRegistry()
    env = registry.create(name="air-gapped", require_local_only=True)
    env = registry.create(name="eu-only", allowed_providers=["groq", "ollama"])
"""
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from lib.hal.home import HAL_HOME as _HAL_HOME

_ENVS_DIR = os.path.join(_HAL_HOME, "state", "environments")


@dataclass
class Environment:
    """An execution environment profile — lightweight, config-driven."""
    env_id: str
    name: str
    description: str = ""
    allowed_tools: list = None
    blocked_tools: list = None
    network_policy: str = "unrestricted"
    allowed_domains: list = None
    max_turns: int = 100
    budget_usd: float = 5.0
    timeout_s: int = 3600
    work_dir: str = ""
    persistent_fs: bool = False
    allowed_providers: list = None
    require_local_only: bool = False
    created_at: float = 0.0

    def __post_init__(self):
        self.allowed_tools = self.allowed_tools or []
        self.blocked_tools = self.blocked_tools or []
        self.allowed_domains = self.allowed_domains or []
        self.allowed_providers = self.allowed_providers or []

    def to_dict(self) -> dict:
        return {
            "env_id": self.env_id,
            "name": self.name,
            "description": self.description,
            "allowed_tools": self.allowed_tools,
            "blocked_tools": self.blocked_tools,
            "network_policy": self.network_policy,
            "allowed_domains": self.allowed_domains,
            "max_turns": self.max_turns,
            "budget_usd": self.budget_usd,
            "timeout_s": self.timeout_s,
            "work_dir": self.work_dir,
            "persistent_fs": self.persistent_fs,
            "allowed_providers": self.allowed_providers,
            "require_local_only": self.require_local_only,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: dict) -> "Environment":
        from lib.hal.io_utils import dataclass_from_dict
        return dataclass_from_dict(Environment, data)


class EnvironmentStore:
    """Persists environment definitions to disk."""

    def __init__(self):
        os.makedirs(_ENVS_DIR, exist_ok=True)

    def save(self, env: Environment) -> None:
        from lib.hal.io_utils import atomic_write
        path = os.path.join(_ENVS_DIR, f"{env.env_id}.json")
        atomic_write(path, json.dumps(env.to_dict(), indent=2))

    def load(self, env_id: str) -> Optional[Environment]:
        path = os.path.join(_ENVS_DIR, f"{env_id}.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return Environment.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError):
            return None

    def load_all(self) -> dict:
        envs = {}
        if not os.path.isdir(_ENVS_DIR):
            return envs
        for f in os.listdir(_ENVS_DIR):
            if f.endswith(".json"):
                env = self.load(f[:-5])
                if env:
                    envs[env.env_id] = env
        return envs

    def remove(self, env_id: str) -> None:
        path = os.path.join(_ENVS_DIR, f"{env_id}.json")
        if os.path.isfile(path):
            os.remove(path)


class EnvironmentRegistry:
    """Thread-safe environment CRUD with disk persistence."""

    def __init__(self, store: Optional[EnvironmentStore] = None):
        self._store = store or EnvironmentStore()
        self._envs: dict[str, Environment] = self._store.load_all()
        self._lock = threading.Lock()

    def create(self, *, name: str, **kwargs) -> Environment:
        """Create a new environment profile."""
        env_id = str(uuid.uuid4())[:8]
        env = Environment(env_id=env_id, name=name, created_at=time.time(), **kwargs)
        with self._lock:
            self._envs[env_id] = env
            self._store.save(env)
        return env

    def get(self, env_id: str) -> Optional[Environment]:
        with self._lock:
            return self._envs.get(env_id)

    def list_environments(self) -> list:
        with self._lock:
            envs = list(self._envs.values())
        envs.sort(key=lambda e: e.name)
        return envs

    def delete(self, env_id: str) -> bool:
        with self._lock:
            if env_id not in self._envs:
                return False
            del self._envs[env_id]
            self._store.remove(env_id)
            return True


def check_tool_allowed(env: Environment, tool_name: str) -> tuple:
    """Check if a tool is allowed by the environment. Returns (allowed, reason)."""
    if env.blocked_tools and tool_name in env.blocked_tools:
        return False, f"Tool '{tool_name}' blocked by environment '{env.name}'"
    if env.allowed_tools and tool_name not in env.allowed_tools:
        return False, f"Tool '{tool_name}' not in environment '{env.name}' allowlist"
    return True, ""


def check_provider_allowed(env: Environment, provider_name: str) -> tuple:
    """Check if a provider is allowed by the environment. Returns (allowed, reason)."""
    if env.require_local_only and provider_name not in ("ollama", "vllm", "llamacpp", "lmstudio", "mlx_vlm"):
        return False, f"Environment '{env.name}' requires local-only providers"
    if env.allowed_providers and provider_name not in env.allowed_providers:
        return False, f"Provider '{provider_name}' not allowed by environment '{env.name}'"
    return True, ""
