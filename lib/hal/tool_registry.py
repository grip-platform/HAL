"""HAL Tool Registry — dynamic tool discovery and registration.

Manages the set of tools available to HAL and its agents. Supports
built-in tools, MCP-discovered tools, and user-registered custom tools.

Usage:
    from lib.hal.tool_registry import ToolRegistry, ToolDef

    registry = ToolRegistry()
    registry.register(ToolDef("search", "Search the web", {"query": "string"}))
    tool = registry.get("search")
    all_tools = registry.list_tools()
"""
import json
import os
from dataclasses import dataclass, field

from lib.hal.home import HAL_HOME as _HAL_HOME

_CUSTOM_TOOLS_PATH = os.path.join(_HAL_HOME, "config", "custom-tools.json")


@dataclass
class ToolDef:
    """Definition of a tool available to HAL."""
    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    source: str = "builtin"  # "builtin", "mcp", "custom", "agent"
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "source": self.source,
            "enabled": self.enabled,
        }


# Built-in tools that HAL always has
_BUILTIN_TOOLS: dict[str, ToolDef] = {
    "read": ToolDef("read", "Read a file from the filesystem", {"file_path": "string"}),
    "write": ToolDef("write", "Write content to a file", {"file_path": "string", "content": "string"}),
    "edit": ToolDef("edit", "Edit a file with string replacement", {"file_path": "string", "old": "string", "new": "string"}),
    "search": ToolDef("search", "Search file contents with regex", {"pattern": "string", "path": "string"}),
    "glob": ToolDef("glob", "Find files by glob pattern", {"pattern": "string"}),
    "bash": ToolDef("bash", "Execute a shell command", {"command": "string"}),
    "chat": ToolDef("chat", "Send a message to an LLM", {"message": "string", "model": "string"}),
    "council": ToolDef("council", "Query the cross-LLM council", {"query": "string"}),
    "websearch": ToolDef(
        "websearch",
        "Search the web using DuckDuckGo. Returns a summary of results.",
        {"query": "string"},
    ),
    "osa_run": ToolDef(
        "osa_run",
        (
            "Run a named macOS AppleScript from the vetted registry. "
            "Library-only — no ad-hoc scripts. Tier-1 scripts run automatically; "
            "tier-2 require a capability token; tier-3 require explicit operator "
            "confirmation via AskUserQuestion."
        ),
        {
            "script": "string",
            "params": "object",
            "host": "string",
        },
    ),
}


class ToolRegistry:
    """Manages tool discovery and registration."""

    def __init__(self):
        self._tools: dict[str, ToolDef] = dict(_BUILTIN_TOOLS)
        self._load_custom()

    def register(self, tool: ToolDef) -> None:
        """Register a tool. Overwrites existing tool with same name."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Unregister a tool. Returns True if removed."""
        if name in self._tools and name not in _BUILTIN_TOOLS:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> ToolDef | None:
        """Get a tool by name."""
        tool = self._tools.get(name)
        if tool and tool.enabled:
            return tool
        return None

    def list_tools(self, source: str = "") -> list[ToolDef]:
        """List all enabled tools, optionally filtered by source."""
        tools = [t for t in self._tools.values() if t.enabled]
        if source:
            tools = [t for t in tools if t.source == source]
        return sorted(tools, key=lambda t: t.name)

    def to_schema(self) -> list[dict]:
        """Export tool definitions as JSON schema (MCP-compatible format)."""
        return [t.to_dict() for t in self.list_tools()]

    def save_custom(self) -> None:
        """Persist custom tools to disk."""
        custom = {
            name: tool.to_dict()
            for name, tool in self._tools.items()
            if tool.source == "custom"
        }
        os.makedirs(os.path.dirname(_CUSTOM_TOOLS_PATH), exist_ok=True)
        with open(_CUSTOM_TOOLS_PATH, "w", encoding="utf-8") as f:
            json.dump(custom, f, indent=2)

    def _load_custom(self) -> None:
        """Load custom tools from disk."""
        if not os.path.exists(_CUSTOM_TOOLS_PATH):
            return
        try:
            with open(_CUSTOM_TOOLS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            for name, d in data.items():
                self._tools[name] = ToolDef(
                    name=name, description=d.get("description", ""),
                    parameters=d.get("parameters", {}),
                    source="custom", enabled=d.get("enabled", True),
                )
        except (json.JSONDecodeError, OSError):
            pass
