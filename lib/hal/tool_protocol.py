"""HAL Tool Protocol — formal interface for all tool implementations.

Ported from TF>> Tool.ts: unified type with execute/validate, input schema,
and concurrency safety flag. All tools implement this protocol for dispatch
by ToolExecutor.

Coexists with existing ToolDispatcher (tool_dispatch.py) — ToolDispatcher
handles the hook enforcement layer, ToolProtocol defines the contract each
tool must satisfy.

Usage:
    from lib.hal.tool_protocol import ToolProtocol, ToolContext, ToolOutput

    class ReadTool(ToolProtocol):
        name = "Read"
        description = "Read a file from disk"
        is_concurrent_safe = True
        input_schema = {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}

        def execute(self, input: dict, context: ToolContext) -> ToolOutput:
            with open(input["file_path"]) as f:
                return ToolOutput(content=f.read())
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolContext:
    """Runtime context passed to tool execution.

    Provides tools with access to session state, other tools, and
    configuration without tight coupling.
    """
    session_id: str = ""
    working_dir: str = ""
    environment_id: str = ""
    max_output_chars: int = 50_000
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolOutput:
    """Result of a tool execution."""
    content: str = ""
    is_error: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "is_error": self.is_error,
            "metadata": self.metadata,
        }


class ToolProtocol(ABC):
    """Abstract base for all HAL tools.

    Every tool declares its name, description, input schema (JSON Schema),
    and whether it's safe to run concurrently with other tools.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this tool does."""

    @property
    def input_schema(self) -> dict:
        """JSON Schema for the tool's input parameters."""
        return {"type": "object", "properties": {}}

    @property
    def is_concurrent_safe(self) -> bool:
        """Whether this tool can run in parallel with other tools.

        Read-only tools (Read, Glob, Grep) are concurrent-safe.
        Write tools (Edit, Write, Bash) are not.
        """
        return False

    @abstractmethod
    def execute(self, tool_input: dict, context: ToolContext) -> ToolOutput:
        """Execute the tool with given input and context."""

    def validate_input(self, tool_input: dict) -> str:
        """Validate input against schema. Returns error string or empty."""
        required = self.input_schema.get("required", [])
        for key in required:
            if key not in tool_input:
                return f"Missing required parameter: {key}"
        return ""

    def to_mcp_tool(self) -> dict:
        """Export as MCP tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
