"""HAL Tool Adapter — bridges the native ToolDispatcher into the ToolProtocol layer.

HAL had two parallel tool surfaces that never met:

- ``lib/hal/tool_dispatch.py`` — 8 native handlers (Bash, Read, Write, Edit,
  Glob, Grep, CodeLookup, CodeSearch) executed through a guard + hook
  enforcement pipeline. Concrete, working, but opaque to ``ToolExecutor``
  because handlers are plain callables keyed by string.
- ``lib/hal/tool_protocol.py`` + ``lib/hal/tool_executor.py`` — a
  ``ToolProtocol`` ABC and a ``ToolExecutor`` that runs concurrent-safe tools
  in parallel via a thread pool. Ported from TF>>. Until now there were no
  concrete implementations — agents could register a ``ToolExecutor`` and get
  nothing back.

This adapter closes the gap without duplicating any execution logic. Each
built-in tool becomes a ``ToolProtocol`` instance whose ``execute`` forwards
to ``ToolDispatcher.execute`` and whose ``is_concurrent_safe`` flag is set
per tool so read-only tools (Read/Grep/Glob/CodeLookup/CodeSearch) fan out in
parallel while mutating tools (Bash/Write/Edit) run serially.

Usage
-----

    from lib.hal.tool_adapter import create_builtin_tools
    from lib.hal.tool_executor import ToolExecutor
    from lib.hal.tool_protocol import ToolContext

    executor = ToolExecutor(tools=create_builtin_tools())
    results = executor.execute_batch(
        calls=[
            ("Read", {"file_path": "plans/active.md"}),
            ("Grep", {"pattern": "TODO", "path": "lib/hal"}),
        ],
        context=ToolContext(session_id="abc123"),
    )
"""
from dataclasses import dataclass

from lib.hal.tool_dispatch import ToolDispatcher
from lib.hal.tool_protocol import ToolContext, ToolOutput, ToolProtocol


@dataclass(frozen=True)
class ToolSpec:
    """Static declaration of a tool's public contract.

    Kept frozen so spec tables are safe to share across adapter instances —
    two ToolExecutors pointed at the same dispatcher still see the same spec
    without defensive copies.
    """
    name: str
    description: str
    input_schema: dict
    is_concurrent_safe: bool


# Built-in tool specs — aligned with the names registered in
# ToolDispatcher._TOOL_REGISTRY. Concurrency flags follow TF>> convention:
# a tool is "concurrent-safe" when two parallel invocations cannot observably
# interfere (read-only FS queries, code index lookups). Anything that mutates
# filesystem or spawns a subprocess runs serially.
_BUILTIN_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="Bash",
        description="Execute a shell command with timeout and optional cwd.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "milliseconds, max 600000"},
                "cwd": {"type": "string"},
            },
            "required": ["command"],
        },
        is_concurrent_safe=False,
    ),
    ToolSpec(
        name="Read",
        description="Read a file from disk.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["file_path"],
        },
        is_concurrent_safe=True,
    ),
    ToolSpec(
        name="Write",
        description="Write content to a file, overwriting if it exists.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
        is_concurrent_safe=False,
    ),
    ToolSpec(
        name="Edit",
        description="Replace an exact string in a file.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
        is_concurrent_safe=False,
    ),
    ToolSpec(
        name="Glob",
        description="Find files matching a glob pattern.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"],
        },
        is_concurrent_safe=True,
    ),
    ToolSpec(
        name="Grep",
        description="Search file contents with regex.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
            },
            "required": ["pattern"],
        },
        is_concurrent_safe=True,
    ),
    ToolSpec(
        name="CodeLookup",
        description="Look up a code symbol in the project index.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "project": {"type": "string"},
            },
            "required": ["symbol"],
        },
        is_concurrent_safe=True,
    ),
    ToolSpec(
        name="CodeSearch",
        description="Full-text search the project code index.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "project": {"type": "string"},
            },
            "required": ["query"],
        },
        is_concurrent_safe=True,
    ),
    ToolSpec(
        name="osa_run",
        description=(
            "Dispatch a registered macOS osascript (AppleScript/JXA) by "
            "name via the HAL osa control layer. Tier-1 scripts (battery_get, "
            "volume_get, frontmost_app_get, idle_time_get, display_count_get, "
            "dnd_status_get) run without authorisation. Tier-2+ scripts "
            "(notification, volume_set, clipboard_get, screen_lock, "
            "dark_mode_toggle, caffeinate_start, speak) require the operator "
            "to set HAL_OSA_TOOL_CHAIN_AUTOMINT=1 so the handler can mint a "
            "capability token. Every dispatch is audit-logged with "
            "provenance=tool_chain."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Registry script name. Discover via `hal osa list`."
                    ),
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Per-script parameters as key/value pairs. Each "
                        "value is validated against the registry schema "
                        "before interpolation."
                    ),
                },
                "timeout_s": {
                    "type": "integer",
                    "description": "Override the registry default timeout.",
                },
            },
            "required": ["name"],
        },
        is_concurrent_safe=False,
    ),
)


class DispatchAdaptedTool(ToolProtocol):
    """``ToolProtocol`` that delegates ``execute`` to a shared ``ToolDispatcher``.

    The adapter owns no execution logic of its own — everything flows through
    the dispatcher's guard + hook pipeline so security invariants (secrets
    redaction, Bash classifier, PreToolUse hooks) apply uniformly whether a
    call lands via the native dispatch path or via a ``ToolExecutor`` batch.
    """

    def __init__(self, spec: ToolSpec, dispatcher: ToolDispatcher):
        self._spec = spec
        self._dispatcher = dispatcher

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def description(self) -> str:
        return self._spec.description

    @property
    def input_schema(self) -> dict:
        return self._spec.input_schema

    @property
    def is_concurrent_safe(self) -> bool:
        return self._spec.is_concurrent_safe

    def execute(self, tool_input: dict, context: ToolContext) -> ToolOutput:
        result = self._dispatcher.execute(self.name, tool_input)
        if result.blocked:
            return ToolOutput(
                content=result.reason or "blocked by guard",
                is_error=True,
                metadata={"tool": self.name, "blocked": True},
            )
        if result.error:
            return ToolOutput(
                content=result.error,
                is_error=True,
                metadata={"tool": self.name, "output": result.output},
            )
        return ToolOutput(content=result.output, metadata={"tool": self.name})


def create_builtin_tools(
    dispatcher: ToolDispatcher | None = None,
) -> list[ToolProtocol]:
    """Return all 8 built-in HAL tools as ``ToolProtocol`` instances.

    If ``dispatcher`` is omitted, a default ``ToolDispatcher`` is constructed.
    Pass an explicit instance when the caller needs custom guard rules or a
    test hook dispatcher.
    """
    dispatcher = dispatcher or ToolDispatcher()
    return [DispatchAdaptedTool(spec, dispatcher) for spec in _BUILTIN_SPECS]


def to_anthropic_tools(
    specs: tuple[ToolSpec, ...] = _BUILTIN_SPECS,
) -> list[dict]:
    """Project HAL tool specs into the Anthropic Messages API tool schema.

    The Messages API expects each tool as a dict with exactly three top-level
    fields: ``name``, ``description``, ``input_schema`` (a JSON Schema object).
    ``ToolSpec`` already stores all three in the right shape, so the bridge
    is a pure projection — no field renaming, no defaults to synthesise,
    and ``is_concurrent_safe`` is intentionally dropped (it's an executor
    concern, not something the model needs to know about).

    Pass a filtered ``specs`` tuple to restrict the toolset offered to the
    model — e.g. ``tuple(s for s in _BUILTIN_SPECS if s.is_concurrent_safe)``
    for a read-only session.

    Usage:
        body["tools"] = to_anthropic_tools()  # all 8 built-ins
    """
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }
        for spec in specs
    ]


def to_openai_tools(
    specs_or_tools,
) -> list[dict]:
    """Project tools into the OpenAI chat-completions tool schema.

    OpenAI-compatible endpoints (Groq, OpenAI, Gemini OpenAI-compat, and
    Ollama's OpenAI-compat bridge) require each tool to be wrapped as::

        {"type": "function",
         "function": {
            "name": <str>, "description": <str>, "parameters": <json-schema>}}

    HAL's native tool registry stores :class:`ToolSpec` instances whose
    ``input_schema`` field holds the JSON Schema. This function accepts
    either the native specs or the Anthropic-shape dicts produced by
    :func:`to_anthropic_tools` — in both cases it emits valid OpenAI-shape
    tool entries so callers can swap provider families without plumbing
    two separate tool tables through the REPL.

    Input tolerance:

      * ``ToolSpec`` instance → read ``name``/``description``/``input_schema``
        directly.
      * Dict with ``input_schema`` key → Anthropic shape, rewrap in OpenAI
        envelope. This is the common path: the REPL builds tools once via
        :func:`to_anthropic_tools` and the OpenAI-compat chat_fn converts
        at request time.
      * Dict with ``type == "function"`` → already OpenAI-shape, passed
        through untouched so double-wrapping is impossible.

    Returns a new list every call (not a view) so callers can safely
    extend the result per-request (e.g. add ``tool_choice`` metadata).

    Issue #200: the OpenAI-compat chat_fn was emitting Anthropic-shape
    tool dicts verbatim, which Groq (and every OpenAI-compat provider)
    rejects with ``'tools.0.type' : property 'type' is missing``. This
    function is the missing wrapping step.
    """
    converted: list[dict] = []
    for entry in specs_or_tools:
        if isinstance(entry, ToolSpec):
            name, description, parameters = (
                entry.name, entry.description, entry.input_schema,
            )
        elif isinstance(entry, dict) and entry.get("type") == "function":
            # Already OpenAI-shape — pass through to avoid double-wrapping.
            converted.append(entry)
            continue
        elif isinstance(entry, dict):
            # Anthropic-shape dict: {name, description, input_schema}.
            name = entry.get("name", "")
            description = entry.get("description", "")
            parameters = entry.get("input_schema") or entry.get("parameters") or {}
        else:
            # Unknown shape — skip rather than raise so one bad entry
            # doesn't poison the whole batch. The provider will reject the
            # request if the resulting list is wrong, which is the visible
            # failure mode we want. Silent drop avoids crashing the REPL.
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        })
    return converted


def parse_tool_use_blocks(raw: object) -> list[dict]:
    """Extract ``tool_use`` blocks from an Anthropic Messages API response.

    When the model invokes tools, the response ``content`` array interleaves
    ``text``, ``thinking``, and ``tool_use`` blocks. ``tool_use`` blocks have
    the shape::

        {"type": "tool_use", "id": "toolu_...", "name": "Read",
         "input": {"file_path": "/tmp/x"}}

    Returns a list of dicts with ``id``, ``name``, ``input`` — the minimum
    the dispatcher needs to execute the call and correlate its result back
    to the conversation via the matching ``tool_result`` block.

    Degrades gracefully on malformed input: non-dict root, missing
    ``content`` key, non-list ``content``, or blocks missing ``name`` /
    ``input`` all yield defaults rather than raising. The caller
    distinguishes "no tool calls" (empty list) from "parse failed" by
    type-checking the input itself.
    """
    if not isinstance(raw, dict):
        return []
    content = raw.get("content")
    if not isinstance(content, list):
        return []
    return [
        {
            "id": block.get("id", ""),
            "name": block.get("name", ""),
            "input": block.get("input", {}),
        }
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


__all__ = [
    "ToolSpec",
    "DispatchAdaptedTool",
    "create_builtin_tools",
    "to_anthropic_tools",
    "to_openai_tools",
    "parse_tool_use_blocks",
]
