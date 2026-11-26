"""HAL Concurrent Tool Executor — parallel read tools, serial write tools.

Ported from TF>> StreamingToolExecutor + toolOrchestration patterns.
Read-only tools (is_concurrent_safe=True) run in parallel via ThreadPoolExecutor.
Write tools run one at a time to prevent conflicts.

Usage:
    from lib.hal.tool_executor import ToolExecutor
    from lib.hal.tool_protocol import ToolProtocol, ToolContext

    executor = ToolExecutor(tools=[ReadTool(), WriteTool(), GrepTool()])
    results = executor.execute_batch(
        calls=[("Read", {"file_path": "a.py"}), ("Grep", {"pattern": "def"})],
        context=ToolContext(session_id="abc"),
    )
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum

from lib.hal.tool_protocol import ToolContext, ToolOutput, ToolProtocol


class ToolState(str, Enum):
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrackedTool:
    """A tool invocation with execution state tracking."""
    tool_name: str
    tool_input: dict
    state: str = ToolState.QUEUED
    output: ToolOutput = field(default_factory=ToolOutput)
    started_at: float = 0.0
    completed_at: float = 0.0
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "state": self.state,
            "output": self.output.to_dict(),
            "duration_ms": round(self.duration_ms, 2),
        }


class ToolExecutor:
    """Executes tool calls with concurrency control.

    Batching strategy (from TF>> toolOrchestration.ts):
    - Multiple concurrent-safe tools in one batch: run in parallel
    - A single non-concurrent-safe tool: run alone, serial
    - Mixed batch: split into concurrent group + serial queue
    """

    def __init__(self, tools: list[ToolProtocol] = None, max_workers: int = 4):
        self._tools: dict[str, ToolProtocol] = {}
        self._max_workers = max_workers
        self._lock = threading.Lock()
        for tool in (tools or []):
            self._tools[tool.name] = tool

    def register(self, tool: ToolProtocol) -> None:
        """Register a tool for execution."""
        with self._lock:
            self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ToolProtocol:
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return sorted(self._tools.keys())

    def execute_single(
        self, tool_name: str, tool_input: dict, context: ToolContext,
    ) -> TrackedTool:
        """Execute a single tool call."""
        tracked = TrackedTool(tool_name=tool_name, tool_input=tool_input)
        tool = self._tools.get(tool_name)
        if tool is None:
            tracked.state = ToolState.FAILED
            tracked.output = ToolOutput(
                content=f"Unknown tool: {tool_name}", is_error=True,
            )
            return tracked

        validation_error = tool.validate_input(tool_input)
        if validation_error:
            tracked.state = ToolState.FAILED
            tracked.output = ToolOutput(content=validation_error, is_error=True)
            return tracked

        tracked.state = ToolState.EXECUTING
        tracked.started_at = time.perf_counter()
        try:
            tracked.output = tool.execute(tool_input, context)
            tracked.state = ToolState.COMPLETED
        except Exception as exc:
            tracked.output = ToolOutput(content=str(exc), is_error=True)
            tracked.state = ToolState.FAILED
        tracked.completed_at = time.perf_counter()
        tracked.duration_ms = (tracked.completed_at - tracked.started_at) * 1000
        return tracked

    def execute_batch(
        self,
        calls: list[tuple[str, dict]],
        context: ToolContext,
    ) -> list[TrackedTool]:
        """Execute a batch of tool calls with optimal concurrency.

        Read-only (concurrent-safe) tools run in parallel.
        Write (non-concurrent-safe) tools run one at a time.
        """
        if not calls:
            return []

        concurrent, serial = self._partition(calls)
        results: list[TrackedTool] = []

        # Run concurrent-safe tools in parallel
        if concurrent:
            results.extend(self._run_parallel(concurrent, context))

        # Run non-concurrent tools serially
        for tool_name, tool_input in serial:
            results.append(self.execute_single(tool_name, tool_input, context))

        return results

    def _partition(
        self, calls: list[tuple[str, dict]],
    ) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
        """Split calls into concurrent-safe and serial groups."""
        concurrent = []
        serial = []
        for tool_name, tool_input in calls:
            tool = self._tools.get(tool_name)
            if tool and tool.is_concurrent_safe:
                concurrent.append((tool_name, tool_input))
            else:
                serial.append((tool_name, tool_input))
        return concurrent, serial

    def _run_parallel(
        self,
        calls: list[tuple[str, dict]],
        context: ToolContext,
    ) -> list[TrackedTool]:
        """Run concurrent-safe tools in a thread pool."""
        results = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self.execute_single, name, inp, context): (name, inp)
                for name, inp in calls
            }
            for future in as_completed(futures):
                results.append(future.result())
        return results
