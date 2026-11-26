"""Tests for HAL Concurrent Tool Executor."""
import time
import unittest

from lib.hal.tool_executor import ToolExecutor, ToolState, TrackedTool
from lib.hal.tool_protocol import ToolProtocol, ToolContext, ToolOutput


class _FastRead(ToolProtocol):
    @property
    def name(self): return "FastRead"
    @property
    def description(self): return "fast read"
    @property
    def is_concurrent_safe(self): return True
    @property
    def input_schema(self):
        return {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    def execute(self, inp, ctx):
        return ToolOutput(content=f"read:{inp['path']}")


class _SlowRead(ToolProtocol):
    @property
    def name(self): return "SlowRead"
    @property
    def description(self): return "slow read"
    @property
    def is_concurrent_safe(self): return True
    def execute(self, inp, ctx):
        time.sleep(0.2)
        return ToolOutput(content="slow-done")


class _WriteTool(ToolProtocol):
    @property
    def name(self): return "Write"
    @property
    def description(self): return "write"
    @property
    def is_concurrent_safe(self): return False
    def execute(self, inp, ctx):
        return ToolOutput(content="written")


class _FailTool(ToolProtocol):
    @property
    def name(self): return "Fail"
    @property
    def description(self): return "always fails"
    @property
    def is_concurrent_safe(self): return True
    def execute(self, inp, ctx):
        raise RuntimeError("intentional failure")


class TestToolExecutorSingle(unittest.TestCase):

    def test_execute_single_success(self):
        executor = ToolExecutor(tools=[_FastRead()])
        ctx = ToolContext(session_id="test")
        result = executor.execute_single("FastRead", {"path": "a.py"}, ctx)
        self.assertEqual(result.state, ToolState.COMPLETED)
        self.assertEqual(result.output.content, "read:a.py")
        self.assertGreater(result.duration_ms, 0)

    def test_execute_single_unknown_tool(self):
        executor = ToolExecutor()
        ctx = ToolContext()
        result = executor.execute_single("NonExistent", {}, ctx)
        self.assertEqual(result.state, ToolState.FAILED)
        self.assertIn("Unknown tool", result.output.content)

    def test_execute_single_validation_error(self):
        executor = ToolExecutor(tools=[_FastRead()])
        ctx = ToolContext()
        result = executor.execute_single("FastRead", {}, ctx)
        self.assertEqual(result.state, ToolState.FAILED)
        self.assertIn("path", result.output.content)

    def test_execute_single_exception(self):
        executor = ToolExecutor(tools=[_FailTool()])
        ctx = ToolContext()
        result = executor.execute_single("Fail", {}, ctx)
        self.assertEqual(result.state, ToolState.FAILED)
        self.assertIn("intentional failure", result.output.content)

    def test_tracked_tool_to_dict(self):
        t = TrackedTool(tool_name="Read", tool_input={"path": "x"})
        d = t.to_dict()
        self.assertEqual(d["tool_name"], "Read")
        self.assertEqual(d["state"], ToolState.QUEUED)


class TestToolExecutorBatch(unittest.TestCase):

    def test_batch_concurrent_reads(self):
        executor = ToolExecutor(tools=[_FastRead(), _SlowRead()])
        ctx = ToolContext()
        results = executor.execute_batch([
            ("FastRead", {"path": "a.py"}),
            ("SlowRead", {}),
            ("FastRead", {"path": "b.py"}),
        ], ctx)
        self.assertEqual(len(results), 3)
        completed = [r for r in results if r.state == ToolState.COMPLETED]
        self.assertEqual(len(completed), 3)

    def test_batch_serial_writes(self):
        executor = ToolExecutor(tools=[_WriteTool()])
        ctx = ToolContext()
        results = executor.execute_batch([
            ("Write", {"content": "a"}),
            ("Write", {"content": "b"}),
        ], ctx)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.state == ToolState.COMPLETED for r in results))

    def test_batch_mixed_concurrent_serial(self):
        executor = ToolExecutor(tools=[_FastRead(), _WriteTool()])
        ctx = ToolContext()
        results = executor.execute_batch([
            ("FastRead", {"path": "a.py"}),
            ("Write", {"content": "x"}),
            ("FastRead", {"path": "b.py"}),
        ], ctx)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.state == ToolState.COMPLETED for r in results))

    def test_batch_empty(self):
        executor = ToolExecutor()
        results = executor.execute_batch([], ToolContext())
        self.assertEqual(results, [])

    def test_parallel_execution_intervals_overlap(self):
        """Two SlowRead tools (200ms each) should run with overlapping execution
        windows, not one strictly after the other."""
        executor = ToolExecutor(tools=[_SlowRead()])
        ctx = ToolContext()
        results = executor.execute_batch([("SlowRead", {}), ("SlowRead", {})], ctx)
        first, second = results[0], results[1]
        self.assertEqual(first.state, ToolState.COMPLETED)
        self.assertEqual(second.state, ToolState.COMPLETED)
        overlap_start = max(first.started_at, second.started_at)
        overlap_end = min(first.completed_at, second.completed_at)
        self.assertGreater(
            overlap_end, overlap_start,
            f"SlowRead calls should overlap, not run serially: {first.to_dict()} vs {second.to_dict()}",
        )


class TestToolExecutorRegistry(unittest.TestCase):

    def test_register_and_list(self):
        executor = ToolExecutor()
        executor.register(_FastRead())
        executor.register(_WriteTool())
        self.assertEqual(executor.list_tools(), ["FastRead", "Write"])

    def test_get_tool(self):
        executor = ToolExecutor(tools=[_FastRead()])
        tool = executor.get_tool("FastRead")
        self.assertEqual(tool.name, "FastRead")
        self.assertIsNone(executor.get_tool("Missing"))


if __name__ == "__main__":
    unittest.main()
