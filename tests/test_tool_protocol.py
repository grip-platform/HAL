"""Tests for HAL Tool Protocol — formal tool interface."""
import os
import sys
import unittest

from lib.hal.tool_protocol import ToolProtocol, ToolContext, ToolOutput


class MockReadTool(ToolProtocol):
    """Test implementation of a read-only tool."""
    @property
    def name(self):
        return "MockRead"

    @property
    def description(self):
        return "Read a test file"

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    @property
    def is_concurrent_safe(self):
        return True

    def execute(self, tool_input, context):
        return ToolOutput(content=f"contents of {tool_input['path']}")


class MockWriteTool(ToolProtocol):
    """Test implementation of a write tool (not concurrent-safe)."""
    @property
    def name(self):
        return "MockWrite"

    @property
    def description(self):
        return "Write to a test file"

    def execute(self, tool_input, context):
        return ToolOutput(content="written")


class TestToolOutput(unittest.TestCase):

    def test_default_values(self):
        out = ToolOutput()
        self.assertEqual(out.content, "")
        self.assertFalse(out.is_error)
        self.assertEqual(out.metadata, {})

    def test_to_dict(self):
        out = ToolOutput(content="hello", is_error=True, metadata={"key": "val"})
        d = out.to_dict()
        self.assertEqual(d["content"], "hello")
        self.assertTrue(d["is_error"])
        self.assertEqual(d["metadata"]["key"], "val")


class TestToolContext(unittest.TestCase):

    def test_default_values(self):
        ctx = ToolContext()
        self.assertEqual(ctx.session_id, "")
        self.assertEqual(ctx.max_output_chars, 50_000)

    def test_custom_values(self):
        ctx = ToolContext(session_id="abc", working_dir="/tmp")
        self.assertEqual(ctx.session_id, "abc")
        self.assertEqual(ctx.working_dir, "/tmp")


class TestToolProtocol(unittest.TestCase):

    def test_read_tool_name(self):
        tool = MockReadTool()
        self.assertEqual(tool.name, "MockRead")

    def test_read_tool_concurrent_safe(self):
        tool = MockReadTool()
        self.assertTrue(tool.is_concurrent_safe)

    def test_write_tool_not_concurrent_safe(self):
        tool = MockWriteTool()
        self.assertFalse(tool.is_concurrent_safe)

    def test_execute_returns_output(self):
        tool = MockReadTool()
        ctx = ToolContext(session_id="test")
        result = tool.execute({"path": "/tmp/test.py"}, ctx)
        self.assertEqual(result.content, "contents of /tmp/test.py")
        self.assertFalse(result.is_error)

    def test_validate_input_missing_required(self):
        tool = MockReadTool()
        err = tool.validate_input({})
        self.assertIn("path", err)

    def test_validate_input_passes(self):
        tool = MockReadTool()
        err = tool.validate_input({"path": "/tmp/file"})
        self.assertEqual(err, "")

    def test_to_mcp_tool(self):
        tool = MockReadTool()
        mcp = tool.to_mcp_tool()
        self.assertEqual(mcp["name"], "MockRead")
        self.assertIn("inputSchema", mcp)
        self.assertEqual(mcp["inputSchema"]["required"], ["path"])

    def test_input_schema_default(self):
        tool = MockWriteTool()
        schema = tool.input_schema
        self.assertEqual(schema["type"], "object")


if __name__ == "__main__":
    unittest.main()
