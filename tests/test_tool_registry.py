"""Tests for HAL tool registry."""
import os
import sys
import unittest

from lib.hal.tool_registry import ToolDef, ToolRegistry, _BUILTIN_TOOLS


class TestToolDef(unittest.TestCase):

    def test_to_dict(self):
        tool = ToolDef("test", "A test tool", {"x": "string"})
        d = tool.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["source"], "builtin")

    def test_default_enabled(self):
        tool = ToolDef("test", "desc")
        self.assertTrue(tool.enabled)


class TestBuiltinTools(unittest.TestCase):

    def test_has_read(self):
        self.assertIn("read", _BUILTIN_TOOLS)

    def test_has_write(self):
        self.assertIn("write", _BUILTIN_TOOLS)

    def test_has_chat(self):
        self.assertIn("chat", _BUILTIN_TOOLS)

    def test_minimum_builtins(self):
        self.assertGreaterEqual(len(_BUILTIN_TOOLS), 6)


class TestToolRegistry(unittest.TestCase):

    def test_get_builtin(self):
        reg = ToolRegistry()
        tool = reg.get("read")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "read")

    def test_get_unknown(self):
        reg = ToolRegistry()
        self.assertIsNone(reg.get("nonexistent_tool"))

    def test_register_custom(self):
        reg = ToolRegistry()
        reg.register(ToolDef("my_tool", "Custom tool", source="custom"))
        tool = reg.get("my_tool")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.source, "custom")

    def test_unregister_custom(self):
        reg = ToolRegistry()
        reg.register(ToolDef("temp", "Temporary", source="custom"))
        self.assertTrue(reg.unregister("temp"))
        self.assertIsNone(reg.get("temp"))

    def test_cannot_unregister_builtin(self):
        reg = ToolRegistry()
        self.assertFalse(reg.unregister("read"))
        self.assertIsNotNone(reg.get("read"))

    def test_list_tools(self):
        reg = ToolRegistry()
        tools = reg.list_tools()
        self.assertGreater(len(tools), 0)
        # Should be sorted
        names = [t.name for t in tools]
        self.assertEqual(names, sorted(names))

    def test_list_by_source(self):
        reg = ToolRegistry()
        reg.register(ToolDef("custom1", "C1", source="custom"))
        custom = reg.list_tools(source="custom")
        self.assertEqual(len(custom), 1)

    def test_disabled_tool_not_returned(self):
        reg = ToolRegistry()
        reg.register(ToolDef("hidden", "Hidden", enabled=False))
        self.assertIsNone(reg.get("hidden"))

    def test_to_schema(self):
        reg = ToolRegistry()
        schema = reg.to_schema()
        self.assertIsInstance(schema, list)
        self.assertGreater(len(schema), 0)
        self.assertIn("name", schema[0])


if __name__ == "__main__":
    unittest.main()
