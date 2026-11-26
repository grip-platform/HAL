"""Tests for HAL MCP server."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.mcp_server import HALMCPServer, _error_response, _TOOLS


class TestMCPToolDefinitions(unittest.TestCase):

    def test_tools_have_required_fields(self):
        for tool in _TOOLS:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("inputSchema", tool)

    def test_tool_names_unique(self):
        names = [t["name"] for t in _TOOLS]
        self.assertEqual(len(names), len(set(names)))

    def test_at_least_fifteen_tools_defined(self):
        """5 original + 4 Fleet Command + 4 Vault + 2 A2A peer tools. Lower bound
        (not exact) so a future tool addition does not break this; drops are
        caught by the per-name membership asserts in test_tool_names."""
        self.assertGreaterEqual(len(_TOOLS), 15)

    def test_tool_names(self):
        names = {t["name"] for t in _TOOLS}
        # Original 5
        self.assertIn("hal_chat", names)
        self.assertIn("hal_council", names)
        self.assertIn("hal_status", names)
        self.assertIn("hal_providers", names)
        self.assertIn("hal_cost", names)
        # Fleet Command 4 (added PR #102)
        self.assertIn("hal_agent_create", names)
        self.assertIn("hal_agent_list", names)
        self.assertIn("hal_fleet", names)
        self.assertIn("hal_task_submit", names)
        # Vault tools 4 (added this sprint)
        self.assertIn("hal_vault_search", names)
        self.assertIn("hal_vault_browse", names)
        self.assertIn("hal_vault_read", names)
        self.assertIn("hal_vault_graph", names)
        # A2A peer delegation 2 (three-harness downstream plane)
        self.assertIn("a2a_ask_peer", names)
        self.assertIn("a2a_list_peers", names)


class TestMCPServer(unittest.TestCase):

    def test_initialize(self):
        server = HALMCPServer()
        resp = server.handle_request({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        self.assertEqual(resp["id"], 1)
        self.assertIn("serverInfo", resp["result"])
        self.assertEqual(resp["result"]["serverInfo"]["name"], "hal")

    def test_tools_list(self):
        server = HALMCPServer()
        resp = server.handle_request({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        })
        self.assertGreaterEqual(len(resp["result"]["tools"]), 15)

    def test_unknown_method(self):
        server = HALMCPServer()
        resp = server.handle_request({
            "jsonrpc": "2.0", "id": 3, "method": "unknown/method", "params": {},
        })
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32601)

    def test_unknown_tool(self):
        server = HALMCPServer()
        server._registry = MagicMock()  # Skip lazy init
        server._conversation = MagicMock()
        server._council = MagicMock()
        resp = server.handle_request({
            "jsonrpc": "2.0", "id": 4,
            "method": "tools/call",
            "params": {"name": "nonexistent", "arguments": {}},
        })
        self.assertTrue(resp["result"]["isError"])


class TestErrorResponse(unittest.TestCase):

    def test_format(self):
        resp = _error_response(42, -32600, "Invalid request")
        self.assertEqual(resp["jsonrpc"], "2.0")
        self.assertEqual(resp["id"], 42)
        self.assertEqual(resp["error"]["code"], -32600)
        self.assertEqual(resp["error"]["message"], "Invalid request")

    def test_null_id(self):
        resp = _error_response(None, -32700, "Parse error")
        self.assertIsNone(resp["id"])


if __name__ == "__main__":
    unittest.main()
