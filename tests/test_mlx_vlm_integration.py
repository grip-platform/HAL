"""Wave 13: tool_dispatch + code_index integration tests.

Tests the full pipeline integration:
- ToolDispatcher routes CodeLookup/CodeSearch to GRIP's AST code index
- Correct fallback when code_index is unavailable (path not found)
- Registry reports CodeLookup and CodeSearch as available tools
- Missing argument validation returns ToolResult.error, not an exception

Also covers the full mlx-vlm pipeline integration (mocked):
- Discovery → registry → routing → intent → multimodal chat
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.tool_dispatch import (
    ToolDispatcher,
    _load_code_index,
    _tool_code_lookup,
    _tool_code_search,
    _TOOL_REGISTRY,
)


class TestCodeLookupTool(unittest.TestCase):
    """Unit tests for _tool_code_lookup."""

    def test_missing_symbol_returns_error(self):
        result = _tool_code_lookup({})
        self.assertTrue(result.error)
        self.assertFalse(result.blocked)

    def test_missing_code_index_returns_error(self):
        # Evict any cached module so the path check is exercised, not the cache hit
        orig = sys.modules.pop("code_index", None)
        try:
            with patch("lib.hal.tool_dispatch._CODE_INDEX_PATH", "/nonexistent/code_index.py"):
                result = _tool_code_lookup({"symbol": "ProviderRegistry"})
        finally:
            if orig is not None:
                sys.modules["code_index"] = orig
        self.assertTrue(result.error)
        self.assertIn("not found", result.error)

    def test_live_lookup_when_index_available(self):
        # Only runs when a real code_index.py exists at HAL's configured path.
        from lib.hal.tool_dispatch import _CODE_INDEX_PATH
        if not os.path.isfile(_CODE_INDEX_PATH):
            self.skipTest(f"code_index.py not found at {_CODE_INDEX_PATH}")

        result = _tool_code_lookup({"symbol": "ProviderRegistry"})
        # Either finds something or "No symbols found" — both are valid outputs
        self.assertFalse(result.error)

    def test_second_load_returns_cached_module(self):
        """_load_code_index() must return the same object on repeated calls (O(1) cache)."""
        from lib.hal.tool_dispatch import _CODE_INDEX_PATH
        if not os.path.isfile(_CODE_INDEX_PATH):
            self.skipTest(f"code_index.py not found at {_CODE_INDEX_PATH}")
        mod1 = _load_code_index()
        mod2 = _load_code_index()
        self.assertIs(mod1, mod2, "Second call must return the cached module object")

    def test_module_exception_returns_error_not_raise(self):
        """Exceptions from the code_index module must be caught as errors."""
        mock_mod = MagicMock()
        mock_mod.CodeIndex.side_effect = RuntimeError("DB locked")

        with patch("lib.hal.tool_dispatch._load_code_index", return_value=mock_mod):
            result = _tool_code_lookup({"symbol": "Foo"})
        self.assertTrue(result.error)
        self.assertIn("DB locked", result.error)


class TestCodeSearchTool(unittest.TestCase):
    """Unit tests for _tool_code_search."""

    def test_missing_query_returns_error(self):
        result = _tool_code_search({})
        self.assertTrue(result.error)

    def test_missing_code_index_returns_error(self):
        orig = sys.modules.pop("code_index", None)
        try:
            with patch("lib.hal.tool_dispatch._CODE_INDEX_PATH", "/nonexistent/code_index.py"):
                result = _tool_code_search({"query": "webhook delivery"})
        finally:
            if orig is not None:
                sys.modules["code_index"] = orig
        self.assertTrue(result.error)
        self.assertIn("not found", result.error)

    def test_returns_results_from_mock_index(self):
        mock_results = [
            {
                "symbol_type": "function",
                "qualified_name": "lib.hal.tool_dispatch._tool_bash",
                "file_path": "~/.hal/lib/hal/tool_dispatch.py",
                "line_start": 71,
                "line_end": 90,
                "signature": "def _tool_bash(args: dict) -> ToolResult:",
                "decorators": "[]",
                "docstring": "Execute a shell command.",
                "django_type": None,
            }
        ]
        mock_mod = MagicMock()
        mock_mod.CodeIndex.return_value.__enter__ = MagicMock()
        idx_instance = MagicMock()
        idx_instance.search_fts.return_value = mock_results
        mock_mod.CodeIndex.return_value = idx_instance
        mock_mod.format_lookup.return_value = "function  lib.hal.tool_dispatch._tool_bash"

        with patch("lib.hal.tool_dispatch._load_code_index", return_value=mock_mod):
            result = _tool_code_search({"query": "bash tool"})

        self.assertFalse(result.error)
        self.assertIn("_tool_bash", result.output)


class TestToolDispatcherRegistration(unittest.TestCase):
    """CodeLookup and CodeSearch must be registered in the tool registry."""

    def test_code_lookup_registered(self):
        self.assertIn("CodeLookup", _TOOL_REGISTRY)

    def test_code_search_registered(self):
        self.assertIn("CodeSearch", _TOOL_REGISTRY)

    def test_available_tools_includes_code_tools(self):
        dispatcher = ToolDispatcher()
        tools = dispatcher.available_tools()
        self.assertIn("CodeLookup", tools)
        self.assertIn("CodeSearch", tools)


class TestFullPipelineIntegration(unittest.TestCase):
    """Full mlx-vlm pipeline: discovery → registry → routing → intent."""

    def test_vlm_discovery_to_routing(self):
        """When mlx_vlm is discovered, vision queries route to it."""
        from lib.hal.cost_router import CostRouter
        from lib.hal.types import ProviderInfo, ProviderType

        registry = MagicMock()
        vlm = ProviderInfo(
            name="mlx_vlm",
            type=ProviderType.LOCAL,
            base_url="http://localhost:8080",
            healthy=True,
            models=("mlx-community/Qwen2-VL-7B-Instruct-4bit",),
        )
        registry.list_providers.return_value = [vlm]
        registry.routing_config = None

        router = CostRouter(registry)
        decision = router.select("Describe the contents of this screenshot")

        self.assertTrue(decision.model.startswith("mlx_vlm/"))
        self.assertEqual(decision.tier, "free")
        self.assertEqual(decision.estimated_cost_usd, 0.0)

    def test_intent_to_routing_agreement(self):
        """IntentDetector and CostRouter agree on vision queries."""
        from lib.hal.cost_router import CostRouter
        from lib.hal.intent_detector import IntentDetector
        from lib.hal.types import ProviderInfo, ProviderType

        registry = MagicMock()
        vlm = ProviderInfo(
            name="mlx_vlm",
            type=ProviderType.LOCAL,
            base_url="http://localhost:8080",
            healthy=True,
            models=("mlx-community/Qwen2-VL-7B-Instruct-4bit",),
        )
        registry.list_providers.return_value = [vlm]
        registry.routing_config = None

        detector = IntentDetector()
        router = CostRouter(registry)

        vision_queries = [
            "describe this image",
            "OCR the text in document.png",
            "detect objects in this photo",
        ]
        for query in vision_queries:
            intent = detector.classify(query)
            decision = router.select(query)
            self.assertEqual(intent.category, "vision", f"Intent wrong for: {query}")
            self.assertTrue(
                decision.model.startswith("mlx_vlm/"),
                f"Router wrong for: {query}",
            )


if __name__ == "__main__":
    unittest.main()
