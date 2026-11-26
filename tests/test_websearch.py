"""Tests for lib/hal/websearch.py — HAL WebSearch tool.

All tests are hermetic — zero real network calls. The HTTP layer is
isolated in WebSearchTool._fetch() and patched throughout.

Covers:
  1. WebSearchResult — summary, to_prompt_text, to_dict, error state
  2. WebSearchTool._parse — abstract, topics, definition extraction
  3. WebSearchTool.run — success, empty query, URL error, timeout, OSError
  4. WebSearchTool.to_openai_schema — schema structure for tool-calling
  5. module-level search() convenience function
  6. ToolRegistry — websearch is registered as a builtin tool
"""
from __future__ import annotations

import os
import sys
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

_HAL_HOME = os.environ.get("HAL_ROOT", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.websearch import WebSearchResult, WebSearchTool, search


# ---------------------------------------------------------------------------
# Fixtures — sample DuckDuckGo API responses
# ---------------------------------------------------------------------------

_FULL_DDG_RESPONSE = {
    "AbstractText": "Claude is an AI assistant made by Anthropic.",
    "AbstractURL": "https://www.anthropic.com/claude",
    "Definition": "An AI assistant.",
    "RelatedTopics": [
        {"Text": "Claude 3 Opus — most capable model"},
        {"Text": "Claude 3 Sonnet — balanced model"},
        {"Text": "Claude 3 Haiku — fastest model"},
        {"Text": "Anthropic — the company behind Claude"},
        {"Text": "Constitutional AI — Anthropic safety method"},
        {"Text": "Should be truncated at _MAX_TOPICS"},
    ],
}

_EMPTY_DDG_RESPONSE = {
    "AbstractText": "",
    "AbstractURL": "",
    "Definition": "",
    "RelatedTopics": [],
}


# ---------------------------------------------------------------------------
# 1. WebSearchResult
# ---------------------------------------------------------------------------

class TestWebSearchResult:
    def test_summary_contains_abstract(self):
        r = WebSearchResult(query="claude", abstract="Claude is an AI assistant.")
        assert "Claude is an AI assistant" in r.summary

    def test_summary_includes_definition(self):
        r = WebSearchResult(query="claude", definition="A language model.")
        assert "A language model" in r.summary

    def test_summary_includes_topics(self):
        r = WebSearchResult(query="q", topics=["topic A", "topic B"])
        assert "topic A" in r.summary

    def test_summary_error_state(self):
        r = WebSearchResult(query="q", error="network unreachable")
        assert "failed" in r.summary.lower() or "error" in r.summary.lower()
        assert "network unreachable" in r.summary

    def test_summary_empty_returns_no_results_message(self):
        r = WebSearchResult(query="obscure thing xyz")
        assert "No results" in r.summary or r.summary  # at least something

    def test_to_prompt_text_includes_header(self):
        r = WebSearchResult(query="anthropic")
        text = r.to_prompt_text()
        assert "[WebSearch: anthropic]" in text

    def test_to_prompt_text_includes_abstract(self):
        r = WebSearchResult(query="q", abstract="Some abstract text.")
        text = r.to_prompt_text()
        assert "Some abstract text" in text

    def test_to_prompt_text_includes_source_url(self):
        r = WebSearchResult(query="q", abstract_url="https://example.com")
        text = r.to_prompt_text()
        assert "https://example.com" in text

    def test_to_prompt_text_error_is_bracketed(self):
        r = WebSearchResult(query="q", error="timeout")
        text = r.to_prompt_text()
        assert "[WebSearch error:" in text
        assert "timeout" in text

    def test_to_dict_has_all_keys(self):
        r = WebSearchResult(query="q", abstract="A", topics=["T"])
        d = r.to_dict()
        assert {"query", "abstract", "abstract_url", "topics", "definition", "error"} <= set(d.keys())

    def test_to_dict_error_field_is_none_on_success(self):
        r = WebSearchResult(query="q", abstract="ok")
        assert r.to_dict()["error"] is None


# ---------------------------------------------------------------------------
# 2. WebSearchTool._parse
# ---------------------------------------------------------------------------

class TestWebSearchToolParse:
    def test_parse_extracts_abstract(self):
        r = WebSearchTool._parse("claude", _FULL_DDG_RESPONSE)
        assert r.abstract == "Claude is an AI assistant made by Anthropic."

    def test_parse_extracts_abstract_url(self):
        r = WebSearchTool._parse("claude", _FULL_DDG_RESPONSE)
        assert r.abstract_url == "https://www.anthropic.com/claude"

    def test_parse_extracts_definition(self):
        r = WebSearchTool._parse("claude", _FULL_DDG_RESPONSE)
        assert r.definition == "An AI assistant."

    def test_parse_topics_capped_at_max(self):
        r = WebSearchTool._parse("claude", _FULL_DDG_RESPONSE)
        assert len(r.topics) <= 5  # _MAX_TOPICS

    def test_parse_topics_content(self):
        r = WebSearchTool._parse("claude", _FULL_DDG_RESPONSE)
        assert any("Opus" in t for t in r.topics)

    def test_parse_empty_response_produces_empty_fields(self):
        r = WebSearchTool._parse("q", _EMPTY_DDG_RESPONSE)
        assert r.abstract == ""
        assert r.topics == []
        assert r.error is None  # empty is not an error

    def test_parse_skips_non_dict_topic_entries(self):
        data = {"AbstractText": "", "AbstractURL": "", "Definition": "",
                "RelatedTopics": [{"Type": "C", "Topics": [{"Text": "nested"}]}, "string"]}
        r = WebSearchTool._parse("q", data)
        assert isinstance(r.topics, list)

    def test_parse_sets_query_correctly(self):
        r = WebSearchTool._parse("my query", _FULL_DDG_RESPONSE)
        assert r.query == "my query"


# ---------------------------------------------------------------------------
# 3. WebSearchTool.run
# ---------------------------------------------------------------------------

class TestWebSearchToolRun:
    def _mock_tool(self, response_data: dict) -> WebSearchTool:
        tool = WebSearchTool()
        tool._fetch = MagicMock(return_value=response_data)
        return tool

    def test_run_success_returns_result(self):
        tool = self._mock_tool(_FULL_DDG_RESPONSE)
        r = tool.run("claude")
        assert r.abstract == "Claude is an AI assistant made by Anthropic."
        assert r.error is None

    def test_run_calls_fetch_once(self):
        tool = self._mock_tool(_FULL_DDG_RESPONSE)
        tool.run("test")
        tool._fetch.assert_called_once()

    def test_run_url_contains_query(self):
        tool = self._mock_tool(_EMPTY_DDG_RESPONSE)
        tool.run("anthropic models")
        url = tool._fetch.call_args[0][0]
        assert "anthropic+models" in url or "anthropic%20models" in url

    def test_run_empty_query_returns_error(self):
        tool = WebSearchTool()
        r = tool.run("")
        assert r.error == "empty query"

    def test_run_whitespace_query_returns_error(self):
        tool = WebSearchTool()
        r = tool.run("   ")
        assert r.error == "empty query"

    def test_run_url_error_returns_error_result(self):
        tool = WebSearchTool()
        tool._fetch = MagicMock(
            side_effect=urllib.error.URLError("connection refused")
        )
        r = tool.run("test")
        assert r.error is not None
        assert "URLError" in r.error

    def test_run_timeout_returns_error_result(self):
        tool = WebSearchTool()
        tool._fetch = MagicMock(side_effect=TimeoutError("timed out"))
        r = tool.run("test")
        assert r.error is not None
        assert "TimeoutError" in r.error

    def test_run_os_error_returns_error_result(self):
        tool = WebSearchTool()
        tool._fetch = MagicMock(side_effect=OSError("socket hang up"))
        r = tool.run("test")
        assert r.error is not None
        assert "OSError" in r.error

    def test_run_never_raises_on_network_errors(self):
        """run() must NEVER propagate network errors — always returns WebSearchResult."""
        for exc in [urllib.error.URLError("x"), TimeoutError("x"), OSError("x")]:
            tool = WebSearchTool()
            tool._fetch = MagicMock(side_effect=exc)
            result = tool.run("q")  # must not raise
            assert isinstance(result, WebSearchResult)
            assert result.error is not None


# ---------------------------------------------------------------------------
# 4. OpenAI schema
# ---------------------------------------------------------------------------

class TestOpenAISchema:
    def test_schema_has_type_function(self):
        schema = WebSearchTool.to_openai_schema()
        assert schema["type"] == "function"

    def test_schema_name_is_websearch(self):
        schema = WebSearchTool.to_openai_schema()
        assert schema["function"]["name"] == "websearch"

    def test_schema_has_description(self):
        schema = WebSearchTool.to_openai_schema()
        assert len(schema["function"]["description"]) > 10

    def test_schema_parameters_type_is_object(self):
        schema = WebSearchTool.to_openai_schema()
        assert schema["function"]["parameters"]["type"] == "object"

    def test_schema_query_is_required(self):
        schema = WebSearchTool.to_openai_schema()
        assert "query" in schema["function"]["parameters"]["required"]

    def test_schema_query_type_is_string(self):
        schema = WebSearchTool.to_openai_schema()
        props = schema["function"]["parameters"]["properties"]
        assert props["query"]["type"] == "string"


# ---------------------------------------------------------------------------
# 5. Module-level search() convenience
# ---------------------------------------------------------------------------

class TestSearchConvenience:
    def test_search_returns_result(self):
        with patch("lib.hal.websearch.WebSearchTool._fetch") as mock_fetch:
            mock_fetch.return_value = _FULL_DDG_RESPONSE
            r = search("claude")
        assert r.abstract == "Claude is an AI assistant made by Anthropic."

    def test_search_passes_query(self):
        with patch("lib.hal.websearch.WebSearchTool._fetch") as mock_fetch:
            mock_fetch.return_value = _EMPTY_DDG_RESPONSE
            r = search("my test query")
        assert r.query == "my test query"


# ---------------------------------------------------------------------------
# 6. ToolRegistry integration
# ---------------------------------------------------------------------------

class TestToolRegistryIntegration:
    def test_websearch_in_builtin_tools(self):
        from lib.hal.tool_registry import ToolRegistry
        registry = ToolRegistry()
        tool = registry.get("websearch")
        assert tool is not None

    def test_websearch_tool_is_enabled(self):
        from lib.hal.tool_registry import ToolRegistry
        registry = ToolRegistry()
        tool = registry.get("websearch")
        assert tool.enabled is True

    def test_websearch_in_list_tools(self):
        from lib.hal.tool_registry import ToolRegistry
        registry = ToolRegistry()
        names = [t.name for t in registry.list_tools()]
        assert "websearch" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
