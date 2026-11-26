"""HAL WebSearch — DuckDuckGo Instant Answers + web search tool.

Zero-dependency web search using the DuckDuckGo Instant Answers API.
No API key required. Results are structured and truncated to a safe
context size for inclusion in LLM prompts.

Architecture:
    WebSearchTool — the callable tool object
    WebSearchResult — structured result dataclass
    search(query) — module-level convenience function

OpenAI tool schema is exposed via WebSearchTool.to_openai_schema() so any
HAL provider that supports function/tool calling can offer web search to
the model without additional wiring.

Usage:
    from lib.hal.websearch import WebSearchTool

    tool = WebSearchTool()
    result = tool.run("latest Anthropic Claude model")
    print(result.summary)           # human-readable summary
    print(result.to_prompt_text())  # formatted for LLM context injection

    schema = WebSearchTool.to_openai_schema()  # dict for tool_choice
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

_DDGQ_URL = "https://api.duckduckgo.com/"
_DEFAULT_TIMEOUT_S = 8
_MAX_TOPICS = 5
_MAX_ABSTRACT_CHARS = 800


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class WebSearchResult:
    """Structured result from a web search query.

    Attributes:
        query:          The original query string.
        abstract:       DuckDuckGo abstract text (may be empty).
        abstract_url:   Source URL for the abstract.
        topics:         Up to _MAX_TOPICS related topic snippets.
        definition:     Short definition if available.
        error:          Set when the search could not be completed.
    """
    query: str
    abstract: str = ""
    abstract_url: str = ""
    topics: list[str] = field(default_factory=list)
    definition: str = ""
    error: Optional[str] = None

    @property
    def summary(self) -> str:
        """One-paragraph summary for display or logging."""
        if self.error:
            return f"Search failed for '{self.query}': {self.error}"
        parts: list[str] = []
        if self.abstract:
            parts.append(self.abstract[:_MAX_ABSTRACT_CHARS])
        if self.definition:
            parts.append(f"Definition: {self.definition}")
        for t in self.topics[:3]:
            parts.append(f"- {t}")
        return "\n".join(parts) if parts else f"No results for '{self.query}'."

    def to_prompt_text(self) -> str:
        """Format for injection into an LLM system or user message."""
        if self.error:
            return f"[WebSearch error: {self.error}]"
        lines: list[str] = [f"[WebSearch: {self.query}]"]
        if self.abstract:
            lines.append(self.abstract[:_MAX_ABSTRACT_CHARS])
        if self.abstract_url:
            lines.append(f"Source: {self.abstract_url}")
        for t in self.topics[:_MAX_TOPICS]:
            lines.append(f"  • {t}")
        if not self.abstract and not self.topics:
            lines.append("(no results)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "abstract": self.abstract,
            "abstract_url": self.abstract_url,
            "topics": self.topics,
            "definition": self.definition,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

class WebSearchTool:
    """DuckDuckGo Instant Answers web search tool.

    Designed to be called by HAL's tool executor or directly from
    conversation pipelines that support function/tool calling.

    The HTTP call is isolated in _fetch() so tests can patch it with
    `unittest.mock.patch("lib.hal.websearch.WebSearchTool._fetch")`.
    """

    name = "websearch"
    description = "Search the web using DuckDuckGo. Returns a summary of results."

    def __init__(self, timeout_s: int = _DEFAULT_TIMEOUT_S) -> None:
        self._timeout = timeout_s

    def run(self, query: str) -> WebSearchResult:
        """Execute a web search and return a structured result.

        Args:
            query: Search query string. Must be non-empty.

        Returns:
            WebSearchResult with abstract, topics, and source URL.
            On network/parse errors, returns a result with .error set.
        """
        if not query or not query.strip():
            return WebSearchResult(query=query, error="empty query")

        params = urllib.parse.urlencode({
            "q": query.strip(),
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })
        url = f"{_DDGQ_URL}?{params}"

        try:
            raw = self._fetch(url)
        except urllib.error.URLError as e:
            return WebSearchResult(query=query, error=f"URLError: {e.reason}")
        except (TimeoutError, OSError) as e:
            return WebSearchResult(query=query, error=f"{type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            return WebSearchResult(query=query, error=f"Unexpected: {type(e).__name__}: {e}")

        return self._parse(query, raw)

    def _fetch(self, url: str) -> dict:
        """Perform the HTTP GET and return parsed JSON. Isolated for testing."""
        req = urllib.request.Request(url, headers={"User-Agent": "HAL/1.0 WebSearch"})
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _parse(query: str, data: dict) -> WebSearchResult:
        """Parse the DuckDuckGo API JSON response."""
        abstract = data.get("AbstractText", "").strip()
        abstract_url = data.get("AbstractURL", "").strip()
        definition = data.get("Definition", "").strip()

        raw_topics = data.get("RelatedTopics", [])
        topics: list[str] = []
        for item in raw_topics:
            if isinstance(item, dict):
                text = item.get("Text", "").strip()
                if text:
                    topics.append(text[:200])
            if len(topics) >= _MAX_TOPICS:
                break

        return WebSearchResult(
            query=query,
            abstract=abstract,
            abstract_url=abstract_url,
            topics=topics,
            definition=definition,
        )

    @staticmethod
    def to_openai_schema() -> dict:
        """Return the OpenAI function-calling schema for this tool.

        Compatible with Anthropic tool_use, OpenAI function calling,
        and any provider that implements the OpenAI tool spec.
        """
        return {
            "type": "function",
            "function": {
                "name": WebSearchTool.name,
                "description": WebSearchTool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up on the web.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def search(query: str, timeout_s: int = _DEFAULT_TIMEOUT_S) -> WebSearchResult:
    """One-liner web search. Creates a tool instance, runs, returns result."""
    return WebSearchTool(timeout_s=timeout_s).run(query)
