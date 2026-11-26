"""HAL Conversation Search — find past conversations by content or topic.

Searches saved session replays for matching content using keyword matching
with TF-based relevance scoring.

Usage:
    from lib.hal.conversation_search import ConversationSearch

    search = ConversationSearch()
    results = search.find("authentication bug fix")
    for result in results:
        print(f"{result.session_id}: {result.snippet}")
"""
import json
import os
import re
from dataclasses import dataclass

from lib.hal.home import HAL_HOME as _HAL_HOME

_REPLAYS_DIR = os.path.join(_HAL_HOME, "state", "hal", "replays")


@dataclass(frozen=True)
class SearchResult:
    """A single search result from conversation history."""
    session_id: str
    snippet: str
    event_type: str
    relevance: float  # 0.0 to 1.0
    timestamp: float


class ConversationSearch:
    """Search across saved session replays."""

    def find(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search all saved replays for matching content."""
        query_terms = _tokenise(query)
        if not query_terms:
            return []

        results: list[SearchResult] = []
        for replay_file in _list_replays():
            results.extend(_search_replay(replay_file, query_terms))

        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:max_results]

    def recent(self, limit: int = 10) -> list[str]:
        """List recent session IDs, newest first."""
        replays = _list_replays()
        replays.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return [
            os.path.splitext(os.path.basename(p))[0]
            for p in replays[:limit]
        ]


def _list_replays() -> list[str]:
    """List all replay JSON files."""
    if not os.path.isdir(_REPLAYS_DIR):
        return []
    return [
        os.path.join(_REPLAYS_DIR, f)
        for f in os.listdir(_REPLAYS_DIR)
        if f.endswith(".json")
    ]


def _search_replay(path: str, query_terms: set[str]) -> list[SearchResult]:
    """Search a single replay file for matching events."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    session_id = data.get("session_id", os.path.basename(path))
    results: list[SearchResult] = []

    for event in data.get("events", []):
        content = event.get("content", "")
        if not content:
            continue

        content_terms = _tokenise(content)
        matches = query_terms & content_terms
        if not matches:
            continue

        relevance = len(matches) / len(query_terms)
        results.append(SearchResult(
            session_id=session_id,
            snippet=_build_snippet(content, matches),
            event_type=event.get("event_type", ""),
            relevance=relevance,
            timestamp=event.get("timestamp", 0.0),
        ))

    return results


_TOKEN_RE = re.compile(r"\b[a-z0-9]{3,}\b")


def _tokenise(text: str) -> set[str]:
    """Extract search tokens from text. Lowercase, 3+ char words."""
    return set(_TOKEN_RE.findall(text.lower()))


def _build_snippet(content: str, matches: set[str], max_len: int = 200) -> str:
    """Build a snippet around the first match region."""
    content_lower = content.lower()
    earliest = len(content)
    for term in matches:
        pos = content_lower.find(term)
        if 0 <= pos < earliest:
            earliest = pos

    start = max(0, earliest - 40)
    end = min(len(content), earliest + max_len - 40)
    snippet = content[start:end].strip()

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    return prefix + snippet + suffix
