"""KONO Bridge — semantic memory recall for HAL's conversation layer.

Bridges HAL to its own vendored KONO substrate (lib/hal/kono/) so every
LLM query, regardless of provider, gets query-relevant context. The
Sentinel pattern: memory that follows every conversation.

HAL used to read GRIP's shared store at ~/.claude/embeddings.db. Since
the 2026-04-15 vendor port, HAL owns its copy at ~/.hal/data/embeddings.db.
One-time seeded from GRIP, then diverges. See plans/kono-standalone-vendor.md.

O(1) coordinator entry point: one call, one formatted string back.
Graceful degradation: returns empty string if KONO unavailable.
"""
import logging
import threading
from typing import Optional

from lib.hal.kono.constants import DB_PATH

logger = logging.getLogger(__name__)

_RECALL_TIMEOUT_S = 1.5
_MAX_RESULTS = 3
_MAX_CONTEXT_CHARS = 1500

# Module-level cache: cleared per conversation turn to avoid stale results
_cache_lock = threading.Lock()
_cache: dict[str, str] = {}


def recall_for_query(
    query: str,
    limit: int = _MAX_RESULTS,
    timeout_s: float = _RECALL_TIMEOUT_S,
) -> str:
    """Recall KONO memories relevant to a query. O(1) from caller.

    Args:
        query: Natural language query (the user's message).
        limit: Max memories to return (default 3).
        timeout_s: Hard timeout for KONO query (default 1.5s).

    Returns:
        Formatted context string with relevant memories, or "" if
        KONO unavailable or no relevant memories found.
    """
    if not query or not query.strip():
        return ""

    # Short-circuit: no KONO installation
    if not _kono_available():
        return ""

    # Check cache (same query in same turn = same result)
    cache_key = f"{query[:200]}:{limit}"
    with _cache_lock:
        if cache_key in _cache:
            return _cache[cache_key]

    # Run recall with timeout protection
    result = _recall_with_timeout(query, limit, timeout_s)

    with _cache_lock:
        _cache[cache_key] = result

    return result


def clear_cache() -> None:
    """Clear the per-turn cache. Call at the start of each conversation turn."""
    with _cache_lock:
        _cache.clear()


def _kono_available() -> bool:
    """Check if HAL's vendored KONO substrate + DB are present."""
    return DB_PATH.exists()


def _recall_with_timeout(query: str, limit: int, timeout_s: float) -> str:
    """Run KONO recall in a thread with timeout protection."""
    result_holder: list[str] = [""]
    error_holder: list[Optional[Exception]] = [None]

    def _do_recall():
        try:
            result_holder[0] = _execute_recall(query, limit)
        except Exception as e:
            error_holder[0] = e

    thread = threading.Thread(target=_do_recall, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        logger.debug("KONO recall timed out after %.1fs", timeout_s)
        return ""

    if error_holder[0]:
        logger.debug("KONO recall error: %s", error_holder[0])
        return ""

    return result_holder[0]


def _execute_recall(query: str, limit: int) -> str:
    """Run semantic recall against HAL's vendored KONO substrate."""
    from lib.hal.kono.substrate import MemorySubstrate

    substrate = MemorySubstrate()
    substrate.connect()

    results = substrate.recall(
        query=query,
        limit=limit,
        min_fitness=0.1,
        expand_spatial=False,  # Speed: skip spatial expansion for per-turn recall
    )

    if not results:
        return ""

    return _format_results(results)


def _format_results(results: list[dict]) -> str:
    """Format KONO recall results as a compact context section.

    Target: ~375 tokens max. Each memory gets 2 lines: content + metadata.
    """
    entries = []

    for mem in results:
        content = mem.get("content", mem.get("text", "")).strip()
        if not content:
            continue

        # Truncate individual memories to keep total under budget
        max_per_item = _MAX_CONTEXT_CHARS // max(len(results), 1)
        if len(content) > max_per_item:
            content = content[:max_per_item].rsplit(" ", 1)[0] + "..."

        fitness = mem.get("fitness_score", mem.get("similarity", 0))
        permanent = mem.get("is_permanent", False)

        meta_parts = [f"fitness={fitness:.2f}"]
        if permanent:
            meta_parts.append("permanent")

        entries.append(f"  {len(entries) + 1}. {content}\n     [{', '.join(meta_parts)}]")

    if not entries:
        return ""

    result = "[KONO Memory — relevant knowledge from past sessions]\n" + "\n".join(entries)
    return result[:_MAX_CONTEXT_CHARS]
