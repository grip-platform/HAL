"""HAL Response Cache — avoid redundant LLM calls for identical queries.

Caches LLM responses with TTL-based expiry. Saves cost when the same
query is sent multiple times (common in workflows, agent loops, and
retry scenarios).

Uses SHA256 of (model + prompt) as cache key. Thread-safe.

Usage:
    from lib.hal.response_cache import ResponseCache

    cache = ResponseCache(ttl_seconds=300)
    cached = cache.get("groq/llama", "What is Python?")
    if cached:
        return cached  # Free!
    response = registry.chat(...)
    cache.put("groq/llama", "What is Python?", response.text)
"""
import hashlib
import threading
import time
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """A cached response with expiry."""
    response: str
    created_at: float
    model: str
    hit_count: int = 0


class ResponseCache:
    """Thread-safe LRU-style response cache with TTL expiry."""

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 100):
        self._ttl = ttl_seconds
        self._max = max_entries
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, model: str, prompt: str) -> str | None:
        """Get cached response, or None if miss/expired."""
        key = _cache_key(model, prompt)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.time() - entry.created_at > self._ttl:
                del self._store[key]
                return None
            entry.hit_count += 1
            return entry.response

    def put(self, model: str, prompt: str, response: str) -> None:
        """Cache a response."""
        key = _cache_key(model, prompt)
        with self._lock:
            if len(self._store) >= self._max:
                self._evict_oldest()
            self._store[key] = CacheEntry(
                response=response, created_at=time.time(), model=model,
            )

    def invalidate(self, model: str, prompt: str) -> bool:
        """Remove a specific cache entry. Returns True if removed."""
        key = _cache_key(model, prompt)
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> int:
        """Clear all cache entries. Returns count cleared."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def stats(self) -> dict:
        """Cache statistics."""
        with self._lock:
            total_hits = sum(e.hit_count for e in self._store.values())
            return {
                "entries": len(self._store),
                "max_entries": self._max,
                "ttl_seconds": self._ttl,
                "total_hits": total_hits,
            }

    def _evict_oldest(self) -> None:
        """Remove the oldest entry. Must be called with lock held."""
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
        del self._store[oldest_key]


def _cache_key(model: str, prompt: str) -> str:
    """Generate cache key from model + prompt."""
    raw = f"{model}:{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
