"""HAL Semantic Response Cache — retrieve on meaning, not string equality.

Replaces SHA256-based exact-match caching with cosine similarity over
text embeddings, so "What is Python?" and "What's python?" can hit the
same cached answer instead of two independent LLM calls.

Design:
  * Embedder is injectable (dependency inversion). The default resolver
    tries ``embeddings.EmbeddingEngine`` (the GRIP-vendored engine) and
    falls back to a pure-stdlib hashed bag-of-words embedder, so the
    cache is always functional even in minimal HAL installs.
  * Cosine threshold (default 0.92) controls the precision/recall
    tradeoff. Below threshold -> miss. Tunable per instance.
  * Scope is per (model, prompt-embedding). A cache entry put under
    model A is never returned to a lookup under model B, even if the
    prompts are semantically identical — different models have
    different response characteristics and conflating them corrupts the
    measurement in Phase γ.
  * Thread-safe around the entry list (same contract as ``ResponseCache``).

Feature flag:
  * ``HAL_SEMANTIC_CACHE=1`` — consumers opt in at wire-up time. The
    module itself has no env coupling; the flag lives in the caller so
    this class can be used deterministically in tests.

Usage:

    from lib.hal.semantic_cache import SemanticCache

    cache = SemanticCache(ttl_seconds=300, similarity_threshold=0.92)
    cached = cache.get("groq/llama", "What is Python?")
    if cached:
        return cached
    response = registry.chat(...)
    cache.put("groq/llama", "What is Python?", response.text)

    # "What's Python?" now hits when similarity >= 0.92.
"""
from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol


class Embedder(Protocol):
    """Interface for any embedding backend.

    Implementations must return L2-normalised float vectors of uniform
    dimension across calls. Cosine similarity is computed as a dot
    product under that assumption, so non-normalised backends will
    produce incorrect scores.
    """

    def embed(self, text: str) -> list[float]:
        ...


class HashedBagOfWordsEmbedder:
    """Pure-stdlib fallback embedder.

    Tokenises on ``\\w+`` boundaries, lowercase, hashes each token into
    one of ``dim`` buckets via the stdlib ``hash``, accumulates counts
    then L2-normalises. Not semantic in the Word2Vec/sentence-transformers
    sense — but it does catch exact-token overlap and survives small
    typos when combined with character n-gram augmentation (see
    :attr:`char_ngram_size`).
    """

    _TOKEN_RE = re.compile(r"\w+")

    def __init__(self, dim: int = 256, char_ngram_size: int = 0):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.char_ngram_size = char_ngram_size

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in self._iter_features(text):
            idx = hash(token) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec  # all-zero input stays all-zero; cosine with anything = 0
        return [v / norm for v in vec]

    def _iter_features(self, text: str):
        lowered = text.lower()
        yield from self._TOKEN_RE.findall(lowered)
        if self.char_ngram_size > 0:
            n = self.char_ngram_size
            for i in range(len(lowered) - n + 1):
                yield lowered[i:i + n]


def _default_embedder() -> Embedder:
    """Resolve the best available embedder, falling back to stdlib."""
    try:
        from embeddings import EmbeddingEngine  # type: ignore[import-not-found]
        engine = EmbeddingEngine()
        return _EngineAdapter(engine)
    except Exception:
        return HashedBagOfWordsEmbedder(dim=256)


class _EngineAdapter:
    """Adapter that coerces the GRIP EmbeddingEngine into the Embedder protocol.

    Normalises the returned vector so the cache can treat cosine as a dot
    product regardless of the backend's convention.

    Falls back permanently to HashedBagOfWordsEmbedder on the first Python-level
    failure of the engine. SIGSEGV from sqlite-lembed CANNOT be caught here
    (it kills the process); that class of failure is prevented upstream by
    ``_ensure_lembed_registered`` in ``substrate.py``. This guard covers the
    softer failures — apsw errors, missing modules, runtime exceptions in
    third-party embedding backends, and malformed return values (None, empty,
    non-numeric) — so a single bad call doesn't turn every subsequent cache
    lookup into an error cascade.
    """

    def __init__(self, engine):
        self._engine = engine
        self._fallback: Embedder | None = None

    def _activate_fallback(self, reason: str) -> list[float]:
        import logging
        logging.getLogger(__name__).warning(
            "EmbeddingEngine.embed_single %s; switching permanently "
            "to HashedBagOfWordsEmbedder for this process",
            reason,
        )
        self._fallback = HashedBagOfWordsEmbedder(dim=256)
        return self._fallback

    def embed(self, text: str) -> list[float]:
        if self._fallback is not None:
            return self._fallback.embed(text)
        try:
            raw = self._engine.embed_single(text)
        except Exception as exc:
            return self._activate_fallback(f"failed ({exc})").embed(text)
        if raw is None:
            return self._activate_fallback("returned None").embed(text)
        try:
            vec = [float(v) for v in raw]
        except (TypeError, ValueError) as exc:
            return self._activate_fallback(f"returned non-numeric vector ({exc})").embed(text)
        if not vec:
            return self._activate_fallback("returned empty vector").embed(text)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two L2-normalised vectors (pure dot product)."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


@dataclass
class SemanticCacheEntry:
    prompt: str
    response: str
    embedding: list[float]
    created_at: float
    model: str
    hit_count: int = 0


@dataclass
class CacheLookup:
    """Returned by ``peek`` — diagnostic data for the hypothesis harness."""

    hit: bool
    best_similarity: float
    best_prompt: str | None
    entries_considered: int


class SemanticCache:
    """Embedding-based response cache with TTL + LRU-style eviction.

    The public shape matches ``ResponseCache`` (``get``/``put``/
    ``invalidate``/``clear``/``stats``) so call sites can swap between
    them based on ``HAL_SEMANTIC_CACHE``.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = 300,
        max_entries: int = 100,
        similarity_threshold: float = 0.92,
        embedder: Embedder | None = None,
        clock: Callable[[], float] = time.time,
    ):
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._ttl = ttl_seconds
        self._max = max_entries
        self._threshold = similarity_threshold
        self._embedder: Embedder = embedder or _default_embedder()
        self._clock = clock
        self._entries: list[SemanticCacheEntry] = []
        self._lock = threading.Lock()
        self._total_hits = 0
        self._total_misses = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def get(self, model: str, prompt: str) -> str | None:
        hit = self.peek(model, prompt)
        return hit[0] if hit else None

    def peek(self, model: str, prompt: str) -> tuple[str, CacheLookup] | None:
        """Like :meth:`get` but also returns the lookup diagnostic tuple."""
        query_vec = self._embedder.embed(prompt)
        now = self._clock()
        with self._lock:
            best_idx = -1
            best_sim = -1.0
            considered = 0
            for i, entry in enumerate(self._entries):
                if entry.model != model:
                    continue
                if now - entry.created_at > self._ttl:
                    continue
                considered += 1
                sim = cosine_similarity(query_vec, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            if best_idx >= 0 and best_sim >= self._threshold:
                entry = self._entries[best_idx]
                entry.hit_count += 1
                self._total_hits += 1
                return entry.response, CacheLookup(
                    hit=True,
                    best_similarity=best_sim,
                    best_prompt=entry.prompt,
                    entries_considered=considered,
                )
            self._total_misses += 1
            return None, CacheLookup(
                hit=False,
                best_similarity=max(best_sim, 0.0),
                best_prompt=self._entries[best_idx].prompt if best_idx >= 0 else None,
                entries_considered=considered,
            )

    def put(self, model: str, prompt: str, response: str) -> None:
        vec = self._embedder.embed(prompt)
        entry = SemanticCacheEntry(
            prompt=prompt,
            response=response,
            embedding=vec,
            created_at=self._clock(),
            model=model,
        )
        with self._lock:
            if len(self._entries) >= self._max:
                self._entries.pop(0)  # evict oldest (FIFO; entries appended)
            self._entries.append(entry)

    def invalidate(self, model: str, prompt: str) -> bool:
        """Remove every entry whose prompt matches ``prompt`` exactly under ``model``.

        Exact-match semantics mirror ``ResponseCache.invalidate``; callers who
        want semantic invalidation should drive it through ``clear`` or via
        their own similarity-scan.
        """
        with self._lock:
            before = len(self._entries)
            self._entries = [
                e for e in self._entries
                if not (e.model == model and e.prompt == prompt)
            ]
            return len(self._entries) != before

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._total_hits = 0
            self._total_misses = 0
            return count

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            total = self._total_hits + self._total_misses
            hit_rate = self._total_hits / total if total > 0 else 0.0
            return {
                "entries": len(self._entries),
                "max_entries": self._max,
                "ttl_seconds": self._ttl,
                "similarity_threshold": self._threshold,
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "hit_rate": hit_rate,
                "embedder": type(self._embedder).__name__,
            }
