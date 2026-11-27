"""HAL Semantic Response Cache -- reuse an answer only for the same request (HAL#965).

A stored answer is returned for a new prompt in two cases:

  1. Both prompts are the same text, character for character (``lib/hal/prompt_equivalence.py``).
     Nothing is ignored, not even whitespace at the ends or how Unicode encodes a character,
     because every rule that ignored something merged two different requests. No model is
     involved.
  2. With a verifier (``lib/hal/cache_verifier.py``), a prompt in other words can hit too.
     The stored prompt closest in embedding space is a candidate when its cosine clears
     ``candidate_floor``. It must then pass a deterministic veto (numbers, negations and word
     order must agree, and the prompts must not differ only invisibly) and the verifier's
     local judge must answer YES.

Why an embedding never decides: this cache first matched on cosine similarity alone, at a
0.92 threshold. HAL#965 measured it on 20 near-miss pairs, prompts that share words but need
different answers. nomic-embed-text cleared the threshold on 6 of 20, keep against discard at
0.973 among them, and the hashed bag-of-words fallback on 2 or 3 of 20, two at 1.000, because
swapping words does not change a bag of words. Embeddings place opposites close together, so
no threshold separates them. With the design above and the qwen3:8b judge, the same
measurement gave 18 of 20 and, on a held-out set from another model, 17 of 30 paraphrase hits
(17 and 12 with the prompts swapped), with no wrong answer on 50 near-miss pairs or 100
unrelated pairs.

Design:
  * Same-text hits are O(1): entries are keyed by (model, canonical form).
  * Scope is per model: an answer stored under model A is never returned under model B.
  * A blank prompt is never stored and never hits.
  * With a verifier, a lookup embeds the prompt only when a live entry for the model has an
    embedding, so a miss in an empty cache costs no Ollama call.
  * The verifier runs outside the lock, and every verifier failure is a miss.
  * TTL expiry and FIFO eviction at ``max_entries``; thread-safe, the same contract as
    ``ResponseCache``.

Feature flags (read by ``lib/hal/conversation.py``, never here):
  * ``HAL_SEMANTIC_CACHE=1`` turns the cache on.
  * ``HAL_SEMANTIC_CACHE_JUDGE=<ollama model>`` adds the verifier (measured: ``qwen3:8b``).

Usage:

    from lib.hal.semantic_cache import SemanticCache

    cache = SemanticCache(ttl_seconds=300)
    cache.put("groq/llama", "What is Python?", answer)
    cache.get("groq/llama", "What is Python?")     # -> answer: the same text
    cache.get("groq/llama", "What is Python? ")    # -> None without a verifier: other text
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Protocol

from lib.hal.prompt_equivalence import canonical_form, structural_difference


class Verifier(Protocol):
    """What the cache needs from a verifier; ``lib/hal/cache_verifier.OllamaVerifier`` is one."""

    name: str

    def embed(self, text: str) -> list[float] | None: ...

    def same_request(self, a: str, b: str) -> bool: ...


@dataclass
class SemanticCacheEntry:
    prompt: str
    response: str
    created_at: float
    model: str
    canon: str
    embedding: list[float] | None = None
    hit_count: int = 0


@dataclass
class CacheLookup:
    """Returned by ``peek``: what the lookup found, for diagnostics."""

    hit: bool
    match: str | None  # "exact", "verified", or None on a miss
    matched_prompt: str | None


class SemanticCache:
    """Response cache that answers only the same request again (HAL#965).

    The public shape matches ``ResponseCache`` (``get``/``put``/``invalidate``/``clear``/
    ``stats``) so call sites can swap between them based on ``HAL_SEMANTIC_CACHE``.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = 300,
        max_entries: int = 100,
        verifier: Verifier | None = None,
        candidate_floor: float = 0.70,
        clock: Callable[[], float] = time.time,
    ):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if not 0.0 < candidate_floor <= 1.0:
            raise ValueError("candidate_floor must be in (0, 1]")
        self._ttl = ttl_seconds
        self._max = max_entries
        self._verifier = verifier
        self._floor = candidate_floor
        self._clock = clock
        self._entries: OrderedDict[tuple[str, str], SemanticCacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._total_hits = 0
        self._total_misses = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def get(self, model: str, prompt: str) -> str | None:
        response, _lookup = self.peek(model, prompt)
        return response

    def peek(self, model: str, prompt: str) -> tuple[str | None, CacheLookup]:
        """Like :meth:`get` but also returns the lookup diagnostic."""
        canon = canonical_form(prompt)
        if not canon:
            return self._miss()
        now = self._clock()
        with self._lock:
            entry = self._live_entry((model, canon), now)
            if entry is not None:
                return self._hit(entry, "exact")
        verified = self._verified_entry(model, prompt, now)
        if verified is not None:
            with self._lock:
                # The judge call ran without the lock: serve the answer only if that same entry
                # is still live, so an invalidate(), clear(), eviction or expiry meanwhile wins.
                if self._live_entry((model, verified.canon), self._clock()) is verified:
                    return self._hit(verified, "verified")
        return self._miss()

    def put(self, model: str, prompt: str, response: str) -> None:
        canon = canonical_form(prompt)
        if not canon:
            return
        embedding = self._verifier.embed(prompt) if self._verifier else None
        entry = SemanticCacheEntry(prompt=prompt, response=response, created_at=self._clock(),
                                   model=model, canon=canon, embedding=embedding)
        with self._lock:
            self._entries.pop((model, canon), None)
            if len(self._entries) >= self._max:
                self._entries.popitem(last=False)  # evict the oldest
            self._entries[(model, canon)] = entry

    def invalidate(self, model: str, prompt: str) -> bool:
        """Remove the stored answer for ``prompt`` under ``model``.

        The match is the same text as :meth:`peek` uses. Returns True when an entry was removed.
        """
        with self._lock:
            return self._entries.pop((model, canonical_form(prompt)), None) is not None

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._total_hits = 0
            self._total_misses = 0
            return count

    # ------------------------------------------------------------------
    # Lookup steps
    # ------------------------------------------------------------------
    def _live_entry(self, key: tuple[str, str], now: float) -> SemanticCacheEntry | None:
        """The entry under ``key``, dropped first if its TTL has passed. Caller holds the lock."""
        entry = self._entries.get(key)
        if entry is not None and now - entry.created_at > self._ttl:
            del self._entries[key]
            return None
        return entry

    def _verified_entry(self, model: str, prompt: str, now: float) -> SemanticCacheEntry | None:
        """The stored entry the verifier confirms as the same request, or None. Takes the lock."""
        if self._verifier is None:
            return None
        with self._lock:
            if not self._has_candidates(model, now):
                return None  # nothing could match: spare the embedding call
        query = self._verifier.embed(prompt)
        if query is None:
            return None
        with self._lock:
            candidate = self._closest_entry(model, query, now)
        if candidate is None or structural_difference(prompt, candidate.prompt):
            return None
        return candidate if self._verifier.same_request(prompt, candidate.prompt) else None

    def _has_candidates(self, model: str, now: float) -> bool:
        """True when a live entry for ``model`` has an embedding. Caller holds the lock."""
        return any(e.model == model and e.embedding is not None and now - e.created_at <= self._ttl
                   for e in self._entries.values())

    def _closest_entry(self, model: str, query: list[float], now: float) -> SemanticCacheEntry | None:
        """The live entry whose embedding is closest to ``query`` above the floor. Caller holds the lock."""
        best, best_score = None, self._floor
        for entry in self._entries.values():
            if entry.model != model or now - entry.created_at > self._ttl:
                continue
            if entry.embedding is None or len(entry.embedding) != len(query):
                continue  # no vector, or one from a model of another size: never a candidate
            score = sum(x * y for x, y in zip(query, entry.embedding))
            if score >= best_score:
                best, best_score = entry, score
        return best

    def _hit(self, entry: SemanticCacheEntry, match: str) -> tuple[str, CacheLookup]:
        """Count a hit and build its result. Caller holds the lock."""
        entry.hit_count += 1
        self._total_hits += 1
        return entry.response, CacheLookup(hit=True, match=match, matched_prompt=entry.prompt)

    def _miss(self) -> tuple[None, CacheLookup]:
        """Count a miss and build its result. Takes the lock."""
        with self._lock:
            self._total_misses += 1
        return None, CacheLookup(hit=False, match=None, matched_prompt=None)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            total = self._total_hits + self._total_misses
            return {
                "entries": len(self._entries),
                "max_entries": self._max,
                "ttl_seconds": self._ttl,
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "hit_rate": self._total_hits / total if total > 0 else 0.0,
                "matcher": "prompt_equivalence",
                "verifier": self._verifier.name if self._verifier else None,
                "candidate_floor": self._floor,
            }
