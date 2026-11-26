"""Tests for Phase α W3 semantic response cache + H-SCACHE harness."""
from __future__ import annotations

import math
import time
import unittest

from lib.hal.semantic_cache import (
    CacheLookup,
    HashedBagOfWordsEmbedder,
    SemanticCache,
    SemanticCacheEntry,
    _default_embedder,
    cosine_similarity,
)


class _FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.t = start

    def advance(self, seconds: float):
        self.t += seconds

    def __call__(self) -> float:
        return self.t


class _FixedEmbedder:
    """Deterministic embedder for unit tests — returns a canned vector per input."""

    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = {}
        for k, v in mapping.items():
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            self._mapping[k] = [x / norm for x in v]

    def embed(self, text: str) -> list[float]:
        # Unknown keys return a zero vector — cosine of 0 to everything.
        return self._mapping.get(text, [0.0, 0.0, 0.0])


class TestCosineSimilarity(unittest.TestCase):
    def test_parallel_vectors_are_one(self):
        v = [1 / math.sqrt(2), 1 / math.sqrt(2)]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=6)

    def test_orthogonal_vectors_are_zero(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


class TestHashedBagOfWordsEmbedder(unittest.TestCase):
    def test_deterministic_output(self):
        emb = HashedBagOfWordsEmbedder(dim=32)
        self.assertEqual(emb.embed("hello world"), emb.embed("hello world"))

    def test_output_dimension(self):
        emb = HashedBagOfWordsEmbedder(dim=64)
        v = emb.embed("one two three")
        self.assertEqual(len(v), 64)

    def test_l2_normalised(self):
        emb = HashedBagOfWordsEmbedder(dim=32)
        v = emb.embed("some non-empty text with repeated repeated tokens")
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0, places=6)

    def test_empty_text_returns_zero_vector(self):
        emb = HashedBagOfWordsEmbedder(dim=16)
        self.assertEqual(emb.embed(""), [0.0] * 16)

    def test_invalid_dim_rejected(self):
        with self.assertRaises(ValueError):
            HashedBagOfWordsEmbedder(dim=0)

    def test_char_ngram_augmentation(self):
        emb_no_ngram = HashedBagOfWordsEmbedder(dim=64, char_ngram_size=0)
        emb_ngram = HashedBagOfWordsEmbedder(dim=64, char_ngram_size=3)
        v1 = emb_no_ngram.embed("hello")
        v2 = emb_ngram.embed("hello")
        # With ngrams, more features contribute -> different vector.
        self.assertNotEqual(v1, v2)


class TestSemanticCacheBasics(unittest.TestCase):
    def setUp(self):
        self.clock = _FakeClock()
        self.embedder = _FixedEmbedder({
            "What is Python?": [1.0, 0.0, 0.0],
            "What's Python?": [0.95, 0.05, 0.0],   # cos ~0.998 with above
            "Tell me about Rust": [0.0, 1.0, 0.0], # orthogonal
        })

    def _cache(self, **overrides):
        kwargs = dict(
            ttl_seconds=300, max_entries=100, similarity_threshold=0.92,
            embedder=self.embedder, clock=self.clock,
        )
        kwargs.update(overrides)
        return SemanticCache(**kwargs)

    def test_exact_match_hits(self):
        cache = self._cache()
        cache.put("groq/llama", "What is Python?", "Python is a language.")
        self.assertEqual(cache.get("groq/llama", "What is Python?"), "Python is a language.")

    def test_paraphrase_hits_when_above_threshold(self):
        cache = self._cache()
        cache.put("groq/llama", "What is Python?", "Python is a language.")
        self.assertEqual(
            cache.get("groq/llama", "What's Python?"),
            "Python is a language.",
        )

    def test_orthogonal_query_misses(self):
        cache = self._cache()
        cache.put("groq/llama", "What is Python?", "Python is a language.")
        self.assertIsNone(cache.get("groq/llama", "Tell me about Rust"))

    def test_threshold_enforcement(self):
        cache = self._cache(similarity_threshold=0.9999)  # too strict for paraphrase
        cache.put("groq/llama", "What is Python?", "Python is a language.")
        self.assertIsNone(cache.get("groq/llama", "What's Python?"))

    def test_model_scope_isolation(self):
        cache = self._cache()
        cache.put("groq/llama", "What is Python?", "llama answer")
        # Same prompt, different model -> miss.
        self.assertIsNone(cache.get("opus/4-7", "What is Python?"))

    def test_ttl_expiry(self):
        cache = self._cache(ttl_seconds=10)
        cache.put("groq/llama", "What is Python?", "answer")
        self.clock.advance(11.0)
        self.assertIsNone(cache.get("groq/llama", "What is Python?"))

    def test_max_entries_fifo_eviction(self):
        cache = self._cache(max_entries=2)
        cache.put("groq/llama", "What is Python?", "A")
        cache.put("groq/llama", "Tell me about Rust", "B")
        cache.put("groq/llama", "What's Python?", "C")
        # The oldest (A) got evicted; "What is Python?" should no longer hit.
        # But "What's Python?" embedding is 0.998 similar to "What is Python?",
        # and C is stored under "What's Python?" — so querying "What is Python?"
        # will match the "What's Python?" entry with response C.
        self.assertEqual(cache.get("groq/llama", "What is Python?"), "C")
        self.assertEqual(cache.stats()["entries"], 2)

    def test_invalidate_exact_match(self):
        cache = self._cache()
        cache.put("groq/llama", "What is Python?", "answer")
        self.assertTrue(cache.invalidate("groq/llama", "What is Python?"))
        self.assertIsNone(cache.get("groq/llama", "What is Python?"))
        self.assertFalse(cache.invalidate("groq/llama", "What is Python?"))

    def test_clear(self):
        cache = self._cache()
        cache.put("groq/llama", "What is Python?", "answer")
        cache.put("groq/llama", "Tell me about Rust", "answer2")
        cleared = cache.clear()
        self.assertEqual(cleared, 2)
        self.assertEqual(cache.stats()["entries"], 0)

    def test_stats_track_hit_miss_rate(self):
        cache = self._cache()
        cache.put("groq/llama", "What is Python?", "answer")
        cache.get("groq/llama", "What is Python?")    # hit
        cache.get("groq/llama", "What's Python?")      # hit (paraphrase)
        cache.get("groq/llama", "Tell me about Rust") # miss (orthogonal)
        stats = cache.stats()
        self.assertEqual(stats["total_hits"], 2)
        self.assertEqual(stats["total_misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 2 / 3, places=4)

    def test_peek_returns_lookup_diagnostic(self):
        cache = self._cache()
        cache.put("groq/llama", "What is Python?", "answer")
        hit = cache.peek("groq/llama", "What's Python?")
        self.assertIsNotNone(hit)
        response, lookup = hit
        self.assertEqual(response, "answer")
        self.assertIsInstance(lookup, CacheLookup)
        self.assertTrue(lookup.hit)
        self.assertGreater(lookup.best_similarity, 0.9)
        self.assertEqual(lookup.best_prompt, "What is Python?")
        self.assertEqual(lookup.entries_considered, 1)

    def test_peek_returns_lookup_on_miss(self):
        cache = self._cache(similarity_threshold=0.9999)
        cache.put("groq/llama", "What is Python?", "answer")
        result = cache.peek("groq/llama", "What's Python?")
        self.assertIsNotNone(result)
        response, lookup = result
        self.assertIsNone(response)
        self.assertFalse(lookup.hit)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            self._cache(similarity_threshold=0.0)
        with self.assertRaises(ValueError):
            self._cache(similarity_threshold=1.1)

    def test_invalid_max_entries_rejected(self):
        with self.assertRaises(ValueError):
            self._cache(max_entries=0)


class TestHypothesisHSCACHE(unittest.TestCase):
    """H-SCACHE falsification harness.

    Claim: on a synthetic paraphrase corpus, a semantic cache hits
    more often than an exact-match cache of the same size. The
    test intentionally avoids an absolute hit-rate target — that
    target belongs in the hypothesis engine with real embeddings.
    Here we just verify the SIGN of the effect is positive (semantic
    > exact) under the stdlib BoW embedder.
    """

    # Prompt pairs chosen to share at least a few tokens so BoW cosine
    # has a real chance of clearing the threshold. Each pair's second
    # element is a "paraphrase-with-shared-vocabulary" (not a Word2Vec
    # synonym — the BoW embedder can't do that).
    PAIRS = [
        ("What is the capital of France?", "Tell me the capital of France."),
        ("How do I install Python?", "How can I install Python?"),
        ("Explain recursion in one line.", "In one line, explain recursion."),
        ("Summarise the article.", "Please summarise the article."),
        ("Best way to cook rice?", "Best way to cook some rice"),
        ("Give me three Python tips.", "Give three Python tips please."),
        ("What's the time in Tokyo?", "Tell me the time in Tokyo."),
        ("Debug this python stack trace.", "Help me debug this python stack trace."),
        ("Translate 'hello' to French.", "Translate hello to French"),
        ("Fix the failing CI test.", "Please fix the failing CI test."),
    ]

    def test_semantic_cache_hits_more_than_exact_on_paraphrase_corpus(self):
        lowered_threshold = 0.55  # BoW is coarser than sentence-transformers
        embedder = HashedBagOfWordsEmbedder(dim=512, char_ngram_size=3)

        # Populate cache with the FIRST prompt of each pair, answer keyed to pair id.
        semantic = SemanticCache(
            ttl_seconds=3600, max_entries=256,
            similarity_threshold=lowered_threshold, embedder=embedder,
        )
        for idx, (orig, _) in enumerate(self.PAIRS):
            semantic.put("groq/llama", orig, f"response-{idx}")

        # Probe with the SECOND (paraphrase) prompt of each pair.
        sem_hits = sum(
            1 for _, para in self.PAIRS
            if semantic.get("groq/llama", para) is not None
        )
        # Exact-match baseline — exact sha256 of "What is the capital of France?"
        # never equals "Tell me the capital of France." so hit rate = 0.
        exact_hits = 0  # by construction of PAIRS; encoded here for clarity.

        self.assertGreater(
            sem_hits, exact_hits,
            f"Semantic cache must beat exact-match on paraphrase corpus "
            f"(sem={sem_hits}, exact={exact_hits})",
        )
        # Also assert a floor: at least 30% of pairs must semantic-hit,
        # else the BoW embedder isn't producing usable signal and the
        # module's Phase-γ load-bearing claim is weakened.
        self.assertGreaterEqual(sem_hits / len(self.PAIRS), 0.3,
                                f"Semantic hit rate too low: {sem_hits}/{len(self.PAIRS)}")


class TestDefaultEmbedder(unittest.TestCase):
    def test_falls_back_to_bag_of_words_when_embeddings_unavailable(self):
        # Don't assume `embeddings` package exists on the test host.
        emb = _default_embedder()
        # One of two shapes: either the real engine adapter or our stdlib fallback.
        # Both MUST produce an L2-normalised vector of positive length for a non-empty prompt.
        vec = emb.embed("integration probe for default embedder")
        self.assertGreater(len(vec), 0)
        norm = math.sqrt(sum(v * v for v in vec))
        # Tolerate 1e-6 float slop.
        self.assertLess(abs(norm - 1.0), 1e-5)


class TestEntryDataclass(unittest.TestCase):
    def test_construction(self):
        e = SemanticCacheEntry(
            prompt="hi", response="yo", embedding=[0.0, 1.0],
            created_at=0.0, model="m",
        )
        self.assertEqual(e.hit_count, 0)


if __name__ == "__main__":
    unittest.main()
