"""Tests for HAL response cache."""
import os
import sys
import time
import unittest

from lib.hal.response_cache import ResponseCache, _cache_key


class TestCacheKey(unittest.TestCase):

    def test_deterministic(self):
        k1 = _cache_key("model", "prompt")
        k2 = _cache_key("model", "prompt")
        self.assertEqual(k1, k2)

    def test_different_for_different_input(self):
        k1 = _cache_key("model_a", "prompt")
        k2 = _cache_key("model_b", "prompt")
        self.assertNotEqual(k1, k2)


class TestResponseCache(unittest.TestCase):

    def test_miss_returns_none(self):
        cache = ResponseCache()
        self.assertIsNone(cache.get("model", "query"))

    def test_put_and_get(self):
        cache = ResponseCache()
        cache.put("model", "query", "response text")
        self.assertEqual(cache.get("model", "query"), "response text")

    def test_ttl_expiry(self):
        cache = ResponseCache(ttl_seconds=0)
        cache.put("model", "query", "response")
        time.sleep(0.01)
        self.assertIsNone(cache.get("model", "query"))

    def test_eviction_at_max(self):
        cache = ResponseCache(max_entries=2)
        cache.put("m", "q1", "r1")
        cache.put("m", "q2", "r2")
        cache.put("m", "q3", "r3")  # Should evict oldest
        self.assertEqual(cache.stats()["entries"], 2)

    def test_invalidate(self):
        cache = ResponseCache()
        cache.put("m", "q", "r")
        self.assertTrue(cache.invalidate("m", "q"))
        self.assertIsNone(cache.get("m", "q"))

    def test_invalidate_missing(self):
        cache = ResponseCache()
        self.assertFalse(cache.invalidate("m", "nonexistent"))

    def test_clear(self):
        cache = ResponseCache()
        cache.put("m", "q1", "r1")
        cache.put("m", "q2", "r2")
        count = cache.clear()
        self.assertEqual(count, 2)
        self.assertEqual(cache.stats()["entries"], 0)

    def test_stats(self):
        cache = ResponseCache(ttl_seconds=60, max_entries=50)
        cache.put("m", "q", "r")
        cache.get("m", "q")
        cache.get("m", "q")
        stats = cache.stats()
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["total_hits"], 2)
        self.assertEqual(stats["ttl_seconds"], 60)

    def test_hit_count_increments(self):
        cache = ResponseCache()
        cache.put("m", "q", "r")
        cache.get("m", "q")
        cache.get("m", "q")
        cache.get("m", "q")
        self.assertEqual(cache.stats()["total_hits"], 3)


if __name__ == "__main__":
    unittest.main()
