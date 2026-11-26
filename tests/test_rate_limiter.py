"""Tests for HAL rate limiter."""
import importlib.util
import os
import sys
import unittest

# Load rate_limiter directly from the worktree path to avoid sys.modules cache
# from ~/.hal (which other tests may have populated before this file runs).
_WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "_hal_rate_limiter_wt",
    os.path.join(_WORKTREE, "lib", "hal", "rate_limiter.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
RateLimiter = _mod.RateLimiter
RateLimitInfo = _mod.RateLimitInfo


class TestRateLimiter(unittest.TestCase):

    def test_allow_unknown_provider(self):
        limiter = RateLimiter()
        self.assertTrue(limiter.allow("unknown_provider"))

    def test_allow_within_limits(self):
        limiter = RateLimiter()
        limiter.configure("test", rpm=10, tpm=10000)
        self.assertTrue(limiter.allow("test", estimated_tokens=100))

    def test_block_over_rpm(self):
        limiter = RateLimiter()
        limiter.configure("test", rpm=2, tpm=0)
        limiter.record("test")
        limiter.record("test")
        self.assertFalse(limiter.allow("test"))

    def test_block_over_tpm(self):
        limiter = RateLimiter()
        limiter.configure("test", rpm=0, tpm=100)
        limiter.record("test", tokens_used=80)
        self.assertFalse(limiter.allow("test", estimated_tokens=50))

    def test_local_unlimited(self):
        limiter = RateLimiter()
        for _ in range(100):
            limiter.record("ollama", tokens_used=1000)
        self.assertTrue(limiter.allow("ollama"))

    def test_wait_time_zero_when_allowed(self):
        limiter = RateLimiter()
        self.assertEqual(limiter.wait_time("ollama"), 0.0)

    def test_wait_time_positive_when_blocked(self):
        limiter = RateLimiter()
        limiter.configure("test", rpm=1, tpm=0)
        limiter.record("test")
        self.assertGreater(limiter.wait_time("test"), 0)

    def test_status(self):
        limiter = RateLimiter()
        limiter.configure("test", rpm=30, tpm=6000)
        limiter.record("test", tokens_used=500)
        s = limiter.status("test")
        self.assertEqual(s["rpm_limit"], 30)
        self.assertEqual(s["rpm_used"], 1)
        self.assertEqual(s["tpm_used"], 500)

    def test_configure_overrides(self):
        limiter = RateLimiter()
        limiter.configure("groq", rpm=100, tpm=50000)
        s = limiter.status("groq")
        self.assertEqual(s["rpm_limit"], 100)


class TestRateLimitInfo(unittest.TestCase):

    def test_rate_limit_info_from_headers(self):
        headers = {
            "x-ratelimit-remaining-requests": "20",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-reset-requests": "1700000000000",
        }
        info = RateLimitInfo.from_headers(headers, "openai")
        self.assertAlmostEqual(info.utilization, 0.80)
        self.assertEqual(info.reset_time_ms, 1700000000000)
        self.assertFalse(info.overage)
        self.assertEqual(info.provider, "openai")

    def test_rate_limit_info_no_headers(self):
        info = RateLimitInfo.from_headers({}, "groq")
        self.assertEqual(info.utilization, 0.0)
        self.assertEqual(info.reset_time_ms, 0)
        self.assertFalse(info.overage)
        self.assertEqual(info.provider, "groq")

    def test_rate_limit_info_overage_when_remaining_zero(self):
        headers = {
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-limit-requests": "50",
        }
        info = RateLimitInfo.from_headers(headers, "anthropic")
        self.assertEqual(info.utilization, 1.0)
        self.assertTrue(info.overage)


class TestEWMALatencyTracker(unittest.TestCase):

    def test_first_record_sets_initial(self):
        from lib.hal.rate_limiter import EWMALatencyTracker
        tracker = EWMALatencyTracker()
        tracker.record("groq", 120.0)
        self.assertAlmostEqual(tracker.get("groq"), 120.0)

    def test_ewma_smooths_subsequent_observations(self):
        from lib.hal.rate_limiter import EWMALatencyTracker
        tracker = EWMALatencyTracker()
        tracker.record("groq", 100.0)
        tracker.record("groq", 200.0)
        val = tracker.get("groq")
        # α=0.3: 0.3*200 + 0.7*100 = 60 + 70 = 130
        self.assertAlmostEqual(val, 130.0)

    def test_fastest_in_returns_lowest(self):
        from lib.hal.rate_limiter import EWMALatencyTracker
        tracker = EWMALatencyTracker()
        tracker.record("groq", 150.0)
        tracker.record("gemini", 80.0)
        self.assertEqual(tracker.fastest_in(["groq", "gemini"]), "gemini")

    def test_fastest_in_none_when_no_data(self):
        from lib.hal.rate_limiter import EWMALatencyTracker
        tracker = EWMALatencyTracker()
        self.assertIsNone(tracker.fastest_in(["groq", "gemini"]))

    def test_sorted_by_latency_unknown_last(self):
        from lib.hal.rate_limiter import EWMALatencyTracker
        tracker = EWMALatencyTracker()
        tracker.record("gemini", 80.0)
        tracker.record("groq", 150.0)
        result = tracker.sorted_by_latency(["groq", "gemini", "openai"])
        self.assertEqual(result[0], "gemini")
        self.assertEqual(result[1], "groq")
        self.assertEqual(result[2], "openai")  # unknown goes last


if __name__ == "__main__":
    unittest.main()
