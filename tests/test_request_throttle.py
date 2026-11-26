"""Tests for HAL request throttle."""
import os, sys, unittest, threading
from lib.hal.request_throttle import RequestThrottle

class TestRequestThrottle(unittest.TestCase):
    def test_acquire_and_release(self):
        throttle = RequestThrottle(max_concurrent=2)
        with throttle.acquire("groq"):
            self.assertEqual(throttle.active, 1)
        self.assertEqual(throttle.active, 0)

    def test_available(self):
        throttle = RequestThrottle(max_concurrent=3)
        self.assertEqual(throttle.available, 3)
        with throttle.acquire():
            self.assertEqual(throttle.available, 2)

    def test_stats(self):
        throttle = RequestThrottle(max_concurrent=5)
        with throttle.acquire():
            pass
        stats = throttle.stats()
        self.assertEqual(stats.total_requests, 1)

    def test_concurrent(self):
        throttle = RequestThrottle(max_concurrent=2)
        results = []
        def worker():
            with throttle.acquire():
                results.append(throttle.active)
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r <= 2 for r in results))

if __name__ == "__main__":
    unittest.main()
