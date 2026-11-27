"""Tests for the local Ollama verifier behind the semantic response cache (HAL#965)."""
from __future__ import annotations

import http.client
import json
import threading
import unittest
import urllib.error
from unittest.mock import patch

from lib.hal.cache_verifier import OllamaVerifier

_REAL_START_WARM = OllamaVerifier.start_warm


class _Reply:
    def __init__(self, payload):
        self._body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _patched(*replies):
    """Patch urlopen to return (or raise) each reply in turn; also return the call log."""
    calls = []

    def fake(request, timeout):
        calls.append((request.full_url, json.loads(request.data), timeout))
        reply = replies[len(calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return _Reply(reply)

    return patch("lib.hal.cache_verifier.urllib.request.urlopen", side_effect=fake), calls


class TestOllamaVerifier(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.verifier = OllamaVerifier("qwen3:8b", timeout_s=5, rest_seconds=60, clock=self.clock)
        self.warm_starts = []
        starter = patch.object(OllamaVerifier, "start_warm", lambda _self: self.warm_starts.append(1))
        starter.start()
        self.addCleanup(starter.stop)

    def test_only_a_whole_yes_is_an_approval(self):
        replies = (("YES", True), ("yes.", True), (" Yes! ", True), ("NO", False), ("Maybe", False),
                   ("", False), ("YES, but they differ in scope", False), ("yes and no", False))
        for text, expected in replies:
            patcher, _ = _patched({"response": text})
            with patcher:
                self.assertEqual(self.verifier.same_request("a", "b"), expected, text)
        self.assertEqual(self.warm_starts, [])

    def test_the_judge_request_is_local_and_deterministic(self):
        patcher, calls = _patched({"response": "NO"})
        with patcher:
            self.verifier.same_request("first", "second")
        url, body, timeout = calls[0]
        self.assertEqual(url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual((body["model"], body["options"], body["think"], body["keep_alive"], timeout),
                         ("qwen3:8b", {"temperature": 0, "seed": 1}, False, "30m", 5))
        self.assertIn("Request A: first\n", body["prompt"])
        self.assertIn("Request B: second\n", body["prompt"])

    def test_a_prompt_over_the_length_cap_is_never_judged(self):
        patcher, calls = _patched()
        with patcher:
            self.assertFalse(self.verifier.same_request("a" * 2001, "b"))
            self.assertFalse(self.verifier.same_request("b", "a" * 2001))
        self.assertEqual((calls, self.warm_starts), ([], []))

    def test_an_unreadable_reply_is_a_no(self):
        for reply in (b"not json", ["a list"], {"response": 7}):
            patcher, _ = _patched(reply)
            with patcher:
                self.assertFalse(self.verifier.same_request("a", "b"), reply)
            self.clock.t += 61  # past any rest that a failure started

    def test_an_http_protocol_error_is_a_no(self):
        patcher, _ = _patched(http.client.IncompleteRead(b"partial"))
        with patcher:
            self.assertFalse(self.verifier.same_request("a", "b"))
        self.assertEqual(self.warm_starts, [1])

    def test_embed_returns_a_unit_vector(self):
        patcher, calls = _patched({"embedding": [3.0, 4.0]})
        with patcher:
            self.assertEqual(self.verifier.embed("text"), [0.6, 0.8])
        self.assertEqual(calls[0][1], {"model": "nomic-embed-text", "prompt": "text"})

    def test_embed_gives_none_for_a_missing_empty_or_non_numeric_vector(self):
        replies = ({}, {"embedding": []}, {"embedding": [0.0, 0.0]}, {"embedding": "x"},
                   {"embedding": ["x"]}, {"embedding": [None]}, {"embedding": [1.0, float("nan")]})
        for reply in replies:
            patcher, _ = _patched(reply)
            with patcher:
                self.assertIsNone(self.verifier.embed("text"), reply)

    def test_a_failure_rests_starts_a_warm_up_then_tries_again(self):
        patcher, calls = _patched(urllib.error.URLError("refused"), {"response": "YES"})
        with patcher:
            self.assertFalse(self.verifier.same_request("a", "b"))
            self.assertEqual(self.warm_starts, [1])
            self.assertIsNone(self.verifier.embed("a"))  # resting: no request is sent
            self.assertEqual(len(calls), 1)
            self.clock.t += 61
            self.assertTrue(self.verifier.same_request("a", "b"))
        self.assertEqual(len(calls), 2)

    def test_a_timeout_is_a_no(self):
        patcher, _ = _patched(TimeoutError("timed out"))
        with patcher:
            self.assertFalse(self.verifier.same_request("a", "b"))
        self.assertEqual(self.warm_starts, [1])

    def test_warm_loads_both_models_and_leaves_the_rest_running(self):
        self.verifier._resting_until = self.clock.t + 60
        patcher, calls = _patched({"done": True}, {"embedding": []}, {"response": "YES"})
        with patcher:
            self.assertTrue(self.verifier.warm())
            self.assertFalse(self.verifier.same_request("a", "b"))  # still resting: nothing is sent
            self.assertEqual(len(calls), 2)
            self.clock.t += 61
            self.assertTrue(self.verifier.same_request("a", "b"))
        self.assertEqual([(url.rsplit("/", 1)[-1], body, timeout) for url, body, timeout in calls[:2]], [
            ("generate", {"model": "qwen3:8b", "prompt": "", "stream": False, "keep_alive": "30m"}, 300.0),
            ("embeddings", {"model": "nomic-embed-text", "prompt": "", "keep_alive": "30m"}, 300.0),
        ])

    def test_a_slow_judge_costs_one_timeout_per_rest_not_one_per_lookup(self):
        # The models load at once, but every real judge call runs past the timeout (HAL#987).
        judge_calls, warm_calls = [], []

        def fake(request, timeout):
            if json.loads(request.data)["prompt"]:  # a real judge call; a warm-up sends ""
                judge_calls.append(self.clock.t)
                raise TimeoutError("timed out")
            warm_calls.append(request.full_url)
            return _Reply({})

        with (patch("lib.hal.cache_verifier.urllib.request.urlopen", side_effect=fake),
              patch.object(OllamaVerifier, "start_warm", lambda verifier: verifier.warm())):
            for _ in range(5):  # five lookups inside one 60 s rest
                self.assertFalse(self.verifier.same_request("a", "b"))
                self.clock.t += 10
            self.assertEqual((len(judge_calls), len(warm_calls)), (1, 2))
            self.clock.t += 20  # the rest that began at the first timeout is over
            self.assertFalse(self.verifier.same_request("a", "b"))
        self.assertEqual((len(judge_calls), len(warm_calls)), (2, 4))

    def test_a_failed_warm_up_keeps_resting(self):
        self.verifier._resting_until = 1e9
        patcher, _ = _patched(urllib.error.URLError("no such model"), {"embedding": []})
        with patcher:
            self.assertFalse(self.verifier.warm())
        self.assertEqual(self.verifier._resting_until, 1e9)

    def test_an_empty_judge_model_is_rejected(self):
        for name in ("", "   "):
            with self.assertRaises(ValueError):
                OllamaVerifier(name)

    def test_only_one_warm_up_runs_at_a_time(self):
        release, started = threading.Event(), []
        verifier = OllamaVerifier("qwen3:8b")

        def slow_warm():
            started.append(1)
            release.wait(5)
            return True

        verifier.warm = slow_warm
        _REAL_START_WARM(verifier)
        _REAL_START_WARM(verifier)
        release.set()
        verifier.warm_thread.join(5)
        self.assertEqual(started, [1])


if __name__ == "__main__":
    unittest.main()
