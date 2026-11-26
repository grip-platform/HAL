"""Regression anchors for Issue #200 — tool loop circuit breaker.

The original failure: a broken tool schema produced a 400 on every
turn and the REPL kept looping the same schema forever. After this PR,
two consecutive tool-schema 400s auto-disable tools so the next turn
at worst falls back to plain text.

Goodhart-protection: tests exercise actual state transitions, not call
counts. A neutered breaker (always returns False, always returns True,
or forgets to reset) will fail at least one assertion.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).parent.parent


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    # Python 3.14's @dataclass inspects ``sys.modules[cls.__module__]`` — the
    # module must be registered BEFORE ``exec_module`` runs, otherwise the
    # dataclass processor crashes on ``NoneType.__dict__``. Safe because
    # the name is a test-only sentinel ("_wt_*") that won't shadow real modules.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_breaker = _load("lib/hal/tool_circuit_breaker.py", "_wt_tool_circuit_breaker")
CircuitBreaker = _breaker.CircuitBreaker
is_tool_schema_400 = _breaker.is_tool_schema_400


class _FakeChatError(Exception):
    """Stand-in for AnthropicChatError — duck-typed match for is_tool_schema_400."""

    def __init__(self, status: int, body: str):
        super().__init__(f"stub {status}")
        self.status = status
        self.body = body


class CircuitBreakerStateTests(unittest.TestCase):
    """Transition invariants: trip only at threshold, reset works, success clears counter."""

    def test_defaults_are_untripped(self):
        b = CircuitBreaker()
        self.assertFalse(b.is_tripped("cat"))

    def test_single_failure_below_threshold_does_not_trip(self):
        b = CircuitBreaker(threshold=2)
        self.assertFalse(b.record_failure("cat"))
        self.assertFalse(b.is_tripped("cat"))

    def test_two_consecutive_failures_trip_at_threshold(self):
        b = CircuitBreaker(threshold=2)
        b.record_failure("cat")
        crossed = b.record_failure("cat")
        self.assertTrue(crossed, "threshold=2 must trip on the second failure")
        self.assertTrue(b.is_tripped("cat"))

    def test_already_tripped_does_not_recross(self):
        """Subsequent failures after trip return False so callers don't re-fire message."""
        b = CircuitBreaker(threshold=2)
        b.record_failure("cat")
        self.assertTrue(b.record_failure("cat"))
        # Third failure: breaker is already tripped, don't re-cross.
        self.assertFalse(b.record_failure("cat"))
        self.assertFalse(b.record_failure("cat"))

    def test_success_resets_consecutive_counter(self):
        """A success between failures resets the streak — one transient 400 doesn't trip."""
        b = CircuitBreaker(threshold=2)
        b.record_failure("cat")
        b.record_success("cat")
        # Now record one failure. Streak is 1, threshold is 2 → must not trip.
        self.assertFalse(b.record_failure("cat"))
        self.assertFalse(b.is_tripped("cat"))

    def test_reset_clears_trip_and_counter(self):
        b = CircuitBreaker(threshold=2)
        b.record_failure("cat")
        b.record_failure("cat")
        self.assertTrue(b.is_tripped("cat"))
        b.reset("cat")
        self.assertFalse(b.is_tripped("cat"))
        # After reset, counter is back to zero — it takes two more to trip.
        self.assertFalse(b.record_failure("cat"))
        self.assertTrue(b.record_failure("cat"))

    def test_reset_all_clears_every_category(self):
        b = CircuitBreaker(threshold=2)
        b.record_failure("a")
        b.record_failure("a")
        b.record_failure("b")
        b.record_failure("b")
        b.reset()
        self.assertFalse(b.is_tripped("a"))
        self.assertFalse(b.is_tripped("b"))

    def test_categories_are_independent(self):
        b = CircuitBreaker(threshold=2)
        b.record_failure("a")
        b.record_failure("b")
        # Two failures, but each in a different category → nothing trips.
        self.assertFalse(b.is_tripped("a"))
        self.assertFalse(b.is_tripped("b"))

    def test_threshold_one_trips_on_first_failure(self):
        b = CircuitBreaker(threshold=1)
        self.assertTrue(b.record_failure("cat"))


class IsToolSchema400Tests(unittest.TestCase):
    """Classifier: only obvious schema-rejection 400s trip the breaker."""

    def test_non_400_not_classified(self):
        self.assertFalse(is_tool_schema_400(_FakeChatError(500, "server error")))
        self.assertFalse(is_tool_schema_400(_FakeChatError(429, "rate limit")))

    def test_400_without_schema_marker_not_classified(self):
        """A plain 400 (e.g. bad user text) must not trip the tool breaker."""
        exc = _FakeChatError(400, "invalid text in user message")
        self.assertFalse(is_tool_schema_400(exc))

    def test_the_exact_issue_200_error_body_is_classified(self):
        """The user's actual error message must be recognised."""
        body = (
            "{\"error\":{\"message\":\"'tools.0.type' : property 'type' is missing\","
            "\"type\":\"invalid_request_error\"}}"
        )
        exc = _FakeChatError(400, body)
        self.assertTrue(is_tool_schema_400(exc))

    def test_case_insensitive_matching(self):
        body = "TOOLS.0.TYPE missing"
        exc = _FakeChatError(400, body)
        self.assertTrue(is_tool_schema_400(exc))

    def test_openai_variant_body_is_classified(self):
        body = "tools[0].type required"
        exc = _FakeChatError(400, body)
        self.assertTrue(is_tool_schema_400(exc))

    def test_missing_attributes_does_not_crash(self):
        """Arbitrary exceptions without status/body must not crash the classifier."""
        self.assertFalse(is_tool_schema_400(ValueError("no status attribute")))


class MutationGuardTests(unittest.TestCase):
    """If someone neuters the breaker, these fail."""

    def test_always_true_record_failure_would_fail(self):
        """If record_failure always returns True, the "first failure below threshold" test breaks."""
        b = CircuitBreaker(threshold=2)
        self.assertFalse(b.record_failure("cat"))

    def test_always_false_record_failure_would_fail(self):
        """If record_failure always returns False, the "second failure trips" test breaks."""
        b = CircuitBreaker(threshold=2)
        b.record_failure("cat")
        self.assertTrue(b.record_failure("cat"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
