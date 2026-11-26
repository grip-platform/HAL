"""ReDoS property tests for HAL secrets guard.

Asserts the single mega-pattern in :mod:`lib.hal.secrets_guard` stays
responsive under pathological inputs. Each case runs with a fixed
wall-clock budget; a regression would blow past the budget and fail.
"""
import time
import unittest

from lib.hal.secrets_guard import (
    MAX_SCAN_BYTES,
    ScanReport,
    has_secrets,
    redact_text,
    scan_text,
    scan_text_report,
)

# Budget per scan. 200ms is ~100x the observed average on M-series silicon
# for clean inputs; catastrophic backtracking would be seconds to minutes.
_BUDGET_S = 0.5


def _timed(fn, *args):
    start = time.monotonic()
    result = fn(*args)
    return result, time.monotonic() - start


class TestAdversarialInputs(unittest.TestCase):
    """Inputs designed to trigger backtracking if patterns were unbounded."""

    def test_long_run_of_alpha(self):
        payload = "a" * (MAX_SCAN_BYTES * 2)  # Well above the cap
        _, elapsed = _timed(has_secrets, payload)
        self.assertLess(elapsed, _BUDGET_S,
                        f"has_secrets took {elapsed:.3f}s on alpha run")

    def test_long_run_of_equals(self):
        payload = "=" * (MAX_SCAN_BYTES * 2)
        _, elapsed = _timed(has_secrets, payload)
        self.assertLess(elapsed, _BUDGET_S)

    def test_long_run_of_word_chars(self):
        payload = ("ABCDEFGH" * 40000)[:MAX_SCAN_BYTES * 2]
        _, elapsed = _timed(scan_text, payload)
        self.assertLess(elapsed, _BUDGET_S)

    def test_many_candidate_prefixes(self):
        # Stress the ENV_VAR_SECRET branch with repeated PASSWORD= starts.
        payload = ("FOO_PASSWORD=" + "x" * 7 + " ") * 5000
        _, elapsed = _timed(scan_text, payload)
        self.assertLess(elapsed, _BUDGET_S * 2)

    def test_nested_quoting(self):
        payload = ('"' * 1000 + "aws_secret_access_key=" + "Z" * 40 + '"' * 1000)
        _, elapsed = _timed(scan_text, payload)
        self.assertLess(elapsed, _BUDGET_S)


class TestBoundedScanBehaviour(unittest.TestCase):

    def test_oversized_input_is_head_scanned(self):
        huge = "x" * (MAX_SCAN_BYTES + 10_000)
        report = scan_text_report(huge)
        self.assertTrue(report.truncated)
        self.assertEqual(report.scanned_bytes, MAX_SCAN_BYTES)

    def test_normal_input_is_not_truncated(self):
        report = scan_text_report("hello world this is a clean message")
        self.assertFalse(report.truncated)

    def test_oversized_redaction_appends_marker(self):
        huge = "x" * (MAX_SCAN_BYTES + 100)
        out = redact_text(huge)
        self.assertIn("[SECRETS SCAN TRUNCATED", out)

    def test_clean_input_redaction_is_noop(self):
        text = "hello world this is a perfectly clean message for testing"
        self.assertEqual(redact_text(text), text)

    def test_secret_in_head_of_oversized_input_still_redacted(self):
        # Secret sits at offset 0; tail is 1 MiB of filler.
        needle = "sk" + "-ant-" + "0123456789abcdef0123456789abcdef"
        payload = needle + " filler " + ("x" * (MAX_SCAN_BYTES + 1000))
        out = redact_text(payload)
        self.assertIn("[REDACTED:ANTHROPIC_KEY]", out)
        self.assertIn("[SECRETS SCAN TRUNCATED", out)

    def test_scan_report_is_immutable(self):
        report = scan_text_report("clean message that is not too short")
        with self.assertRaises(Exception):  # frozen dataclass
            report.truncated = True  # type: ignore


class TestWorstCaseTiming(unittest.TestCase):
    """A fuzz-lite smoke check: scan random-ish content quickly."""

    def test_mixed_content_within_budget(self):
        import random
        rng = random.Random(1234)
        chunks = []
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789_-= :/"
        for _ in range(2000):
            chunks.append("".join(rng.choice(alphabet) for _ in range(32)))
        payload = " ".join(chunks)
        _, elapsed = _timed(scan_text, payload)
        self.assertLess(elapsed, _BUDGET_S)


if __name__ == "__main__":
    unittest.main()
