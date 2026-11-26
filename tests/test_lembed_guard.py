"""Regression tests for the sqlite-lembed input guard (Phase 0a).

These tests ensure the guard catches every W3-reproduced SIGSEGV input
BEFORE the input reaches ``SELECT lembed(...)``. The tests do NOT actually
call ``lembed()`` — the whole point of the guard is that we never have to.

If any of these tests starts failing because the guard no longer fires on
W3 crash inputs, the HAL process is once again vulnerable to SIGSEGV on
user-supplied text. Do not relax thresholds without a signed replacement
strategy (e.g. full SWAP-LEMBED migration).

Origin: investigation/apsw-lembed-stack W5 (2026-04-18).
"""
import unittest

from lib.hal.kono.lembed_guard import (
    LembedInputTooRisky,
    _max_repeat_run,
    is_likely_crash_input,
)


class TestCrashInputGuard(unittest.TestCase):
    """The guard must fire on every input W3 reproduced SIGSEGV on."""

    def test_repeat_a_1100_is_blocked(self):
        """Exact W3 crash input: 1100 'a' chars."""
        payload = "a" * 1100
        self.assertTrue(is_likely_crash_input(payload))

    def test_repeat_a_5000_is_blocked(self):
        """Larger W3 crash input: 5000 'a' chars."""
        payload = "a" * 5000
        self.assertTrue(is_likely_crash_input(payload))

    def test_long_lorem_repetition_is_blocked(self):
        """100x Lorem ipsum (~5700 chars) — W3 second crash input."""
        payload = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100
        # Unique ratio is not catastrophic for Lorem, but max_repeat_run
        # across repeated sentences can still breach — and if length alone
        # is the only signal breached, that's fine too. Current impl uses
        # unique_ratio + max_repeat_run for > threshold length; this test
        # encodes observed behaviour: this specific corpus is over 900 chars
        # with many repeated whitespace/punctuation runs.
        # If the test starts failing, investigate whether the W3 crash
        # still reproduces with updated thresholds.
        self.assertTrue(is_likely_crash_input(payload))


class TestSafeInputsPassThrough(unittest.TestCase):
    """The guard must not reject normal content."""

    def test_short_safe_input(self):
        self.assertFalse(is_likely_crash_input("hello world"))

    def test_normal_english_long_passage(self):
        """Mixed-entropy English text of moderate length is safe."""
        passage = (
            "The Harness Abstraction Layer is a provider-agnostic routing "
            "substrate. It composes local and remote model endpoints under "
            "a uniform interface. Conversations carry context forward via "
            "the KONO memory substrate, which assigns 4D spatial coordinates "
            "to every embedding and tracks attestation consensus across "
            "instance lifetimes."
        )
        # This is well under the length threshold.
        self.assertFalse(is_likely_crash_input(passage))

    def test_long_but_varied_english(self):
        """Long varied English text stays under the unique-ratio threshold."""
        # Build a 1200-char string of varied content — under the unique-ratio
        # floor should keep it safe.
        base = (
            "Python 3.14 introduces free-threaded builds. HAL uses apsw + "
            "sqlite-vec + sqlite-lembed for local embedding. The nomic-embed "
            "Q8 GGUF model produces 768-float vectors per input. "
        )
        # Repeat but with enough variation — ensure unique-char-ratio > 0.1.
        payload = base * 6  # ~1300 chars, uses the full English alphabet
        self.assertFalse(is_likely_crash_input(payload))

    def test_empty_string(self):
        self.assertFalse(is_likely_crash_input(""))

    def test_unicode_short(self):
        self.assertFalse(is_likely_crash_input("αβγ 中文 עברית 🌊"))


class TestMaxRepeatRun(unittest.TestCase):
    """Helper function contract."""

    def test_empty(self):
        self.assertEqual(_max_repeat_run(""), 0)

    def test_single_char(self):
        self.assertEqual(_max_repeat_run("x"), 1)

    def test_uniform(self):
        self.assertEqual(_max_repeat_run("aaaa"), 4)

    def test_mixed(self):
        self.assertEqual(_max_repeat_run("aabbbccccdee"), 4)

    def test_no_repeats(self):
        self.assertEqual(_max_repeat_run("abcdef"), 1)

    def test_long_run_inside_text(self):
        payload = "hello " + ("x" * 250) + " world"
        self.assertEqual(_max_repeat_run(payload), 250)


class TestExceptionType(unittest.TestCase):
    """LembedInputTooRisky is a ValueError subclass (callers may catch either)."""

    def test_subclass_of_value_error(self):
        self.assertTrue(issubclass(LembedInputTooRisky, ValueError))

    def test_instantiable(self):
        err = LembedInputTooRisky("bad input")
        self.assertEqual(str(err), "bad input")


if __name__ == "__main__":
    unittest.main()
