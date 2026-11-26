"""Tests for HAL Output Quality Filter — dismissal pattern detection."""
import os
import sys
import unittest

from lib.hal.output_quality import OutputQualityChecker, QualityWarning


class TestQualityWarning(unittest.TestCase):

    def test_to_dict(self):
        w = QualityWarning(pattern="pre-existing", matched_text="pre-existing", suggestion="fix it")
        d = w.to_dict()
        self.assertEqual(d["pattern"], "pre-existing")
        self.assertEqual(d["suggestion"], "fix it")
        self.assertEqual(d["severity"], "advisory")


class TestDismissalDetection(unittest.TestCase):

    def setUp(self):
        self.checker = OutputQualityChecker()

    def test_detects_pre_existing(self):
        warnings = self.checker.check("This is a pre-existing issue in the test suite.")
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].pattern, "pre-existing")

    def test_detects_not_related_to_sprint(self):
        warnings = self.checker.check("This failure is not related to this sprint.")
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].pattern, "not-related-to")

    def test_detects_not_related_to_change(self):
        warnings = self.checker.check("The error is not related to the change we made.")
        self.assertEqual(len(warnings), 1)

    def test_detects_unrelated_failure(self):
        warnings = self.checker.check("There is an unrelated failure in test_foo.py.")
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].pattern, "unrelated-issue")

    def test_detects_out_of_scope(self):
        warnings = self.checker.check("Fixing this is out of scope for this task.")
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].pattern, "out-of-scope")

    def test_detects_not_introduced_by(self):
        warnings = self.checker.check("This bug was not introduced by our changes.")
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].pattern, "not-introduced-by")

    def test_detects_already_broken(self):
        warnings = self.checker.check("The test was already broken before we started.")
        self.assertEqual(len(warnings), 2)  # "already broken" + "was already...before"
        patterns = {w.pattern for w in warnings}
        self.assertIn("already-broken", patterns)
        self.assertIn("was-already-before", patterns)

    def test_clean_text_no_warnings(self):
        warnings = self.checker.check("Found 3 additional test failures. Investigating root causes now.")
        self.assertEqual(len(warnings), 0)

    def test_replacement_language_clean(self):
        """The suggested replacement phrases should not trigger warnings."""
        clean_phrases = [
            "Found issue in test_foo.py — investigating now.",
            "Also found a failing test in the council module.",
            "Additional failure detected in session_registry tests.",
            "This issue needs attention regardless of when it was introduced.",
        ]
        for phrase in clean_phrases:
            warnings = self.checker.check(phrase)
            self.assertEqual(len(warnings), 0, f"False positive on: {phrase}")

    def test_has_dismissals_true(self):
        self.assertTrue(self.checker.has_dismissals("This is a pre-existing issue."))

    def test_has_dismissals_false(self):
        self.assertFalse(self.checker.has_dismissals("Found an issue. Fixing now."))

    def test_case_insensitive(self):
        warnings = self.checker.check("This is a PRE-EXISTING issue.")
        self.assertEqual(len(warnings), 1)

    def test_rewrite_suggestion(self):
        text = "This is a pre-existing issue."
        result = self.checker.rewrite_suggestion(text)
        self.assertIn("[FIX:", result)
        self.assertIn("found issue", result.lower())


if __name__ == "__main__":
    unittest.main()
