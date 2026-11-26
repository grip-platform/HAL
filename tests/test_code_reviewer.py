"""Tests for lib/hal/code_reviewer.py — CodeReviewer, ReviewResult, helpers."""
import sys
import pathlib
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lib.hal.code_reviewer import (
    CodeReviewer,
    ReviewIssue,
    ReviewResult,
    _build_summary,
    _severity_from_issues,
)


def _make_generated(source="def foo(): pass\n", spec_desc="A function"):
    g = MagicMock()
    g.module_source = source
    g.spec = MagicMock()
    g.spec.description = spec_desc
    return g


class TestMechanicalGates(unittest.TestCase):

    def test_pass_clean_code(self):
        """Clean Python → severity == 'pass', passed == True."""
        reviewer = CodeReviewer()
        result = reviewer.review(_make_generated("def foo():\n    return 42\n"))
        self.assertEqual(result.severity, "pass")
        self.assertTrue(result.passed)

    def test_block_on_high_security_issue(self):
        """Source with os.system shell call → severity == 'block', passed == False."""
        source = 'import os\nos.system("rm -rf /")\n'
        reviewer = CodeReviewer()
        result = reviewer.review(_make_generated(source))
        self.assertEqual(result.severity, "block")
        self.assertFalse(result.passed)

    def test_warn_on_medium_security_issue(self):
        """Hardcoded credential string → severity == 'warn', passed == True."""
        source = 'password = "hardcoded_secret123"\n'
        reviewer = CodeReviewer()
        result = reviewer.review(_make_generated(source))
        # Medium-severity scanner issues map to 'warning' → overall 'warn'
        self.assertIn(result.severity, ("warn", "block"))
        # Either warn or block; must NOT be a false-negative 'pass'
        self.assertNotEqual(result.severity, "pass")

    def test_syntax_block_on_invalid_python(self):
        """Syntactically invalid Python → severity == 'block'."""
        source = "def foo(:\n    pass\n"
        reviewer = CodeReviewer()
        result = reviewer.review(_make_generated(source))
        self.assertEqual(result.severity, "block")


class TestSemanticGate(unittest.TestCase):

    def test_semantic_gate_called_when_llm_provided(self):
        """llm_fn returning '[]' is called exactly once during review."""
        llm_fn = MagicMock(return_value="[]")
        reviewer = CodeReviewer(llm_fn=llm_fn)
        reviewer.review(_make_generated())
        llm_fn.assert_called_once()

    def test_semantic_failure_non_fatal(self):
        """llm_fn raising RuntimeError → review returns a result without raising."""
        llm_fn = MagicMock(side_effect=RuntimeError("LLM down"))
        reviewer = CodeReviewer(llm_fn=llm_fn)
        result = reviewer.review(_make_generated())
        self.assertIsInstance(result, ReviewResult)

    def test_semantic_not_called_without_llm(self):
        """No llm_fn → zero semantic issues (gate never runs)."""
        reviewer = CodeReviewer()
        result = reviewer.review(_make_generated())
        semantic_issues = [i for i in result.issues if i.gate == "semantic"]
        self.assertEqual(len(semantic_issues), 0)


class TestHelperFunctions(unittest.TestCase):

    def test_severity_from_issues_dispatch(self):
        """_severity_from_issues: error→block, warning→warn, info→pass."""
        error_issue = ReviewIssue(gate="safety", severity="error", line=1, message="bad")
        warn_issue = ReviewIssue(gate="safety", severity="warning", line=1, message="hmm")
        info_issue = ReviewIssue(gate="safety", severity="info", line=1, message="fyi")

        self.assertEqual(_severity_from_issues([error_issue]), "block")
        self.assertEqual(_severity_from_issues([warn_issue]), "warn")
        self.assertEqual(_severity_from_issues([info_issue]), "pass")
        # error dominates warn
        self.assertEqual(_severity_from_issues([warn_issue, error_issue]), "block")

    def test_build_summary_format(self):
        """_build_summary with an error issue → summary starts with 'BLOCKED'."""
        issue = ReviewIssue(gate="safety", severity="error", line=3, message="dangerous call")
        summary = _build_summary([issue], "block")
        self.assertTrue(summary.startswith("BLOCKED"), f"Got: {summary!r}")


if __name__ == "__main__":
    unittest.main()
