"""Tests for HAL diagnostic engine."""
import unittest
from collections import namedtuple
from unittest import mock

from lib.hal.diagnostic import (
    DiagnosticEngine,
    DiagnosticIssue,
    DiagnosticReport,
    _check_python_version,
)

# Shaped like the real sys.version_info structseq (major/minor attributes,
# tuple comparison and slicing) rather than a bare tuple, because
# _check_python_version reads .major/.minor directly -- a bare tuple stub
# raises AttributeError here even though it works for a check that only
# slices sys.version_info[:2] (AGORA, qwen4-max, PR #946 round 2).
_FakeVersion = namedtuple("_FakeVersion", "major minor micro releaselevel serial")


def _fake_version(major, minor):
    return _FakeVersion(major, minor, 0, "final", 0)


class TestDiagnosticEngine(unittest.TestCase):

    def test_runs_without_error(self):
        engine = DiagnosticEngine()
        report = engine.run_all()
        self.assertIsInstance(report, DiagnosticReport)

    def test_finds_some_results(self):
        engine = DiagnosticEngine()
        report = engine.run_all()
        # Should find at least some info-level items (git status, missing config)
        self.assertIsInstance(report.issues, list)


class TestDiagnosticReport(unittest.TestCase):

    def test_summary_all_clear(self):
        report = DiagnosticReport(issues=[])
        self.assertIn("all clear", report.summary)
        self.assertFalse(report.has_errors)

    def test_has_errors(self):
        report = DiagnosticReport(issues=[
            DiagnosticIssue("test", "error", "broken", "fix it"),
        ])
        self.assertTrue(report.has_errors)

    def test_summary_with_issues(self):
        report = DiagnosticReport(issues=[
            DiagnosticIssue("test", "error", "broken", "fix"),
            DiagnosticIssue("test", "warning", "iffy", "check"),
            DiagnosticIssue("test", "info", "fyi", "note"),
        ])
        self.assertIn("1 errors", report.summary)
        self.assertIn("1 warnings", report.summary)


class TestCheckPythonVersion(unittest.TestCase):
    """V>> raised HAL's floor to 3.11 on 2026-11-26 (HAL#929). Mutation: put
    the threshold back to (3, 10) in _check_python_version -- the (3, 10)
    case below stops being flagged as an error -> RED.
    """

    def test_below_floor_flags_error(self):
        for major, minor in ((3, 9), (3, 10)):
            with self.subTest(version=f"{major}.{minor}"):
                with mock.patch("sys.version_info", _fake_version(major, minor)):
                    issues = _check_python_version()
                self.assertEqual(len(issues), 1)
                self.assertEqual(issues[0].severity, "error")
                self.assertEqual(issues[0].category, "dependency")
                self.assertIn(f"{major}.{minor}", issues[0].description)
                self.assertIn("3.11", issues[0].description)
                self.assertIn("3.11", issues[0].suggested_fix)

    def test_at_or_above_floor_passes(self):
        for major, minor in ((3, 11), (3, 12)):
            with self.subTest(version=f"{major}.{minor}"):
                with mock.patch("sys.version_info", _fake_version(major, minor)):
                    issues = _check_python_version()
                self.assertEqual(issues, [])


class TestDiagnosticIssue(unittest.TestCase):

    def test_frozen(self):
        issue = DiagnosticIssue("cat", "error", "desc", "fix")
        with self.assertRaises(AttributeError):
            issue.severity = "info"

    def test_fields(self):
        issue = DiagnosticIssue("config", "warning", "Missing file", "Create it")
        self.assertEqual(issue.category, "config")
        self.assertEqual(issue.severity, "warning")


if __name__ == "__main__":
    unittest.main()
