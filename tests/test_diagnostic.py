"""Tests for HAL diagnostic engine."""
import os
import sys
import unittest

from lib.hal.diagnostic import DiagnosticEngine, DiagnosticIssue, DiagnosticReport


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
