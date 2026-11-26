"""Tests for lib/hal/git_ops.py — GitHubOps merge/status methods."""
import json
import subprocess
import sys
import pathlib
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lib.hal.git_ops import GitHubOps


def _pr_status_json(conclusions=("SUCCESS",), mergeable="MERGEABLE"):
    checks = [{"conclusion": c} for c in conclusions]
    return json.dumps({
        "state": "OPEN",
        "mergeable": mergeable,
        "statusCheckRollup": checks,
        "mergeStateStatus": "CLEAN",
    })


class TestPrNumberFromUrl(unittest.TestCase):

    def test_pr_number_from_url_valid(self):
        ops = GitHubOps()
        self.assertEqual(ops.pr_number_from_url("https://github.com/org/repo/pull/42"), 42)

    def test_pr_number_from_url_malformed(self):
        ops = GitHubOps()
        self.assertEqual(ops.pr_number_from_url("https://github.com/org/repo"), 0)

    def test_pr_number_from_url_zero_on_no_match(self):
        ops = GitHubOps()
        self.assertEqual(ops.pr_number_from_url(""), 0)


class TestMergePr(unittest.TestCase):

    def test_merge_pr_returns_true_on_success(self):
        """subprocess.run returncode=0 → True."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            ops = GitHubOps()
            self.assertTrue(ops.merge_pr(42))

    def test_merge_pr_returns_false_on_failure(self):
        """subprocess.run returncode=1 → False."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            ops = GitHubOps()
            self.assertFalse(ops.merge_pr(42))

    def test_merge_pr_returns_false_on_timeout(self):
        """subprocess.run raising TimeoutExpired → False."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="gh", timeout=60)):
            ops = GitHubOps()
            self.assertFalse(ops.merge_pr(42))


class TestGetPrStatus(unittest.TestCase):

    def test_get_pr_status_ci_passing(self):
        """All conclusions SUCCESS → ci_passing=True."""
        with patch("lib.hal.git_ops._run", return_value=_pr_status_json(["SUCCESS", "SUCCESS"])):
            ops = GitHubOps()
            status = ops.get_pr_status(42)
        self.assertTrue(status["ci_passing"])

    def test_get_pr_status_ci_failing(self):
        """One FAILURE conclusion → ci_passing=False."""
        with patch("lib.hal.git_ops._run", return_value=_pr_status_json(["SUCCESS", "FAILURE"])):
            ops = GitHubOps()
            status = ops.get_pr_status(42)
        self.assertFalse(status["ci_passing"])

    def test_get_pr_status_mergeable_field(self):
        """mergeable='MERGEABLE' in JSON → mergeable=True."""
        with patch("lib.hal.git_ops._run", return_value=_pr_status_json(mergeable="MERGEABLE")):
            ops = GitHubOps()
            status = ops.get_pr_status(42)
        self.assertTrue(status["mergeable"])


if __name__ == "__main__":
    unittest.main()
