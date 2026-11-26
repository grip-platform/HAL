"""Tests for lib/hal/self_build.py — build_with_review(), _attempt_merge(), helpers."""
import sys
import pathlib
import time
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lib.hal.self_build import SelfBuilder, BuildConfig, BuildResult, _review_issues_to_refine_prompt
from lib.hal.code_reviewer import ReviewIssue, ReviewResult
from lib.hal.build_loop import LoopIteration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_builder(auto_review=True, auto_merge=False, max_review_passes=2):
    registry = MagicMock()
    code_gen = MagicMock()
    git = MagicMock()
    gh = MagicMock()
    gate = MagicMock()
    gate.check.return_value = True
    config = BuildConfig(
        auto_review=auto_review,
        auto_merge_on_review_pass=auto_merge,
        max_review_passes=max_review_passes,
    )
    builder = SelfBuilder(registry, code_gen, git, gh, gate, config)
    return builder, code_gen, git, gh, gate


def _base_result(tests_passed=True, pr_url="https://github.com/org/repo/pull/99"):
    return BuildResult(
        proposal_action="add",
        module_path="/tmp/mod.py",
        test_path="/tmp/test_mod.py",
        validation_passed=True,
        tests_passed=tests_passed,
        pr_url=pr_url,
    )


def _review_pass():
    return ReviewResult(passed=True, issues=(), severity="pass", summary="All gates passed.")


def _review_block():
    issue = ReviewIssue(gate="safety", severity="error", line=1, message="dangerous call")
    return ReviewResult(passed=False, issues=(issue,), severity="block", summary="BLOCKED: 1 error(s)")


# ---------------------------------------------------------------------------
# build_with_review tests
# ---------------------------------------------------------------------------

class TestBuildWithReview(unittest.TestCase):

    def _run_with_reviewer(self, builder, base, review_results):
        """Patch build_from_proposal + CodeReviewer.review and call build_with_review."""
        proposal = MagicMock()
        builder._last_generated = MagicMock()
        builder._code_gen.refine.return_value = builder._last_generated

        review_iter = iter(review_results)

        with patch.object(builder, "build_from_proposal", return_value=base), \
             patch("lib.hal.code_reviewer.CodeReviewer") as MockReviewer:
            instance = MockReviewer.return_value
            instance.review.side_effect = lambda g: next(review_iter)
            result = builder.build_with_review(proposal)

        return result, instance

    def test_review_passes_first_round(self):
        """Reviewer returns pass first try → review_passed=True, review_rounds=1."""
        builder, *_ = _make_builder()
        result, _ = self._run_with_reviewer(builder, _base_result(), [_review_pass()])
        self.assertTrue(result.review_passed)
        self.assertEqual(result.review_rounds, 1)

    def test_review_block_then_pass(self):
        """Reviewer returns block then pass → review_rounds=2, review_passed=True."""
        builder, *_ = _make_builder(max_review_passes=3)
        with patch.object(builder, "_write_and_test", return_value=True):
            result, _ = self._run_with_reviewer(
                builder, _base_result(), [_review_block(), _review_pass()]
            )
        self.assertTrue(result.review_passed)
        self.assertEqual(result.review_rounds, 2)

    def test_always_blocks_max_passes(self):
        """Reviewer always blocks → review_passed=False, review_rounds==max_review_passes."""
        builder, *_ = _make_builder(max_review_passes=2)
        with patch.object(builder, "_write_and_test", return_value=True):
            result, _ = self._run_with_reviewer(
                builder, _base_result(), [_review_block(), _review_block()]
            )
        self.assertFalse(result.review_passed)
        self.assertEqual(result.review_rounds, 2)

    def test_auto_review_false_skips_reviewer(self):
        """auto_review=False → CodeReviewer.review never called."""
        builder, *_ = _make_builder(auto_review=False)
        proposal = MagicMock()
        builder._last_generated = MagicMock()

        with patch.object(builder, "build_from_proposal", return_value=_base_result()), \
             patch("lib.hal.code_reviewer.CodeReviewer") as MockReviewer:
            builder.build_with_review(proposal)
            MockReviewer.return_value.review.assert_not_called()

    def test_tests_fail_skips_review(self):
        """tests_passed=False in base result → reviewer never called."""
        builder, *_ = _make_builder()
        proposal = MagicMock()

        with patch.object(builder, "build_from_proposal", return_value=_base_result(tests_passed=False)), \
             patch("lib.hal.code_reviewer.CodeReviewer") as MockReviewer:
            builder.build_with_review(proposal)
            MockReviewer.return_value.review.assert_not_called()


# ---------------------------------------------------------------------------
# _attempt_merge tests
# ---------------------------------------------------------------------------

class TestAttemptMerge(unittest.TestCase):

    def test_gate_blocks_merge_at_approve_pr(self):
        """gate.check('merge_pr',...) returns False → merged=False."""
        builder, _, _, gh, gate = _make_builder(auto_merge=True)
        gate.check.return_value = False
        result = builder._attempt_merge("https://github.com/org/repo/pull/99")
        self.assertFalse(result)

    def test_gate_passes_at_full_auto_merges(self):
        """gate passes, CI passing, mergeable → merge_pr called, returns True."""
        builder, _, _, gh, gate = _make_builder(auto_merge=True)
        gate.check.return_value = True
        gh.pr_number_from_url.return_value = 99
        gh.get_pr_status.return_value = {"ci_passing": True, "mergeable": True}
        gh.merge_pr.return_value = True

        result = builder._attempt_merge("https://github.com/org/repo/pull/99")
        self.assertTrue(result)
        gh.merge_pr.assert_called_once_with(99)

    def test_ci_timeout_merge_false(self):
        """get_pr_status always returns ci_passing=False → merged=False after deadline."""
        builder, _, _, gh, gate = _make_builder(auto_merge=True)
        gate.check.return_value = True
        gh.pr_number_from_url.return_value = 99
        gh.get_pr_status.return_value = {"ci_passing": False, "mergeable": True}

        # Advance time past the 120s deadline without sleeping
        fake_times = [0, 0, 121]  # first call=start, deadline calc, loop check
        fake_time_iter = iter(fake_times)

        def fake_time():
            try:
                return next(fake_time_iter)
            except StopIteration:
                return 999  # always past deadline

        with patch("lib.hal.self_build.time") as mock_time:
            mock_time.time.side_effect = fake_time
            mock_time.sleep = MagicMock()
            result = builder._attempt_merge("https://github.com/org/repo/pull/99")

        self.assertFalse(result)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestReviewIssuesToRefinePrompt(unittest.TestCase):

    def test_review_issues_to_refine_prompt_errors_only(self):
        """Has error and warning issues → only errors appear in output."""
        error = ReviewIssue(gate="safety", severity="error", line=5, message="bad call")
        warn = ReviewIssue(gate="safety", severity="warning", line=10, message="risky")
        review = ReviewResult(
            passed=False,
            issues=(error, warn),
            severity="block",
            summary="BLOCKED: 1 error(s), 1 warning(s)",
        )
        output = _review_issues_to_refine_prompt(review)
        self.assertIn("bad call", output)
        self.assertNotIn("risky", output)

    def test_review_issues_to_refine_prompt_no_errors(self):
        """Only warning issues → returns review.summary."""
        warn = ReviewIssue(gate="safety", severity="warning", line=10, message="risky")
        review = ReviewResult(
            passed=True,
            issues=(warn,),
            severity="warn",
            summary="WARNED: 1 warning(s)",
        )
        output = _review_issues_to_refine_prompt(review)
        self.assertEqual(output, review.summary)


# ---------------------------------------------------------------------------
# LoopIteration dataclass test
# ---------------------------------------------------------------------------

class TestLoopIteration(unittest.TestCase):

    def test_loop_iteration_reviews_passed_field(self):
        """LoopIteration.reviews_passed defaults to 0."""
        iteration = LoopIteration(
            iteration=1,
            proposals_found=2,
            builds_attempted=2,
            builds_succeeded=1,
            modules_added=1,
            tests_added=1,
            cost_usd=0.01,
            duration_s=3.5,
        )
        self.assertEqual(iteration.reviews_passed, 0)


if __name__ == "__main__":
    unittest.main()
