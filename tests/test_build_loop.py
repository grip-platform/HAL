"""Tests for HAL Build Loop."""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.hal.build_loop import BuildLoop, LoopIteration, LoopState, _CONVERGENCE_WINDOW
from lib.hal.rsi import Proposal
from lib.hal.self_build import BuildResult


class TestLoopIteration(unittest.TestCase):
    """Test LoopIteration frozen dataclass."""

    def test_creation(self):
        it = LoopIteration(
            iteration=1, proposals_found=3, builds_attempted=3,
            builds_succeeded=2, modules_added=2, tests_added=2,
            cost_usd=0.30, duration_s=5.0,
        )
        self.assertEqual(it.iteration, 1)
        self.assertEqual(it.builds_succeeded, 2)

    def test_frozen(self):
        it = LoopIteration(
            iteration=1, proposals_found=0, builds_attempted=0,
            builds_succeeded=0, modules_added=0, tests_added=0,
            cost_usd=0.0, duration_s=0.0,
        )
        with self.assertRaises(AttributeError):
            it.iteration = 2


class TestLoopState(unittest.TestCase):
    """Test LoopState frozen dataclass."""

    def test_converged(self):
        state = LoopState(
            iterations=(), total_modules=5, total_tests=5,
            total_cost_usd=0.50, converged=True,
            convergence_reason="No proposals",
        )
        self.assertTrue(state.converged)

    def test_not_converged(self):
        state = LoopState(
            iterations=(), total_modules=0, total_tests=0,
            total_cost_usd=0.0, converged=False,
            convergence_reason="Max iterations",
        )
        self.assertFalse(state.converged)


class TestBuildLoop(unittest.TestCase):
    """Test BuildLoop with mocked dependencies."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.state_path = os.path.join(self.tmp_dir, "build-loop.json")
        self.registry = MagicMock()
        self.rsi_engine = MagicMock()
        self.self_builder = MagicMock()

        self.loop = BuildLoop(
            self.registry, self.rsi_engine, self.self_builder,
            cost_per_build=0.10,
            state_path=self.state_path,
        )

    def tearDown(self):
        if os.path.exists(self.state_path):
            os.unlink(self.state_path)
        os.rmdir(self.tmp_dir)

    def _make_proposal(self, action: str = "test action") -> Proposal:
        return Proposal(
            type="skill", description=f"Pattern: {action}",
            pattern_id="abc123", confidence=0.8,
            action=action, auto_apply=False,
        )

    def _make_success_result(self) -> BuildResult:
        return BuildResult(
            proposal_action="test", module_path="/tmp/mod.py",
            test_path="/tmp/test_mod.py",
            validation_passed=True, tests_passed=True,
            branch="feat/test", commit_hash="abc123",
        )

    def _make_failure_result(self) -> BuildResult:
        return BuildResult(
            proposal_action="test", module_path="", test_path="",
            validation_passed=False, tests_passed=False,
            error="Failed",
        )

    def test_run_once_no_proposals(self):
        self.rsi_engine.analyse_for_build.return_value = []
        iteration = self.loop.run_once()
        self.assertEqual(iteration.proposals_found, 0)
        self.assertEqual(iteration.builds_attempted, 0)
        self.assertEqual(iteration.iteration, 1)

    def test_run_once_with_proposals(self):
        proposals = [self._make_proposal("a"), self._make_proposal("b")]
        self.rsi_engine.analyse_for_build.return_value = proposals
        self.self_builder.build_from_proposal.return_value = self._make_success_result()

        iteration = self.loop.run_once()
        self.assertEqual(iteration.proposals_found, 2)
        self.assertEqual(iteration.builds_attempted, 2)
        self.assertEqual(iteration.builds_succeeded, 2)
        self.assertEqual(iteration.modules_added, 2)

    def test_run_once_mixed_results(self):
        proposals = [self._make_proposal("a"), self._make_proposal("b")]
        self.rsi_engine.analyse_for_build.return_value = proposals
        self.self_builder.build_from_proposal.side_effect = [
            self._make_success_result(),
            self._make_failure_result(),
        ]

        iteration = self.loop.run_once()
        self.assertEqual(iteration.builds_succeeded, 1)
        self.assertEqual(iteration.modules_added, 1)

    def test_run_once_increments_iteration_count(self):
        self.rsi_engine.analyse_for_build.return_value = []
        self.loop.run_once()
        self.loop.run_once()
        iteration = self.loop.run_once()
        self.assertEqual(iteration.iteration, 3)

    def test_run_until_converge_empty(self):
        self.rsi_engine.analyse_for_build.return_value = []
        state = self.loop.run_until_converge(max_iterations=5)
        self.assertTrue(state.converged)
        self.assertIn("No proposals", state.convergence_reason)
        self.assertEqual(len(state.iterations), _CONVERGENCE_WINDOW)

    def test_run_until_converge_budget_exhausted(self):
        proposals = [self._make_proposal()]
        self.rsi_engine.analyse_for_build.return_value = proposals
        self.self_builder.build_from_proposal.return_value = self._make_success_result()

        state = self.loop.run_until_converge(max_iterations=100, budget_usd=0.15)
        self.assertFalse(state.converged)
        self.assertIn("Budget", state.convergence_reason)

    def test_run_until_converge_max_iterations(self):
        proposals = [self._make_proposal()]
        self.rsi_engine.analyse_for_build.return_value = proposals
        self.self_builder.build_from_proposal.return_value = self._make_success_result()

        state = self.loop.run_until_converge(max_iterations=3, budget_usd=100.0)
        self.assertFalse(state.converged)
        self.assertIn("Max iterations", state.convergence_reason)
        self.assertEqual(len(state.iterations), 3)

    def test_run_until_converge_accumulates_modules(self):
        proposals = [self._make_proposal()]
        self.rsi_engine.analyse_for_build.return_value = proposals
        self.self_builder.build_from_proposal.return_value = self._make_success_result()

        state = self.loop.run_until_converge(max_iterations=3, budget_usd=100.0)
        self.assertEqual(state.total_modules, 3)
        self.assertEqual(state.total_tests, 3)

    def test_get_state_before_any_runs(self):
        state = self.loop.get_state()
        self.assertFalse(state.converged)
        self.assertEqual(state.total_modules, 0)
        self.assertEqual(state.convergence_reason, "Not started")

    def test_get_state_after_run(self):
        self.rsi_engine.analyse_for_build.return_value = []
        self.loop.run_once()
        state = self.loop.get_state()
        self.assertEqual(len(state.iterations), 1)
        self.assertEqual(state.convergence_reason, "In progress")

    def test_state_persisted_to_disk(self):
        self.rsi_engine.analyse_for_build.return_value = []
        self.loop.run_once()
        self.assertTrue(os.path.exists(self.state_path))

    def test_convergence_resets_on_new_proposals(self):
        """If proposals appear after an empty streak, convergence resets."""
        proposals = [self._make_proposal()]
        # Iteration 1: empty, 2: has proposals, 3: empty, 4: empty -> converge
        self.rsi_engine.analyse_for_build.side_effect = [
            [],
            proposals,
            [],
            [],
        ]
        self.self_builder.build_from_proposal.return_value = self._make_success_result()

        state = self.loop.run_until_converge(max_iterations=10, budget_usd=100.0)
        self.assertTrue(state.converged)
        self.assertEqual(len(state.iterations), 4)


class TestRSIAnalyseForBuild(unittest.TestCase):
    """Test the RSI engine's analyse_for_build method."""

    def test_filters_by_type_and_confidence(self):
        from lib.hal.rsi import RSIEngine
        registry = MagicMock()
        engine = RSIEngine(registry)

        # Mock the base analyse to return mixed proposals
        with patch.object(engine, "analyse") as mock_analyse:
            mock_analyse.return_value = [
                Proposal(type="skill", description="s1", pattern_id="a",
                         confidence=0.8, action="a1"),
                Proposal(type="config", description="s2", pattern_id="b",
                         confidence=0.9, action="a2"),
                Proposal(type="hook", description="s3", pattern_id="c",
                         confidence=0.5, action="a3"),
                Proposal(type="skill", description="s4", pattern_id="d",
                         confidence=0.3, action="a4"),
            ]

            results = engine.analyse_for_build([])

        # Only type=skill/hook with confidence >= 0.7
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].type, "skill")
        self.assertEqual(results[0].action, "a1")


if __name__ == "__main__":
    unittest.main()
