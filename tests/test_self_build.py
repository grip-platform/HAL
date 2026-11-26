"""Tests for HAL Self-Build Orchestrator."""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.hal.approval_gate import ApprovalGate, ApprovalLevel
from lib.hal.code_gen import CodeSpec, GeneratedCode, ValidationIssue, ValidationReport
from lib.hal.git_ops import GitMutator
from lib.hal.self_build import BuildConfig, BuildResult, SelfBuilder


class TestBuildConfig(unittest.TestCase):
    """Test BuildConfig frozen dataclass."""

    def test_defaults(self):
        config = BuildConfig()
        self.assertTrue(config.auto_branch)
        self.assertTrue(config.auto_commit)
        self.assertFalse(config.auto_pr)
        self.assertFalse(config.auto_merge)
        self.assertEqual(config.max_refinement_passes, 3)

    def test_frozen(self):
        config = BuildConfig()
        with self.assertRaises(AttributeError):
            config.auto_pr = True


class TestBuildResult(unittest.TestCase):
    """Test BuildResult frozen dataclass."""

    def test_success_result(self):
        result = BuildResult(
            proposal_action="add retry logic",
            module_path="/tmp/retry.py",
            test_path="/tmp/test_retry.py",
            validation_passed=True,
            tests_passed=True,
            branch="feat/retry",
            commit_hash="abc123",
        )
        self.assertTrue(result.validation_passed)
        self.assertTrue(result.tests_passed)
        self.assertEqual(result.branch, "feat/retry")
        self.assertEqual(result.error, "")

    def test_failure_result(self):
        result = BuildResult(
            proposal_action="broken",
            module_path="", test_path="",
            validation_passed=False, tests_passed=False,
            error="Syntax error",
        )
        self.assertFalse(result.validation_passed)
        self.assertEqual(result.error, "Syntax error")

    def test_frozen(self):
        result = BuildResult(
            proposal_action="x", module_path="", test_path="",
            validation_passed=True, tests_passed=True,
        )
        with self.assertRaises(AttributeError):
            result.error = "changed"


class TestSelfBuilder(unittest.TestCase):
    """Test SelfBuilder with mocked dependencies."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.registry = MagicMock()
        self.code_gen = MagicMock()
        self.git_mutator = MagicMock(spec=GitMutator)
        self.github_ops = MagicMock()

        self.gate = ApprovalGate(
            state_path=os.path.join(self.tmp_dir, "level.json"),
            pending_path=os.path.join(self.tmp_dir, "pending.json"),
        )
        self.gate.set_level(ApprovalLevel.FULL_AUTO)

        self.config = BuildConfig(auto_branch=True, auto_commit=True, auto_pr=False)
        self.builder = SelfBuilder(
            self.registry, self.code_gen, self.git_mutator,
            self.github_ops, self.gate, self.config,
        )

    def tearDown(self):
        for f in os.listdir(self.tmp_dir):
            os.unlink(os.path.join(self.tmp_dir, f))
        os.rmdir(self.tmp_dir)

    def _make_spec(self, name: str = "test_mod") -> CodeSpec:
        return CodeSpec(
            module_name=name, description="A test module",
            public_api=("class Foo",), depends_on=(), test_cases=("test_create",),
        )

    def _make_generated(self, spec: CodeSpec) -> GeneratedCode:
        return GeneratedCode(
            module_path=os.path.join(self.tmp_dir, f"{spec.module_name}.py"),
            test_path=os.path.join(self.tmp_dir, f"test_{spec.module_name}.py"),
            module_source="class Foo:\n    pass\n",
            test_source="import unittest\nclass T(unittest.TestCase):\n    def test_a(self): pass\n",
            spec=spec, model_used="test",
        )

    def test_build_from_spec_validation_fails(self):
        spec = self._make_spec()
        generated = self._make_generated(spec)
        report = ValidationReport(
            passed=False,
            issues=(ValidationIssue("ast", "critical", "Syntax error"),),
            ast_valid=False, security_clean=True, compiles=False,
        )
        self.code_gen.generate_and_validate.return_value = (generated, report)

        result = self.builder.build_from_spec(spec)
        self.assertFalse(result.validation_passed)
        self.assertIn("Validation failed", result.error)

    def test_build_from_spec_validation_passes(self):
        spec = self._make_spec()
        generated = self._make_generated(spec)
        report = ValidationReport(
            passed=True, issues=(),
            ast_valid=True, security_clean=True, compiles=True,
        )
        self.code_gen.generate_and_validate.return_value = (generated, report)
        self.git_mutator.create_branch.return_value = "feat/test-mod-123"
        self.git_mutator.add_and_commit.return_value = "abc1234"

        with patch.object(self.builder, "_run_tests", return_value=True):
            result = self.builder.build_from_spec(spec)

        self.assertTrue(result.validation_passed)
        self.assertTrue(result.tests_passed)
        self.assertEqual(result.branch, "feat/test-mod-123")
        self.assertEqual(result.commit_hash, "abc1234")

    def test_build_from_spec_blocked_by_gate(self):
        self.gate.set_level(ApprovalLevel.FULL_MANUAL)

        spec = self._make_spec()
        generated = self._make_generated(spec)
        report = ValidationReport(
            passed=True, issues=(),
            ast_valid=True, security_clean=True, compiles=True,
        )
        self.code_gen.generate_and_validate.return_value = (generated, report)

        result = self.builder.build_from_spec(spec)
        self.assertTrue(result.validation_passed)
        self.assertFalse(result.tests_passed)
        self.assertIn("approval gate", result.error)

    def test_build_from_proposal(self):
        proposal = MagicMock()
        proposal.action = "Create retry logic"

        spec = self._make_spec("retry")
        self.code_gen.spec_from_proposal.return_value = spec

        generated = self._make_generated(spec)
        report = ValidationReport(
            passed=True, issues=(),
            ast_valid=True, security_clean=True, compiles=True,
        )
        self.code_gen.generate_and_validate.return_value = (generated, report)
        self.git_mutator.create_branch.return_value = "feat/retry-123"
        self.git_mutator.add_and_commit.return_value = "def5678"

        with patch.object(self.builder, "_run_tests", return_value=True):
            result = self.builder.build_from_proposal(proposal)

        self.assertTrue(result.validation_passed)
        self.assertEqual(result.proposal_action, "Create retry logic")

    def test_build_from_proposal_exception(self):
        proposal = MagicMock()
        proposal.action = "broken"
        self.code_gen.spec_from_proposal.side_effect = RuntimeError("LLM down")

        result = self.builder.build_from_proposal(proposal)
        self.assertFalse(result.validation_passed)
        self.assertIn("LLM down", result.error)

    def test_no_branch_when_tests_fail(self):
        spec = self._make_spec()
        generated = self._make_generated(spec)
        report = ValidationReport(
            passed=True, issues=(),
            ast_valid=True, security_clean=True, compiles=True,
        )
        self.code_gen.generate_and_validate.return_value = (generated, report)

        with patch.object(self.builder, "_run_tests", return_value=False):
            result = self.builder.build_from_spec(spec)

        self.assertFalse(result.tests_passed)
        self.assertEqual(result.branch, "")
        self.assertEqual(result.commit_hash, "")

    def test_dry_run_does_not_write(self):
        proposal = MagicMock()
        proposal.action = "test dry run"

        spec = self._make_spec()
        generated = self._make_generated(spec)
        report = ValidationReport(
            passed=True, issues=(), ast_valid=True,
            security_clean=True, compiles=True,
        )
        self.code_gen.spec_from_proposal.return_value = spec
        self.code_gen.generate_and_validate.return_value = (generated, report)

        gen, rep = self.builder.dry_run(proposal)
        self.assertTrue(rep.passed)
        self.git_mutator.create_branch.assert_not_called()
        self.git_mutator.add_and_commit.assert_not_called()


class TestGitMutator(unittest.TestCase):
    """Test GitMutator class."""

    def test_add_and_commit_empty_files(self):
        mutator = GitMutator()
        self.assertEqual(mutator.add_and_commit([], "msg"), "")


if __name__ == "__main__":
    unittest.main()
