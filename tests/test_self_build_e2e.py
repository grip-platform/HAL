"""End-to-end integration test for HAL self-build pipeline.

Tests the full loop: RSI proposal -> CodeSpec -> code generation ->
validation -> file write -> test execution. Uses mocked LLM responses
to simulate the pipeline without requiring live API keys.
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.hal.approval_gate import ApprovalGate, ApprovalLevel
from lib.hal.build_loop import BuildLoop
from lib.hal.code_gen import CodeGenerator, CodeSpec, GeneratedCode
from lib.hal.git_ops import GitMutator
from lib.hal.rsi import Proposal, RSIEngine
from lib.hal.self_build import BuildConfig, SelfBuilder


# Realistic generated module source — this is what the LLM would produce
_MOCK_MODULE_SOURCE = '''\
"""HAL Retry Logic — exponential backoff with jitter."""
import random
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behaviour."""
    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter: bool = True


@dataclass(frozen=True)
class RetryResult:
    """Outcome of a retry sequence."""
    succeeded: bool
    attempts: int
    last_error: str = ""


def retry(fn, config: RetryConfig | None = None) -> RetryResult:
    """Execute fn with exponential backoff. Returns RetryResult."""
    config = config or RetryConfig()
    last_error = ""
    for attempt in range(1, config.max_attempts + 1):
        try:
            fn()
            return RetryResult(succeeded=True, attempts=attempt)
        except Exception as e:
            last_error = str(e)[:200]
            if attempt < config.max_attempts:
                delay = min(config.base_delay_s * (2 ** (attempt - 1)), config.max_delay_s)
                if config.jitter:
                    delay *= random.uniform(0.5, 1.5)
                time.sleep(delay)
    return RetryResult(succeeded=False, attempts=config.max_attempts, last_error=last_error)
'''

# Realistic generated test source
_MOCK_TEST_SOURCE = '''\
"""Tests for retry logic module."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.hal.retry_logic import RetryConfig, RetryResult, retry


class TestRetryConfig(unittest.TestCase):
    def test_defaults(self):
        config = RetryConfig()
        self.assertEqual(config.max_attempts, 3)
        self.assertEqual(config.base_delay_s, 1.0)
        self.assertTrue(config.jitter)

    def test_frozen(self):
        config = RetryConfig()
        with self.assertRaises(AttributeError):
            config.max_attempts = 5


class TestRetryResult(unittest.TestCase):
    def test_success(self):
        result = RetryResult(succeeded=True, attempts=1)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.last_error, "")

    def test_failure(self):
        result = RetryResult(succeeded=False, attempts=3, last_error="timeout")
        self.assertFalse(result.succeeded)
        self.assertEqual(result.last_error, "timeout")


class TestRetry(unittest.TestCase):
    def test_succeeds_first_try(self):
        result = retry(lambda: None, RetryConfig(max_attempts=3, base_delay_s=0.0))
        self.assertTrue(result.succeeded)
        self.assertEqual(result.attempts, 1)

    def test_fails_then_succeeds(self):
        call_count = [0]
        def flaky():
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("transient")
        result = retry(flaky, RetryConfig(max_attempts=3, base_delay_s=0.001, jitter=False))
        self.assertTrue(result.succeeded)
        self.assertEqual(result.attempts, 2)

    def test_exhausts_retries(self):
        def always_fail():
            raise RuntimeError("permanent")
        result = retry(always_fail, RetryConfig(max_attempts=2, base_delay_s=0.001, jitter=False))
        self.assertFalse(result.succeeded)
        self.assertEqual(result.attempts, 2)
        self.assertIn("permanent", result.last_error)


if __name__ == "__main__":
    unittest.main()
'''


class TestE2ECodeGeneration(unittest.TestCase):
    """Test that CodeGenerator produces valid, testable code."""

    def setUp(self):
        self.registry = MagicMock()
        provider = MagicMock()
        provider.name = "groq"
        provider.healthy = True
        self.registry.list_providers.return_value = [provider]

        self.gen = CodeGenerator(self.registry)

    def test_generated_module_passes_ast(self):
        """The mock module source should pass AST validation."""
        import ast
        tree = ast.parse(_MOCK_MODULE_SOURCE)
        class_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        self.assertIn("RetryConfig", class_names)
        self.assertIn("RetryResult", class_names)

    def test_generated_test_passes_ast(self):
        """The mock test source should pass AST validation."""
        import ast
        ast.parse(_MOCK_TEST_SOURCE)

    def test_validate_mock_module(self):
        """Full validation pipeline on mock module."""
        spec = CodeSpec(
            module_name="retry_logic",
            description="Retry with exponential backoff",
            public_api=("class RetryConfig", "class RetryResult", "def retry()"),
            depends_on=(), test_cases=("test_succeeds", "test_fails"),
        )
        generated = GeneratedCode(
            module_path="/tmp/retry_logic.py",
            test_path="/tmp/test_retry_logic.py",
            module_source=_MOCK_MODULE_SOURCE,
            test_source=_MOCK_TEST_SOURCE,
            spec=spec, model_used="groq/llama",
        )
        report = self.gen.validate(generated)
        self.assertTrue(report.ast_valid, f"AST issues: {report.issues}")
        self.assertTrue(report.security_clean, f"Security issues: {report.issues}")
        self.assertTrue(report.compiles, f"Compile issues: {report.issues}")
        self.assertTrue(report.passed)


class TestE2ESelfBuildPipeline(unittest.TestCase):
    """Test the full self-build pipeline with mock LLM."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.hal_tmp = os.path.join(self.tmp_dir, "hal")
        self.lib_dir = os.path.join(self.hal_tmp, "lib", "hal")
        self.test_dir = os.path.join(self.hal_tmp, "tests")
        os.makedirs(self.lib_dir)
        os.makedirs(self.test_dir)

        self.registry = MagicMock()
        provider = MagicMock()
        provider.name = "groq"
        provider.healthy = True
        self.registry.list_providers.return_value = [provider]

        # Mock LLM returns realistic code
        self.registry.chat.side_effect = [
            MagicMock(text=f'```python\n{_MOCK_MODULE_SOURCE}\n```'),
            MagicMock(text=f'```python\n{_MOCK_TEST_SOURCE}\n```'),
        ]

        self.code_gen = CodeGenerator(self.registry)
        self.git_mutator = MagicMock(spec=GitMutator)
        self.git_mutator.create_branch.return_value = "feat/retry-logic-123"
        self.git_mutator.add_and_commit.return_value = "abc1234"
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
        shutil.rmtree(self.tmp_dir)

    def test_full_pipeline_from_spec(self):
        """Build a module from spec: generate -> validate -> write -> (mock) git."""
        spec = CodeSpec(
            module_name="retry_logic",
            description="Retry with exponential backoff",
            public_api=("class RetryConfig", "class RetryResult", "def retry()"),
            depends_on=(), test_cases=("test_succeeds", "test_fails"),
        )

        # Override paths to use tmp dir
        with patch("lib.hal.code_gen._MODULE_DIR", self.lib_dir), \
             patch("lib.hal.code_gen._TEST_DIR", self.test_dir):
            generated, report = self.code_gen.generate_and_validate(spec)

        self.assertTrue(report.passed, f"Validation failed: {report.issues}")
        self.assertIn("RetryConfig", generated.module_source)
        self.assertIn("class RetryResult", generated.module_source)
        self.assertGreater(generated.confidence, 0.5)

    def test_full_pipeline_from_proposal(self):
        """Build from an RSI proposal (spec_from_proposal + generate)."""
        proposal = MagicMock()
        proposal.action = "Create a skill codifying: retry with backoff"
        proposal.description = "Pattern seen 5x: retry with backoff"

        # Add spec_from_proposal LLM response
        spec_response = MagicMock(
            text='{"module_name": "retry_logic", "description": "Retry with exponential backoff", "public_api": ["class RetryConfig", "def retry()"], "depends_on": [], "test_cases": ["test_succeeds_first_try", "test_exhausts_retries"]}'
        )
        self.registry.chat.side_effect = [
            spec_response,
            MagicMock(text=f'```python\n{_MOCK_MODULE_SOURCE}\n```'),
            MagicMock(text=f'```python\n{_MOCK_TEST_SOURCE}\n```'),
        ]

        with patch("lib.hal.code_gen._MODULE_DIR", self.lib_dir), \
             patch("lib.hal.code_gen._TEST_DIR", self.test_dir), \
             patch.object(self.builder, "_run_tests", return_value=True):
            result = self.builder.build_from_proposal(proposal)

        self.assertTrue(result.validation_passed)
        self.assertTrue(result.tests_passed)
        self.assertEqual(result.branch, "feat/retry-logic-123")
        self.assertEqual(result.commit_hash, "abc1234")


class TestE2EBuildLoop(unittest.TestCase):
    """Test the build loop converges correctly."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.loop_patch = patch(
            "lib.hal.build_loop._STATE_PATH",
            os.path.join(self.tmp_dir, "loop.json"),
        )
        self.loop_patch.start()

        self.registry = MagicMock()
        self.rsi = MagicMock(spec=RSIEngine)
        self.builder = MagicMock(spec=SelfBuilder)

    def tearDown(self):
        self.loop_patch.stop()
        shutil.rmtree(self.tmp_dir)

    def test_loop_converges_after_work(self):
        """Simulate: 2 iterations with proposals, then convergence."""
        from lib.hal.self_build import BuildResult

        proposal = Proposal(
            type="skill", description="Retry pattern",
            pattern_id="abc", confidence=0.8,
            action="Create retry skill",
        )

        success = BuildResult(
            proposal_action="test", module_path="/tmp/m.py",
            test_path="/tmp/t.py", validation_passed=True,
            tests_passed=True, branch="feat/x", commit_hash="abc",
        )

        self.rsi.analyse_for_build.side_effect = [
            [proposal],
            [proposal],
            [],
            [],
        ]
        self.builder.build_from_proposal.return_value = success

        loop = BuildLoop(self.registry, self.rsi, self.builder, cost_per_build=0.10)
        state = loop.run_until_converge(max_iterations=10, budget_usd=5.0)

        self.assertTrue(state.converged)
        self.assertEqual(state.total_modules, 2)
        self.assertEqual(len(state.iterations), 4)
        self.assertLessEqual(state.total_cost_usd, 0.50)


if __name__ == "__main__":
    unittest.main()
