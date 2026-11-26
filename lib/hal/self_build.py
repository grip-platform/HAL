"""HAL Self-Build Orchestrator — wire RSI proposals through code gen to PRs.

Takes an RSI Proposal, generates code via CodeGenerator, validates through
safety gates, commits to a branch, and creates a PR. The pipeline that
closes the HAL self-improvement loop.

Usage:
    from lib.hal.self_build import SelfBuilder, BuildConfig

    builder = SelfBuilder(registry, code_gen, git_mutator, github_ops, gate)
    result = builder.build_from_proposal(proposal)
    if result.pr_url:
        print(f"PR created: {result.pr_url}")
"""
import os
import re
import subprocess
import time
from dataclasses import dataclass

from lib.hal.approval_gate import ApprovalGate
from lib.hal.code_gen import CodeGenerator, CodeSpec, GeneratedCode, ValidationReport
from lib.hal.git_ops import GitHubOps, GitMutator
from lib.hal.home import HAL_HOME as _HAL_HOME


@dataclass(frozen=True)
class BuildConfig:
    """Configuration for self-build operations."""

    auto_branch: bool = True
    auto_commit: bool = True
    auto_pr: bool = False
    auto_merge: bool = False
    max_refinement_passes: int = 3
    auto_review: bool = True
    max_review_passes: int = 2
    auto_merge_on_review_pass: bool = False


@dataclass(frozen=True)
class BuildResult:
    """Result of a self-build operation."""

    proposal_action: str
    module_path: str
    test_path: str
    validation_passed: bool
    tests_passed: bool
    branch: str = ""
    commit_hash: str = ""
    pr_url: str = ""
    error: str = ""
    review_passed: bool = False
    review_rounds: int = 0
    merged: bool = False


class SelfBuilder:
    """Orchestrates the full self-build pipeline."""

    def __init__(
        self,
        registry: "ProviderRegistry",
        code_gen: CodeGenerator,
        git_mutator: GitMutator,
        github_ops: GitHubOps,
        approval_gate: ApprovalGate,
        config: BuildConfig | None = None,
    ):
        self._registry = registry
        self._code_gen = code_gen
        self._git = git_mutator
        self._gh = github_ops
        self._gate = approval_gate
        self._config = config or BuildConfig()
        self._last_generated: GeneratedCode | None = None

    def build_from_proposal(self, proposal: "Proposal") -> BuildResult:
        """Full pipeline: proposal -> spec -> generate -> validate -> git -> PR."""
        try:
            spec = self._code_gen.spec_from_proposal(proposal)
            return self.build_from_spec(spec, proposal_action=proposal.action)
        except Exception as e:
            return BuildResult(
                proposal_action=getattr(proposal, "action", ""),
                module_path="", test_path="",
                validation_passed=False, tests_passed=False,
                error=str(e)[:500],
            )

    def build_from_spec(self, spec: CodeSpec, proposal_action: str = "") -> BuildResult:
        """Build from a structured spec (skip proposal-to-spec LLM call)."""
        generated, report = self._code_gen.generate_and_validate(spec)
        if not report.passed:
            error_msg = "; ".join(i.message for i in report.issues[:5])
            return BuildResult(
                proposal_action=proposal_action,
                module_path=generated.module_path,
                test_path=generated.test_path,
                validation_passed=False, tests_passed=False,
                error=f"Validation failed: {error_msg}",
            )
        if not self._gate.check("write_module", f"Write {spec.module_name}"):
            return BuildResult(
                proposal_action=proposal_action,
                module_path=generated.module_path,
                test_path=generated.test_path,
                validation_passed=True, tests_passed=False,
                error="Blocked by approval gate: write_module",
            )
        self._last_generated = generated  # captured for build_with_review()
        tests_passed = self._write_and_test(generated)
        return self._git_phase(spec, generated, proposal_action, tests_passed)

    def _git_phase(
        self, spec: CodeSpec, generated: GeneratedCode, proposal_action: str, tests_passed: bool,
    ) -> BuildResult:
        """Orchestrate branch, commit, PR from a completed test run."""
        branch = ""
        commit_hash = ""
        pr_url = ""
        if tests_passed and self._config.auto_branch:
            branch = self._create_branch(spec.module_name)
        if tests_passed and branch and self._config.auto_commit:
            if self._gate.check("commit", f"Commit {spec.module_name}"):
                commit_hash = self._commit(generated, spec)
        if commit_hash and self._config.auto_pr:
            if self._gate.check("create_pr", f"PR for {spec.module_name}"):
                pr_url = self._create_pr(spec, branch)
        return BuildResult(
            proposal_action=proposal_action,
            module_path=generated.module_path,
            test_path=generated.test_path,
            validation_passed=True,
            tests_passed=tests_passed,
            branch=branch,
            commit_hash=commit_hash,
            pr_url=pr_url,
        )

    def build_with_review(self, proposal: "Proposal") -> BuildResult:
        """Extended pipeline: build → review loop → optional merge.

        Phase 1: build_from_proposal (existing flow — may create PR)
        Phase 2: _run_review_loop (only if tests_passed AND auto_review)
        Phase 3: _attempt_merge() if auto_merge_on_review_pass AND pr_url
        """
        from lib.hal.code_reviewer import CodeReviewer

        base = self.build_from_proposal(proposal)
        if not base.tests_passed or not self._config.auto_review:
            return base
        generated = self._last_generated
        if generated is None:
            return base

        reviewer = CodeReviewer()
        review_passed, review_rounds, _ = self._run_review_loop(generated, reviewer)

        merged = False
        if review_passed and base.pr_url and self._config.auto_merge_on_review_pass:
            merged = self._attempt_merge(base.pr_url)

        return BuildResult(
            proposal_action=base.proposal_action,
            module_path=base.module_path,
            test_path=base.test_path,
            validation_passed=base.validation_passed,
            tests_passed=base.tests_passed,
            branch=base.branch,
            commit_hash=base.commit_hash,
            pr_url=base.pr_url,
            error=base.error,
            review_passed=review_passed,
            review_rounds=review_rounds,
            merged=merged,
        )

    def _run_review_loop(
        self, generated: GeneratedCode, reviewer: "CodeReviewer",
    ) -> tuple[bool, int, GeneratedCode]:
        """Run up to max_review_passes review + refine cycles.

        Returns (review_passed, rounds_taken, final_generated).
        Each blocked round: gate check → refine → re-test before next review.
        """
        review_passed = False
        review_rounds = 0
        for _ in range(self._config.max_review_passes):
            review_rounds += 1
            result = reviewer.review(generated)
            if result.severity != "block":
                review_passed = True
                break
            if not self._gate.check("code_review", "Fix review-blocking issues"):
                break
            error_str = _review_issues_to_refine_prompt(result)
            try:
                generated = self._code_gen.refine(generated, error_str)
                self._last_generated = generated
                if not self._write_and_test(generated):
                    break
            except Exception:
                break
        return review_passed, review_rounds, generated

    def _attempt_merge(self, pr_url: str) -> bool:
        """Poll PR status and merge when CI passes. Hard deadline: 120 s.

        Gated by approval_gate("merge_pr") — auto-approved only at FULL_AUTO.
        """
        if not self._gate.check("merge_pr", f"Merge PR {pr_url}"):
            return False
        pr_number = self._gh.pr_number_from_url(pr_url)
        if not pr_number:
            return False
        deadline = time.time() + 120
        while time.time() < deadline:
            status = self._gh.get_pr_status(pr_number)
            if status["ci_passing"] and status["mergeable"]:
                return self._gh.merge_pr(pr_number)
            time.sleep(10)
        return False

    def dry_run(self, proposal: "Proposal") -> tuple[GeneratedCode, ValidationReport]:
        """Generate and validate without writing to disk."""
        spec = self._code_gen.spec_from_proposal(proposal)
        return self._code_gen.generate_and_validate(spec)

    def _write_and_test(self, generated: GeneratedCode) -> bool:
        """Write files to disk and run tests."""
        os.makedirs(os.path.dirname(generated.module_path), exist_ok=True)
        os.makedirs(os.path.dirname(generated.test_path), exist_ok=True)

        with open(generated.module_path, "w") as f:
            f.write(generated.module_source)
        with open(generated.test_path, "w") as f:
            f.write(generated.test_source)

        return self._run_tests(generated.test_path)

    def _run_tests(self, test_path: str) -> bool:
        """Run the specific test file."""
        try:
            result = subprocess.run(
                ["python3", "-m", "unittest", test_path, "-q"],
                capture_output=True, text=True, timeout=60,
                cwd=_HAL_HOME,
                env={**os.environ, "HAL_HOME": _HAL_HOME},
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def _create_branch(self, module_name: str) -> str:
        """Create a feature branch for the new module."""
        slug = re.sub(r"[^a-z0-9]+", "-", module_name.lower()).strip("-")
        return self._git.create_branch(f"feat/self-build-{slug}-{int(time.time())}")

    def _commit(self, generated: GeneratedCode, spec: CodeSpec) -> str:
        """Stage and commit the generated files."""
        files = [generated.module_path, generated.test_path]
        message = f"feat({spec.module_name}): auto-generated by HAL self-build\n\n{spec.description}"
        return self._git.add_and_commit(files, message)

    def _create_pr(self, spec: CodeSpec, branch: str) -> str:
        """Push branch and create PR."""
        self._git.push(branch)
        title = f"feat({spec.module_name}): HAL self-build"
        body = (
            f"## Summary\n"
            f"- Auto-generated module: `{spec.module_name}`\n"
            f"- Description: {spec.description}\n\n"
            f"## Test plan\n"
            f"- [ ] Verify module passes py_compile\n"
            f"- [ ] Verify all tests pass\n"
        )
        return self._gh.create_pr(title, body)


def _review_issues_to_refine_prompt(review: "ReviewResult") -> str:
    """Format error-severity review issues into a refine() error string.

    Module-level pure function — keeps SelfBuilder at ≤7 public methods (ISP).
    Only "error" severity issues are surfaced; warnings are not blocking.
    """
    errors = [i for i in review.issues if i.severity == "error"]
    if not errors:
        return review.summary
    lines = [f"Line {i.line}: [{i.gate}] {i.message}" for i in errors]
    return "Review blocking issues:\n" + "\n".join(lines)
