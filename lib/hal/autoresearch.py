"""AutoResearch loop for HAL — Karpathy Loop applied to routing config.

Autonomous hill-climbing: propose config delta → benchmark → keep/discard.
Runs overnight, unattended. Human directs via config/rsi-direction.md.

Architecture:
- Single optimization target: routing config + model catalog
- Objective metric: composite score = pass_rate * (1 - normalised_cost)
- Fixed time budget per experiment (configurable, default 300s)
- Greedy hill-climbing: keep improvements, revert regressions
- Goodhart protection: self-reflection gate on every candidate
"""
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)
_DEFAULT_BUDGET_S = 300      # 5-minute window per experiment (Karpathy pattern)
_OVERFIT_SIGNALS = ["hardcod", "magic number", "specific test", "rubric"]


@dataclass
class Experiment:
    """Result of one autoresearch iteration."""

    iteration: int
    description: str
    score_before: float
    score_after: float
    kept: bool
    duration_s: float
    reflection_passed: bool = True

    @property
    def delta(self) -> float:
        return self.score_after - self.score_before


@dataclass
class AutoResearchResult:
    """Aggregate result of a hill-climbing session."""

    experiments: list[Experiment] = field(default_factory=list)
    best_score: float = 0.0
    total_duration_s: float = 0.0

    def summary(self) -> str:
        kept = sum(1 for e in self.experiments if e.kept)
        return (
            f"{len(self.experiments)} experiments, {kept} kept, "
            f"best score {self.best_score:.3f}"
        )


def _parse_test_output(output: str) -> tuple[int, int]:
    """Return (total, failed) from unittest output."""
    total = 0
    failed = 0
    for line in output.splitlines():
        if line.startswith("Ran "):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    total = int(parts[1])
                except ValueError:
                    pass
        if "FAILED" in line:
            nums = re.findall(r"\d+", line)
            failed = sum(int(n) for n in nums)
    return total, failed


def _run_test_suite(repo_root: str, timeout_s: float = _DEFAULT_BUDGET_S) -> float:
    """Run the HAL test suite and return a score in [0.0, 1.0].

    Score = passing_tests / total_tests.
    Returns 0.0 on timeout or suite error.
    """
    try:
        result = subprocess.run(
            ["python3", "-m", "unittest", "discover", "-s", "tests"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        output = result.stderr + result.stdout
        total, failed = _parse_test_output(output)
        if total == 0:
            return 0.0
        if "OK" in output and "FAILED" not in output:
            return 1.0
        return max(0.0, (total - failed) / total)
    except subprocess.TimeoutExpired:
        _log.warning("Test suite timed out after %.0fs", timeout_s)
        return 0.0
    except Exception as exc:
        _log.exception("Test suite run failed: %s", exc)
        return 0.0


class AutoResearchLoop:
    """Karpathy Loop applied to HAL's routing configuration.

    Greedy hill-climbing: propose → score → keep/discard.
    Human direction comes from config/rsi-direction.md (program.md equivalent).
    """

    def __init__(
        self,
        repo_root: str | None = None,
        budget_s: float = _DEFAULT_BUDGET_S,
    ) -> None:
        self._repo = repo_root or os.path.expanduser("~/.hal")
        self._budget_s = budget_s
        self._direction = self._load_direction()

    def _load_direction(self) -> str:
        path = os.path.join(self._repo, "config", "rsi-direction.md")
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "Improve HAL's routing reliability and cost efficiency."

    def score(self) -> float:
        """Current quality score of the repo (0.0 – 1.0)."""
        return _run_test_suite(self._repo, self._budget_s)

    def _goodhart_check(self, description: str) -> bool:
        """Self-reflection gate (AutoAgent insight: agents overfit).

        Rejects changes that signal overfitting to specific test details.
        """
        lower = description.lower()
        return not any(sig in lower for sig in _OVERFIT_SIGNALS)

    def run_experiment(
        self,
        description: str,
        apply_fn,
        revert_fn,
    ) -> Experiment:
        """Run one hill-climbing step.

        apply_fn(): applies the proposed change.
        revert_fn(): reverts it if score did not improve.
        """
        score_before = self.score()
        start = time.monotonic()

        reflection_passed = self._goodhart_check(description)
        if not reflection_passed:
            _log.warning("Goodhart gate blocked experiment: %s", description)
            return Experiment(
                iteration=0,
                description=description,
                score_before=score_before,
                score_after=score_before,
                kept=False,
                duration_s=time.monotonic() - start,
                reflection_passed=False,
            )

        apply_fn()
        score_after = self.score()
        kept = score_after >= score_before

        if not kept:
            revert_fn()
            _log.info("Reverted: %s (%.3f → %.3f)", description, score_before, score_after)
        else:
            _log.info("Kept: %s (%.3f → %.3f)", description, score_before, score_after)

        return Experiment(
            iteration=0,
            description=description,
            score_before=score_before,
            score_after=score_after,
            kept=kept,
            duration_s=time.monotonic() - start,
            reflection_passed=reflection_passed,
        )

    def hill_climb(self, candidates: list[tuple]) -> AutoResearchResult:
        """Run multiple experiments sequentially, building on each kept improvement."""
        result = AutoResearchResult()
        start = time.monotonic()
        result.best_score = self.score()

        for i, (description, apply_fn, revert_fn) in enumerate(candidates):
            exp = self.run_experiment(description, apply_fn, revert_fn)
            result.experiments.append(Experiment(
                iteration=i,
                description=exp.description,
                score_before=exp.score_before,
                score_after=exp.score_after,
                kept=exp.kept,
                duration_s=exp.duration_s,
                reflection_passed=exp.reflection_passed,
            ))
            if exp.kept:
                result.best_score = max(result.best_score, exp.score_after)

        result.total_duration_s = time.monotonic() - start
        return result
