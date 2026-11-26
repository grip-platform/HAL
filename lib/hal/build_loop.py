"""HAL Build Loop — continuous self-build with convergence detection.

Top-level loop that ties RSI analysis to self-build. Runs iterations
until convergence (no new proposals), budget exhaustion, or max iterations.

Usage:
    from lib.hal.build_loop import BuildLoop

    loop = BuildLoop(registry, rsi_engine, self_builder)
    state = loop.run_until_converge(max_iterations=10, budget_usd=5.0)
    print(f"Generated {state.total_modules} modules, converged={state.converged}")
"""
import json
import logging
import os
import time
from dataclasses import dataclass

from lib.hal.home import HAL_HOME as _HAL_HOME
from lib.hal.types import ChatMessage

_log = logging.getLogger("hal.build_loop")

_STATE_PATH = os.path.join(_HAL_HOME, "state", "hal", "build-loop.json")
_CONVERGENCE_WINDOW = 2  # Consecutive empty iterations before convergence


@dataclass(frozen=True)
class LoopIteration:
    """Summary of a single build loop iteration."""

    iteration: int
    proposals_found: int
    builds_attempted: int
    builds_succeeded: int
    modules_added: int
    tests_added: int
    cost_usd: float
    duration_s: float
    score: float = 0.0
    reviews_passed: int = 0


@dataclass(frozen=True)
class LoopState:
    """Overall state of the build loop."""

    iterations: tuple[LoopIteration, ...]
    total_modules: int
    total_tests: int
    total_cost_usd: float
    converged: bool
    convergence_reason: str


class BuildLoop:
    """Continuous self-build loop with convergence detection."""

    def __init__(
        self,
        registry: "ProviderRegistry",
        rsi_engine: "RSIEngine",
        self_builder: "SelfBuilder",
        cost_per_build: float = 0.10,
        state_path: str | None = None,
    ):
        self._registry = registry
        self._rsi = rsi_engine
        self._builder = self_builder
        self._cost_per_build = cost_per_build
        self._state_path = state_path or _STATE_PATH
        self._iterations: list[LoopIteration] = []

    def run_once(self, history: list[ChatMessage] | None = None) -> LoopIteration:
        """Single iteration: RSI analyse -> filter -> build each proposal."""
        start = time.time()
        history = history or []

        try:
            proposals = self._rsi.analyse_for_build(history)
        except Exception:
            _log.exception("RSI analyse_for_build failed")
            proposals = []

        # Strict `is True` guards against MagicMock / non-bool truthy values in tests
        use_review = getattr(getattr(self._builder, "_config", None), "auto_review", False) is True
        build_fn = self._builder.build_with_review if use_review else self._builder.build_from_proposal
        attempted, succeeded, modules, tests, reviews = self._execute_proposals(proposals, build_fn)

        iteration = LoopIteration(
            iteration=len(self._iterations) + 1,
            proposals_found=len(proposals),
            builds_attempted=attempted,
            builds_succeeded=succeeded,
            modules_added=modules,
            tests_added=tests,
            cost_usd=attempted * self._cost_per_build,
            duration_s=time.time() - start,
            score=succeeded / max(attempted, 1),
            reviews_passed=reviews,
        )
        self._iterations.append(iteration)
        self._save_state()
        return iteration

    def _execute_proposals(
        self, proposals: list, build_fn
    ) -> tuple[int, int, int, int, int]:
        """Execute each proposal through build_fn.

        Returns (attempted, succeeded, modules_added, tests_added, reviews_passed).
        Failures are logged and skipped — never bubble up to the loop.
        """
        attempted = succeeded = modules = tests = reviews = 0
        for proposal in proposals:
            attempted += 1
            try:
                result = build_fn(proposal)
            except Exception:
                _log.exception("build failed for %s", proposal.action[:80])
                continue
            if result.tests_passed:
                succeeded += 1
                modules += bool(result.module_path)
                tests += bool(result.test_path)
            if getattr(result, "review_passed", False):
                reviews += 1
        return attempted, succeeded, modules, tests, reviews

    def run_until_converge(
        self,
        max_iterations: int = 10,
        budget_usd: float = 5.0,
        history: list[ChatMessage] | None = None,
    ) -> LoopState:
        """Run iterations until convergence, budget exhaustion, or max iterations."""
        empty_streak = 0
        total_cost = 0.0

        for _ in range(max_iterations):
            iteration = self.run_once(history)
            total_cost += iteration.cost_usd

            if iteration.proposals_found == 0:
                empty_streak += 1
            else:
                empty_streak = 0

            if empty_streak >= _CONVERGENCE_WINDOW:
                return self._build_state(True, "No proposals for 2 consecutive iterations")

            if total_cost >= budget_usd:
                return self._build_state(False, f"Budget exhausted: ${total_cost:.2f} >= ${budget_usd:.2f}")

        return self._build_state(False, f"Max iterations ({max_iterations}) reached")

    def get_state(self) -> LoopState:
        """Get current loop state."""
        return self._build_state(
            converged=False,
            reason="In progress" if self._iterations else "Not started",
        )

    def _build_state(self, converged: bool, reason: str) -> LoopState:
        """Construct LoopState from accumulated iterations."""
        return LoopState(
            iterations=tuple(self._iterations),
            total_modules=sum(i.modules_added for i in self._iterations),
            total_tests=sum(i.tests_added for i in self._iterations),
            total_cost_usd=sum(i.cost_usd for i in self._iterations),
            converged=converged,
            convergence_reason=reason,
        )

    def _save_state(self) -> None:
        """Persist loop state to disk."""
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        data = {
            "iterations": len(self._iterations),
            "total_modules": sum(i.modules_added for i in self._iterations),
            "total_tests": sum(i.tests_added for i in self._iterations),
            "total_cost_usd": sum(i.cost_usd for i in self._iterations),
            "last_updated": time.time(),
        }
        with open(self._state_path, "w") as f:
            json.dump(data, f, indent=2)


def register_with_scheduler(scheduler: "Scheduler", registry: "ProviderRegistry") -> None:
    """Wire build loop into HAL's scheduler for periodic autonomous RSI.

    Registers a 'self_build' handler that runs one iteration of the
    build loop. Default schedule: every 3600s (1 hour).
    """
    from lib.hal.approval_gate import ApprovalGate
    from lib.hal.code_gen import CodeGenerator
    from lib.hal.git_ops import GitHubOps, GitMutator
    from lib.hal.rsi import RSIEngine
    from lib.hal.scheduler import Schedule
    from lib.hal.self_build import SelfBuilder

    code_gen = CodeGenerator(registry)
    rsi = RSIEngine(registry)
    gate = ApprovalGate()
    builder = SelfBuilder(registry, code_gen, GitMutator(), GitHubOps(), gate)
    loop = BuildLoop(registry, rsi, builder)

    scheduler.register_handler("self_build", lambda: loop.run_once())
    scheduler.add(Schedule(
        name="self_build",
        interval_s=3600,
        command="self_build",
        enabled=False,  # Disabled by default — enable via scheduler API
    ))
