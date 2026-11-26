"""The ladder runner: direct first, escalate with a reason (DFI-FIB).

The deterministic rung always runs first, at no cost, and its notes travel up
as evidence. From the floor upwards, each rung either settles the task -- it
decided, with enough agreement -- or the runner climbs one rung and records why.
It stops at the ceiling, or before a rung whose estimate would overrun the
budget, and then returns the best synthesis it has, flagged and explained,
never silently.

The returned synthesis lists every seat that sat on any rung, so its cost is
everything the ladder spent.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

from lib.hal.synthesis.contract import (
    Effort,
    Stakes,
    Synthesis,
    SynthesisRung,
    SynthesisTask,
    contract_violations,
    undecided,
)

#: The lowest effort a task's stakes allow to settle it.
STAKES_FLOOR = {
    Stakes.NORMAL: Effort.DETERMINISTIC,
    Stakes.HIGH: Effort.HIGH,
    Stakes.BOOTSTRAP: Effort.XHIGH,
}


def run_ladder(task: SynthesisTask, rungs: Sequence[SynthesisRung], *,
               floor: Effort = Effort.DETERMINISTIC, ceiling: Effort = Effort.XHIGH,
               budget_usd: float = 0.0, agreement: float = 1.0) -> Synthesis:
    """Climb ``rungs`` until one settles ``task``, the ceiling, or the budget."""
    floor = max(floor, STAKES_FLOOR[task.stakes])
    if floor > ceiling:
        return undecided(floor, f"{task.stakes.value} stakes need {floor.name}, "
                                f"above the ceiling {ceiling.name}")
    climb = _Climb(task, floor, budget_usd, agreement)
    for rung in sorted(rungs, key=lambda r: r.effort):
        if rung.effort > ceiling:
            break
        if climb.step(rung):
            return climb.result()
    return climb.result(f"nothing settled it at or below {ceiling.name}")


class _Climb:
    """The state of one climb: what has run, what it cost, what was learnt."""

    def __init__(self, task: SynthesisTask, floor: Effort, budget_usd: float, agreement: float):
        self.task, self.floor = task, floor
        self.budget_usd, self.agreement = budget_usd, agreement
        self.trail: list[str] = []
        self.seats: list = []
        self.prior: Optional[Synthesis] = None
        self.best: Optional[Synthesis] = None
        self.capped = False
        self.poisoned = False

    @property
    def spent(self) -> float:
        return sum(s.cost_usd for s in self.seats)

    def step(self, rung: SynthesisRung) -> bool:
        """Run ``rung`` if it applies; True when the climb is over."""
        if Effort.DETERMINISTIC < rung.effort < self.floor:
            return False
        if not _available(rung, self.task):
            self.trail.append(f"{rung.effort.name}: unavailable")
            return False
        estimate = _estimate(rung, self.task)
        if self.spent + estimate > self.budget_usd + 1e-12:
            self.trail.append(f"{rung.effort.name}: needs up to ${estimate:.4f}, "
                              f"${max(0.0, self.budget_usd - self.spent):.4f} left")
            self.capped = True
            return True
        result, broke_contract = _synthesise(rung, self.task, self.prior)
        return self._record(rung, result, broke_contract)

    def _record(self, rung: SynthesisRung, result: Synthesis, broke_contract: bool) -> bool:
        """Keep what ``rung`` produced; True when the climb should stop here."""
        self.seats.extend(result.seats)
        self.prior = result
        if result.decided or self.best is None or not self.best.decided:
            self.best = result
        if broke_contract:
            # A rung that breaks its contract cannot be trusted to have honoured
            # the vendor exclusions either, so the climb stops rather than spend more.
            self.poisoned = True
            return True
        settled = (rung.effort >= self.floor and result.decided
                   and result.agreement >= self.agreement)
        if not settled:
            self.trail.append(f"{rung.effort.name}: escalating ({_why(result, self.agreement)})")
        return settled

    def result(self, closing: str = "") -> Synthesis:
        best = self.best or undecided(self.floor, "no rung ran")
        trail = tuple(self.trail) + ((closing,) if closing else ())
        settled = (best.decided and not self.capped and not self.poisoned and not closing)
        reasons = trail + best.reasons
        if self.poisoned:
            reasons += ("a rung seated an excluded vendor or broke the contract, "
                        "so this ladder cannot decide",)
        return replace(best, decided=settled, seats=tuple(self.seats), cost_usd=self.spent,
                       reasons=reasons, budget_capped=self.capped)


def _available(rung: SynthesisRung, task: SynthesisTask) -> bool:
    try:
        return bool(rung.available(task))
    except Exception:  # noqa: BLE001 -- a broken presence check means unavailable
        return False


def _estimate(rung: SynthesisRung, task: SynthesisTask) -> float:
    try:
        return max(0.0, float(rung.estimate(task)))
    except Exception:  # noqa: BLE001 -- an unknown cost cannot fit any budget
        return float("inf")


def _synthesise(rung: SynthesisRung, task: SynthesisTask,
                prior: Optional[Synthesis]) -> tuple[Synthesis, bool]:
    """Run one rung under the contract; returns (result, contract broken).

    A raise becomes an undecided result and the climb goes on. A returned result
    that breaks the contract is kept, seats and all -- the money was spent -- but
    marks the climb as unable to decide.
    """
    try:
        result = rung.synthesise(task, prior)
    except Exception as exc:  # noqa: BLE001 -- the contract says never raise; enforce it
        return undecided(rung.effort, f"raised {type(exc).__name__}: {exc}"), False
    problems = contract_violations(result, task, rung.effort)
    if not problems:
        return result, False
    seats = result.seats if isinstance(result, Synthesis) else ()
    return undecided(rung.effort, "broke the contract: " + "; ".join(problems), seats=seats), True


def _why(result: Synthesis, agreement: float) -> str:
    if not result.decided:
        return "; ".join(result.reasons) or "undecided"
    return f"agreement {result.agreement:.2f} below {agreement:.2f}"
