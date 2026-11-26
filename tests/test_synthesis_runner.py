"""The ladder runner: escalate only when needed, stop at the ceiling, honour the budget.

Fake rungs make each property observable: which rungs were called, with what
prior, and what the climb returned.
"""
from dataclasses import dataclass, field

import pytest

from lib.hal.synthesis import (
    Billing,
    Effort,
    Seat,
    Stakes,
    Synthesis,
    SynthesisTask,
    run_ladder,
)

TASK = SynthesisTask(question="q", verdicts=("MERGE", "REQUEST_CHANGES"))


@dataclass
class FakeRung:
    effort: Effort
    decided: bool = True
    agreement: float = 1.0
    cost: float = 0.0
    est: float = 0.0
    vendor: str = "google"
    up: bool = True
    raises: bool = False
    priors: list = field(default_factory=list)

    def available(self, task):
        return self.up

    def estimate(self, task):
        return self.est

    def synthesise(self, task, prior):
        self.priors.append(prior)
        if self.raises:
            raise RuntimeError("boom")
        seat = Seat(model=f"m/{self.effort.name}", vendor=self.vendor, role="r",
                    billing=Billing.POSTPAID, verdict="MERGE", cost_usd=self.cost)
        return Synthesis(effort=self.effort, decided=self.decided,
                         verdict="MERGE" if self.decided else "", agreement=self.agreement,
                         seats=(seat,), cost_usd=self.cost)


def _ladder(**overrides):
    rungs = {e: FakeRung(e, decided=False) for e in (Effort.DETERMINISTIC, Effort.LOW,
                                                      Effort.MEDIUM, Effort.HIGH, Effort.XHIGH)}
    for effort, rung in overrides.items():
        rungs[Effort[effort]] = rung
    return rungs


def test_the_first_settled_rung_ends_the_climb():
    rungs = _ladder(LOW=FakeRung(Effort.LOW))
    result = run_ladder(TASK, list(rungs.values()), budget_usd=1.0)
    assert result.decided and result.effort == Effort.LOW
    assert rungs[Effort.MEDIUM].priors == []


def test_an_undecided_rung_escalates_and_says_why():
    rungs = _ladder(MEDIUM=FakeRung(Effort.MEDIUM))
    result = run_ladder(TASK, list(rungs.values()), budget_usd=1.0)
    assert result.decided and result.effort == Effort.MEDIUM
    assert any(r.startswith("LOW: escalating") for r in result.reasons)


def test_partial_agreement_escalates_when_unanimity_is_required():
    rungs = _ladder(HIGH=FakeRung(Effort.HIGH, agreement=2 / 3),
                    XHIGH=FakeRung(Effort.XHIGH))
    result = run_ladder(TASK, list(rungs.values()), floor=Effort.HIGH, budget_usd=1.0)
    assert result.effort == Effort.XHIGH
    assert any("agreement 0.67 below 1.00" in r for r in result.reasons)


def test_the_ceiling_is_never_crossed():
    rungs = _ladder()
    result = run_ladder(TASK, list(rungs.values()), ceiling=Effort.HIGH, budget_usd=1.0)
    assert rungs[Effort.XHIGH].priors == []
    assert not result.decided
    assert result.reasons[-1] == "nothing settled it at or below HIGH"


def test_a_rung_that_would_overrun_the_budget_never_runs():
    rungs = _ladder(MEDIUM=FakeRung(Effort.MEDIUM, cost=0.004, est=0.005),
                    HIGH=FakeRung(Effort.HIGH, est=0.05))
    rungs[Effort.MEDIUM].decided = False
    result = run_ladder(TASK, list(rungs.values()), budget_usd=0.02)
    assert rungs[Effort.HIGH].priors == []
    assert result.budget_capped and not result.decided
    assert result.cost_usd == pytest.approx(0.004)
    assert any("needs up to $0.0500" in r for r in result.reasons)


def test_the_deterministic_rung_runs_first_and_feeds_the_next():
    rungs = _ladder(HIGH=FakeRung(Effort.HIGH))
    run_ladder(TASK, list(rungs.values()), floor=Effort.HIGH, budget_usd=1.0)
    assert len(rungs[Effort.DETERMINISTIC].priors) == 1
    assert rungs[Effort.LOW].priors == [] and rungs[Effort.MEDIUM].priors == []
    (prior,) = rungs[Effort.HIGH].priors
    assert prior.effort == Effort.DETERMINISTIC


def test_bootstrap_stakes_raise_the_floor_to_xhigh():
    task = SynthesisTask(question="q", verdicts=("MERGE",), stakes=Stakes.BOOTSTRAP)
    rungs = _ladder(HIGH=FakeRung(Effort.HIGH), XHIGH=FakeRung(Effort.XHIGH))
    result = run_ladder(task, list(rungs.values()), budget_usd=1.0)
    assert result.effort == Effort.XHIGH
    assert rungs[Effort.HIGH].priors == []


def test_bootstrap_stakes_under_a_low_ceiling_decide_nothing():
    task = SynthesisTask(question="q", verdicts=("MERGE",), stakes=Stakes.BOOTSTRAP)
    rungs = _ladder(HIGH=FakeRung(Effort.HIGH))
    result = run_ladder(task, list(rungs.values()), ceiling=Effort.HIGH, budget_usd=1.0)
    assert not result.decided
    assert "bootstrap stakes need XHIGH" in result.reasons[0]
    assert all(r.priors == [] for r in rungs.values())


def test_a_rung_that_raises_is_skipped_not_fatal():
    rungs = _ladder(LOW=FakeRung(Effort.LOW, raises=True), MEDIUM=FakeRung(Effort.MEDIUM))
    result = run_ladder(TASK, list(rungs.values()), budget_usd=1.0)
    assert result.decided and result.effort == Effort.MEDIUM
    assert any("raised RuntimeError: boom" in r for r in result.reasons)


def test_a_rung_that_seats_an_excluded_vendor_stops_the_climb():
    task = SynthesisTask(question="q", verdicts=("MERGE",), excluded_vendors=frozenset({"xai"}))
    rungs = _ladder(LOW=FakeRung(Effort.LOW, vendor="xai", cost=0.001),
                    MEDIUM=FakeRung(Effort.MEDIUM))
    result = run_ladder(task, list(rungs.values()), budget_usd=1.0)
    assert not result.decided
    assert rungs[Effort.MEDIUM].priors == []
    assert result.cost_usd == pytest.approx(0.001)  # the money it spent is still shown
    assert any("cannot decide" in r for r in result.reasons)


def test_the_result_lists_every_seat_and_costs_their_sum():
    rungs = _ladder(LOW=FakeRung(Effort.LOW, decided=False, cost=0.001),
                    MEDIUM=FakeRung(Effort.MEDIUM, cost=0.002))
    result = run_ladder(TASK, list(rungs.values()), budget_usd=1.0)
    assert [s.model for s in result.seats] == ["m/DETERMINISTIC", "m/LOW", "m/MEDIUM"]
    assert result.cost_usd == pytest.approx(0.003)


def test_an_unavailable_rung_is_passed_over():
    rungs = _ladder(LOW=FakeRung(Effort.LOW, up=False), MEDIUM=FakeRung(Effort.MEDIUM))
    result = run_ladder(TASK, list(rungs.values()), budget_usd=1.0)
    assert result.effort == Effort.MEDIUM
    assert "LOW: unavailable" in result.reasons
