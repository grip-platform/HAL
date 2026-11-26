"""The Synthesis Ladder contract: one task in, one synthesis out, at a priced effort.

Every rung -- deterministic gates, a local model, a cheap cloud seat, a council, a
synthesiser -- honours the same contract, so one can stand in for another (Liskov)
and the runner can climb from the cheapest rung that might answer to the costliest
it is allowed. Plan: GRIP "the Synthesis Ladder", approved by V>> on 2026-10-01.

The contract every rung keeps, checked by ``contract_violations`` and by the
shared suite in tests/test_synthesis_contract.py:

- it accepts any ``SynthesisTask`` and returns a ``Synthesis`` of its own effort;
- a provider failure yields ``decided=False`` with reasons -- it never raises;
- it never seats a vendor the task excludes;
- ``estimate()`` is an upper bound on the money ``synthesise()`` then spends.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional, Protocol, runtime_checkable


class Effort(IntEnum):
    """Rungs in cost order, named after Claude Code's effort levels."""

    DETERMINISTIC = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    XHIGH = 4
    MAX = 5
    ULTRA = 6


class Billing(str, Enum):
    """How a seat is paid for.

    LOCAL and SUBSCRIPTION seats cost no money per call (a subscription spends
    the operator's quota instead, which the governor meters separately).
    PREPAID draws down a balance already paid; POSTPAID adds to a bill. A
    FREE_TIER seat is never eligible: the free tiers checked so far use content
    for training.
    """

    LOCAL = "local"
    SUBSCRIPTION = "subscription"
    PREPAID = "prepaid"
    POSTPAID = "postpaid"
    FREE_TIER = "free_tier"


#: Seats whose calls add money to a sitting's cost.
MONEY_BILLING = frozenset({Billing.PREPAID, Billing.POSTPAID})


class Stakes(str, Enum):
    NORMAL = "normal"
    HIGH = "high"
    BOOTSTRAP = "bootstrap"


class Privacy(str, Enum):
    LOCAL_ONLY = "local_only"
    PRIVATE_CLOUD = "private_cloud"


@dataclass(frozen=True)
class SynthesisTask:
    """What to decide, the evidence to decide it on, and the rules around it.

    ``verdicts`` lists the allowed verdict tokens (AGORA's MERGE,
    REQUEST_CHANGES, DEFER_CHANGES); empty means a free-form answer.
    ``excluded_vendors`` names families that may not sit, such as the PR
    author's and the speakers'.
    """

    question: str
    evidence: tuple[str, ...] = ()
    verdicts: tuple[str, ...] = ()
    stakes: Stakes = Stakes.NORMAL
    privacy: Privacy = Privacy.PRIVATE_CLOUD
    excluded_vendors: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Seat:
    """One model's part in a synthesis: what it said, and what it cost."""

    model: str
    vendor: str
    role: str
    billing: Billing
    verdict: str = ""
    text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    ok: bool = True
    note: str = ""


@dataclass(frozen=True)
class Synthesis:
    """A rung's answer. ``cost_usd`` is always the sum of its seats' costs."""

    effort: Effort
    decided: bool
    verdict: str = ""
    answer: str = ""
    agreement: float = 0.0
    dissent: tuple[str, ...] = ()
    seats: tuple[Seat, ...] = ()
    cost_usd: float = 0.0
    reasons: tuple[str, ...] = ()
    budget_capped: bool = False
    prompt_sha256: str = ""


@runtime_checkable
class SynthesisRung(Protocol):
    """The interface every rung implements."""

    effort: Effort

    def available(self, task: SynthesisTask) -> bool:
        """True when the rung could serve ``task`` now (keys, models, vendors)."""

    def estimate(self, task: SynthesisTask) -> float:
        """An upper bound, in USD, on the money ``synthesise`` would spend."""

    def synthesise(self, task: SynthesisTask, prior: Optional[Synthesis]) -> Synthesis:
        """Answer ``task``, given the previous rung's synthesis when there was one."""


def undecided(effort: Effort, *reasons: str, seats: tuple[Seat, ...] = ()) -> Synthesis:
    """The shape every failure takes: never raise, say why, keep what was spent."""
    return Synthesis(effort=effort, decided=False, seats=tuple(seats),
                     cost_usd=sum(s.cost_usd for s in seats),
                     reasons=tuple(r for r in reasons if r))


def contract_violations(result: object, task: SynthesisTask, effort: Effort) -> list[str]:
    """Every way ``result`` breaks the contract for ``task``; empty when it holds."""
    if not isinstance(result, Synthesis):
        return [f"returned {type(result).__name__}, not a Synthesis"]
    problems = []
    if result.effort != effort:
        problems.append(f"effort {result.effort!r} from a {effort.name} rung")
    if result.decided and task.verdicts and result.verdict not in task.verdicts:
        problems.append(f"verdict {result.verdict!r} is not one of {task.verdicts}")
    if not 0.0 <= result.agreement <= 1.0:
        problems.append(f"agreement {result.agreement} outside [0, 1]")
    seated = sorted({s.vendor for s in result.seats} & task.excluded_vendors)
    if seated:
        problems.append(f"seated excluded vendor(s) {seated}")
    if abs(result.cost_usd - sum(s.cost_usd for s in result.seats)) > 1e-9:
        problems.append("cost_usd is not the sum of its seats' costs")
    return problems
