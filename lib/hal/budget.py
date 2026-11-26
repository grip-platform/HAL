"""HAL Token Budget — set spending limits per session or conversation.

Prevents runaway costs by enforcing configurable USD or token budgets.
Integrates with the conversation engine to halt queries when the budget
is exhausted.

Usage:
    from lib.hal.budget import Budget

    budget = Budget(max_usd=1.00)
    budget.spend(0.05)  # Record a query cost
    if budget.exhausted:
        print("Budget reached — switch to a cheaper provider or stop.")
"""
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetStatus:
    """Current budget state."""
    max_usd: float
    spent_usd: float
    remaining_usd: float
    exhausted: bool
    usage_pct: float
    queries: int


class Budget:
    """Enforces spending limits on LLM queries. Thread-safe."""

    def __init__(self, max_usd: float = 5.0):
        self._max_usd = max_usd
        self._spent_usd = 0.0
        self._queries = 0
        self._lock = threading.Lock()

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._spent_usd >= self._max_usd

    @property
    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self._max_usd - self._spent_usd)

    def spend(self, amount_usd: float) -> bool:
        """Record spending. Returns True if budget still has funds."""
        with self._lock:
            self._spent_usd += amount_usd
            self._queries += 1
            return self._spent_usd < self._max_usd

    def can_afford(self, estimated_cost: float) -> bool:
        """Check if a query fits within remaining budget."""
        with self._lock:
            return (self._spent_usd + estimated_cost) <= self._max_usd

    def status(self) -> BudgetStatus:
        """Get current budget status."""
        with self._lock:
            remaining = max(0.0, self._max_usd - self._spent_usd)
            pct = self._spent_usd / self._max_usd if self._max_usd > 0 else 1.0
            return BudgetStatus(
                max_usd=self._max_usd,
                spent_usd=round(self._spent_usd, 4),
                remaining_usd=round(remaining, 4),
                exhausted=self._spent_usd >= self._max_usd,
                usage_pct=round(pct, 3),
                queries=self._queries,
            )

    def reset(self, new_max: float = 0) -> None:
        """Reset budget. Optionally set a new limit."""
        with self._lock:
            self._spent_usd = 0.0
            self._queries = 0
            if new_max > 0:
                self._max_usd = new_max
