"""Transport-free equivalence types shared by RAILLM and gateway policy."""
import enum
from dataclasses import dataclass


class ModelTier(enum.Enum):
    """Capability tiers for the RAILLM cascade.

    Cascade order (best availability / largest context first):
    FRONTIER -> FAST -> FREE_TIER -> LOCAL. A primary call in FRONTIER
    will peer-fail within FRONTIER first, then drop to FAST, then
    FREE_TIER, then LOCAL, exhausting the full registered chain before
    raising `RAILLMExhausted`.
    """

    FRONTIER = "frontier"
    FAST = "fast"
    FREE_TIER = "free_tier"
    LOCAL = "local"


@dataclass
class ModelEquivalenceClass:
    """A set of model addresses that are functionally interchangeable."""

    tier: ModelTier
    members: list[str]
    context_window: int


