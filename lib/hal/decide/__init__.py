"""HAL decide: typed decisions from decision models, in the System One wire format.

    from lib.hal.decide import decide, noul, choice
    result = decide("Payouts have failed for 3 days.",
                    {"urgent": noul("Is this urgent?"),
                     "team": choice("Who should handle it?", {"billing": None, "technical": None})})
    if result.ok:
        result.answers["urgent"]["noul"]      # the probability of yes
    else:
        result.reason                         # why not, from a closed set

What a decision model is and is not, and how HAL guards every send: docs/decide.md.
"""
from lib.hal.decide.contract import (REASONS, DecisionModel, DecisionProvider, DecisionResult,
                                     InvalidDecision, ProviderSpec, choice, noul, read_answers,
                                     score, validate)
from lib.hal.decide.dispatch import DEFAULT_MODEL, decide

__all__ = [
    "DEFAULT_MODEL", "REASONS", "DecisionModel", "DecisionProvider", "DecisionResult",
    "InvalidDecision", "ProviderSpec", "choice", "decide", "noul", "read_answers", "score",
    "validate",
]
