"""The Synthesis Ladder: cost-tiered multi-LLM synthesis behind one contract.

    from lib.hal.synthesis import Effort, SynthesisTask, run_ladder

See ``contract`` for the contract every rung keeps, ``rungs`` for the two rung
shapes, and ``runner`` for the climb.
"""
from lib.hal.synthesis.contract import (
    MONEY_BILLING,
    Billing,
    Effort,
    Privacy,
    Seat,
    Stakes,
    Synthesis,
    SynthesisRung,
    SynthesisTask,
    contract_violations,
    undecided,
)
from lib.hal.synthesis.rungs import (
    ChatFn,
    ChatReply,
    DeterministicRung,
    SeatRung,
    SeatSpec,
    parse_verdict,
    registry_chat,
)
from lib.hal.synthesis.runner import STAKES_FLOOR, run_ladder

__all__ = [
    "MONEY_BILLING", "Billing", "Effort", "Privacy", "Seat", "Stakes", "Synthesis",
    "SynthesisRung", "SynthesisTask", "contract_violations", "undecided",
    "ChatFn", "ChatReply", "DeterministicRung", "SeatRung", "SeatSpec",
    "parse_verdict", "registry_chat", "STAKES_FLOOR", "run_ladder",
]
