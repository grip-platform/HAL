"""HAL hypothesis engine — standalone falsification tracker (H155).

Phase B port of GRIP's ``lib/hypothesis_engine.py``. All state lives
under ``$HAL_HOME/learning/`` so HAL can be run with no GRIP install
present. Re-exports the public API from :mod:`lib.hal.hypothesis.engine`.
"""
from lib.hal.hypothesis.engine import (
    Hypothesis,
    HYPOTHESIS_FILE,
    falsify,
    register,
    status,
    verify_all,
)

__all__ = [
    "Hypothesis",
    "HYPOTHESIS_FILE",
    "register",
    "verify_all",
    "falsify",
    "status",
]
