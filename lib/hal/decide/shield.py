"""GRIP's decision-body redaction shield, loaded the way lib/hal/cascade_policy.py loads GRIP's.

Nothing reaches a decision model unless GRIP's ``shield_decision_payload`` is
present. A GRIP checkout too old to have it would scrub a decision body only
under its prose keys and let ``state`` out whole (GRIP#7933), so its absence
refuses the send. There is deliberately no fallback to ``shield_payload``.
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

#: A failed load is retried after this many seconds, never cached forever: one
#: missing file or import blip must not disable decisions for the life of a server.
RETRY_SECONDS = 60.0
_lock = threading.Lock()
_state: dict = {"shield": None, "retry_at": 0.0}


class ShieldUnavailable(RuntimeError):
    """GRIP's decision shield could not be loaded, so nothing was sent."""


def load(now: Optional[float] = None) -> Optional[Callable]:
    """``shield_decision_payload`` from GRIP's checkout, or None. A success is cached; a failure retried after 60 s."""
    clock = time.monotonic() if now is None else now
    with _lock:
        if _state["shield"] is not None:
            return _state["shield"]
        if clock < _state["retry_at"]:
            return None
        try:
            _state["shield"] = _import_shield()
        except Exception:  # noqa: BLE001 -- any failure means no shield, which refuses the send
            _state["retry_at"] = clock + RETRY_SECONDS
            return None
        return _state["shield"]


def _import_shield() -> Callable:
    directory = (Path(os.environ.get("CLAUDE_DIR", "~/.claude")).expanduser() / "lib").resolve()
    if not (directory / "review_redaction.py").is_file():
        raise FileNotFoundError(directory)
    if str(directory) not in sys.path:
        sys.path.append(str(directory))
    module = importlib.import_module("review_redaction")
    if Path(module.__file__).resolve() != directory / "review_redaction.py":
        raise ImportError("import priority changed")
    return module.shield_decision_payload  # AttributeError on a GRIP without it


def scrub(body: dict, *, to: str) -> dict:
    """``body`` as GRIP's decision shield rewrites it for the destination ``to``.

    Raises ShieldUnavailable when there is no shield. Whatever the shield raises
    (an OSError: a failed scrub, a key it would have to change, a billing
    refusal) propagates, so the caller refuses the send.
    """
    shield = load()
    if shield is None:
        raise ShieldUnavailable("GRIP's decision shield is not available")
    return shield(body, to=to)


def reset() -> None:
    """Forget a loaded shield and any retry wait (tests, or after GRIP is updated in place)."""
    with _lock:
        _state.update(shield=None, retry_at=0.0)
