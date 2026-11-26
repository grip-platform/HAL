#!/usr/bin/env python3
"""KONO session counter store + operational-content filtering (HAL port).

Vendored and simplified from GRIP's lib/kono_counters.py. HAL's KONO bridge
is read-only, so counters are in-memory best-effort only — no file locking,
no cross-process persistence. If HAL grows a write path, re-vendor the full
GRIP implementation (lib/kono_counters.py + lib/file_lock.py + lib/_compat.py).

Interface preserved for substrate.py: `store.stored += 1`, `store.recalled`,
`store.reset()`, `store.to_dict()` all still work via dynamic attribute access.
"""

import logging
import re
from typing import Dict

from lib.hal.kono.constants import HAL_DIR

logger = logging.getLogger(__name__)

CLAUDE_DIR = HAL_DIR

OPERATIONAL_CONTENT_PATTERNS = [
    re.compile(r"^INTEGRATION_TEST_MARKER_"),
    re.compile(r"^CHECKPOINT_\d+"),
    re.compile(r"^SESSION_CHECKPOINT:"),
    re.compile(r"^TEST_PROBE_\d+"),
]


def is_operational_content(content: str) -> bool:
    """True if content is a transient marker that should not be recalled."""
    return any(p.search(content) for p in OPERATIONAL_CONTENT_PATTERNS)


class SessionCounterStore:
    """In-memory counter store with dict-backed dynamic attribute access.

    Supports `store.stored`, `store.stored += 1`, `store.to_dict()`, `store.reset()`.
    No persistence — HAL's read-only bridge does not need cross-process coherence.
    """

    _COUNTER_KEYS = ('stored', 'recalled', 'attested', 'promoted')

    def __init__(self, path=None):
        object.__setattr__(self, '_values', {k: 0 for k in self._COUNTER_KEYS})
        object.__setattr__(self, '_path', path)

    def __getattr__(self, name):
        values = object.__getattribute__(self, '_values')
        if name in values:
            return values[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        values = object.__getattribute__(self, '_values')
        if name in values:
            values[name] = value
        else:
            object.__setattr__(self, name, value)

    def to_dict(self) -> Dict[str, int]:
        return dict(self._values)

    def reset(self) -> None:
        for k in self._COUNTER_KEYS:
            self._values[k] = 0
