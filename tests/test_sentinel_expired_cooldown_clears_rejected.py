"""An expired cooldown outranks a stale ``status: rejected`` (HAL#612).

The shape found live on 2026-09-22 in ``~/.hal/state/hal/limits.json``::

    "openrouter": {"status": "rejected", "cooldown_until": <7.6 days ago>,
                   "resets_at": null, "consecutive_429s": 0, "last_429_at": null}

while ``hal --status`` reported the same provider healthy. Anything routing off
the file skipped a working provider.

WHY IT STUCK, read from the code rather than guessed. Every ``rejected`` write
in ``_on_error`` sets ``status`` and ``cooldown_until`` and never ``resets_at``.
The only recovery inside ``_is_near_limit`` was gated on
``state.resets_at and now >= state.resets_at`` -- a field written solely from
Anthropic's ``anthropic-ratelimit-unified-*-reset`` headers. For every other
provider ``resets_at`` is ``None`` forever, so the ``rejected`` branch returned
True unconditionally. ``anthropic`` in the same file recovered from FOUR
consecutive 429s because it is the one provider that receives that header.
That is the control that made this a defect rather than the design.

The success path (``_update_from_headers`` clearing status on a good response)
cannot rescue it either: a rejected provider is never called, so the response
that would clear the flag never arrives. Only wiping the state file recovered it.

The fix is on the READ side, which is the reversible half: an expired
``cooldown_until`` is authoritative over a stale ``status``. The write path is
untouched, so nothing about how a rejection is RECORDED changes -- only how long
the reader believes it after the evidence behind it has lapsed.

Tests here construct the sentinel exactly as ``test_sentinel_dispatch.py`` does,
with ``_load_state`` patched out so the real ``limits.json`` is never read --
AND with ``_STATE_PATH`` redirected to a per-test temp file, because the repaired
read path now PERSISTS the cleared status via ``_save_state``. The older sentinel
tests never needed that redirect: nothing they exercised wrote. Without it, a
fixture holding one provider would overwrite the live file and erase every other
provider's state -- the same test-writes-live-state shape this suite exists to
keep out of the router.
"""
from __future__ import annotations

import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lib.hal.sentinel as sentinel_mod  # noqa: E402
from lib.hal.sentinel import LimitSentinel, ProviderLimitState  # noqa: E402

PROVIDER = "openrouter"


@pytest.fixture(autouse=True)
def _isolated_state_path(tmp_path, monkeypatch):
    """Every test writes its persisted state under tmp_path, never ~/.hal."""
    monkeypatch.setattr(
        sentinel_mod, "_STATE_PATH", str(tmp_path / "state" / "hal" / "limits.json")
    )


def _sentinel_with(state: ProviderLimitState) -> LimitSentinel:
    reg = MagicMock()
    reg.list_providers.return_value = []
    with patch("lib.hal.sentinel.LimitSentinel._load_state"):
        sentinel = LimitSentinel(reg)
    sentinel._state[PROVIDER] = state
    return sentinel


def _stale_rejected(expired_seconds_ago: float) -> ProviderLimitState:
    """The exact on-disk shape from HAL#612: rejected, evidence lapsed, no reset header."""
    state = ProviderLimitState()
    state.status = "rejected"
    state.cooldown_until = time.time() - expired_seconds_ago
    state.resets_at = None          # non-Anthropic providers never receive one
    state.consecutive_429s = 0
    state.last_429_at = None
    return state


def test_expired_cooldown_clears_a_stale_rejection():
    """The HAL#612 shape must read as usable, not near-limit."""
    sentinel = _sentinel_with(_stale_rejected(expired_seconds_ago=182 * 3600))
    assert sentinel._is_near_limit(PROVIDER) is False


def test_clearing_is_persisted_to_the_state_object_not_just_returned():
    """The reader repairs the record it read, so the next reader need not re-derive it.

    Asserting only on the return value would let a fix that recomputes the
    answer on every call while leaving ``status: rejected`` on disk pass -- the
    file would keep lying to every other consumer (``hal --status`` reads it too).
    """
    state = _stale_rejected(expired_seconds_ago=3600)
    sentinel = _sentinel_with(state)
    sentinel._is_near_limit(PROVIDER)
    assert state.status == "allowed"
    assert state.cooldown_until is None
    # And the repair reached the FILE, not only the in-memory object: read the
    # artefact the operation acted on rather than trusting the object it
    # returned. The on-disk record is what hal --status reads.
    with open(sentinel_mod._STATE_PATH) as f:
        on_disk = f.read()
    assert '"status": "allowed"' in on_disk
    assert '"rejected"' not in on_disk


def test_an_unexpired_cooldown_still_rejects():
    """Control: the fix must not become 'ignore status'. Live evidence still blocks."""
    state = ProviderLimitState()
    state.status = "rejected"
    state.cooldown_until = time.time() + 600      # ten minutes of real cooldown left
    state.resets_at = None
    sentinel = _sentinel_with(state)
    assert sentinel._is_near_limit(PROVIDER) is True
    assert state.status == "rejected", "a live rejection must not be cleared early"


def test_rejected_with_no_cooldown_and_no_reset_is_not_stuck_forever():
    """A ``rejected`` carrying NO expiry evidence at all has nothing to wait for.

    Before the fix this shape was permanent: neither gate could ever fire. No
    write path in ``_on_error`` produces it -- every rejection sets a cooldown --
    so it can only arrive from a hand-edited or older-format state file. It is
    pinned so that the repaired predicate cannot regress into an infinite hold
    on a record the code itself never writes.
    """
    state = ProviderLimitState()
    state.status = "rejected"
    state.cooldown_until = None
    state.resets_at = None
    sentinel = _sentinel_with(state)
    assert sentinel._is_near_limit(PROVIDER) is False


def test_anthropic_style_reset_header_path_still_recovers():
    """The pre-existing recovery (the one anthropic used) keeps working unchanged."""
    state = ProviderLimitState()
    state.status = "rejected"
    state.cooldown_until = time.time() - 10
    state.resets_at = time.time() - 5             # header-driven reset has passed
    sentinel = _sentinel_with(state)
    assert sentinel._is_near_limit(PROVIDER) is False
    assert state.status == "allowed"


def test_reset_header_in_the_future_holds_even_when_cooldown_lapsed():
    """If the server named a reset time that has not arrived, honour it.

    The cooldown is HAL's own backoff estimate; ``resets_at`` is the provider
    telling HAL when its window actually resets. When both exist and disagree,
    the server's word is the stronger evidence, so a lapsed cooldown alone does
    not release a rejection the provider says is still in force.
    """
    state = ProviderLimitState()
    state.status = "rejected"
    state.cooldown_until = time.time() - 10       # HAL's estimate has lapsed
    state.resets_at = time.time() + 900           # provider says: 15 more minutes
    sentinel = _sentinel_with(state)
    assert sentinel._is_near_limit(PROVIDER) is True
    assert state.status == "rejected"
