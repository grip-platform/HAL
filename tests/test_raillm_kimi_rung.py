"""Goodhart-resistant test: the RAILLM Kimi rung carries an id Moonshot serves.

Origin 2026-05-19: forced moonshot calls silently failed over (groq/CCH)
because the ladder carried ``moonshot/kimi-k2`` -> HTTP 404. Repointed to
``kimi-k2.5`` (#358), then to ``kimi-k3`` on 2026-09-07 after Moonshot retired
kimi-k2.5 and the moonshot-v1 family on 2026-08-31
(platform.kimi.ai/docs/models.md) -- those ids return 404 as well.

The two temperature-forcing tests that lived here were deleted on 2026-09-07:
they asserted a subscription-path ``temperature=1`` override that PR #511 moved
into lib/hal/model_quirks.py (the adapter now OMITS temperature for the whole
Kimi lineage), so one was red against the shipped behaviour and both are
covered by tests/test_openai_compat_locked_params.py and
tests/test_gateway_kimi_temperature.py. Mutation sensitivity: repoint the rung
to any retired id and the test below fails.
"""
from lib.hal.raillm_ladder import GRIP_LADDER_ADDRESSES

RETIRED_KIMI_ADDRESSES = ("moonshot/kimi-k2", "moonshot/kimi-k2.5")


def test_ladder_uses_live_kimi_model_id():
    assert "moonshot/kimi-k3" in GRIP_LADDER_ADDRESSES
    for retired in RETIRED_KIMI_ADDRESSES:
        assert retired not in GRIP_LADDER_ADDRESSES
