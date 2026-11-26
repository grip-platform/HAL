"""The AGORA speaker roster, seatable on any machine.

On a GitHub runner with no provider secrets, ``_apikey_auth_or_absent`` in
``lib/hal_council_dispatch.py`` marks every rung absent (an empty auth mapping),
and ``_build_role_chains`` in ``lib/agora_pr_review.py`` drops absent rungs, so
the live roster seats no role. Tests that pin the roster's SHAPE must hold on any
machine, so they give each absent rung a fake key (HAL#885: the public newcomer
check failed on a keyless runner). A rung that already has auth keeps it.

Use this only for shape tests. Capability tests (which auth scheme a rung uses,
whether HAL can route it) must keep reading the live roster.
"""
from __future__ import annotations

from lib.hal_council_dispatch import _AGORA_SPEAKER_ROSTER

FAKE_AUTH = {"scheme": "apikey", "token": "env:HAL_TEST_FAKE_KEY"}


def seated_roster(roster=None) -> list:
    """The roster (the live AGORA roster by default) with each absent rung given
    its own copy of a fake key."""
    return [(e[0], e[1], e[2] or dict(FAKE_AUTH), *e[3:])
            for e in (_AGORA_SPEAKER_ROSTER if roster is None else roster)]
