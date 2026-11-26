"""P1 of v1.1 alignment — envelope parse acceptance tests.

H424: After P1 ships, every happi/1.0 envelope still parses without error AND
happi/1.1 envelopes are accepted identically at the parse layer. v1.2 (and any
other unknown future version) MUST still be rejected — the SUPPORTED_VERSIONS
set is closed, not open.

These tests are Goodhart-proof: each one asserts specific structural fields and
echoed values, not merely "parses without exception". A regression that silently
coerces the version, drops args, or swallows flags would fail at least one
assertion.
"""
from __future__ import annotations

import pytest

from lib.hal.happi.envelope import (
    SUPPORTED_VERSIONS,
    EnvelopeError,
    HappiEnvelope,
)


def test_supported_versions_is_a_closed_contiguous_set():
    """Assert the PROPERTIES of the accepted set, not its membership list.

    This test previously pinned the set literally: "happi/1.2" not in it, and
    len() == 2. Both were true at HAPPI/1.1 and became false at 1.2 and 1.3, so
    the test failed on main for the version bump rather than for a defect --
    demanding the old policy and making the correct code look broken.

    Enumerating the set guarantees a red test on every release by construction.
    These assertions hold across bumps and still catch the things worth catching:
    back-compat is never dropped, the set stays a closed contiguous range from
    1.0, and an unknown version is refused.
    """
    # Back-compat is a HAPPI guarantee: 1.0 must never be dropped.
    assert "happi/1.0" in SUPPORTED_VERSIONS

    minors = sorted(int(v.split("/1.")[1]) for v in SUPPORTED_VERSIONS)
    # Contiguous from 1.0 with no gaps -- a hole would silently strand clients.
    assert minors == list(range(len(minors))), SUPPORTED_VERSIONS
    # Closed, not open-ended: the version above the highest is NOT accepted.
    assert f"happi/1.{minors[-1] + 1}" not in SUPPORTED_VERSIONS


def test_envelope_accepts_v11_minimal():
    """Minimal v1.1 envelope parses; v field echoes client's declaration."""
    raw = '{"v":"happi/1.1","id":"x","cmd":"version"}'
    env = HappiEnvelope.parse(raw)
    assert env.v == "happi/1.1"  # echoed, not coerced to v1.0
    assert env.id == "x"
    assert env.cmd == "version"
    assert env.args == ()
    assert env.flags == {}


def test_envelope_accepts_v11_with_args_and_flags():
    """Full v1.1 envelope: args + flags.audit=true + extra flag fields preserved."""
    raw = (
        '{"v":"happi/1.1","id":"audit-1","cmd":"echo",'
        '"args":["a","b"],'
        '"flags":{"audit":true,"stream":false}}'
    )
    env = HappiEnvelope.parse(raw)
    assert env.v == "happi/1.1"
    assert env.id == "audit-1"
    assert env.cmd == "echo"
    assert env.args == ("a", "b")
    # Flag values preserved verbatim — no boolean coercion drift.
    assert env.flags == {"audit": True, "stream": False}
    assert env.flags["audit"] is True


def test_envelope_rejects_an_unknown_future_version():
    """Closed-set boundary: a version above the accepted range MUST be rejected.

    Previously hardcoded to happi/1.2, which the runtime has accepted since the
    1.2 release -- so the test demanded a rejection the parser was right not to
    make, and failed on main for being stale rather than for a defect.

    The target is now DERIVED from SUPPORTED_VERSIONS, so it stays one step
    beyond whatever the current range is and never needs editing on a bump.
    """
    highest = max(int(v.split("/1.")[1]) for v in SUPPORTED_VERSIONS)
    unknown = f"happi/1.{highest + 1}"
    raw = '{"v":"%s","id":"x","cmd":"version"}' % unknown

    with pytest.raises(EnvelopeError) as excinfo:
        HappiEnvelope.parse(raw)
    # Message contract: the rejection cites the offending version.
    assert "unsupported protocol version" in str(excinfo.value)
    assert unknown in str(excinfo.value)


def test_envelope_accepts_every_version_in_the_supported_set():
    """The Goodhart anchor for the two tests above.

    They assert what must be REFUSED; without this, narrowing SUPPORTED_VERSIONS
    to a single version would satisfy both while breaking every existing client.
    This pins the other side: everything in the set actually parses.
    """
    for version in sorted(SUPPORTED_VERSIONS):
        env = HappiEnvelope.parse('{"v":"%s","id":"x","cmd":"version"}' % version)
        # Echoed, never coerced -- the envelope reports the CLIENT's version.
        assert env.v == version


def test_envelope_v10_still_parses_unchanged():
    """Backward-compat regression guard: v1.0 envelope parses identically to pre-P1.

    The v field echoes the client's declared version (NOT a normalised form);
    no automatic upgrade to v1.1.
    """
    raw = (
        '{"v":"happi/1.0","id":"r-echo","cmd":"echo",'
        '"args":["hi"],"flags":{"stream":true}}'
    )
    env = HappiEnvelope.parse(raw)
    assert env.v == "happi/1.0"  # NOT coerced upward
    assert env.id == "r-echo"
    assert env.cmd == "echo"
    assert env.args == ("hi",)
    assert env.flags == {"stream": True}


def test_envelope_rejects_missing_v_field():
    """Regression: envelope without v still fails (None not in SUPPORTED_VERSIONS)."""
    raw = '{"id":"x","cmd":"version"}'
    with pytest.raises(EnvelopeError) as excinfo:
        HappiEnvelope.parse(raw)
    assert "unsupported protocol version" in str(excinfo.value)
