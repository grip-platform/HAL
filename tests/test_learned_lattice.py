"""Goodhart-proof tests for ``lib.hal.learned_lattice`` (HAL #420).

Numpy-free port of the gridlm capability-lattice ladder. These assert the
proven mathematical properties — Banach contraction < 1, iterate == closed
form, ladder is a feasible permutation, probe DI overrides declared rpm with
per-provider fail-open, and the boot guard refuses a non-contracting layout —
not call counts. numpy is ABSENT in this environment by design; the module is
stdlib-only, so these run unmodified.
"""
import math

import pytest

from lib.hal.learned_lattice import (
    LearnedLattice,
    Provider,
    learned_ladder_order,
)
from lib.hal.raillm_ladder import GRIP_LADDER_ADDRESSES


def test_contraction_modulus_below_one():
    """The damped row-stochastic walk is an l1 contraction (modulus = alpha
    < 1) — the Banach guarantee of a unique, geometrically-reached fixed point.
    Bounded by alpha for EVERY layout because ``||A^T||_1 = 1`` for a
    row-stochastic ``A``; the sup norm carries no such bound (live ladder:
    ``||a A^T||_inf = 1.275``), which is what the pre-2026-09-07 guard measured
    and why it went red (0.8601) when the kimi-k3 signature landed."""
    lat = LearnedLattice()
    assert lat.contraction < 1.0
    # modulus is bounded by alpha (rows of A sum to 1 => unit column sums of
    # A^T) and positive.
    assert 0.0 < lat.contraction <= lat.alpha + 1e-9


def test_fixed_point_iterate_equals_closed_form():
    """The 200-step iterate equals the closed-form solve to tolerance — the
    convergence proof, re-derived numpy-free."""
    lat = LearnedLattice()
    for i in range(lat.n):
        p_it, p_cl = lat.diffuse(i)
        err = max(abs(a - b) for a, b in zip(p_it, p_cl))
        assert err < 1e-8, (i, err)
        # personalised PageRank mass is a probability-like vector: non-negative.
        assert all(x >= -1e-12 for x in p_it)


def test_ladder_is_permutation_primary_first():
    """``ladder`` returns a permutation of the input — every rung preserved,
    none dropped — with the requested primary first."""
    lat = LearnedLattice()
    order = lat.ladder(GRIP_LADDER_ADDRESSES[0])
    assert order[0] == GRIP_LADDER_ADDRESSES[0]
    assert sorted(order) == sorted(GRIP_LADDER_ADDRESSES)
    assert len(set(order)) == len(GRIP_LADDER_ADDRESSES)


def test_learned_ladder_order_permutation():
    """The public ``learned_ladder_order`` is a rung-preserving permutation."""
    order = learned_ladder_order(GRIP_LADDER_ADDRESSES)
    assert sorted(order) == sorted(GRIP_LADDER_ADDRESSES)
    assert order[0] == GRIP_LADDER_ADDRESSES[0]


def test_ladder_deterministic():
    """No RNG anywhere — repeated builds give the byte-identical order."""
    a = learned_ladder_order(GRIP_LADDER_ADDRESSES)
    b = learned_ladder_order(GRIP_LADDER_ADDRESSES)
    assert a == b


def test_unknown_primary_falls_back_to_index_zero():
    """An unknown primary address is fail-safe — defaults to rung 0, never
    raises, never drops a rung."""
    lat = LearnedLattice()
    order = lat.ladder("does/not-exist")
    assert order[0] == GRIP_LADDER_ADDRESSES[0]
    assert sorted(order) == sorted(GRIP_LADDER_ADDRESSES)


def test_throughput_probe_overrides_declared():
    """An injected throughput_probe feeds measured rpm; identical contract to
    gridlm_probe.make_probes (fn(provider)->rpm|None).

    The un-probed assertion below compares against ``lat.P[opus_i].throughput``
    -- whatever THIS run's lattice actually declared for the address -- not an
    independently hardcoded ``Provider("x", "subscription", ...)`` (the prior
    shape of this test). That hardcoded comparison passed even while
    _LADDER_PROVIDERS still keyed the CCH primary under the pre-HAL#610
    "anthropic/claude-opus-4-7" address: the lookup silently fell through to
    _provider_for's local/text/32k default, and TIER_THROUGHPUT happens to
    give "subscription" and "local" the same value (2), so the assertion could
    not tell a correct signature from a wrong one. Comparing against
    ``lat.P[opus_i].throughput`` makes it discriminate on the actual property
    this test is about (fail-open preserves THE DECLARED VALUE, whatever it
    is) instead of passing by that numeric coincidence.

    HAL#687 also fixed that stale key; the test below pins the CCH primary's
    declared signature directly, which this throughput test cannot see.
    """
    def probe(p):
        return 999.0 if "kimi" in p.address else None
    lat = LearnedLattice(throughput_probe=probe)
    kimi_i = lat.addresses.index("moonshot/kimi-k3")
    assert lat.throughput[kimi_i] == 999.0
    # un-probed providers keep declared throughput (per-provider fail-open).
    opus_i = lat.addresses.index("anthropic/claude-opus-5")
    assert lat.throughput[opus_i] == lat.P[opus_i].throughput
    assert lat.P[opus_i].throughput == Provider(
        "x", "subscription", "text", 1).throughput  # true today; see docstring


def test_the_cch_primary_has_its_declared_signature():
    """The CCH primary is described by _LADDER_PROVIDERS, not by the local,
    text-only, 32K default. Under its pre-HAL#610 key it fell through to that
    default, and the throughput test above could not tell, because the
    subscription and local tiers share a throughput value."""
    lat = LearnedLattice()
    primary = lat.P[lat.addresses.index(GRIP_LADDER_ADDRESSES[0])]
    assert (primary.tier, primary.modality, primary.context_window) == (
        "subscription", "text+vision", 1_000_000)


def test_throughput_probe_failopen_on_none_and_nan():
    """None / NaN / inf / bool readings fail open to the declared value —
    a garbage probe can never corrupt the lattice."""
    declared_sub = Provider("x", "subscription", "text", 1).throughput
    for bad in (None, float("nan"), float("inf"), True):
        lat = LearnedLattice(throughput_probe=lambda p: bad)
        assert all(t == lat.P[i].throughput for i, t in enumerate(lat.throughput))
    # sanity: subscription rung declared throughput is the tier value.
    lat = LearnedLattice(throughput_probe=lambda p: None)
    assert lat.throughput[0] == declared_sub


def test_boot_guard_raises_on_non_contracting_layout():
    """A degenerate layout whose damping is >= 1 is non-contracting — the boot
    guard raises RuntimeError loudly, so a mis-wired lattice cannot ship
    (acceptance: 'fails loudly at boot')."""
    with pytest.raises(RuntimeError):
        LearnedLattice(GRIP_LADDER_ADDRESSES, alpha=1.0)
    with pytest.raises(RuntimeError):
        LearnedLattice(GRIP_LADDER_ADDRESSES, alpha=1.5)


def test_feature_vector_shape_and_bounds():
    """Capability feature vector matches the gridlm schema: 8 dims, cost and
    log-context normalised, scheme one-hot, modality multi-hot."""
    p = Provider("anthropic/claude-opus-4-7", "subscription", "text+vision", 1_000_000)
    f = p.feature()
    assert len(f) == 8
    assert f[0] == 0.0                      # subscription cost 0
    assert math.isclose(f[1], math.log10(1_000_000) / 7.0)
    assert tuple(f[2:5]) == (1.0, 0.0, 0.0)  # subscription scheme one-hot
    assert f[5] == 1.0 and f[6] == 1.0 and f[7] == 0.0  # text+vision, no audio
