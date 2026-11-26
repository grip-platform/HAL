"""Goodhart-proof P3 W1 — raillm_ladder.pin_primary is pure, total,
order-preserving, fail-safe; ladder_cascade_order stays semantics-exact.

Builds the REAL ladder via build_grip_ladder() and derives the address
to pin from it (no fabricated address strings — verify-canonical).

Mutation-survival (each fails a test if the invariant breaks):
  - address not moved first / rest scrambled   -> test_pins_first_preserving_rest FAILS
  - input ladder or an eqc mutated in place    -> test_input_not_mutated FAILS
  - unknown address raises / alters the ladder -> test_unknown_address_total FAILS
  - other tiers / tier-key order changed       -> test_other_tiers_preserved FAILS
  - returns the input dict / same pinned eqc   -> test_fresh_objects_for_pinned_tier FAILS
  - pinning the already-primary reorders       -> test_pin_current_primary_noop FAILS
  - cascade dedup/order regressed by the O(n) refactor -> test_cascade_order_unchanged FAILS
"""
from __future__ import annotations

from lib.hal.raillm import ModelTier
from lib.hal.raillm_ladder import (
    build_grip_ladder,
    ladder_cascade_order,
    pin_primary,
)


def _frontier(ladder):
    return ladder[ModelTier.FRONTIER].members


def test_pins_first_preserving_rest():
    L = build_grip_ladder()
    fm = _frontier(L)
    assert len(fm) >= 2, "need a non-first FRONTIER member to pin"
    target = fm[1]
    out = pin_primary(L, target)
    om = out[ModelTier.FRONTIER].members
    assert om[0] == target                                  # pinned first
    assert om[1:] == [m for m in fm if m != target]         # rest stable


def test_input_not_mutated():
    L = build_grip_ladder()
    snap = {t: list(e.members) for t, e in L.items()}
    ids = {t: id(e) for t, e in L.items()}
    pin_primary(L, _frontier(L)[1])
    assert {t: list(e.members) for t, e in L.items()} == snap   # values intact
    assert {t: id(e) for t, e in L.items()} == ids              # objs not swapped


def test_unknown_address_total():
    L = build_grip_ladder()
    snap = {t: list(e.members) for t, e in L.items()}
    out = pin_primary(L, "nonexistent/model-xyz")           # must NOT raise
    assert out == L                                          # unchanged mapping
    assert {t: list(e.members) for t, e in out.items()} == snap


def test_other_tiers_preserved():
    L = build_grip_ladder()
    out = pin_primary(L, _frontier(L)[1])
    assert list(out.keys()) == list(L.keys())               # tier-key order
    for t in L:
        if t is ModelTier.FRONTIER:
            continue
        assert out[t].members == L[t].members                # byte-identical
        assert out[t].context_window == L[t].context_window


def test_fresh_objects_for_pinned_tier():
    L = build_grip_ladder()
    out = pin_primary(L, _frontier(L)[1])
    assert out is not L                                      # new dict
    fr_in, fr_out = L[ModelTier.FRONTIER], out[ModelTier.FRONTIER]
    assert fr_out is not fr_in                               # new eqc
    assert fr_out.members is not fr_in.members               # new list
    assert fr_out.tier == ModelTier.FRONTIER                 # carried faithfully
    assert fr_out.context_window == fr_in.context_window


def test_pin_current_primary_noop():
    L = build_grip_ladder()
    first = _frontier(L)[0]
    out = pin_primary(L, first)
    assert out[ModelTier.FRONTIER].members == L[ModelTier.FRONTIER].members


def test_cascade_order_unchanged():
    """The O(n) refactor of ladder_cascade_order must preserve the exact
    FRONTIER->FAST->FREE_TIER->LOCAL emission order and first-wins dedup.
    Recompute the reference the old (O(n^2)) way and assert equality."""
    L = build_grip_ladder()
    ref: list[str] = []
    for t in (ModelTier.FRONTIER, ModelTier.FAST,
              ModelTier.FREE_TIER, ModelTier.LOCAL):
        for addr in L[t].members:
            if addr not in ref:
                ref.append(addr)
    assert ladder_cascade_order() == ref
    assert len(ladder_cascade_order()) == len(set(ladder_cascade_order()))
