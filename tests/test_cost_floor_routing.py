"""Routine work must never reach the Top tier — the cost floor, enforced.

Two $230 subscriptions are *quota*, not money-per-call. The operator does not run
out of money mid-task; he runs out of tokens. The fix is not to cancel them, it
is to stop spending their quota on work that free and metered providers do just
as well — so the subscriptions are still full when the work actually needs them.

That is a routing property, and routing properties rot silently: one plausible
"just add a stronger model to the fallback chain" edit, and every routine call
starts draining subscription quota. This test is the floor's regression anchor.

**Top tier** = models that cost the operator either flat-fee subscription quota
(Anthropic: Fable/Opus/Claude) or premium per-token rates (Astra). **Routine
paths** = anything a normal request can reach without an explicit escalation:
the default model, the general fallback chain, the free/cheap/mid tier chains,
the simple/standard quality-cost routes, and the council seats.

Top-tier models *may* appear in explicitly-escalated paths (the premium tier, or
a deliberate frontier call) — that is the point of a Top tier. They may not be
*ambient*.

Design note (AGORA concerns on #603, all addressed): addresses are extracted
**recursively**, so a dict-valued entry (a weighted fallback, a per-model block)
cannot hide a leak from the check; every routine section must be **present**, so
a config that omits one fails loudly instead of passing vacuously; and marker
matching is done per address **segment**, not by raw substring, so an unrelated
name that merely contains a marker cannot be misread as Top tier.

Falsifier for the cost-floor plan itself: if subscription quota still empties
each week with this test green, the floor is not being respected somewhere this
test does not look — which is a finding, not a pass.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PROVIDERS = ROOT / "config" / "providers.yaml"

#: A model address is Top tier when a **whole segment** of
#: ``provider/model`` (split on ``/`` and ``:``) is one of these. Segment
#: matching, not substring matching: the address ``xai/grok-4.6`` must not be
#: flagged for containing "rok", and a future non-canonical name must be added
#: here deliberately rather than matched by accident.
TOP_TIER_SEGMENTS = frozenset({
    "anthropic", "claude", "claude-fable-5", "opus", "fable", "sonnet", "haiku",
    "gpt-6-astra", "astra",
})

#: Routing keys a routine request reaches with no explicit escalation.
ROUTINE_KEYS = ("default_model", "fallback_chain", "council")
#: tier_chains that must stay below the Top tier.
ROUTINE_TIERS = ("free", "cheap", "mid")
#: quality_cost routes that must stay below the Top tier.
ROUTINE_QUALITY = ("simple", "standard", "local")


def _routing() -> dict:
    """The routing block, or a loud failure — never a silent empty dict."""
    doc = yaml.safe_load(PROVIDERS.read_text())
    assert isinstance(doc, dict), f"{PROVIDERS} is not a mapping"
    routing = doc.get("routing")
    assert isinstance(routing, dict), (
        f"{PROVIDERS} has no `routing` mapping — the cost floor cannot be "
        "verified, which is a failure, not a pass"
    )
    return routing


def _addresses(value) -> list[str]:
    """Every model address anywhere in *value*, recursively.

    Dicts are traversed, not dropped: a routing section that switches to a
    structured form must not be able to hide a Top-tier model from this check
    (AGORA #603).
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [a for item in value for a in _addresses(item)]
    if isinstance(value, dict):
        # Keys AND values. A weighted fallback may key ON the model address
        # (`{anthropic/claude-opus: 0.6, ...}`), so inspecting only values would
        # hide exactly the leak this function exists to surface (AGORA #603).
        return ([a for key in value.keys() for a in _addresses(key)]
                + [a for item in value.values() for a in _addresses(item)])
    return []


def _is_top_tier(address: str) -> bool:
    """True when any whole segment of the address is a Top-tier marker."""
    segments = address.replace(":", "/").split("/")
    return any(seg.strip().lower() in TOP_TIER_SEGMENTS for seg in segments)


def _top_tier(addresses: list[str]) -> list[str]:
    return [a for a in addresses if _is_top_tier(a)]


def _require_section(routing: dict, path: str, value) -> list[str]:
    """The addresses at *path*, failing loudly when the section is absent.

    An absent section must fail rather than yield an empty list — otherwise a
    config that omits `fallback_chain` would pass this test while providing no
    coverage of the very path it exists to guard (AGORA #603).
    """
    assert value is not None, (
        f"routing.{path} is missing — a routine routing path that does not exist "
        "cannot be verified, and its absence is itself a finding"
    )
    addresses = _addresses(value)
    assert addresses, (
        f"routing.{path} is present but carries no addresses — an EMPTY routine "
        "path is not a verified path; it passes vacuously, which is the failure "
        "this guard exists to prevent (AGORA #603)"
    )
    return addresses


def test_no_top_tier_model_in_any_routine_path():
    """A top-tier model in an ambient path drains subscription quota silently."""
    routing = _routing()
    leaks: dict[str, list[str]] = {}

    for key in ROUTINE_KEYS:
        found = _top_tier(_require_section(routing, key, routing.get(key)))
        if found:
            leaks[key] = found

    chains = routing.get("tier_chains")
    assert isinstance(chains, dict), "routing.tier_chains is not a mapping"
    for tier in ROUTINE_TIERS:
        found = _top_tier(_require_section(routing, f"tier_chains.{tier}",
                                            (chains or {}).get(tier)))
        if found:
            leaks[f"tier_chains.{tier}"] = found

    quality = routing.get("quality_cost")
    assert isinstance(quality, dict), "routing.quality_cost is not a mapping"
    for route in ROUTINE_QUALITY:
        found = _top_tier(_require_section(routing, f"quality_cost.{route}",
                                           (quality or {}).get(route)))
        if found:
            leaks[f"quality_cost.{route}"] = found

    assert not leaks, (
        "Top-tier models are ambient in routine routing — they will consume "
        f"subscription quota on ordinary requests: {leaks}"
    )


def test_the_floor_is_the_default():
    """The cheapest capable route is what an unqualified call gets.

    Mutation: point default_model at any Top-tier model → red. This is the single
    edit that would move the operator's whole routine load onto quota.
    """
    default = _routing().get("default_model")
    assert isinstance(default, str) and default, (
        "routing.default_model is missing — every unqualified call would fall "
        "through to an unverified path"
    )
    assert not _is_top_tier(default), (
        f"default_model is Top tier ({default}); every unqualified call would "
        "spend subscription quota"
    )


def test_the_floor_is_actually_free_or_cheap():
    """A floor that is merely 'not Anthropic' is not a floor.

    The default must resolve *inside* the free/cheap tier chains, so "floor" is
    checked against the tier taxonomy rather than against a marker list.
    """
    routing = _routing()
    default = routing.get("default_model")
    assert isinstance(default, str) and default, "routing.default_model is missing"
    chains = routing.get("tier_chains") or {}
    floor = _addresses(chains.get("free")) + _addresses(chains.get("cheap"))
    assert floor, (
        "tier_chains.free and tier_chains.cheap are both empty or missing — "
        "there is no floor to assert against"
    )
    assert default in floor, (
        f"default_model {default!r} is not in the free/cheap tier chains — the "
        "floor is asserted, not configured"
    )
