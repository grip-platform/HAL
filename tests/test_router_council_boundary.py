"""COUNCIL boundary tests for lib/hal/router.py.

Canonical design (verify-canonical of tests/test_router.py lines ~67-129, plus
PR #491 / GRIP ADR 0012 which placed council at the :4010 gateway):
RoutingStrategy.COUNCIL is a first-class enum member but is NOT executed by the
Router. A council returns a multi-vote deliberation (CouncilDeliberation /
CouncilResult), not a single ChatResponse, and needs a LIST of member models the
single-`model` strategy path does not carry. Council routing is owned by:

  - lib/hal/council_agent.py::CouncilRunner.deliberate  (agent layer)
  - lib/hal/moe.py::HALCouncil                          (3-phase engine)
  - lib/hal/gateway_openai.py::_relay_council           (:4010 wire layer, ADR 0012)

Router.route's COUNCIL branch (`_route_council`) is therefore a PRESENT-TENSE
boundary error that names those homes — replacing the old "not yet implemented"
deferral scrub. These tests lock that in.

This file is a focused sibling of tests/test_router.py (mirroring the existing
tests/test_router_local_jurisdiction.py split) so the boundary contract is
tested in isolation. It uses a stub registry — no network, no providers.yaml.

Goodhart-proofing: each test asserts on the actual raised message and FAILS on
the pre-fix code (which raised 'Routing strategy ... not yet implemented').
"""
from __future__ import annotations

import pytest

from lib.hal.router import Router, _COUNCIL_BOUNDARY_MESSAGE
from lib.hal.types import RoutingStrategy


# Deferral / scrub phrasings a correct-by-construction boundary must NEVER carry.
_SCRUB_PHRASES: frozenset[str] = frozenset({
    "not yet implemented", "not implemented yet", "not yet wired",
    "coming soon", "to be implemented", "placeholder", "todo", "wip",
})


def _route_error(strategy) -> NotImplementedError:
    """Route a probe under ``strategy`` against a stub registry and return the
    raised NotImplementedError.

    The stub's ``chat`` raises if reached — a boundary strategy must fail at the
    dispatch layer, before any provider call.
    """
    class _StubRegistry:
        routing = type("R", (), {"default_strategy": RoutingStrategy.FIXED})()

        def chat(self, *_a, **_kw):
            raise RuntimeError("stub must not be reached for a boundary strategy")

    router = Router(_StubRegistry())
    with pytest.raises(NotImplementedError) as exc_info:
        router.route("probe", strategy=strategy)
    return exc_info.value


def test_council_strategy_raises_honest_boundary_not_scrub():
    """RoutingStrategy.COUNCIL raises an HONEST, present-tense boundary that names
    a canonical council home — never the 'not yet implemented' scrub.

    Mutation-surviving: reverting _route_council to any deferral phrasing fails
    the scrub assertion; dropping the canonical-home reference fails the naming
    assertion. FAILS on the pre-fix code, PASSES after the fix.
    """
    err = _route_error(RoutingStrategy.COUNCIL)
    msg = str(err).lower()
    scrubs = [p for p in _SCRUB_PHRASES if p in msg]
    assert not scrubs, f"council boundary carries scrub phrasing {scrubs}: {err}"
    assert ("council_agent" in msg or "halcouncil" in msg
            or "_relay_council" in msg), (
        f"council boundary must name a canonical council home (council_agent.py "
        f"/ moe.py HALCouncil / gateway _relay_council) so the caller is "
        f"redirected, not dead-ended: {err}"
    )


def test_council_strategy_is_explicit_handler_not_generic_fallthrough():
    """COUNCIL must be an EXPLICIT dispatch-table entry (its own _route_council
    boundary), not the generic 'no registered handler' fallthrough.

    Mutation-surviving: removing RoutingStrategy.COUNCIL from Router.route's
    dispatch dict drops it to the generic fallthrough, whose message lacks
    'council' and 'chatresponse' — failing both assertions below.
    """
    msg = str(_route_error(RoutingStrategy.COUNCIL)).lower()
    assert "council" in msg, f"COUNCIL handler missing council-specific text: {msg}"
    assert "chatresponse" in msg, (
        f"COUNCIL boundary should explain the single-ChatResponse contract "
        f"mismatch — proving it is the dedicated _route_council handler, not the "
        f"generic fallthrough: {msg}"
    )


def test_unknown_strategy_fallthrough_is_present_tense_not_scrub():
    """A strategy value with no dispatch-table entry hits the fallthrough, which
    must be a present-tense wiring error — not the old deferral scrub. Guards the
    second scrub removed from Router.route. A raw non-member string routes to the
    fallthrough because it matches no RoutingStrategy value.
    """
    err = _route_error("phantom-strategy-xyz")
    msg = str(err).lower()
    scrubs = [p for p in _SCRUB_PHRASES if p in msg]
    assert not scrubs, f"fallthrough carries scrub phrasing {scrubs}: {err}"
    assert "no registered handler" in msg or "dispatch table" in msg, (
        f"fallthrough should name the dispatch-table wiring gap: {err}"
    )


def test_unknown_strategy_hits_generic_fallthrough_not_council_arm():
    """An unknown/unhandled strategy must reach the GENERIC dispatch-table
    fallthrough — NOT the COUNCIL boundary — even now that COUNCIL is an explicit
    dispatch entry (PR #500 concern 4a).

    Mutation-surviving: if Router.route's ``dispatch.get(strat)`` were changed to
    ``dispatch.get(strat, self._route_council)`` (routing unknowns to the COUNCIL
    arm), an unknown strategy would raise the COUNCIL boundary message and the
    negative-marker assertions below would fail. It also fails if the two
    boundaries collapse into one message, or if the fallthrough regresses to a
    scrub/deferral phrasing.
    """
    unknown = str(_route_error("phantom-strategy-xyz")).lower()
    council = str(_route_error(RoutingStrategy.COUNCIL)).lower()

    # It hit the generic fallthrough, which names the dispatch-table wiring gap.
    assert "no registered handler" in unknown or "dispatch table" in unknown, (
        f"unknown strategy should name the dispatch-table gap: {unknown}"
    )
    # ...and it is NOT the COUNCIL boundary (no council-home / ChatResponse text).
    for marker in ("council_agent", "_relay_council", "halcouncil", "chatresponse"):
        assert marker not in unknown, (
            f"unknown-strategy fallthrough leaked COUNCIL-boundary marker "
            f"'{marker}' — it routed to _route_council, not the generic arm: {unknown}"
        )
    # The two boundaries are genuinely distinct messages.
    assert unknown != council, (
        "unknown-strategy and COUNCIL must raise DISTINCT messages — a shared "
        "message means the COUNCIL arm swallowed the generic fallthrough"
    )
    scrubs = [p for p in _SCRUB_PHRASES if p in unknown]
    assert not scrubs, f"fallthrough carries scrub phrasing {scrubs}: {unknown}"


def test_council_config_error_raises_exact_boundary_message():
    """The 'config error' path — a registry whose ``routing.default_strategy`` is
    COUNCIL (i.e. ``routing.default: council`` in providers.yaml) — must raise the
    EXACT canonical boundary constant, both when COUNCIL is the resolved default
    (no explicit strategy) and when it is passed explicitly (PR #500 concern 4b).

    This is the scenario the pre-existing tests leave untested: they always pass
    ``strategy=`` explicitly against a FIXED default. Mutation-surviving: removing
    ``RoutingStrategy.COUNCIL: self._route_council`` from Router.route's dispatch
    drops both paths to the generic fallthrough, whose message !=
    ``_COUNCIL_BOUNDARY_MESSAGE`` — failing the exact-equality assertion.
    """
    class _CouncilDefaultRegistry:
        routing = type("R", (), {"default_strategy": RoutingStrategy.COUNCIL})()

        def chat(self, *_a, **_kw):
            raise RuntimeError("stub must not be reached for a boundary strategy")

    router = Router(_CouncilDefaultRegistry())

    # (1) resolved from the council-related default (no explicit strategy)
    with pytest.raises(NotImplementedError) as exc_default:
        router.route("probe")
    # (2) passed explicitly
    with pytest.raises(NotImplementedError) as exc_explicit:
        router.route("probe", strategy=RoutingStrategy.COUNCIL)

    for exc in (exc_default, exc_explicit):
        assert str(exc.value) == _COUNCIL_BOUNDARY_MESSAGE, (
            f"COUNCIL config-error path must raise the exact canonical boundary "
            f"constant, got: {exc.value}"
        )

    # Concern 3 anchor: the docstring and the raised message are the SAME shared
    # constant, so they can never drift apart.
    assert Router._route_council.__doc__ == _COUNCIL_BOUNDARY_MESSAGE, (
        "_route_council.__doc__ must be sourced from _COUNCIL_BOUNDARY_MESSAGE "
        "so docstring and error string cannot drift"
    )

    # Non-tautology guard: the shared constant is itself a real, non-scrub,
    # canonical-home-naming boundary (so the exact-equality checks above cannot
    # pass on a constant that was silently mutated into a deferral scrub).
    canonical = _COUNCIL_BOUNDARY_MESSAGE.lower()
    leaked = [p for p in _SCRUB_PHRASES if p in canonical]
    assert not leaked, f"boundary constant carries scrub phrasing {leaked}"
    assert ("council_agent" in canonical or "halcouncil" in canonical
            or "_relay_council" in canonical), (
        f"boundary constant must name a canonical council home: "
        f"{_COUNCIL_BOUNDARY_MESSAGE}"
    )
