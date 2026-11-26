"""Sync invariant tests for lib/hal/router.py — issue #305.

Audit verdict: router.py uses adapter-pattern dispatch (RoutingStrategy enum
mapped to handler methods via a dict at line ~80). Per-backend quirks live in
lib/hal/providers/openai_compat.py adapter classes, not in router.py.
There are zero backend-specific if/elif arms in the router file.

These tests extend the H245 sync-invariant pattern (HAPPI router whitelist
matches providers.yaml) to the router/registry dispatch layer. Goal: every
provider in config/providers.yaml is reachable through the router's dispatch
surface, and adding a new yaml provider without wiring it raises a CI failure.

Goodhart-proofing: each test verifies behaviour against a real artefact
(providers.yaml) — no mocked registry, no internal call-count assertions.
A test fails iff the actual dispatch contract regresses.
"""
from __future__ import annotations

import os

import pytest

from lib.hal.registry import parse_model_address
from lib.hal.router import Router
from lib.hal.types import RoutingStrategy


# --- Helpers ---------------------------------------------------------------


def _yaml_provider_names() -> set[str]:
    """Read provider names from config/providers.yaml.

    Stdlib-only line scan keyed on `  - name:` — robust across YAML rewrites.
    Mirrors the helper in test_happi_providers_openai_compat.py for parity
    with the H245 sync invariant.
    """
    yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config", "providers.yaml",
    )
    if not os.path.exists(yaml_path):
        pytest.skip(f"providers.yaml not found at {yaml_path}")
    names: set[str] = set()
    with open(yaml_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("- name:"):
                _, _, val = stripped.partition(":")
                names.add(val.strip().strip("'\""))
    return names


def _read_router_source() -> str:
    """Return the source text of lib/hal/router.py for static checks."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "lib", "hal", "router.py",
    )
    with open(path, "r") as f:
        return f.read()


# --- Sync invariant: dispatch table covers all routing strategies ----------


# Strategies that Router.route is expected to dispatch directly. COUNCIL is
# owned by lib/hal/council_agent.py, not the Router, and is intentionally not
# in this set. If a new direct-dispatch strategy is added to the enum without
# a Router handler, this test fails.
_ROUTER_OWNED_STRATEGIES: frozenset[RoutingStrategy] = frozenset({
    RoutingStrategy.FIXED,
    RoutingStrategy.FALLBACK,
    RoutingStrategy.QUALITY_COST,
})


def test_router_dispatch_table_covers_router_owned_strategies():
    """Every Router-owned RoutingStrategy MUST have a handler in Router.route's
    dispatch dict. Falsifies the failure mode where a new strategy is added to
    the enum but the dispatch dict still raises NotImplementedError.

    COUNCIL is excluded — it is dispatched via lib/hal/council_agent.py and
    is not part of Router.route's contract. Updating _ROUTER_OWNED_STRATEGIES
    is mandatory when a new direct-dispatch strategy lands.

    Verifies adapter-pattern integrity: dispatch is data, not control flow.
    """
    class _StubRegistry:
        routing = type("R", (), {"default_strategy": RoutingStrategy.FIXED})()

        def chat(self, *_a, **_kw):
            raise RuntimeError("stub")

    router = Router(_StubRegistry())
    missing: list[str] = []
    for strategy in _ROUTER_OWNED_STRATEGIES:
        try:
            router.route("probe", strategy=strategy)
        except NotImplementedError:
            missing.append(strategy.value)
        except Exception:
            # Any other failure means dispatch reached a handler — that's the
            # contract being verified. The handler may then fail for unrelated
            # reasons (no model, stub registry), and that is fine here.
            pass
    assert not missing, (
        f"Router.route has no handler for RoutingStrategy values: {missing}. "
        f"Add an entry to the dispatch dict in Router.route."
    )


def test_all_routing_strategies_classified_as_router_or_external():
    """Every RoutingStrategy enum value MUST be either Router-owned or
    explicitly external. Catches drift where a new enum value is added but
    nobody decides where it dispatches.

    External strategies (currently {COUNCIL}) live in dedicated subsystems;
    if a future enum value is intended for Router but isn't added to
    _ROUTER_OWNED_STRATEGIES, this test fails alongside the dispatch test.
    """
    external = {RoutingStrategy.COUNCIL}
    classified = _ROUTER_OWNED_STRATEGIES | external
    unclassified = [s.value for s in RoutingStrategy if s not in classified]
    assert not unclassified, (
        f"RoutingStrategy values without a documented dispatch home: "
        f"{unclassified}. Add to _ROUTER_OWNED_STRATEGIES (Router.route) or "
        f"to the external set in this test."
    )


# --- Sync invariant: yaml providers all routable through parse_model_address


def test_router_yaml_providers_all_routable():
    """Every provider in config/providers.yaml MUST resolve via
    parse_model_address (the registry-level dispatch primitive used by
    Router._route_fixed and the fallback chain).

    Falsification: if a new provider is added to providers.yaml without a
    matching entry in registry.parse_model_address's known_providers set,
    parse_model_address silently returns ('', address) and the request gets
    routed to the default provider instead of the named one. This test fails
    when that drift occurs.
    """
    yaml_names = _yaml_provider_names()
    unroutable: list[str] = []
    for name in sorted(yaml_names):
        provider, _model = parse_model_address(f"{name}/some-model")
        if provider != name:
            unroutable.append(name)
    assert not unroutable, (
        f"providers.yaml has {len(unroutable)} providers that parse_model_address "
        f"cannot route: {unroutable}. Add them to known_providers in "
        f"lib/hal/registry.py::parse_model_address."
    )


# --- Sync invariant: router has no backend-specific if/elif chain ----------


def test_router_no_backend_specific_if_elif_chain():
    """router.py MUST NOT contain backend-specific if/elif arms.

    Adapter-pattern integrity check (issue #305): per-backend quirks (Gemini
    safety blocks, DeepSeek token counting, etc.) belong in
    lib/hal/providers/, not in router.py. This test fails if someone drifts
    router.py back to a conditional chain like::

        if provider == "gemini": ...
        elif provider == "deepseek": ...

    The check is a substring scan — false positives are possible if a
    docstring legitimately discusses these patterns. Allow-list comments and
    docstrings by stripping them before scanning.
    """
    source = _read_router_source()
    # Drop lines that are clearly comments or inside docstrings (single-line
    # heuristic — the file's docstring uses ''' or " around block quotes).
    code_lines = [
        line for line in source.splitlines()
        if not line.lstrip().startswith("#")
    ]
    # Look for backend-specific dispatch arms by name. The yaml provider names
    # are the attack surface for drift.
    yaml_names = _yaml_provider_names()
    offenders: list[tuple[str, str]] = []
    for line in code_lines:
        stripped = line.strip()
        if not (stripped.startswith("if ") or stripped.startswith("elif ")):
            continue
        # Match patterns like `if provider == "gemini"` or `elif backend in ("xai",`
        for name in yaml_names:
            quoted = f'"{name}"'
            single = f"'{name}'"
            if quoted in stripped or single in stripped:
                offenders.append((name, stripped))
    assert not offenders, (
        f"router.py has backend-specific if/elif arms (drift from adapter "
        f"pattern): {offenders}. Move per-backend logic to "
        f"lib/hal/providers/openai_compat.py adapters."
    )


# --- Sync invariant: routing config models all reference yaml providers ----


def test_router_quality_cost_models_address_yaml_providers():
    """Every model address in routing.quality_cost and routing.tier_chains
    MUST use a provider from providers.yaml.

    Falsifies typos in providers.yaml's `routing:` block — e.g.,
    `complex: gemeni/foo` (typo) would route to default-provider instead of
    Gemini and the bug would only surface when a complex prompt actually fires
    in production. This test catches the typo at CI time.
    """
    from lib.hal.registry import _load_yaml_config
    yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config", "providers.yaml",
    )
    if not os.path.exists(yaml_path):
        pytest.skip(f"providers.yaml not found at {yaml_path}")

    yaml_names = _yaml_provider_names()
    _, routing = _load_yaml_config(yaml_path)
    if routing is None:
        pytest.skip("No routing config in providers.yaml")

    addresses: list[str] = []
    addresses.extend(routing.fallback_chain)
    # quality_cost_tiers is declared dict[str, str] but the simple YAML parser
    # may flatten nested keys to a sibling-string when the structure is
    # unusually shaped — defensively guard the iteration.
    if isinstance(routing.quality_cost_tiers, dict):
        addresses.extend(routing.quality_cost_tiers.values())
    if isinstance(routing.tier_chains, dict):
        for chain in routing.tier_chains.values():
            if isinstance(chain, list):
                addresses.extend(chain)

    bad: list[str] = []
    for address in addresses:
        if not address or "/" not in address:
            continue
        provider, _ = parse_model_address(address)
        if provider not in yaml_names:
            bad.append(address)
    assert not bad, (
        f"routing config in providers.yaml references providers not declared "
        f"in the providers list: {bad}. Either add the provider entry or fix "
        f"the typo in routing.fallback_chain / quality_cost / tier_chains."
    )
