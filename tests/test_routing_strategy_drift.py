"""Routing-strategy vocabulary drift guard — enum <-> MCP schema <-> evolver agree.

Origin: reconciliation of a real drift. The ``hal_agent_create`` MCP tool schema
advertised ``routing_strategy`` = ["fixed", "fallback", "sentinel", "council"]
while the canonical ``RoutingStrategy`` enum (lib/hal/types.py) is
{fixed, fallback, quality_cost, council}. "sentinel" is NOT a routing strategy —
the Limit-Sentinel (lib/hal/sentinel.py) is an always-on failover layer wrapping
every dispatch — and "quality_cost" (a real, Router-executed member) was omitted
from the schema. The evolver (self_improving_agent.py) also suggested the
non-enum value "sentinel".

This test prevents that ENTIRE CLASS of drift from recurring: the canonical enum
is the single source of truth; every surface that advertises OR suggests a
routing_strategy value MUST agree with it.

Goodhart-proof: asserts the advertised set EQUALS the enum set (not a subset),
so re-adding a non-enum value ("sentinel") OR dropping a real member
("quality_cost") both fail with a named diff. The evolver test drives a REAL
suggest_improvements() call (not a mock of its output), so any suggestion of a
non-enum value fails.

Falsifier: this test passes while a production surface advertises or suggests a
routing_strategy value absent from RoutingStrategy — grep for the enum literal
across lib/ after it passes; only types.py should define the members.
"""
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# Pin HAL_HOME to this checkout so mcp_server's import-time sys.path insert
# resolves lib.hal.* here, never a sibling ~/.hal install (shadowing guard).
os.environ["HAL_HOME"] = str(REPO_ROOT)

from lib.hal.types import RoutingStrategy  # noqa: E402
from lib.hal.mcp_server import _TOOLS  # noqa: E402
from lib.hal.self_improving_agent import PerformanceAnalyser  # noqa: E402

CANONICAL = {s.value for s in RoutingStrategy}


def _advertised_routing_strategies() -> set:
    """The routing_strategy enum advertised by the hal_agent_create MCP tool."""
    for tool in _TOOLS:
        if tool.get("name") == "hal_agent_create":
            props = tool["inputSchema"]["properties"]
            return set(props["routing_strategy"]["enum"])
    raise AssertionError("hal_agent_create tool not found in _TOOLS")


class _FakeCounters:
    """Minimal fleet-counters double — suggest_improvements only calls snapshot()."""

    def snapshot(self) -> dict:
        return {"zero_cost_sessions": 0}


class RoutingStrategyDriftTest(unittest.TestCase):
    def test_mcp_schema_matches_enum_exactly(self):
        advertised = _advertised_routing_strategies()
        self.assertEqual(
            advertised, CANONICAL,
            f"routing_strategy drift: advertised={sorted(advertised)} "
            f"canonical={sorted(CANONICAL)} "
            f"only={sorted(advertised - CANONICAL)} "
            f"missing={sorted(CANONICAL - advertised)}",
        )

    def test_every_advertised_value_is_a_valid_enum_member(self):
        for value in _advertised_routing_strategies():
            RoutingStrategy(value)  # raises ValueError on a non-member

    def test_sentinel_is_not_advertised_as_a_strategy(self):
        # The Limit-Sentinel is an always-on failover layer, not a strategy.
        self.assertNotIn("sentinel", _advertised_routing_strategies())

    def test_quality_cost_is_advertised(self):
        # Real, Router-executed member the drift had omitted.
        self.assertIn("quality_cost", _advertised_routing_strategies())

    def test_evolver_only_suggests_valid_enum_strategies(self):
        analyser = PerformanceAnalyser(_FakeCounters())
        suggestions = analyser.suggest_improvements(
            routing_strategy="fixed", fallback_chain=[], budget_usd=0.0,
        )
        strat_values = [
            s["value"] for s in suggestions if s["field"] == "routing_strategy"
        ]
        self.assertTrue(
            strat_values,
            "expected a routing_strategy suggestion for fixed strategy + no chain",
        )
        for value in strat_values:
            self.assertIn(
                value, CANONICAL,
                f"evolver suggested non-enum routing_strategy {value!r}",
            )


if __name__ == "__main__":
    unittest.main()
