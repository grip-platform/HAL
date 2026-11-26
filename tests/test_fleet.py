import os
import sys
import unittest
from unittest.mock import MagicMock

from lib.hal.fleet import FleetAggregator, FleetCounters, FleetManager


class TestFleetCountersSnapshot(unittest.TestCase):
    def test_fleet_counters_snapshot(self):
        counters = FleetCounters()
        counters.record_sentinel()
        counters.record_sentinel()
        counters.record_council()
        counters.record_task_complete()
        counters.record_task_complete()
        counters.record_task_complete()
        self.assertEqual(counters.snapshot(), {
            "sentinel_interventions": 2,
            "council_deliberations": 1,
            "zero_cost_sessions": 0,
            "total_tasks_completed": 3,
            "gateway_messages_by_transport": (),
        })


class TestFleetGatewayCounter(unittest.TestCase):
    """#116 follow-up — FleetCounters observes per-transport gateway traffic."""

    def test_record_gateway_message_increments_per_transport(self):
        counters = FleetCounters()
        counters.record_gateway_message("telegram")
        counters.record_gateway_message("slack")
        counters.record_gateway_message("telegram")
        snap = counters.snapshot()
        # sorted by transport name so the tuple is deterministic
        self.assertEqual(
            snap["gateway_messages_by_transport"],
            (("slack", 1), ("telegram", 2)),
        )

    def test_empty_transport_is_ignored(self):
        """Defensive: empty string shouldn't create a phantom bucket."""
        counters = FleetCounters()
        counters.record_gateway_message("")
        self.assertEqual(counters.snapshot()["gateway_messages_by_transport"], ())

    def test_snapshot_is_thread_safe_across_concurrent_transports(self):
        """Multiple threads recording simultaneously produce consistent totals."""
        import threading
        counters = FleetCounters()

        def hammer(transport, n):
            for _ in range(n):
                counters.record_gateway_message(transport)

        threads = [
            threading.Thread(target=hammer, args=("telegram", 50)),
            threading.Thread(target=hammer, args=("slack", 30)),
            threading.Thread(target=hammer, args=("telegram", 20)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = dict(counters.snapshot()["gateway_messages_by_transport"])
        self.assertEqual(snap["telegram"], 70)
        self.assertEqual(snap["slack"], 30)


class TestFleetAggregator(unittest.TestCase):
    def _make_aggregator(self, agent_defn, sessions, cost_total=0.0, providers=(), models=()):
        agent_registry = MagicMock()
        agent_registry.get.return_value = agent_defn
        session_registry = MagicMock()
        session_registry.list_sessions.return_value = sessions
        cost_tracker = MagicMock()
        cost_tracker.total_usd = cost_total
        cost_tracker.per_provider.return_value = providers
        cost_tracker.per_model.return_value = models
        return FleetAggregator(agent_registry, session_registry, cost_tracker)

    def test_aggregator_agent_stats_found(self):
        defn = MagicMock()
        defn.name = "test-agent"
        agg = self._make_aggregator(defn, sessions=[])
        result = agg.agent_stats("abc")
        self.assertIsNotNone(result)
        self.assertEqual(result.agent_name, "test-agent")
        self.assertEqual(result.total_sessions, 0)

    def test_aggregator_agent_stats_not_found(self):
        agg = self._make_aggregator(agent_defn=None, sessions=[])
        self.assertIsNone(agg.agent_stats("xyz"))

    def test_aggregator_cost_report(self):
        defn = MagicMock()
        defn.name = "any"
        agg = self._make_aggregator(
            defn,
            sessions=[],
            cost_total=1.5,
            providers=[("groq", 1.0), ("ollama", 0.5)],
            models=[("llama", 1.5)],
        )
        report = agg.cost_report()
        self.assertEqual(report["total_usd"], 1.5)
        self.assertEqual(len(report["by_provider"]), 2)


class TestFleetManagerSnapshot(unittest.TestCase):
    def test_fleet_manager_snapshot(self):
        agent_registry = MagicMock()
        agent_registry.list_agents.return_value = [MagicMock(), MagicMock()]

        mock_session = MagicMock()
        mock_session.status = "active"
        session_registry = MagicMock()
        session_registry.list_sessions.return_value = [mock_session]

        cost_tracker = MagicMock()
        cost_tracker.total_usd = 0.5
        cost_tracker.per_provider.return_value = [("groq", 0.5)]
        cost_tracker.per_model.return_value = [("llama", 0.5)]

        manager = FleetManager(agent_registry, session_registry, cost_tracker)
        manager.counters.record_zero_cost_session()
        manager.counters.record_task_complete()

        snap = manager.snapshot()
        self.assertEqual(snap.total_agents, 2)
        self.assertEqual(snap.total_sessions, 1)
        self.assertEqual(snap.active_sessions, 1)
        self.assertEqual(snap.zero_cost_ratio, 1.0)
        self.assertEqual(snap.provider_diversity, 1)
        self.assertEqual(snap.completed_tasks, 1)


if __name__ == "__main__":
    unittest.main()
