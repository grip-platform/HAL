import os
import sys
import unittest
from unittest.mock import patch
import tempfile
import json

import lib.hal.agent_registry as _registry_mod
from lib.hal.agent_registry import (
    AgentDefinition,
    AgentDefinitionRegistry,
    AgentVersionStore,
)


def _make_registry(tmp_dir: str) -> AgentDefinitionRegistry:
    """Return a registry backed by tmp_dir, not the real state directory."""
    with patch.object(_registry_mod, "_AGENTS_STATE_DIR", tmp_dir):
        store = AgentVersionStore()
        # Patch again so __init__ makedirs + load_all_latest use tmp_dir.
        with patch.object(_registry_mod, "_AGENTS_STATE_DIR", tmp_dir):
            return AgentDefinitionRegistry(store=store)


class TestAgentDefinitionRoundtrip(unittest.TestCase):
    def test_to_dict_roundtrip(self):
        defn = AgentDefinition(
            agent_id="abcd1234",
            name="roundtrip-agent",
            version=3,
            model="groq/llama-3.3-70b",
            system_prompt="You are a tester.",
            tools=["bash"],
            skills=["coding"],
            mcp_servers=["grip-channel"],
            max_turns=50,
            budget_usd=2.5,
            routing_strategy="council",
            fallback_chain=["gpt-4o"],
            council_models=["claude-3", "gpt-4o"],
            environment_id="env-99",
            metadata={"owner": "test"},
            created_at=1000.0,
            updated_at=2000.0,
        )
        data = defn.to_dict()
        restored = AgentDefinition.from_dict(data)

        self.assertEqual(restored.agent_id, "abcd1234")
        self.assertEqual(restored.name, "roundtrip-agent")
        self.assertEqual(restored.version, 3)
        self.assertEqual(restored.model, "groq/llama-3.3-70b")
        self.assertEqual(restored.system_prompt, "You are a tester.")
        self.assertEqual(restored.tools, ["bash"])
        self.assertEqual(restored.skills, ["coding"])
        self.assertEqual(restored.mcp_servers, ["grip-channel"])
        self.assertEqual(restored.max_turns, 50)
        self.assertAlmostEqual(restored.budget_usd, 2.5)
        self.assertEqual(restored.routing_strategy, "council")
        self.assertEqual(restored.fallback_chain, ["gpt-4o"])
        self.assertEqual(restored.council_models, ["claude-3", "gpt-4o"])
        self.assertEqual(restored.environment_id, "env-99")
        self.assertEqual(restored.metadata, {"owner": "test"})
        self.assertAlmostEqual(restored.created_at, 1000.0)
        self.assertAlmostEqual(restored.updated_at, 2000.0)


class TestAgentVersionStore(unittest.TestCase):
    def test_version_store_persist_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(_registry_mod, "_AGENTS_STATE_DIR", tmp_dir):
                store = AgentVersionStore()
                defn = AgentDefinition(
                    agent_id="store0001",
                    name="test-agent",
                    version=1,
                    created_at=1234.0,
                    updated_at=1234.0,
                )
                store.persist(defn)
                loaded = store.load_latest("store0001")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "test-agent")
        self.assertEqual(loaded.version, 1)


class TestAgentDefinitionRegistry(unittest.TestCase):

    def test_create_agent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(_registry_mod, "_AGENTS_STATE_DIR", tmp_dir):
                registry = AgentDefinitionRegistry(store=AgentVersionStore())
                agent = registry.create(name="my-new-agent", model="claude-sonnet-4-6")

        self.assertEqual(len(agent.agent_id), 8)
        self.assertEqual(agent.version, 1)
        self.assertGreater(agent.created_at, 0)
        self.assertEqual(agent.name, "my-new-agent")

    def test_update_creates_new_version(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(_registry_mod, "_AGENTS_STATE_DIR", tmp_dir):
                registry = AgentDefinitionRegistry(store=AgentVersionStore())
                agent = registry.create(name="update-agent")
                updated = registry.update(agent.agent_id, {"budget_usd": 10.0})

        self.assertIsNotNone(updated)
        self.assertEqual(updated.version, 2)
        self.assertAlmostEqual(updated.budget_usd, 10.0)
        self.assertGreaterEqual(updated.updated_at, agent.created_at)

    def test_rollback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(_registry_mod, "_AGENTS_STATE_DIR", tmp_dir):
                registry = AgentDefinitionRegistry(store=AgentVersionStore())
                agent = registry.create(
                    name="rollback-agent", system_prompt="original prompt"
                )
                v1_prompt = agent.system_prompt
                registry.update(agent.agent_id, {"system_prompt": "changed prompt"})
                rolled = registry.rollback(agent.agent_id, 1)

        self.assertIsNotNone(rolled)
        self.assertEqual(rolled.version, 3)
        self.assertEqual(rolled.system_prompt, v1_prompt)

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(_registry_mod, "_AGENTS_STATE_DIR", tmp_dir):
                registry = AgentDefinitionRegistry(store=AgentVersionStore())
                agent = registry.create(name="delete-agent")
                agent_id = agent.agent_id

                result_first = registry.delete(agent_id)
                after_delete = registry.get(agent_id)
                result_second = registry.delete(agent_id)

        self.assertTrue(result_first)
        self.assertIsNone(after_delete)
        self.assertFalse(result_second)

    def test_list_agents_sorted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(_registry_mod, "_AGENTS_STATE_DIR", tmp_dir):
                registry = AgentDefinitionRegistry(store=AgentVersionStore())
                registry.create(name="zebra")
                registry.create(name="alpha")
                agents = registry.list_agents()

        self.assertEqual(agents[0].name, "alpha")
        self.assertEqual(agents[1].name, "zebra")

    def test_summary_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(_registry_mod, "_AGENTS_STATE_DIR", tmp_dir):
                registry = AgentDefinitionRegistry(store=AgentVersionStore())
                agent = registry.create(name="summary-agent", model="ollama/mistral:7b")
                summary = agent.summary()

        expected_keys = {
            "agent_id", "name", "version", "model",
            "routing_strategy", "created_at", "updated_at",
        }
        self.assertEqual(set(summary.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
