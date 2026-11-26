import os
import sys
import unittest
from unittest.mock import patch
import tempfile
import time

from lib.hal.environment import (
    Environment,
    EnvironmentRegistry,
    EnvironmentStore,
    check_tool_allowed,
    check_provider_allowed,
)


def _make_env(**kwargs) -> Environment:
    defaults = dict(
        env_id="env12345",
        name="test-env",
        created_at=time.time(),
    )
    defaults.update(kwargs)
    return Environment(**defaults)


class TestEnvironment(unittest.TestCase):

    def test_post_init_none_to_list(self):
        env = _make_env(allowed_tools=None)
        self.assertEqual(env.allowed_tools, [])

    def test_to_dict_roundtrip(self):
        original = _make_env(
            description="desc",
            allowed_tools=["bash"],
            blocked_tools=["rm"],
            network_policy="restricted",
            allowed_domains=["example.com"],
            max_turns=50,
            budget_usd=2.5,
            timeout_s=1800,
            work_dir="/tmp",
            persistent_fs=True,
            allowed_providers=["ollama"],
            require_local_only=True,
        )
        restored = Environment.from_dict(original.to_dict())
        self.assertEqual(restored.env_id, original.env_id)
        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.allowed_tools, original.allowed_tools)
        self.assertEqual(restored.blocked_tools, original.blocked_tools)
        self.assertEqual(restored.require_local_only, original.require_local_only)
        self.assertEqual(restored.budget_usd, original.budget_usd)

    def test_registry_create(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("lib.hal.environment._ENVS_DIR", tmpdir):
                registry = EnvironmentRegistry()
                env = registry.create(name="my-env")
        self.assertEqual(len(env.env_id), 8)
        self.assertGreater(env.created_at, 0)
        self.assertEqual(env.name, "my-env")

    def test_registry_list_sorted_by_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("lib.hal.environment._ENVS_DIR", tmpdir):
                registry = EnvironmentRegistry()
                registry.create(name="zulu")
                registry.create(name="alpha")
                envs = registry.list_environments()
        self.assertEqual(envs[0].name, "alpha")
        self.assertEqual(envs[1].name, "zulu")

    def test_check_tool_blocked(self):
        env = _make_env(blocked_tools=["bash"])
        allowed, reason = check_tool_allowed(env, "bash")
        self.assertFalse(allowed)
        self.assertIsInstance(reason, str)
        self.assertTrue(len(reason) > 0)

    def test_check_provider_local_only(self):
        env = _make_env(require_local_only=True)
        allowed_groq, reason_groq = check_provider_allowed(env, "groq")
        self.assertFalse(allowed_groq)
        self.assertIsInstance(reason_groq, str)

        allowed_ollama, reason_ollama = check_provider_allowed(env, "ollama")
        self.assertTrue(allowed_ollama)
        self.assertEqual(reason_ollama, "")

    def test_delete_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("lib.hal.environment._ENVS_DIR", tmpdir):
                registry = EnvironmentRegistry()
                result = registry.delete("nonexistent")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
