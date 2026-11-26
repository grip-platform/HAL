"""Tests for HAL async provider adapter."""
import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock

from lib.hal.async_provider import AsyncAdapter, AsyncChatResult


class TestAsyncChatResult(unittest.TestCase):

    def test_success(self):
        r = AsyncChatResult(model="test", response=None, success=True, latency_ms=100)
        self.assertTrue(r.success)

    def test_failure(self):
        r = AsyncChatResult(model="test", response=None, success=False, error="timeout")
        self.assertFalse(r.success)

    def test_frozen(self):
        r = AsyncChatResult(model="test", response=None, success=True)
        with self.assertRaises(AttributeError):
            r.model = "changed"


class TestAsyncAdapter(unittest.TestCase):

    def test_chat_success(self):
        mock_registry = MagicMock()
        mock_registry.chat.return_value = MagicMock(text="Hello")
        adapter = AsyncAdapter(mock_registry)
        result = asyncio.run(adapter.chat("test/model", "Hi"))
        self.assertTrue(result.success)
        self.assertGreater(result.latency_ms, 0)

    def test_chat_failure(self):
        mock_registry = MagicMock()
        mock_registry.chat.side_effect = RuntimeError("refused")
        adapter = AsyncAdapter(mock_registry)
        result = asyncio.run(adapter.chat("test/model", "Hi"))
        self.assertFalse(result.success)
        self.assertIn("refused", result.error)

    def test_parallel_chat(self):
        mock_registry = MagicMock()
        mock_registry.chat.return_value = MagicMock(text="ok")
        adapter = AsyncAdapter(mock_registry)
        results = asyncio.run(adapter.parallel_chat([("m1", "Q1"), ("m2", "Q2")]))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.success for r in results))

    def test_race_returns_first_success(self):
        mock_registry = MagicMock()
        mock_registry.chat.return_value = MagicMock(text="winner")
        adapter = AsyncAdapter(mock_registry)
        result = asyncio.run(adapter.race(["m1", "m2", "m3"], "test"))
        self.assertTrue(result.success)

    def test_race_empty_models(self):
        adapter = AsyncAdapter(MagicMock())
        result = asyncio.run(adapter.race([], "test"))
        self.assertFalse(result.success)

    def test_health_check(self):
        mock_registry = MagicMock()
        mock_registry.chat.return_value = MagicMock(text="pong")
        adapter = AsyncAdapter(mock_registry)
        health = asyncio.run(adapter.health_check(["m1", "m2"]))
        self.assertEqual(len(health), 2)
        self.assertTrue(all(health.values()))


if __name__ == "__main__":
    unittest.main()
