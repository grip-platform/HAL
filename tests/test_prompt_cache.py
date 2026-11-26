"""Tests for Phase γ W3 prompt-cache builder (H-PCACHE)."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from lib.hal.prompt_cache import (
    CacheControlConfig,
    apply_cache_control,
    has_cache_markers,
)


class TestSystemMarker(unittest.TestCase):
    def test_string_system_upgraded_to_block_with_marker(self):
        body = {"system": "long system prompt", "messages": []}
        out = apply_cache_control(body)
        self.assertIsInstance(out["system"], list)
        self.assertEqual(out["system"][0]["type"], "text")
        self.assertEqual(out["system"][0]["cache_control"], {"type": "ephemeral"})

    def test_list_system_marks_last_block(self):
        body = {"system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        apply_cache_control(body)
        self.assertNotIn("cache_control", body["system"][0])
        self.assertEqual(body["system"][1]["cache_control"], {"type": "ephemeral"})

    def test_missing_system_is_noop(self):
        body = {"messages": []}
        apply_cache_control(body)
        self.assertNotIn("system", body)


class TestToolsMarker(unittest.TestCase):
    def test_tools_list_marks_last_tool(self):
        body = {"tools": [{"name": "a"}, {"name": "b"}]}
        apply_cache_control(body)
        self.assertNotIn("cache_control", body["tools"][0])
        self.assertEqual(body["tools"][1]["cache_control"], {"type": "ephemeral"})

    def test_empty_tools_is_noop(self):
        body = {"tools": []}
        apply_cache_control(body)
        self.assertEqual(body["tools"], [])

    def test_missing_tools_is_noop(self):
        body = {"system": "x"}
        apply_cache_control(body)
        self.assertNotIn("tools", body)


class TestConfigToggles(unittest.TestCase):
    def test_mark_system_false_skips_system(self):
        body = {"system": "x", "tools": [{"name": "t"}]}
        apply_cache_control(body, config=CacheControlConfig(mark_system=False))
        self.assertEqual(body["system"], "x")  # unchanged
        self.assertIn("cache_control", body["tools"][0])

    def test_mark_tools_false_skips_tools(self):
        body = {"system": "x", "tools": [{"name": "t"}]}
        apply_cache_control(body, config=CacheControlConfig(mark_tools=False))
        self.assertNotIn("cache_control", body["tools"][0])

    def test_env_disable_overrides_all(self):
        body = {"system": "x", "tools": [{"name": "t"}]}
        with mock.patch.dict(os.environ, {"HAL_DISABLE_PROMPT_CACHE": "1"}):
            apply_cache_control(body)
        self.assertEqual(body["system"], "x")
        self.assertNotIn("cache_control", body["tools"][0])


class TestHasCacheMarkers(unittest.TestCase):
    def test_empty_body(self):
        self.assertFalse(has_cache_markers({}))

    def test_body_without_markers(self):
        body = {"system": [{"type": "text", "text": "x"}], "tools": [{"name": "t"}]}
        self.assertFalse(has_cache_markers(body))

    def test_body_with_system_marker(self):
        body = {"system": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]}
        self.assertTrue(has_cache_markers(body))

    def test_body_with_tool_marker(self):
        body = {"tools": [{"name": "t", "cache_control": {"type": "ephemeral"}}]}
        self.assertTrue(has_cache_markers(body))


class TestRoundTrip(unittest.TestCase):
    def test_apply_then_has_markers_true(self):
        body = {"system": "x" * 100, "tools": [{"name": "Read"}]}
        apply_cache_control(body)
        self.assertTrue(has_cache_markers(body))

    def test_returned_body_is_same_object(self):
        body = {"system": "x"}
        self.assertIs(apply_cache_control(body), body)


if __name__ == "__main__":
    unittest.main()
