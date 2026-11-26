"""Adapter-level tests for Ollama tool-loop with dynamic num_ctx (W5, #1228).

Goodhart-safe: each test mocks the Ollama HTTP surface and asserts a concrete
observable behaviour (URLs POSTed, body shape, policy decisions). A regression
that reverts the keep_alive invariant or breaks the VRAM guard would fail at
least one test here.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.providers import ollama_tool_adapter as adapter  # noqa: E402


class TestReloadInvariants(unittest.TestCase):
    def test_reload_passes_keep_alive_verbatim(self):
        with patch.object(adapter, "_post_json", return_value={}) as post:
            adapter.reload_with_num_ctx(
                model="gemma4:e4b",
                new_ctx=16384,
                keep_alive="2h30m",
                ollama_host="http://localhost:11434",
            )
        self.assertEqual(post.call_count, 1)
        url, body = post.call_args.args[0], post.call_args.args[1]
        self.assertIn("/api/generate", url)
        self.assertEqual(body["keep_alive"], "2h30m")
        self.assertEqual(body["options"]["num_ctx"], 16384)
        self.assertEqual(body["model"], "gemma4:e4b")

    def test_reload_omits_keep_alive_when_none(self):
        with patch.object(adapter, "_post_json", return_value={}) as post:
            adapter.reload_with_num_ctx(
                model="gemma4:e4b", new_ctx=32768, keep_alive=None,
                ollama_host="http://localhost:11434",
            )
        body = post.call_args.args[1]
        self.assertNotIn("keep_alive", body)
        self.assertEqual(body["options"]["num_ctx"], 32768)


class TestVRAMPreflight(unittest.TestCase):
    def test_allows_small_enlargement_ample_memory(self):
        with patch.object(adapter, "_free_vram_mb", return_value=500):
            self.assertTrue(adapter._default_vram_preflight(4096, 16384))

    def test_blocks_large_enlargement_under_pressure(self):
        with patch.object(adapter, "_free_vram_mb", return_value=200):
            self.assertFalse(adapter._default_vram_preflight(4096, 131072))

    def test_fails_open_when_free_unknown(self):
        with patch.object(adapter, "_free_vram_mb", return_value=None):
            self.assertTrue(adapter._default_vram_preflight(4096, 131072))

    def test_cost_estimator_matches_25kb_calibration(self):
        cost = adapter._estimate_reload_cost_mb(4096, 16384)
        self.assertGreaterEqual(cost, 250)
        self.assertLessEqual(cost, 350)

    def test_cost_estimator_zero_on_shrink_or_equal(self):
        self.assertEqual(adapter._estimate_reload_cost_mb(16384, 16384), 0)
        self.assertEqual(adapter._estimate_reload_cost_mb(16384, 4096), 0)


class TestEscalationDecision(unittest.TestCase):
    def _build_adapter(self, *, vram_preflight=None):
        with patch("lib.hal.tool_runtime.make_openai_compat_chat_fn") as factory:
            factory.return_value = MagicMock(
                return_value={"content": [], "stop_reason": "end_turn"},
            )
            return adapter.OllamaToolAdapter(
                base_url="http://localhost:11434/v1",
                model="gemma4:e4b",
                vram_preflight=vram_preflight or (lambda _o, _n: True),
            )

    def test_small_prompt_no_reload(self):
        a = self._build_adapter()
        with patch.object(adapter, "fetch_loaded_ctx", return_value=(16384, "1h30m")), \
             patch.object(adapter, "fetch_arch_max", return_value=131072), \
             patch.object(adapter, "reload_with_num_ctx") as reload_fn:
            a.chat_fn(messages=[{"role": "user", "content": "hi"}], tools=[])
        reload_fn.assert_not_called()

    def test_large_prompt_reloads_preserving_keep_alive(self):
        a = self._build_adapter()
        with patch.object(adapter, "fetch_loaded_ctx", return_value=(4096, "1h30m")), \
             patch.object(adapter, "fetch_arch_max", return_value=131072), \
             patch.object(adapter, "reload_with_num_ctx") as reload_fn:
            big = "x" * 60000
            a.chat_fn(messages=[{"role": "user", "content": big}], tools=[])
        reload_fn.assert_called_once()
        kwargs = reload_fn.call_args.kwargs
        self.assertEqual(kwargs["keep_alive"], "1h30m")
        self.assertGreaterEqual(kwargs["new_ctx"], 16384)
        self.assertEqual(kwargs["model"], "gemma4:e4b")

    def test_vram_block_raises_ollama_ctx_error(self):
        a = self._build_adapter(vram_preflight=lambda _o, _n: False)
        with patch.object(adapter, "fetch_loaded_ctx", return_value=(4096, "1h30m")), \
             patch.object(adapter, "fetch_arch_max", return_value=131072), \
             patch.object(adapter, "reload_with_num_ctx") as reload_fn:
            big = "y" * 60000
            with self.assertRaises(adapter.OllamaCtxError):
                a.chat_fn(messages=[{"role": "user", "content": big}], tools=[])
        reload_fn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
