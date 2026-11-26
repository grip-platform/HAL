"""Phase δ W1 — semantic cache + citation chip wiring into Conversation.

Regression anchors for the H-CACHE-WIRED hypothesis. The invariants under test:

  * HAL_SEMANTIC_CACHE=1 opt-in — cache is None otherwise.
  * Identical prompt under the same model hits the cache on the second send.
  * Cache hit short-circuits the pipeline — LimitSentinel.dispatch fires once.
  * ``last_cache_hit`` and ``last_recall_count`` are populated per turn.
  * ``_count_kono_entries`` correctly parses the kono_bridge format.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import unittest
from unittest.mock import MagicMock, patch

_REPO = pathlib.Path(__file__).parent.parent


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_conv = _load("lib/hal/conversation.py", "_wt_conversation_cache")
Conversation = _conv.Conversation
_count_kono_entries = _conv._count_kono_entries
_maybe_build_semantic_cache = _conv._maybe_build_semantic_cache


def _make_registry(model_addr: str = "groq/llama-3.3-70b-versatile") -> MagicMock:
    provider = MagicMock()
    provider.name = model_addr.split("/")[0]
    provider.models = [model_addr.split("/", 1)[1]]
    provider.type.value = "cloud"

    routing = MagicMock()
    routing.default_model = model_addr

    registry = MagicMock()
    registry.list_providers.return_value = [provider]
    registry.routing = routing
    return registry


def _make_dispatch_response(model: str = "groq/llama-3.3-70b-versatile", text: str = "Python is a language.") -> MagicMock:
    response = MagicMock()
    response.text = text
    response.model = model
    response.provider = model.split("/")[0]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    response.latency_ms = 42.0
    response.cost_usd = 0.0001
    return response


class _SemanticCacheEnv:
    """Context manager that sets HAL_SEMANTIC_CACHE deterministically."""

    def __init__(self, value: str | None):
        self.value = value
        self._prev: str | None = None

    def __enter__(self):
        self._prev = os.environ.get("HAL_SEMANTIC_CACHE")
        if self.value is None:
            os.environ.pop("HAL_SEMANTIC_CACHE", None)
        else:
            os.environ["HAL_SEMANTIC_CACHE"] = self.value
        return self

    def __exit__(self, *_):
        if self._prev is None:
            os.environ.pop("HAL_SEMANTIC_CACHE", None)
        else:
            os.environ["HAL_SEMANTIC_CACHE"] = self._prev


def _patched_conv(registry):
    """Common test patches — return the context managers as a tuple."""
    return (
        patch.object(_conv, "get_registry", return_value=registry),
        patch.object(_conv, "LimitSentinel"),
        patch.object(_conv, "SessionManager"),
        patch.object(_conv, "SessionRecorder"),
        patch.object(_conv, "IntentDetector", return_value=MagicMock()),
    )


class TestKonoCountHelper(unittest.TestCase):

    def test_counts_numbered_entries(self):
        formatted = (
            "[KONO Memory — relevant knowledge from past sessions]\n"
            "  1. first memory\n"
            "     [fitness=0.85, permanent]\n"
            "  2. second memory\n"
            "     [fitness=0.72]\n"
            "  3. third memory\n"
            "     [fitness=0.65]"
        )
        self.assertEqual(_count_kono_entries(formatted), 3)

    def test_empty_returns_zero(self):
        self.assertEqual(_count_kono_entries(""), 0)

    def test_header_only_no_entries(self):
        self.assertEqual(
            _count_kono_entries("[KONO Memory — relevant knowledge from past sessions]"),
            0,
        )

    def test_non_numbered_lines_ignored(self):
        # Metadata lines ("     [fitness=...]") start with whitespace but not a digit.
        self.assertEqual(
            _count_kono_entries("  1. one\n     [fitness=0.9]\n  notes: ignored"),
            1,
        )


class TestCacheFactory(unittest.TestCase):

    def test_cache_off_when_env_unset(self):
        with _SemanticCacheEnv(None):
            self.assertIsNone(_maybe_build_semantic_cache())

    def test_cache_on_when_env_one(self):
        with _SemanticCacheEnv("1"):
            cache = _maybe_build_semantic_cache()
            self.assertIsNotNone(cache)
            self.assertEqual(cache.stats()["entries"], 0)

    def test_cache_off_for_other_values(self):
        with _SemanticCacheEnv("true"):
            self.assertIsNone(_maybe_build_semantic_cache())
        with _SemanticCacheEnv("0"):
            self.assertIsNone(_maybe_build_semantic_cache())


class TestSemanticCacheWiring(unittest.TestCase):
    """Identical prompts hit the cache on the second send, skipping dispatch."""

    def test_second_identical_send_hits_cache(self):
        registry = _make_registry()
        with _SemanticCacheEnv("1"), \
             patch.object(_conv, "get_registry", return_value=registry), \
             patch.object(_conv, "LimitSentinel") as mock_sentinel_cls, \
             patch.object(_conv, "SessionManager"), \
             patch.object(_conv, "SessionRecorder"), \
             patch.object(_conv, "IntentDetector", return_value=MagicMock()):

            dispatch = mock_sentinel_cls.return_value.dispatch
            dispatch.return_value = _make_dispatch_response(text="Python is a language.")

            conv = Conversation(registry=registry)
            self.assertIsNotNone(conv._semantic_cache)

            r1 = conv.send("What is Python?")
            self.assertFalse(conv.last_cache_hit)
            self.assertEqual(dispatch.call_count, 1)

            r2 = conv.send("What is Python?")
            self.assertTrue(conv.last_cache_hit)
            self.assertEqual(dispatch.call_count, 1, "dispatch must not fire on cache hit")
            self.assertEqual(r2.provider, "cache")
            self.assertEqual(r2.text, r1.text)

    def test_cache_disabled_always_dispatches(self):
        registry = _make_registry()
        with _SemanticCacheEnv(None), \
             patch.object(_conv, "get_registry", return_value=registry), \
             patch.object(_conv, "LimitSentinel") as mock_sentinel_cls, \
             patch.object(_conv, "SessionManager"), \
             patch.object(_conv, "SessionRecorder"), \
             patch.object(_conv, "IntentDetector", return_value=MagicMock()):

            dispatch = mock_sentinel_cls.return_value.dispatch
            dispatch.return_value = _make_dispatch_response()

            conv = Conversation(registry=registry)
            self.assertIsNone(conv._semantic_cache)

            conv.send("What is Python?")
            conv.send("What is Python?")
            self.assertEqual(dispatch.call_count, 2, "no cache means both turns dispatch")

    def test_cache_hit_preserves_history_shape(self):
        registry = _make_registry()
        with _SemanticCacheEnv("1"), \
             patch.object(_conv, "get_registry", return_value=registry), \
             patch.object(_conv, "LimitSentinel") as mock_sentinel_cls, \
             patch.object(_conv, "SessionManager"), \
             patch.object(_conv, "SessionRecorder"), \
             patch.object(_conv, "IntentDetector", return_value=MagicMock()):

            mock_sentinel_cls.return_value.dispatch.return_value = _make_dispatch_response(text="Answer.")
            conv = Conversation(registry=registry)
            conv.send("Q")
            conv.send("Q")  # cache hit
            # Expect: [user-Q, assistant-A, user-Q, assistant-A-cached] — 4 entries.
            self.assertEqual(len(conv.history), 4)
            self.assertEqual(conv.history[-1].role, "assistant")
            self.assertEqual(conv.history[-1].content, "Answer.")


if __name__ == "__main__":
    unittest.main()
