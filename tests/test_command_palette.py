"""Tests for Phase α W4 command palette."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from lib.hal.command_palette import (
    PaletteItem,
    PaletteProvider,
    build_default_provider,
    fuzzy_match,
    fuzzy_score,
    install_palette,
    prompt_toolkit_available,
)


# ---------------------------------------------------------------------------
# fuzzy_score
# ---------------------------------------------------------------------------
class TestFuzzyScore(unittest.TestCase):
    def test_empty_query_scores_zero(self):
        self.assertEqual(fuzzy_score("", "anything"), 0.0)

    def test_empty_candidate_scores_zero(self):
        self.assertEqual(fuzzy_score("x", ""), 0.0)

    def test_missing_char_scores_zero(self):
        self.assertEqual(fuzzy_score("z", "/mode"), 0.0)

    def test_exact_prefix_beats_middle_match(self):
        self.assertGreater(fuzzy_score("mo", "model"), fuzzy_score("mo", "command"))

    def test_consecutive_beats_scattered(self):
        self.assertGreater(fuzzy_score("abc", "abc"), fuzzy_score("abc", "aXbXc"))

    def test_word_boundary_bonus(self):
        self.assertGreater(fuzzy_score("c", "/cost"), fuzzy_score("c", "exec"))

    def test_case_insensitive(self):
        self.assertEqual(fuzzy_score("MODE", "/mode"), fuzzy_score("mode", "/MODE"))

    def test_subsequence_requires_correct_order(self):
        self.assertEqual(fuzzy_score("mo", "om"), 0.0)
        self.assertGreater(fuzzy_score("mo", "mode"), 0.0)

    def test_long_candidate_penalises_skipped_chars(self):
        self.assertGreater(fuzzy_score("xy", "xy"), fuzzy_score("xy", "xAAAAAAAAAAAAAy"))


# ---------------------------------------------------------------------------
# fuzzy_match
# ---------------------------------------------------------------------------
class TestFuzzyMatch(unittest.TestCase):
    def _items(self, *values: str) -> list[PaletteItem]:
        return [PaletteItem(v, v, "cmd") for v in values]

    def test_top_match_is_best_ranked(self):
        items = self._items("/model", "/mode", "/cost", "/history")
        results = fuzzy_match("mod", items)
        self.assertEqual(results[0][0].value, "/mode")
        self.assertEqual(results[1][0].value, "/model")

    def test_zero_score_items_omitted(self):
        items = self._items("/model", "/cost", "/clear")
        self.assertEqual(fuzzy_match("xyz", items), [])

    def test_limit_caps_result_count(self):
        items = self._items(*[f"/item{i}" for i in range(20)])
        self.assertEqual(len(fuzzy_match("item", items, limit=5)), 5)

    def test_tie_break_by_value_length(self):
        items = [
            PaletteItem("/model", "/model", "cmd"),
            PaletteItem("/modelCompare", "/modelCompare", "cmd"),
        ]
        self.assertEqual(fuzzy_match("mo", items)[0][0].value, "/model")

    def test_scores_against_label_when_better_than_value(self):
        items = [PaletteItem("m", "rusty-lang-model", "model")]
        results = fuzzy_match("rust", items)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0][1], 0.0)


# ---------------------------------------------------------------------------
# PaletteProvider + build_default_provider
# ---------------------------------------------------------------------------
class TestPaletteProvider(unittest.TestCase):
    def test_empty_provider(self):
        prov = PaletteProvider()
        self.assertEqual(prov.all_items(), [])
        self.assertEqual(prov.search("anything"), [])

    def test_all_items_concatenates_sources(self):
        prov = PaletteProvider(
            commands=[PaletteItem("/a", "/a", "cmd")],
            models=[PaletteItem("m", "m", "model")],
            tools=[PaletteItem("Read", "Read", "tool")],
        )
        items = prov.all_items()
        self.assertEqual(len(items), 3)
        self.assertEqual({i.category for i in items}, {"cmd", "model", "tool"})

    def test_search_returns_ranked(self):
        prov = build_default_provider()
        results = prov.search("mode")
        self.assertGreater(len(results), 0)
        top3 = [r[0].value for r in results[:3]]
        self.assertIn("/mode", top3)

    def test_build_default_provider_includes_models(self):
        prov = build_default_provider(model_names=["groq/a", "opus-4-7"])
        model_items = [i for i in prov.all_items() if i.category == "model"]
        self.assertEqual({i.value for i in model_items}, {"groq/a", "opus-4-7"})

    def test_build_default_provider_includes_core_commands(self):
        prov = build_default_provider()
        cmd_values = {i.value for i in prov.all_items() if i.category == "cmd"}
        for expected in ("/model", "/mode", "/cost", "/council", "/quit"):
            self.assertIn(expected, cmd_values)

    def test_fuzzy_finds_model_in_top_3(self):
        prov = build_default_provider(model_names=[
            "groq/llama-3.3-70b-versatile",
            "opus-4-7",
            "haiku-4-5",
            "gemma4:e4b",
        ])
        results = prov.search("llama", limit=3)
        self.assertIn("groq/llama-3.3-70b-versatile", [r[0].value for r in results])


# ---------------------------------------------------------------------------
# install_palette — optional prompt_toolkit path
# ---------------------------------------------------------------------------
class _FakeStream:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class TestInstallPalette(unittest.TestCase):
    def setUp(self):
        os.environ.pop("HAL_NO_PALETTE", None)

    def test_returns_false_on_non_tty(self):
        prov = build_default_provider()
        self.assertFalse(install_palette(session=None, provider=prov,
                                         stream=_FakeStream(tty=False)))

    def test_returns_false_when_hal_no_palette_set(self):
        prov = build_default_provider()
        with mock.patch.dict(os.environ, {"HAL_NO_PALETTE": "1"}):
            self.assertFalse(install_palette(
                session=None, provider=prov, stream=_FakeStream(tty=True)
            ))

    def test_returns_false_when_prompt_toolkit_missing(self):
        prov = build_default_provider()
        import builtins
        real_import = builtins.__import__

        def fail_prompt_toolkit(name, *a, **kw):
            if name.startswith("prompt_toolkit"):
                raise ImportError("simulated: no prompt_toolkit")
            return real_import(name, *a, **kw)

        with mock.patch.object(builtins, "__import__", side_effect=fail_prompt_toolkit):
            self.assertFalse(install_palette(
                session=None, provider=prov, stream=_FakeStream(tty=True)
            ))

    def test_prompt_toolkit_available_probe_returns_bool(self):
        self.assertIsInstance(prompt_toolkit_available(), bool)


# ---------------------------------------------------------------------------
# Integration — typical queries hit known items in top-3
# ---------------------------------------------------------------------------
class TestFuzzyIntegrationTop3(unittest.TestCase):
    def test_command_query_top_3(self):
        prov = build_default_provider()
        for query, expected in [
            ("mo", "/mode"),
            ("co", "/cost"),
            ("pr", "/providers"),
            ("hi", "/history"),
            ("cl", "/clear"),
            ("qu", "/quit"),
        ]:
            with self.subTest(query=query):
                values = [r[0].value for r in prov.search(query, limit=3)]
                self.assertIn(expected, values,
                              f"'{expected}' not in top 3 for query '{query}': {values}")

    def test_model_query_top_3(self):
        prov = build_default_provider(model_names=[
            "groq/llama-3.3-70b-versatile",
            "opus-4-7",
            "haiku-4-5",
            "gemma4:e4b",
            "qwen2.5vl:7b",
        ])
        for query, expected in [
            ("llama", "groq/llama-3.3-70b-versatile"),
            ("opus", "opus-4-7"),
            ("gemma", "gemma4:e4b"),
            ("qwen", "qwen2.5vl:7b"),
        ]:
            with self.subTest(query=query):
                values = [r[0].value for r in prov.search(query, limit=3)]
                self.assertIn(expected, values)


if __name__ == "__main__":
    unittest.main()
