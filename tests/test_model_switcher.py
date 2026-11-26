"""Tests for HAL natural-language model switcher and fuzzy resolver."""
import unittest
from unittest.mock import MagicMock

from lib.hal.model_switcher import detect_model_switch, fuzzy_resolve_model


def _make_registry(*providers):
    """Build a minimal registry mock from (name, [models]) tuples."""
    infos = []
    for name, models in providers:
        p = MagicMock()
        p.name = name
        p.models = models
        infos.append(p)
    registry = MagicMock()
    registry.list_providers.return_value = infos
    return registry


_REG = _make_registry(
    ("groq", ["llama-3.3-70b-versatile", "qwen3-32b"]),
    ("deepseek", ["deepseek-flash", "deepseek-v4-pro"]),
    ("gemini", ["gemini-2.5-flash"]),
    ("ollama", []),
)


class TestDetectModelSwitch(unittest.TestCase):
    def _detect(self, text):
        return detect_model_switch(text, _REG)

    def test_switch_to_deepseek(self):
        self.assertEqual(self._detect("switch to deepseek"), "deepseek/deepseek-flash")

    def test_use_groq(self):
        self.assertEqual(self._detect("use groq"), "groq/llama-3.3-70b-versatile")

    def test_change_to_gemini(self):
        self.assertEqual(self._detect("change to gemini"), "gemini/gemini-2.5-flash")

    def test_model_suffix_stripped(self):
        self.assertEqual(self._detect("try deepseek model"), "deepseek/deepseek-flash")

    def test_slash_commands_ignored(self):
        self.assertIsNone(self._detect("/model groq"))

    def test_long_input_ignored(self):
        self.assertIsNone(self._detect("x" * 81))

    def test_ordinary_question_ignored(self):
        self.assertIsNone(self._detect("what is the capital of France?"))

    def test_exact_address_resolves(self):
        self.assertEqual(self._detect("use groq/qwen3-32b"), "groq/qwen3-32b")

    def test_case_insensitive(self):
        self.assertEqual(self._detect("Switch To DeepSeek"), "deepseek/deepseek-flash")


class TestFuzzyResolveModel(unittest.TestCase):
    def _resolve(self, fragment):
        return fuzzy_resolve_model(fragment, _REG)

    def test_substring_match_full_address(self):
        self.assertEqual(self._resolve("deepseek-flash"), "deepseek/deepseek-flash")

    def test_provider_name_returns_first_model(self):
        self.assertEqual(self._resolve("groq"), "groq/llama-3.3-70b-versatile")

    def test_provider_no_models_returns_name(self):
        self.assertEqual(self._resolve("ollama"), "ollama")

    def test_provider_partial_prefix(self):
        # "deep" prefix-matches provider name "deepseek" via tier-2 startswith check
        self.assertEqual(self._resolve("deep"), "deepseek/deepseek-flash")

    def test_no_match_returns_none(self):
        self.assertIsNone(self._resolve("zzznomatch"))

    def test_partial_provider_prefix(self):
        self.assertEqual(self._resolve("gem"), "gemini/gemini-2.5-flash")

    def test_model_substring_wins(self):
        self.assertEqual(self._resolve("qwen3"), "groq/qwen3-32b")


if __name__ == "__main__":
    unittest.main()
