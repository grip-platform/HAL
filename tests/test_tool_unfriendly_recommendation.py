"""H339 — Auto-disable message must recommend a tool-friendly alternative
when the failing model is in the known-unfriendly registry.

Goodhart-proof: tests assert positive (recommendation present for known
bad models) AND negative (no recommendation for unknown / unmapped models)
so a fix that ALWAYS prints "try X" doesn't pass the test suite.
"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.hal.cli import (
    _auto_disable_tools,
    _recommended_tool_friendly_alternative,
    _TOOL_UNFRIENDLY_RECOMMENDATIONS,
)


class _FakeException(Exception):
    """Stand-in for the schema-rejection exception."""
    provider = "groq"
    body = "Failed to call a function. Please adjust your prompt."


class TestRecommendationLookup(unittest.TestCase):
    """Pure-function tests on the lookup helper."""

    def test_known_bad_model_returns_alternative(self) -> None:
        alt = _recommended_tool_friendly_alternative("groq/llama-3.3-70b-versatile")
        self.assertEqual(alt, "groq/llama-4-scout-17b-16e-instruct")

    def test_unknown_model_returns_none(self) -> None:
        self.assertIsNone(_recommended_tool_friendly_alternative("openai/gpt-4o"))
        self.assertIsNone(_recommended_tool_friendly_alternative("anthropic/claude-opus-4-7"))

    def test_none_or_empty_model_returns_none(self) -> None:
        self.assertIsNone(_recommended_tool_friendly_alternative(None))
        self.assertIsNone(_recommended_tool_friendly_alternative(""))

    def test_registry_has_at_least_the_documented_entry(self) -> None:
        """Regression guard: if someone removes llama-3.3 from the registry,
        this test catches it. The empirical evidence (user session showing
        tool_use_failed) justifies the entry's presence."""
        self.assertIn(
            "groq/llama-3.3-70b-versatile",
            _TOOL_UNFRIENDLY_RECOMMENDATIONS,
            "H339 regression: llama-3.3 was removed from registry without "
            "evidence that it's now tool-friendly",
        )


class TestAutoDisableMessageRecommends(unittest.TestCase):
    """Output-side tests: recommendation appears in user-facing message."""

    def _capture(self, model: str | None) -> str:
        state: dict = {"enabled": True, "chat_fn": object(), "auth": "test"}
        buf = io.StringIO()
        with redirect_stdout(buf):
            _auto_disable_tools(state, _FakeException(), model=model)
        return buf.getvalue()

    def test_known_bad_model_shows_suggestion(self) -> None:
        """Load-bearing positive: groq/llama-3.3 must trigger the suggestion."""
        output = self._capture("groq/llama-3.3-70b-versatile")
        self.assertIn(
            "llama-4-scout",
            output,
            "H339 regression: auto-disable message missing the tool-friendly "
            f"alternative for groq/llama-3.3-70b-versatile. Output: {output!r}",
        )
        self.assertIn("/model", output, "suggestion must reference the /model command")
        # Existing message should still appear (no regression).
        self.assertIn("auto-disabled", output)

    def test_unknown_model_no_suggestion(self) -> None:
        """False-positive guard: openai/gpt-4o is fine for tools, no suggestion."""
        output = self._capture("openai/gpt-4o")
        self.assertNotIn(
            "Tip:",
            output,
            f"H339 false-positive: showed suggestion for unmapped model. Output: {output!r}",
        )
        self.assertNotIn("llama-4-scout", output)
        # Existing message must still render.
        self.assertIn("auto-disabled", output)

    def test_none_model_no_suggestion(self) -> None:
        """Defensive: model=None must not crash and must not suggest."""
        output = self._capture(None)
        self.assertNotIn("Tip:", output)
        self.assertIn("auto-disabled", output, "base message must still print")

    def test_disable_state_set_regardless_of_recommendation(self) -> None:
        """Side effect on tool_state must happen for both known and unknown
        models (the recommendation is a UI add-on, not a state change)."""
        for model in ("groq/llama-3.3-70b-versatile", "openai/gpt-4o", None):
            state: dict = {"enabled": True, "chat_fn": object(), "auth": "test"}
            with redirect_stdout(io.StringIO()):
                _auto_disable_tools(state, _FakeException(), model=model)
            self.assertFalse(state["enabled"], f"model={model}")
            self.assertIsNone(state["chat_fn"], f"model={model}")
            self.assertIsNone(state["auth"], f"model={model}")


if __name__ == "__main__":
    unittest.main()
