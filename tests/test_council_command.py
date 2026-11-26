"""Tests for the /council command — arg parsing, standard + cascade modes."""
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.council_command import (
    _parse_council_args,
    council_handler,
    register_council_command,
)
from lib.hal.command_router import CommandRouter
from lib.hal.types import ChatResponse, CouncilResult, CouncilVote, TokenUsage


def _mock_registry():
    """Build a mock registry that returns canned responses."""
    registry = MagicMock()
    registry.chat.return_value = ChatResponse(
        text="This is a council response about the topic.",
        model="llama-3.3-70b-versatile",
        provider="groq",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        cost_usd=0.001,
    )
    registry.get.return_value = MagicMock()  # provider exists
    registry.routing = MagicMock()
    registry.routing.tier_chains = {
        "premium": ["gemini/gemini-3.1-pro-preview"],
        "mid": ["gemini/gemini-2.5-flash"],
        "cheap": ["groq/llama-3.3-70b-versatile"],
        "free": ["ollama/gemma4:26b"],
    }
    return registry


class TestParseCouncilArgs(unittest.TestCase):
    """Unit tests for argument parsing."""

    def test_simple_query(self):
        query, opts = _parse_council_args("What is RAILLM?")
        self.assertEqual(query, "What is RAILLM?")
        self.assertFalse(opts["cascade"])
        self.assertEqual(opts["rounds"], 1)
        self.assertIsNone(opts["budget"])
        self.assertIsNone(opts["models"])

    def test_cascade_flag(self):
        query, opts = _parse_council_args("--cascade Explain failover")
        self.assertEqual(query, "Explain failover")
        self.assertTrue(opts["cascade"])

    def test_rounds_flag(self):
        query, opts = _parse_council_args("--rounds 3 Deep analysis")
        self.assertEqual(query, "Deep analysis")
        self.assertEqual(opts["rounds"], 3)

    def test_budget_flag(self):
        query, opts = _parse_council_args("--budget 0.05 Quick check")
        self.assertEqual(query, "Quick check")
        self.assertEqual(opts["budget"], 0.05)

    def test_models_flag(self):
        query, opts = _parse_council_args("--models groq/llama,gemini/flash Query")
        self.assertEqual(query, "Query")
        self.assertEqual(opts["models"], ["groq/llama", "gemini/flash"])

    def test_combined_flags(self):
        query, opts = _parse_council_args(
            "--cascade --rounds 2 --budget 0.10 Is this safe?"
        )
        self.assertEqual(query, "Is this safe?")
        self.assertTrue(opts["cascade"])
        self.assertEqual(opts["rounds"], 2)
        self.assertEqual(opts["budget"], 0.10)

    def test_empty_input(self):
        query, opts = _parse_council_args("")
        self.assertEqual(query, "")


class TestCouncilHandler(unittest.TestCase):
    """Tests for the handler function."""

    def test_empty_args_returns_usage(self):
        registry = _mock_registry()
        handler = council_handler(registry)
        result = handler("")
        self.assertIn("Usage:", result)
        self.assertIn("--cascade", result)

    def test_no_query_after_flags(self):
        registry = _mock_registry()
        handler = council_handler(registry)
        result = handler("--cascade")
        self.assertIn("Error:", result)

    @patch("lib.hal.council_command.HALCouncil")
    def test_standard_mode_calls_deliberate_parallel(self, mock_council_cls):
        """Standard mode delegates to HALCouncil.deliberate_parallel."""
        mock_council = MagicMock()
        mock_council.deliberate_parallel.return_value = CouncilResult(
            merged_text="The synthesised answer.",
            votes=(
                CouncilVote(
                    model="groq/llama",
                    provider="groq",
                    response=ChatResponse(
                        text="Response A", model="llama", provider="groq",
                        usage=TokenUsage(10, 10, 20),
                    ),
                ),
            ),
            agreement_score=0.85,
        )
        mock_council_cls.return_value = mock_council

        registry = _mock_registry()
        handler = council_handler(registry)
        result = handler("What is RAILLM?")

        mock_council.deliberate_parallel.assert_called_once()
        self.assertIn("Council", result)
        self.assertIn("85%", result)
        self.assertIn("The synthesised answer.", result)

    @patch("lib.hal.council_command.RAILLMSentinel")
    def test_cascade_mode_uses_raillm(self, mock_sentinel_cls):
        """Cascade mode dispatches through RAILLMSentinel."""
        mock_sentinel = MagicMock()
        mock_response = ChatResponse(
            text="Cascade response about RAILLM.",
            model="llama-3.3-70b",
            provider="groq",
            usage=TokenUsage(100, 50, 150),
        )
        mock_sentinel.dispatch_with_history.return_value = mock_response
        mock_sentinel.thread = MagicMock()
        mock_sentinel.thread.messages = []
        mock_sentinel.thread.system_prompt = ""
        mock_sentinel_cls.return_value = mock_sentinel

        registry = _mock_registry()
        handler = council_handler(registry)
        result = handler("--cascade Explain RAILLM")

        mock_sentinel.dispatch_with_history.assert_called()
        self.assertIn("Cascade Council", result)
        self.assertIn("Cascade response about RAILLM.", result)

    @patch("lib.hal.council_command.RAILLMSentinel")
    def test_cascade_with_rounds(self, mock_sentinel_cls):
        """Cascade with --rounds 2 dispatches multiple times."""
        mock_sentinel = MagicMock()
        mock_response = ChatResponse(
            text="Round result", model="llama", provider="groq",
            usage=TokenUsage(10, 10, 20),
        )
        mock_sentinel.dispatch_with_history.return_value = mock_response
        mock_sentinel.thread = MagicMock()
        mock_sentinel.thread.messages = []
        mock_sentinel.thread.system_prompt = ""
        mock_sentinel_cls.return_value = mock_sentinel

        registry = _mock_registry()
        handler = council_handler(registry)
        handler("--cascade --rounds 2 Test query")

        # 1 initial + 2 rounds = 3 dispatches
        self.assertEqual(mock_sentinel.dispatch_with_history.call_count, 3)


class TestRegisterCouncilCommand(unittest.TestCase):
    """Tests for command router integration."""

    def test_register_adds_council_command(self):
        router = CommandRouter()
        registry = _mock_registry()
        register_council_command(router, registry)

        commands = {c.name for c in router.list_commands()}
        self.assertIn("council", commands)

    def test_registered_command_executes(self):
        router = CommandRouter()
        registry = _mock_registry()
        register_council_command(router, registry)

        result = router.execute("/council")
        self.assertTrue(result.success)
        self.assertIn("Usage:", result.output)

    def test_help_includes_council(self):
        router = CommandRouter()
        registry = _mock_registry()
        register_council_command(router, registry)

        help_text = router.execute("/help")
        self.assertIn("council", help_text.output)


if __name__ == "__main__":
    unittest.main()
