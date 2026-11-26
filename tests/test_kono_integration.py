"""Integration tests for KONO memory wiring into Conversation and Council."""
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.types import ChatMessage, ChatResponse, CouncilVote, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(text: str = "mock response", model: str = "test/mock") -> ChatResponse:
    """Create a minimal ChatResponse for testing."""
    return ChatResponse(
        text=text,
        model=model,
        provider="test",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        cost_usd=0.0,
        latency_ms=50,
    )


def _make_registry(responses: dict[str, str] | None = None) -> MagicMock:
    """Create a mock ProviderRegistry."""
    registry = MagicMock()
    registry.routing = MagicMock()
    registry.routing.default_model = "test/default"
    registry.routing.fallback_chain = ["test/default"]
    registry.routing.tier_chains = {"premium": ["test/default"], "mid": [], "cheap": [], "free": []}
    registry.list_providers.return_value = []

    if responses:
        def chat_side_effect(model, prompt, **kwargs):
            text = responses.get(model, f"response from {model}")
            return _make_response(text=text, model=model)
        registry.chat.side_effect = chat_side_effect
    else:
        registry.chat.return_value = _make_response()

    return registry


# ---------------------------------------------------------------------------
# Conversation + KONO
# ---------------------------------------------------------------------------

class TestConversationKonoIntegration(unittest.TestCase):
    """Test that Conversation.send() injects KONO context."""

    @patch("lib.hal.kono_bridge._kono_available", return_value=True)
    @patch("lib.hal.kono_bridge._recall_with_timeout", return_value="[KONO Memory]\n  1. Billing uses Stripe")
    @patch("lib.hal.conversation._build_identity", return_value="You are HAL")
    def test_kono_context_in_system_prompt(self, _id, _recall, _avail):
        """send() should include KONO recall in the system prompt sent to provider."""
        from lib.hal.conversation import Conversation

        registry = _make_registry()
        conv = Conversation(model="test/default", registry=registry)

        # Mock the sentinel dispatch to capture the system prompt
        mock_response = _make_response(text="billing answer")
        conv._sentinel.dispatch = MagicMock(return_value=mock_response)

        conv.send("How does billing work?")

        # The system prompt passed to dispatch should contain KONO memory
        call_args = conv._sentinel.dispatch.call_args
        system_kwarg = call_args.kwargs.get("system", "")
        self.assertIn("KONO Memory", system_kwarg,
                       "KONO context should be injected into system prompt")

    @patch("lib.hal.kono_bridge._kono_available", return_value=False)
    @patch("lib.hal.conversation._build_identity", return_value="You are HAL")
    def test_graceful_without_kono(self, _id, _avail):
        """send() works normally when KONO is unavailable."""
        from lib.hal.conversation import Conversation

        registry = _make_registry()
        conv = Conversation(model="test/default", registry=registry)

        mock_response = _make_response(text="hello back")
        conv._sentinel.dispatch = MagicMock(return_value=mock_response)

        response = conv.send("Hello")
        self.assertIsNotNone(response)
        # System prompt should NOT contain KONO when unavailable
        system_kwarg = conv._sentinel.dispatch.call_args.kwargs.get("system", "")
        self.assertNotIn("KONO Memory", system_kwarg)


# ---------------------------------------------------------------------------
# Council + KONO
# ---------------------------------------------------------------------------

class TestCouncilKonoIntegration(unittest.TestCase):
    """Test that HALCouncil injects KONO context."""

    @patch("lib.hal.kono_bridge._kono_available", return_value=True)
    @patch("lib.hal.kono_bridge._recall_with_timeout", return_value="[KONO Memory]\n  1. Contract law basics")
    def test_sequential_council_includes_kono(self, _recall, _avail):
        """deliberate() should inject KONO context into system prompt."""
        from lib.hal.moe import HALCouncil

        responses = {"m/a": "answer A", "m/b": "answer B"}
        registry = _make_registry(responses)
        council = HALCouncil(registry)

        result = council.deliberate("What is contract law?", models=["m/a", "m/b"])

        # Verify KONO context was passed to chat calls
        for call in registry.chat.call_args_list:
            system = call.kwargs.get("system", "")
            self.assertIn("KONO Memory", system,
                          "Council members should receive KONO context in system prompt")

    @patch("lib.hal.kono_bridge._kono_available", return_value=True)
    @patch("lib.hal.kono_bridge._recall_with_timeout", return_value="[KONO Memory]\n  1. Tax law precedent")
    def test_parallel_council_includes_kono(self, _recall, _avail):
        """deliberate_parallel() should inject KONO context."""
        from lib.hal.moe import HALCouncil

        responses = {"m/a": "answer A", "m/b": "answer B"}
        registry = _make_registry(responses)
        council = HALCouncil(registry)

        result = council.deliberate_parallel(
            "Tax implications?", models=["m/a", "m/b"], rounds=0,
        )

        # At least one chat call should have KONO context
        systems = [call.kwargs.get("system", "") for call in registry.chat.call_args_list]
        kono_present = any("KONO Memory" in s for s in systems)
        self.assertTrue(kono_present, "At least one council call should have KONO context")

    @patch("lib.hal.kono_bridge._kono_available", return_value=False)
    def test_council_works_without_kono(self, _avail):
        """Council should function normally when KONO is unavailable."""
        from lib.hal.moe import HALCouncil

        responses = {"m/a": "answer A"}
        registry = _make_registry(responses)
        council = HALCouncil(registry)

        result = council.deliberate("Hello", models=["m/a"])
        self.assertIn("answer A", result.merged_text)


# ---------------------------------------------------------------------------
# GRIP Bridge + KONO
# ---------------------------------------------------------------------------

class TestGripBridgeKonoRecall(unittest.TestCase):
    """Test the new load_grip_recall function."""

    # load_grip_recall returns "" when no GRIP install folder exists. A stranger's
    # machine has none, so stand one in; the KONO calls are patched below (HAL#885).
    @patch("lib.hal.grip_bridge.CLAUDE_DIR", MagicMock(exists=lambda: True))
    @patch("lib.hal.kono_bridge._kono_available", return_value=True)
    @patch("lib.hal.kono_bridge._recall_with_timeout", return_value="[KONO Memory]\n  1. Test recall")
    def test_load_grip_recall_returns_context(self, _recall, _avail):
        from lib.hal.grip_bridge import load_grip_recall
        result = load_grip_recall("test query")
        self.assertIn("KONO Memory", result)

    def test_load_grip_recall_empty_query(self):
        from lib.hal.grip_bridge import load_grip_recall
        self.assertEqual(load_grip_recall(""), "")

    @patch("lib.hal.kono_bridge._kono_available", return_value=False)
    def test_load_grip_recall_graceful_degradation(self, _avail):
        from lib.hal.grip_bridge import load_grip_recall
        result = load_grip_recall("anything")
        self.assertEqual(result, "")

    def test_load_grip_context_still_works(self):
        """Existing static context function unchanged."""
        from lib.hal.grip_bridge import load_grip_context
        # Should return a string (possibly empty if GRIP not fully configured)
        result = load_grip_context()
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
