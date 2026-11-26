"""Tests for HAL core types."""
import os
import sys
import unittest

from lib.hal.types import (
    AuthMethod,
    ChatMessage,
    ChatResponse,
    CostSnapshot,
    CouncilResult,
    CouncilVote,
    ModelInfo,
    ProviderConfig,
    ProviderInfo,
    ProviderType,
    RoutingConfig,
    RoutingStrategy,
    TokenUsage,
)


class TestChatMessage(unittest.TestCase):

    def test_frozen(self):
        msg = ChatMessage(role="user", content="hello")
        with self.assertRaises(AttributeError):
            msg.role = "assistant"

    def test_fields(self):
        msg = ChatMessage(role="assistant", content="hi", name="hal")
        self.assertEqual(msg.role, "assistant")
        self.assertEqual(msg.name, "hal")


class TestChatResponse(unittest.TestCase):

    def test_defaults(self):
        resp = ChatResponse(text="ok", model="test", provider="groq")
        self.assertEqual(resp.finish_reason, "stop")
        self.assertIsNone(resp.cost_usd)

    def test_with_usage(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp = ChatResponse(text="ok", model="m", provider="p", usage=usage)
        self.assertEqual(resp.usage.total_tokens, 15)


class TestTokenUsage(unittest.TestCase):

    def test_defaults_zero(self):
        usage = TokenUsage()
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.cache_read_tokens, 0)


class TestCostSnapshot(unittest.TestCase):

    def test_defaults(self):
        snap = CostSnapshot()
        self.assertEqual(snap.total_usd, 0.0)
        self.assertEqual(snap.input_tokens, 0)


class TestProviderEnums(unittest.TestCase):

    def test_provider_type_values(self):
        self.assertEqual(ProviderType.CLOUD.value, "cloud")
        self.assertEqual(ProviderType.LOCAL.value, "local")

    def test_routing_strategy_values(self):
        self.assertEqual(RoutingStrategy.FALLBACK.value, "fallback")
        self.assertEqual(RoutingStrategy.COUNCIL.value, "council")

    def test_auth_method_values(self):
        self.assertEqual(AuthMethod.API_KEY.value, "api_key")
        self.assertEqual(AuthMethod.NONE.value, "none")


class TestProviderConfig(unittest.TestCase):

    def test_defaults(self):
        config = ProviderConfig(
            name="test", type=ProviderType.CLOUD, base_url="http://localhost",
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.timeout_s, 120.0)


class TestRoutingConfigPrimary(unittest.TestCase):

    def test_primary_returns_first_chain_entry(self):
        config = RoutingConfig(
            fallback_chain=["groq/llama-3.3-70b-versatile", "gemini/gemini-2.5-flash"],
            default_model="fallback/model",
        )
        self.assertEqual(config.primary, "groq/llama-3.3-70b-versatile")

    def test_primary_falls_back_to_default_model_when_chain_empty(self):
        config = RoutingConfig(fallback_chain=[], default_model="groq/llama-3.3-70b-versatile")
        self.assertEqual(config.primary, "groq/llama-3.3-70b-versatile")

    def test_primary_returns_empty_string_when_both_unset(self):
        config = RoutingConfig()
        self.assertEqual(config.primary, "")

    def test_primary_single_entry_chain(self):
        config = RoutingConfig(fallback_chain=["ollama/gemma4:e4b"], default_model="other/model")
        self.assertEqual(config.primary, "ollama/gemma4:e4b")


class TestCouncilResult(unittest.TestCase):

    def test_empty(self):
        result = CouncilResult(merged_text="none")
        self.assertEqual(len(result.votes), 0)
        self.assertEqual(result.agreement_score, 0.0)


if __name__ == "__main__":
    unittest.main()
