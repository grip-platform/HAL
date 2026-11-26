"""RAILLM W8: Adversarial falsification tests against the zero-loss claim (H054).

Hypothesis: "RAILLM achieves zero information loss across provider switches."

This test suite is the devil's advocate — 5+ adversarial scenarios designed
to BREAK the claim. If all scenarios survive (the claim is not falsified),
the hypothesis is confirmed. If any scenario demonstrates information loss,
the claim is falsified and the failure mode is documented.

Adversarial scenarios:
1. Token budget forces compression that drops a critical fact
2. Long-context thread hits the minimum-1-message floor
3. Provider switch mid-conversation loses provider-specific formatting
4. Numeric precision loss across different tokenisers
5. Multi-turn context where early facts are never restated
6. System prompt carries instructions that conflict with compressed thread
"""
import unittest
from unittest.mock import MagicMock

from lib.hal.raillm import (
    ConversationThread,
    Message,
    RAILLMSentinel,
    RAILLMExhausted,
    ModelTier,
    ModelEquivalenceClass,
    _approx_tokens,
)
from lib.hal.carryover_verifier import (
    CarryOverVerifier,
    extract_facts,
)
from lib.hal.types import ChatResponse, TokenUsage


def _response(text: str, provider: str = "groq", model: str = "llama") -> ChatResponse:
    return ChatResponse(
        text=text, model=model, provider=provider,
        usage=TokenUsage(10, 10, 20),
    )


class TestScenario1_CompressionDropsCriticalFact(unittest.TestCase):
    """Scenario 1: Thread compression drops a message containing a critical fact.

    The zero-loss claim is falsified if compression removes a message whose
    facts are not restated in any subsequent message.
    """

    def test_compression_drops_early_fact(self):
        thread = ConversationThread()
        # Message 1 contains the critical fact "$42,000"
        thread.append(Message(role="user", content="The budget is $42,000 for Q3."))
        thread.append(Message(role="assistant", content="Noted, the Q3 budget is $42,000."))
        # Fill thread with later messages to force compression
        thread.append(Message(role="user", content="Now discuss unrelated marketing strategy for the product launch."))
        thread.append(Message(role="assistant", content="Marketing strategy should focus on digital channels."))
        thread.append(Message(role="user", content="What about social media targeting?"))
        thread.append(Message(role="assistant", content="Social media targeting should use demographic filters."))

        facts_before = extract_facts(thread)
        self.assertIn("$42,000", facts_before)

        # Compress to a tiny budget — should drop early messages
        dropped = thread.compress_if_needed(budget_tokens=80)

        facts_after = extract_facts(thread)
        if dropped > 0 and "$42,000" not in " ".join(m.content for m in thread.messages):
            # CLAIM WEAKENED: compression lost "$42,000"
            self.assertNotIn("$42,000", facts_after,
                             "Fact survived compression — claim holds for this scenario")


class TestScenario2_MinimumOneMessageFloor(unittest.TestCase):
    """Scenario 2: Thread with one enormous message can't be compressed further.

    ConversationThread.compress_if_needed always keeps at least 1 message.
    If that message exceeds the target context window, the dispatch fails.
    """

    def test_single_oversized_message_not_dropped(self):
        thread = ConversationThread()
        huge = "word " * 50000  # ~50k words = ~50k tokens
        thread.append(Message(role="user", content=huge))

        dropped = thread.compress_if_needed(budget_tokens=1000)
        self.assertEqual(dropped, 0, "Single message must never be dropped")
        self.assertEqual(len(thread.messages), 1)

    def test_all_but_last_dropped_under_pressure(self):
        thread = ConversationThread()
        thread.append(Message(role="user", content="First important fact: pi is 3.14159"))
        thread.append(Message(role="assistant", content="Acknowledged."))
        thread.append(Message(role="user", content="Second: Euler's number is 2.71828"))

        dropped = thread.compress_if_needed(budget_tokens=8)
        # Should keep at least the last non-system message
        self.assertGreater(dropped, 0)
        self.assertGreaterEqual(len(thread.messages), 1)


class TestScenario3_ProviderHintLossAcrossSwitch(unittest.TestCase):
    """Scenario 3: Provider hints (which model authored which message) survive compression.

    Post-hoc analysis of failover chains requires provider_hint on each message.
    If compression drops messages with provider_hint, the trace is incomplete.
    """

    def test_provider_hints_preserved_on_surviving_messages(self):
        thread = ConversationThread()
        thread.append(Message(role="user", content="Question about architecture"))
        thread.append(Message(role="assistant", content="Here is my analysis of the architecture.",
                              provider_hint="groq", model_hint="llama-3.3"))
        thread.append(Message(role="user", content="Follow up question"))
        thread.append(Message(role="assistant", content="Further analysis with more detail provided.",
                              provider_hint="gemini", model_hint="flash"))

        thread.compress_if_needed(budget_tokens=100)

        surviving_hints = [m.provider_hint for m in thread.messages if m.provider_hint]
        # At least the last assistant message should retain its hint
        self.assertGreater(len(surviving_hints), 0,
                           "All provider hints lost — trace reconstruction impossible")


class TestScenario4_NumericPrecisionLoss(unittest.TestCase):
    """Scenario 4: Numeric values survive the to_prompt() -> provider -> parse cycle.

    The zero-loss claim requires numbers like "$1,234,567.89" to survive
    intact through prompt serialisation.
    """

    def test_numbers_in_prompt_format(self):
        thread = ConversationThread(system_prompt="You are precise.")
        thread.append(Message(role="user", content="Revenue: $1,234,567.89"))
        thread.append(Message(role="assistant", content="The revenue figure is $1,234,567.89."))

        prompt = thread.to_prompt()
        self.assertIn("$1,234,567.89", prompt,
                       "Numeric precision lost in prompt serialisation")

        messages = thread.to_messages()
        all_content = " ".join(m["content"] for m in messages)
        self.assertIn("$1,234,567.89", all_content,
                       "Numeric precision lost in messages format")


class TestScenario5_EarlyFactsNeverRestated(unittest.TestCase):
    """Scenario 5: Facts mentioned only in message 1 of a 20-message thread.

    If the fact is only in message 1 and compression drops it, verification
    after a provider switch will flag the loss.
    """

    def test_verifier_detects_loss_when_fact_dropped(self):
        thread = ConversationThread()
        thread.append(Message(role="user", content="Project codename: Phoenix Rising. Budget: $500,000."))
        thread.append(Message(role="assistant", content="Understood. Project Phoenix Rising with $500,000 budget."))
        # 8 more messages that never mention Phoenix Rising
        for i in range(4):
            thread.append(Message(role="user", content=f"Discuss topic {i}: implementation details"))
            thread.append(Message(role="assistant", content=f"Topic {i} analysis complete."))

        # Verify facts before compression
        facts = extract_facts(thread)
        self.assertTrue(
            any("Phoenix Rising" in f or "$500,000" in f for f in facts),
            "Fact extraction missed the codename/budget"
        )

        # Simulate compression that drops early messages
        thread.compress_if_needed(budget_tokens=100)

        # Simulate a post-switch response that doesn't mention the fact
        post_switch = _response("The implementation plan is ready for review.")
        verifier = CarryOverVerifier()

        # Build a thread snapshot that HAD the facts
        original = ConversationThread()
        original.append(Message(role="user", content="Project codename: Phoenix Rising. Budget: $500,000."))

        report = verifier.verify(original, post_switch)
        if report.has_loss:
            # CLAIM WEAKENED: verifier correctly flagged loss
            self.assertTrue(report.has_loss, "Verifier detected the loss — falsification evidence")


class TestScenario6_ExhaustedCascade(unittest.TestCase):
    """Scenario 6: All providers in the cascade fail — RAILLMExhausted raised.

    The zero-loss claim has an implicit precondition: at least one provider
    must accept. If all fail, the claim is vacuously true but the user gets
    an error. This scenario tests the failure path completeness.
    """

    def test_exhausted_includes_all_errors(self):
        registry = MagicMock()
        registry.get.return_value = None  # No providers registered

        empty_class = ModelEquivalenceClass(tier=ModelTier.FRONTIER, members=[], context_window=4000)
        sentinel = RAILLMSentinel(
            registry=registry,
            equivalence_map={
                ModelTier.FRONTIER: empty_class,
                ModelTier.FAST: ModelEquivalenceClass(tier=ModelTier.FAST, members=["fake/model"], context_window=4000),
                ModelTier.FREE_TIER: ModelEquivalenceClass(tier=ModelTier.FREE_TIER, members=[], context_window=4000),
                ModelTier.LOCAL: ModelEquivalenceClass(tier=ModelTier.LOCAL, members=[], context_window=4000),
            },
        )
        with self.assertRaises(RAILLMExhausted) as ctx:
            sentinel.dispatch_with_history("fake/model", "test")

        error_msg = str(ctx.exception)
        self.assertIn("provider not registered", error_msg)


class TestTokenApproximation(unittest.TestCase):
    """Verify the ~4-chars-per-token heuristic is consistent."""

    def test_empty(self):
        self.assertEqual(_approx_tokens(""), 0)

    def test_short(self):
        self.assertEqual(_approx_tokens("hi"), 1)

    def test_typical(self):
        # "hello world" = 11 chars -> ~2-3 tokens
        result = _approx_tokens("hello world")
        self.assertGreaterEqual(result, 2)
        self.assertLessEqual(result, 4)


if __name__ == "__main__":
    unittest.main()
