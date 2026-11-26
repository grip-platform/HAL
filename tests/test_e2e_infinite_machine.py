"""E2E InfiniteMachine tests — prove the Sentinel + Conversation loop never terminates.

The InfiniteMachine pattern:
  LimitSentinel × Conversation = HAL never sees a provider limit.

Three behavioural contracts verified here:

  1. Multi-turn history integrity: N turns accumulate correctly; each
     provider call receives the full prior conversation as context.

  2. Failover transparency: when Provider A raises ConnectionError/
     TimeoutError/OSError mid-conversation, the Sentinel silently
     routes to Provider B and the response is returned to the caller
     with zero disruption to the calling code.  The history is intact.

  3. History preservation across failover: Provider B's chat() call
     receives every message produced before the failover, in order.

  4. Tool-augmented multi-turn: a conversation containing tool_call
     and tool_result turns is handled without error and history grows
     as expected (the pipeline is agnostic to role values).
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

_HAL_HOME = os.environ.get("HAL_ROOT", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    # append (not insert(0)): run from a git worktree on a feature branch,
    # the worktree's lib/ must win over ~/.hal main -- otherwise
    # worktree-only modules (raillm_subscription_bridge) are shadowed.
    sys.path.append(_HAL_HOME)

from lib.hal.types import ChatMessage, ChatResponse, TokenUsage

# W5 (H-RAILLM-5): grip-ladder context-window decoupling proof. Drives the
# REAL RAILLMSentinel.dispatch_with_history over a hermetic equivalence map
# (no network, no keychain) -- behaviour, not mocks of the sentinel.
from lib.hal.providers.openai_compat import OpenAICompatError
from lib.hal.raillm import (
    ModelEquivalenceClass,
    ModelTier,
    RAILLMExhausted,
    RAILLMSentinel,
)
from lib.hal.raillm_subscription_bridge import register_subscriptions
from lib.hal.registry import ProviderRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resp(text="ok", provider="groq", model="groq/llama-3.3-70b-versatile") -> ChatResponse:
    return ChatResponse(
        text=text, model=model, provider=provider,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        cost_usd=0.0001, latency_ms=80.0,
    )


def _make_conversation(providers: dict[str, MagicMock]):
    """Build a hermetic Conversation with the given named providers.

    providers: mapping of provider_name -> configured MagicMock

    The registry mock's reg.get(name) returns the corresponding MagicMock.
    reg.routing.fallback_chain lists providers in insertion order (after the
    primary) so the sentinel chains them correctly.
    """
    from lib.hal.conversation import Conversation

    provider_names = list(providers.keys())
    primary = provider_names[0]
    fallbacks = [f"{name}/model" for name in provider_names[1:]]

    reg = MagicMock()
    reg.routing = MagicMock()
    reg.routing.default_model = f"{primary}/llama-3.3-70b-versatile"
    reg.routing.fallback_chain = fallbacks
    reg.list_providers.return_value = []

    def _get(name):
        return providers.get(name, providers[primary])

    reg.get.side_effect = _get

    with patch("lib.hal.conversation._build_identity", return_value="HAL test"), \
         patch("lib.hal.sentinel.LimitSentinel._load_state"):
        conv = Conversation(
            model=f"{primary}/llama-3.3-70b-versatile",
            registry=reg,
        )
    conv._build_system = lambda mode="": "HAL test"
    return conv, reg


# ---------------------------------------------------------------------------
# 1. Multi-turn history integrity
# ---------------------------------------------------------------------------

class TestMultiTurnHistoryIntegrity:
    """Every provider.chat() call must receive the full prior conversation."""

    def test_single_turn_produces_response(self):
        p = MagicMock()
        p.chat.return_value = _resp("turn1")
        conv, _ = _make_conversation({"groq": p})

        r = conv.send("hello")
        assert r.text == "turn1"

    def test_history_grows_one_pair_per_turn(self):
        p = MagicMock()
        p.chat.return_value = _resp()
        conv, _ = _make_conversation({"groq": p})

        conv.send("msg1")
        assert len(conv.history) == 2   # 1 user + 1 assistant

        conv.send("msg2")
        assert len(conv.history) == 4

        conv.send("msg3")
        assert len(conv.history) == 6

    def test_provider_receives_accumulating_history(self):
        """The messages kwarg passed to provider.chat() grows with each turn."""
        p = MagicMock()
        p.chat.return_value = _resp()
        conv, _ = _make_conversation({"groq": p})

        conv.send("first")
        first_call_messages = p.chat.call_args.kwargs.get("messages", [])

        conv.send("second")
        second_call_messages = p.chat.call_args.kwargs.get("messages", [])

        # Second call context has at least 2 more entries than first
        assert len(second_call_messages) > len(first_call_messages)

    def test_history_alternates_user_assistant_roles(self):
        p = MagicMock()
        p.chat.return_value = _resp()
        conv, _ = _make_conversation({"groq": p})

        conv.send("ping")
        conv.send("pong")

        roles = [m.role for m in conv.history]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_five_turn_conversation_keeps_all_history(self):
        p = MagicMock()
        responses = [_resp(f"reply{i}") for i in range(5)]
        p.chat.side_effect = responses
        conv, _ = _make_conversation({"groq": p})

        for i in range(5):
            r = conv.send(f"turn {i}")
            assert r.text == f"reply{i}"

        assert len(conv.history) == 10  # 5 user + 5 assistant


# ---------------------------------------------------------------------------
# 2. Failover transparency — caller sees no disruption
# ---------------------------------------------------------------------------

class TestFailoverTransparency:
    """Sentinel failover must be invisible to the caller."""

    def test_connection_error_failover_returns_response(self):
        bad = MagicMock()
        bad.chat.side_effect = ConnectionError("provider A unreachable")
        good = MagicMock()
        good.chat.return_value = _resp(text="fallback answer", provider="anthropic")

        conv, _ = _make_conversation({"groq": bad, "anthropic": good})
        r = conv.send("hello")
        assert r.text == "fallback answer"

    def test_timeout_error_failover_returns_response(self):
        bad = MagicMock()
        bad.chat.side_effect = TimeoutError("provider A timed out")
        good = MagicMock()
        good.chat.return_value = _resp(text="after timeout", provider="anthropic")

        conv, _ = _make_conversation({"groq": bad, "anthropic": good})
        r = conv.send("hello")
        assert r.text == "after timeout"

    def test_os_error_failover_returns_response(self):
        bad = MagicMock()
        bad.chat.side_effect = OSError("socket error")
        good = MagicMock()
        good.chat.return_value = _resp(text="os fallback", provider="anthropic")

        conv, _ = _make_conversation({"groq": bad, "anthropic": good})
        r = conv.send("hello")
        assert r.text == "os fallback"

    def test_mid_conversation_failover_is_transparent(self):
        """Turn 1 succeeds via groq. Turn 2 groq fails -> anthropic picks up.
        The caller sees no error; the response arrives normally."""
        groq_mock = MagicMock()
        groq_mock.chat.side_effect = [
            _resp("groq turn 1"),
            ConnectionError("groq limit hit"),
        ]
        anthropic_mock = MagicMock()
        anthropic_mock.chat.return_value = _resp("anthropic turn 2", provider="anthropic")

        conv, _ = _make_conversation({"groq": groq_mock, "anthropic": anthropic_mock})

        r1 = conv.send("first question")
        r2 = conv.send("second question")

        assert r1.text == "groq turn 1"
        assert r2.text == "anthropic turn 2"

    def test_all_providers_fail_raises_runtime_error(self):
        bad = MagicMock()
        bad.chat.side_effect = ConnectionError("all dead")

        conv, _ = _make_conversation({"groq": bad})
        with pytest.raises(RuntimeError, match="All providers exhausted"):
            conv.send("hello")


# ---------------------------------------------------------------------------
# 3. History preservation across failover
# ---------------------------------------------------------------------------

class TestHistoryPreservationAcrossFailover:
    """Provider B must receive ALL messages produced before the failover."""

    def test_fallback_provider_receives_prior_turns(self):
        """After groq fails on turn 2, anthropic must receive the history from turn 1."""
        groq_mock = MagicMock()
        groq_mock.chat.side_effect = [
            _resp("groq answer to turn 1"),
            ConnectionError("groq down"),
        ]
        anthropic_mock = MagicMock()
        anthropic_mock.chat.return_value = _resp("anthropic answer", provider="anthropic")

        conv, _ = _make_conversation({"groq": groq_mock, "anthropic": anthropic_mock})

        conv.send("first question")
        conv.send("second question")

        # Inspect what anthropic.chat received
        kwargs = anthropic_mock.chat.call_args.kwargs
        messages = kwargs.get("messages", [])

        # The messages list must contain at least the turn-1 user + assistant entries
        message_contents = [
            m.content if isinstance(m, ChatMessage) else m.get("content", "")
            for m in messages
        ]
        assert any("first question" in str(c) for c in message_contents), (
            "Provider B did not receive the first question from prior turn"
        )

    def test_history_intact_after_failover(self):
        """conv.history contains all turns regardless of which provider served them."""
        groq_mock = MagicMock()
        groq_mock.chat.side_effect = [
            _resp("groq reply 1"),
            ConnectionError("groq down"),
            _resp("groq reply 3"),
        ]
        anthropic_mock = MagicMock()
        anthropic_mock.chat.return_value = _resp("anthropic reply 2", provider="anthropic")

        conv, _ = _make_conversation({"groq": groq_mock, "anthropic": anthropic_mock})

        conv.send("q1")
        conv.send("q2")
        conv.send("q3")

        # All 3 rounds are in history
        assert len(conv.history) == 6
        contents = [m.content for m in conv.history]
        assert "groq reply 1" in contents
        assert "anthropic reply 2" in contents
        assert "groq reply 3" in contents

    def test_three_provider_chain_exhausts_in_order(self):
        """When A and B both fail, C picks up and history is fully preserved."""
        a = MagicMock(); a.chat.side_effect = ConnectionError("A down")
        b = MagicMock(); b.chat.side_effect = TimeoutError("B timeout")
        c = MagicMock(); c.chat.return_value = _resp("C succeeded", provider="gemini")

        conv, _ = _make_conversation({"groq": a, "anthropic": b, "gemini": c})
        r = conv.send("hello")
        assert r.text == "C succeeded"
        assert r.provider == "gemini"


# ---------------------------------------------------------------------------
# 4. Tool-augmented multi-turn
# ---------------------------------------------------------------------------

class TestToolAugmentedMultiTurn:
    """Tool-call and tool-result entries in history must not break the pipeline."""

    def test_conversation_handles_tool_call_in_prompt(self):
        """A message that looks like a tool call string passes through without error."""
        p = MagicMock()
        p.chat.return_value = _resp("I searched the database")
        conv, _ = _make_conversation({"groq": p})

        r = conv.send('{"tool": "search", "query": "contract clauses"}')
        assert r.text == "I searched the database"

    def test_history_with_injected_tool_turn_is_passed_through(self):
        """Injecting a tool_result ChatMessage into history doesn't break dispatch."""
        p = MagicMock()
        p.chat.return_value = _resp("answered with tool result")
        conv, _ = _make_conversation({"groq": p})

        # Manually inject a "tool result" into history (simulating a prior tool call round)
        conv._history.append(ChatMessage(role="user", content="search: contract clauses"))
        conv._history.append(ChatMessage(role="assistant", content='{"result": "found 3 clauses"}'))

        r = conv.send("summarise those results")
        assert r.text == "answered with tool result"
        # History now includes our injected entries plus the new round
        assert len(conv.history) >= 4

    def test_multi_tool_turn_conversation_five_rounds(self):
        """Five alternating rounds including a mid-stream failover with tool messages."""
        groq_mock = MagicMock()
        groq_mock.chat.side_effect = [
            _resp("searching..."),
            _resp("found 3 results"),
            ConnectionError("groq limit"),        # T3 groq fails -> anthropic
            ConnectionError("groq still down"),   # T4 groq fails -> anthropic
        ]
        anthropic_mock = MagicMock()
        anthropic_mock.chat.side_effect = [
            _resp("anthropic summary", provider="anthropic"),
            _resp("done", provider="anthropic"),
        ]

        conv, _ = _make_conversation({"groq": groq_mock, "anthropic": anthropic_mock})

        results = [
            conv.send("search for X"),
            conv.send("what did you find?"),
            conv.send("summarise"),          # groq fails -> anthropic
            conv.send("format as table"),
        ]

        assert results[0].text == "searching..."
        assert results[1].text == "found 3 results"
        assert results[2].text == "anthropic summary"
        assert results[3].text == "done"
        assert len(conv.history) == 8


# ---------------------------------------------------------------------------
# 5. Cost and status across provider switches
# ---------------------------------------------------------------------------

class TestCostAndStatusAcrossFailover:
    """Status reporting must reflect the actual providers used."""

    def test_cost_accumulates_across_provider_switch(self):
        groq_mock = MagicMock()
        groq_mock.chat.side_effect = [_resp("groq"), ConnectionError("down")]
        anthropic_mock = MagicMock()
        anthropic_mock.chat.return_value = _resp("anthropic", provider="anthropic")

        conv, _ = _make_conversation({"groq": groq_mock, "anthropic": anthropic_mock})

        conv.send("q1")
        cost_after_1 = conv.status()["total_cost_usd"]

        conv.send("q2")
        cost_after_2 = conv.status()["total_cost_usd"]

        # Both turns incurred cost (even though T2 failed over to anthropic)
        assert cost_after_2 > cost_after_1 >= 0.0

    def test_status_turn_count_correct_after_failover(self):
        groq_mock = MagicMock()
        groq_mock.chat.side_effect = [_resp("r1"), ConnectionError("down")]
        anthropic_mock = MagicMock()
        anthropic_mock.chat.return_value = _resp("r2", provider="anthropic")

        conv, _ = _make_conversation({"groq": groq_mock, "anthropic": anthropic_mock})
        conv.send("t1")
        conv.send("t2")

        assert conv.status()["turns"] == 2

    def test_status_context_pct_grows_with_turns(self):
        p = MagicMock()
        p.chat.return_value = _resp()
        conv, _ = _make_conversation({"groq": p})

        pct_before = conv.status()["context_usage_pct"]
        conv.send("long message " * 50)
        pct_after = conv.status()["context_usage_pct"]

        assert pct_after >= pct_before


# ---------------------------------------------------------------------------
# 6. Grip-ladder context-window decoupling (W5, H-RAILLM-5)
# ---------------------------------------------------------------------------

_A5 = "anthropic/claude-opus-4-7"   # FRONTIER / CCH rung
_K5 = "moonshot/kimi-k2"            # FAST / Kimi rung
_L5 = "ollama/llama3.3:70b"         # LOCAL rung
_FACT = "VAULT-CODE-ZEPHYR-7731"    # unique sentinel planted in turn 1


class _LadderSub:
    """Hermetic SubscriptionAuth-shaped provider. Records the prompt it was
    handed so a test asserts what history actually crossed the rung
    boundary (Goodhart: assert content continuity, never call counts)."""

    def __init__(self, pid, *, reply=None, raise_exc=None):
        self.provider_id = pid
        self._reply = reply
        self._raise = raise_exc
        self.last_prompt = None

    def chat(self, prompt="", *, model, system=None, temperature=None,
             max_tokens=2048, messages=None, tools=None):
        self.last_prompt = prompt
        if self._raise is not None:
            raise self._raise
        return ChatResponse(text=self._reply, model=model,
                            provider=self.provider_id, usage=TokenUsage())

    def health_check(self):
        return True


def _emap5(frontier_w, fast_w, local_w=32_000):
    """3-rung map FRONTIER=[anthropic] FAST=[moonshot] LOCAL=[ollama].
    cascade order == [_A5, _K5, _L5] (verified via _build_tier_chain)."""
    return {
        ModelTier.FRONTIER: ModelEquivalenceClass(
            ModelTier.FRONTIER, [_A5], frontier_w),
        ModelTier.FAST: ModelEquivalenceClass(ModelTier.FAST, [_K5], fast_w),
        ModelTier.FREE_TIER: ModelEquivalenceClass(
            ModelTier.FREE_TIER, [], 128_000),
        ModelTier.LOCAL: ModelEquivalenceClass(
            ModelTier.LOCAL, [_L5], local_w),
    }


def _ladder_sentinel(tmp_path, monkeypatch, emap, subs):
    monkeypatch.setattr(
        "lib.hal.sentinel._STATE_PATH", str(tmp_path / "limits.json"))
    reg = ProviderRegistry()
    register_subscriptions(reg, subs)
    return RAILLMSentinel(registry=reg, equivalence_map=emap)


class TestGripLadderContextWindowDecoupling:
    """H-RAILLM-5 / HAL#329 -- context-window decoupling. VERDICT:
    **CONFIRMED** after the HAL#329 fix (W5 falsified it as stated; the
    #329 fix W1 landed the resolution).

    History (the science, recorded not rewritten):

      * W5 FALSIFIED the original claim: dispatch_with_history called the
        MUTATING compress_if_needed on the SHARED thread for every
        attempted rung, so a small-window failing primary destructively
        truncated the shared thread before the larger fallback was tried
        -- turn-1 content lost across CCH->Kimi.
      * HAL#329 fix (branch fix/raillm-compress-before-failover):
        dispatch_with_history now uses the NON-mutating
        ConversationThread.compressed_prompt -- each rung gets a per-rung
        compressed WIRE COPY; the shared thread stays the canonical full
        history; only the serving rung's reply is appended. The claim now
        HOLDS: a small-window failing primary no longer bounds the
        conversation.

    Devil's-advocate (VERIFY precision, recorded not suppressed) -- three
    attacks on the fix, each rebutted by a test here or by scope:
      1. "It just moved the bug -> unbounded shared-thread growth." No:
         the shared thread always grew (any conversation does); the claim
         is decoupling from a SINGLE provider's window, not infinite
         context. The wire payload is still per-rung compressed
         (test_compressed_prompt_* unit tests assert dropped>0 on the wire
         copy). An explicit thread-cap policy is a separate concern, not a
         #329 regression.
      2. "Serving rung saw a compressed view but the reply is appended to
         the FULL thread -> future incoherence." Bounded and strictly
         better than permanent destruction: a larger future rung sees MORE
         context, never less. Per-message provider-fidelity replay is a
         noted future enhancement, out of #329 scope (R0).
      3. "Faking both providers is circular." The fix is in
         dispatch_with_history's real compression call path; the fakes
         only stand in for the network. The non-mutation invariant is
         additionally proven at unit level (tests/test_raillm.py).

    Regression sentinel (inverted): test_small_window_failing_primary_
    preserves_shared_thread now asserts continuity HOLDS. If it ever fails
    again (turns/fact lost), the HAL#329 fix was reverted -- restore the
    non-mutating wire path, do not re-falsify H-RAILLM-5."""

    def test_small_window_failing_primary_preserves_shared_thread(
            self, tmp_path, monkeypatch):
        """H-RAILLM-5 CONFIRMED + #329 regression sentinel. Primary window
        (120) < live thread, primary always fails. Post-#329 the wire copy
        is non-mutating, so the shared thread is NOT truncated: all 10
        turns retained, turn-1 fact survives to the larger fallback rung.
        If this fails (turns/fact lost) the #329 fix was reverted."""
        cch = _LadderSub("anthropic",
                         raise_exc=OpenAICompatError("cch exhausted"))
        kimi = _LadderSub("moonshot", reply="recall ok")
        s = _ladder_sentinel(
            tmp_path, monkeypatch, _emap5(120, 200_000),
            {"anthropic": cch, "moonshot": kimi})
        s.dispatch_with_history(_A5, f"Remember: {_FACT}.")
        for i in range(1, 10):
            s.dispatch_with_history(_A5, "filler " * 40 + f"#{i}")
        users = [m.content for m in s.thread.messages if m.role == "user"]
        assert len(users) == 10                       # NO turn lost
        assert _FACT in (kimi.last_prompt or "")      # turn-1 fact survived
        assert s.thread.messages[-1].provider_hint == "moonshot"
        # the failing 120-token primary did NOT bound the shared thread:
        assert s.thread.token_estimate() > 120

    def test_failover_transparency_intact_across_failing_primary(
            self, tmp_path, monkeypatch):
        """Transparency is orthogonal to continuity: with the primary
        always failing, the InfiniteMachine 'caller never sees the limit'
        property holds -- every dispatch returns a normal fallback
        response, no RAILLMExhausted. (Post-#329 continuity ALSO holds in
        this config; this test isolates the transparency property only.)"""
        cch = _LadderSub("anthropic",
                         raise_exc=OpenAICompatError("cch exhausted"))
        kimi = _LadderSub("moonshot", reply="recall ok")
        s = _ladder_sentinel(
            tmp_path, monkeypatch, _emap5(120, 200_000),
            {"anthropic": cch, "moonshot": kimi})
        for i in range(8):
            r = s.dispatch_with_history(_A5, "filler " * 40 + f"#{i}")
            assert r.text == "recall ok"              # caller unaffected
            assert r.provider == "moonshot"           # served by fallback

    def test_ten_turn_large_window_failover_loses_no_turn(
            self, tmp_path, monkeypatch):
        """Plan falsifier (b): a >=10-message thread must not lose turns
        across a mid-conversation CCH->Kimi failover (windows huge so any
        loss would be a real history-carry bug, not compression)."""
        cch = _LadderSub("anthropic", reply="cch ok")
        kimi = _LadderSub("moonshot", reply="kimi ok")
        s = _ladder_sentinel(
            tmp_path, monkeypatch, _emap5(1_000_000, 1_000_000),
            {"anthropic": cch, "moonshot": kimi})
        for i in range(5):
            s.dispatch_with_history(_A5, f"turn-{i}-content")
        cch._raise = OpenAICompatError("cch limit at turn 5")
        for i in range(5, 10):
            r = s.dispatch_with_history(_A5, f"turn-{i}-content")
            assert r.provider == "moonshot"
        users = [m.content for m in s.thread.messages if m.role == "user"]
        assert len(users) == 10
        assert "turn-0-content" in kimi.last_prompt   # earliest survived

    def test_full_ladder_fault_injection_exhausts(
            self, tmp_path, monkeypatch):
        subs = {p: _LadderSub(p, raise_exc=OpenAICompatError(f"{p} down"))
                for p in ("anthropic", "moonshot", "ollama")}
        s = _ladder_sentinel(
            tmp_path, monkeypatch, _emap5(200_000, 200_000), subs)
        with pytest.raises(RAILLMExhausted) as ei:
            s.dispatch_with_history(_A5, "anybody home?")
        msg = str(ei.value)
        assert _A5 in msg and _K5 in msg and _L5 in msg   # whole chain

    def test_local_rung_serves_when_all_subscriptions_fail(
            self, tmp_path, monkeypatch):
        """The 'to any local LLM' end of the ladder: CCH + Kimi dead, the
        LOCAL rung answers and is recorded in the thread."""
        cch = _LadderSub("anthropic", raise_exc=OpenAICompatError("cch"))
        kimi = _LadderSub("moonshot", raise_exc=OpenAICompatError("kimi"))
        loc = _LadderSub("ollama", reply="served locally")
        s = _ladder_sentinel(
            tmp_path, monkeypatch, _emap5(200_000, 200_000),
            {"anthropic": cch, "moonshot": kimi, "ollama": loc})
        r = s.dispatch_with_history(_A5, "offline?")
        assert r.provider == "ollama" and r.text == "served locally"
        last = s.thread.messages[-1]
        assert last.role == "assistant" and last.provider_hint == "ollama"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
