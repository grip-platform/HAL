"""Goodhart-proof P3 W4 (VERIFY) — H-LLMSW end-to-end: switching the
RAILLM primary rung BETWEEN turns is context-loss-free.

This is the falsifiable core the whole P3 design rests on: a "switch
subscription / LLM" is just "make rung X the primary for the next
turn", and it loses zero context because the ONE reused
ConversationThread carries history across provider switches (HAL#329).
The W2 marker wire (#348, unit-tested there) is merely the *trigger*;
this proves the *guarantee* the trigger relies on, exercised through
the real RAILLMSentinel.dispatch_with_history on origin/main.

Harness is the canonical test_raillm.py seam reused verbatim (Rule 21):
_mock_registry + a MagicMock provider whose .chat returns a fake
ChatResponse and records the prompt it received.

Mutation-survival:
  - sentinel/thread reconstructed between turns -> turn-1 fact lost at
    the post-switch rung                         -> test_fact_survives_switch FAILS
  - history not carried into the dispatched prompt -> test_post_switch_prompt_has_turn1 FAILS
  - the thread object is not the single reused one -> test_one_session_thread_identity FAILS
  - a no-switch baseline diverges from switched    -> test_switch_matches_noswitch_baseline FAILS
"""
from __future__ import annotations

from unittest.mock import MagicMock

from lib.hal.raillm import ConversationThread, RAILLMSentinel
from lib.hal.types import ChatResponse

# Real ladder addresses — a CCH(Anthropic) -> Kimi -> CCH switch sequence.
_CCH = "anthropic/claude-opus-4-7"
_KIMI = "moonshot/kimi-k2"
_FACT = "ZEBRA-FACT-42-the-vault-code-is-octopus"   # unique turn-1 token


def _mock_registry():
    reg = MagicMock()
    reg.get = MagicMock(return_value=None)
    reg.list_providers = MagicMock(return_value=[])
    reg.routing = MagicMock(fallback_chain=[])
    reg._configs = {}
    return reg


def _capturing_provider() -> tuple[MagicMock, list]:
    """A provider whose .chat records every prompt it is handed and
    returns a deterministic fake response."""
    seen: list[str] = []

    def _chat(*args, **kwargs):
        prompt = kwargs.get("prompt")
        if prompt is None and args:
            prompt = args[0]
        seen.append(prompt or "")
        return ChatResponse(text="ack", model="fake/model",
                            provider="fake", raw={"_headers": {}})

    p = MagicMock()
    p.chat = MagicMock(side_effect=_chat)
    return p, seen


def _drive(sentinel: RAILLMSentinel, seq):
    """Run one (model_address, user_message) turn-list through the SAME
    sentinel (the operator switching the primary rung between turns)."""
    for addr, msg in seq:
        sentinel.dispatch_with_history(model_address=addr, user_message=msg)


def test_fact_survives_switch():
    reg = _mock_registry()
    provider, seen = _capturing_provider()
    reg.get.return_value = provider
    s = RAILLMSentinel(registry=reg)
    _drive(s, [(_CCH, f"remember this: {_FACT}"),
               (_KIMI, "what did I ask you to remember?"),
               (_CCH, "confirm it once more")])
    # the post-switch (turn 3, back on CCH) prompt still carries turn-1
    assert _FACT in seen[-1], "turn-1 fact lost after CCH->Kimi->CCH switch"


def test_post_switch_prompt_has_turn1():
    reg = _mock_registry()
    provider, seen = _capturing_provider()
    reg.get.return_value = provider
    s = RAILLMSentinel(registry=reg)
    _drive(s, [(_CCH, f"fact: {_FACT}"), (_KIMI, "follow-up turn on a different rung")])
    # turn 2 ran on a DIFFERENT primary, yet sees turn-1's content
    assert _FACT in seen[-1]
    assert len(seen) == 2                         # both turns dispatched


def test_one_session_thread_identity():
    reg = _mock_registry()
    provider, _ = _capturing_provider()
    reg.get.return_value = provider
    s = RAILLMSentinel(registry=reg)
    tid = id(s.thread)
    _drive(s, [(_CCH, "t1"), (_KIMI, "t2"), (_CCH, "t3")])
    assert id(s.thread) == tid                    # never reconstructed
    # 3 user + 3 assistant accumulated in the single reused thread
    assert len(s.thread.messages) == 6
    assert s.thread.messages[0].content == "t1"   # turn-1 still anchored


def test_switch_matches_noswitch_baseline():
    """The opted switch path must not lose anything the no-switch path
    keeps — same turn-1 carry with and without changing model_address."""
    reg_a = _mock_registry()
    pa, seen_a = _capturing_provider()
    reg_a.get.return_value = pa
    s_a = RAILLMSentinel(registry=reg_a)
    _drive(s_a, [(_CCH, f"baseline {_FACT}"), (_CCH, "same rung turn 2")])

    reg_b = _mock_registry()
    pb, seen_b = _capturing_provider()
    reg_b.get.return_value = pb
    s_b = RAILLMSentinel(registry=reg_b)
    _drive(s_b, [(_CCH, f"baseline {_FACT}"), (_KIMI, "switched rung turn 2")])

    assert (_FACT in seen_a[-1]) and (_FACT in seen_b[-1])  # carry both ways
