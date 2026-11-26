"""Unit tests for RAILLM — Redundant Array of Independent LLMs.

Covers W1 surface:
- Message dataclass
- ConversationThread (append, estimate, compress, to_prompt, to_messages)
- classify_model + DEFAULT_EQUIVALENCE_MAP
- RAILLMSentinel tier cascade + dispatch_with_history

Tests verify output VALUES (not call counts) per Goodhart guard rule.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

try:
    import pytest
except ImportError:  # pragma: no cover — pytest is an optional dev dep
    raise unittest.SkipTest("pytest not installed — skipping raillm test module")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.hal.raillm import (  # noqa: E402
    DEFAULT_EQUIVALENCE_MAP,
    ConversationThread,
    Message,
    ModelEquivalenceClass,
    ModelTier,
    RAILLMExhausted,
    RAILLMSentinel,
    _approx_tokens,
    classify_model,
)
from lib.hal.types import ChatResponse  # noqa: E402


def test_approx_tokens_empty_string_is_zero():
    assert _approx_tokens("") == 0


def test_approx_tokens_four_chars_approximates_one_token():
    assert _approx_tokens("abcd") == 1
    assert _approx_tokens("a" * 40) == 10
    assert _approx_tokens("a" * 400) == 100


def test_approx_tokens_short_string_rounds_up_to_at_least_one():
    assert _approx_tokens("hi") == 1


def test_message_defaults_timestamp_and_hints():
    m = Message(role="user", content="hello")
    assert m.role == "user"
    assert m.content == "hello"
    assert m.timestamp > 0
    assert m.provider_hint is None
    assert m.model_hint is None


def test_message_carries_provider_and_model_hints_when_set():
    m = Message(
        role="assistant",
        content="hi",
        provider_hint="groq",
        model_hint="llama-3.3-70b-versatile",
    )
    assert m.provider_hint == "groq"
    assert m.model_hint == "llama-3.3-70b-versatile"


def test_thread_append_extends_messages_in_order():
    t = ConversationThread()
    t.append(Message(role="user", content="hi"))
    t.append(Message(role="assistant", content="hello"))
    assert [m.role for m in t.messages] == ["user", "assistant"]
    assert t.messages[0].content == "hi"
    assert t.messages[1].content == "hello"


def test_thread_token_estimate_sums_system_and_messages():
    t = ConversationThread(system_prompt="a" * 40)  # ~10 tokens
    t.append(Message(role="user", content="b" * 400))  # ~100 tokens
    t.append(Message(role="assistant", content="c" * 800))  # ~200 tokens
    est = t.token_estimate()
    assert 300 <= est <= 320


def test_thread_compress_no_op_when_under_budget():
    t = ConversationThread()
    t.append(Message(role="user", content="hi"))
    t.append(Message(role="assistant", content="hello"))
    dropped = t.compress_if_needed(budget_tokens=10_000)
    assert dropped == 0
    assert len(t.messages) == 2


def test_thread_compress_drops_oldest_non_system_until_fit():
    t = ConversationThread(system_prompt="sys" * 4)
    for _ in range(10):
        t.append(Message(role="user", content="x" * 400))
    dropped = t.compress_if_needed(budget_tokens=500)
    assert dropped > 0
    assert t.token_estimate() <= 600


def test_thread_compress_preserves_last_non_system_message_even_if_oversized():
    t = ConversationThread()
    t.append(Message(role="user", content="x" * 10_000))
    dropped = t.compress_if_needed(budget_tokens=100)
    assert len(t.messages) == 1
    assert dropped == 0


def test_thread_compress_preserves_all_system_messages():
    t = ConversationThread()
    for i in range(5):
        t.append(Message(role="system", content=f"system directive {i}"))
    for _ in range(10):
        t.append(Message(role="user", content="x" * 400))
    t.compress_if_needed(budget_tokens=200)
    system_count = sum(1 for m in t.messages if m.role == "system")
    assert system_count == 5


def test_compressed_prompt_does_not_mutate_thread():
    """HAL#329 core invariant: compressed_prompt returns a fitted COPY and
    leaves the shared thread untouched (oldest content still present)."""
    t = ConversationThread()
    t.append(Message(role="user", content="OLDEST-MARKER " + "x" * 400))
    for _ in range(8):
        t.append(Message(role="user", content="y" * 400))
    t.append(Message(role="user", content="NEWEST-MARKER " + "z" * 400))
    before_len = len(t.messages)
    before_tok = t.token_estimate()

    prompt, dropped = t.compressed_prompt(budget_tokens=300)

    assert dropped > 0
    assert "NEWEST-MARKER" in prompt          # recent kept on the wire
    assert "OLDEST-MARKER" not in prompt      # oldest dropped from the wire
    # ...but the SHARED thread is intact (the whole point of #329):
    assert len(t.messages) == before_len
    assert t.token_estimate() == before_tok
    assert any("OLDEST-MARKER" in m.content for m in t.messages)


def test_compressed_prompt_no_op_under_budget_equals_to_prompt():
    t = ConversationThread(system_prompt="be helpful")
    t.append(Message(role="user", content="hi"))
    t.append(Message(role="assistant", content="hello"))
    prompt, dropped = t.compressed_prompt(budget_tokens=10_000)
    assert dropped == 0
    assert prompt == t.to_prompt()            # full render, nothing dropped
    assert len(t.messages) == 2               # unchanged


def test_compressed_prompt_keeps_system_and_last_message():
    t = ConversationThread(system_prompt="SYS-DIRECTIVE")
    t.append(Message(role="system", content="SYS-RULE"))
    for _ in range(6):
        t.append(Message(role="user", content="x" * 400))
    t.append(Message(role="user", content="LAST-MARKER"))
    prompt, dropped = t.compressed_prompt(budget_tokens=120)
    assert "SYS-DIRECTIVE" in prompt and "SYS-RULE" in prompt
    assert "LAST-MARKER" in prompt            # most-recent never dropped
    assert len(t.messages) == 8               # non-mutating (1 sys + 7 user)


def test_compress_if_needed_still_mutates_after_refactor():
    """Regression guard for the DRY extraction: the mutating contract that
    direct callers depend on is unchanged."""
    t = ConversationThread()
    for _ in range(10):
        t.append(Message(role="user", content="x" * 400))
    dropped = t.compress_if_needed(budget_tokens=300)
    assert dropped > 0
    assert len(t.messages) < 10               # DID mutate (oldest gone)
    assert t.token_estimate() <= 400


def test_thread_to_prompt_emits_xml_role_tags():
    t = ConversationThread(system_prompt="be helpful")
    t.append(Message(role="user", content="hi"))
    t.append(Message(role="assistant", content="hello"))
    prompt = t.to_prompt()
    assert "<system>\nbe helpful\n</system>" in prompt
    assert "<user>\nhi\n</user>" in prompt
    assert "<assistant>\nhello\n</assistant>" in prompt


def test_thread_to_messages_returns_openai_format_with_system_leading():
    t = ConversationThread(system_prompt="be helpful")
    t.append(Message(role="user", content="hi"))
    out = t.to_messages()
    assert out == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
    ]


def test_thread_to_messages_omits_system_when_empty():
    t = ConversationThread()
    t.append(Message(role="user", content="hi"))
    out = t.to_messages()
    assert out == [{"role": "user", "content": "hi"}]


def test_classify_frontier_model():
    assert classify_model("gemini/gemini-4-carbon") == ModelTier.FRONTIER


def test_classify_fast_model():
    assert classify_model("groq/llama-3.3-70b-versatile") == ModelTier.FAST


def test_classify_local_model():
    assert classify_model("ollama/gemma4:e4b") == ModelTier.LOCAL


def test_classify_unknown_model_defaults_to_fast():
    assert classify_model("madeup/never-heard-of-this-1") == ModelTier.FAST


def test_classify_with_custom_equivalence_map():
    custom = {
        ModelTier.FRONTIER: DEFAULT_EQUIVALENCE_MAP[ModelTier.FRONTIER],
        ModelTier.FAST: ModelEquivalenceClass(
            tier=ModelTier.FAST,
            members=["custom/cool-model"],
            context_window=64_000,
        ),
        ModelTier.FREE_TIER: DEFAULT_EQUIVALENCE_MAP[ModelTier.FREE_TIER],
        ModelTier.LOCAL: DEFAULT_EQUIVALENCE_MAP[ModelTier.LOCAL],
    }
    assert classify_model("custom/cool-model", custom) == ModelTier.FAST


def _mock_registry():
    reg = MagicMock()
    reg.get = MagicMock(return_value=None)
    reg.list_providers = MagicMock(return_value=[])
    reg.routing = MagicMock(fallback_chain=[])
    reg._configs = {}
    return reg


def _fake_chat_response(text: str) -> ChatResponse:
    return ChatResponse(
        text=text,
        model="fake/model",
        provider="fake",
        raw={"_headers": {}},
    )


def test_sentinel_init_carries_supplied_thread():
    thread = ConversationThread()
    thread.append(Message(role="user", content="preload"))
    s = RAILLMSentinel(registry=_mock_registry(), thread=thread)
    assert s.thread is thread
    assert len(s.thread.messages) == 1


def test_sentinel_init_creates_empty_thread_by_default():
    s = RAILLMSentinel(registry=_mock_registry())
    assert isinstance(s.thread, ConversationThread)
    assert s.thread.messages == []


def test_build_tier_chain_primary_address_first():
    s = RAILLMSentinel(registry=_mock_registry())
    chain = s._build_tier_chain(
        primary_address="groq/llama-3.3-70b-versatile",
        primary_tier=ModelTier.FAST,
        explicit_fallback=None,
    )
    assert chain[0] == "groq/llama-3.3-70b-versatile"


def test_build_tier_chain_primary_tier_peers_before_cascade():
    s = RAILLMSentinel(registry=_mock_registry())
    chain = s._build_tier_chain(
        primary_address="groq/llama-3.3-70b-versatile",
        primary_tier=ModelTier.FAST,
        explicit_fallback=None,
    )
    qwen_idx = chain.index("qwen/qwen3.5-flash")
    local_idx = chain.index("ollama/gemma4:e4b")
    assert qwen_idx < local_idx


def test_build_tier_chain_explicit_fallback_order_preserved():
    s = RAILLMSentinel(registry=_mock_registry())
    chain = s._build_tier_chain(
        primary_address="groq/llama-3.3-70b-versatile",
        primary_tier=ModelTier.FAST,
        explicit_fallback=["openai/gpt-4o"],
    )
    openai_idx = chain.index("openai/gpt-4o")
    gemini_idx = chain.index("gemini/gemini-4-carbon")
    assert openai_idx < gemini_idx


def test_build_tier_chain_no_duplicates():
    s = RAILLMSentinel(registry=_mock_registry())
    chain = s._build_tier_chain(
        primary_address="groq/llama-3.3-70b-versatile",
        primary_tier=ModelTier.FAST,
        explicit_fallback=[
            "groq/llama-3.3-70b-versatile",
            "gemini/gemini-2.5-flash",
        ],
    )
    assert chain.count("groq/llama-3.3-70b-versatile") == 1
    assert chain.count("gemini/gemini-2.5-flash") == 1


def test_dispatch_with_history_raises_when_no_providers_registered():
    reg = _mock_registry()
    reg.get.return_value = None
    s = RAILLMSentinel(registry=reg)
    with pytest.raises(RAILLMExhausted) as excinfo:
        s.dispatch_with_history(
            model_address="groq/llama-3.3-70b-versatile",
            user_message="hi",
        )
    assert "provider not registered" in str(excinfo.value)


def test_dispatch_with_history_appends_user_message_before_attempting():
    reg = _mock_registry()
    reg.get.return_value = None
    s = RAILLMSentinel(registry=reg)
    with pytest.raises(RAILLMExhausted):
        s.dispatch_with_history(
            model_address="groq/llama-3.3-70b-versatile",
            user_message="this should survive",
        )
    assert len(s.thread.messages) == 1
    assert s.thread.messages[0].role == "user"
    assert s.thread.messages[0].content == "this should survive"


def test_dispatch_with_history_success_appends_assistant_with_hints():
    reg = _mock_registry()
    provider = MagicMock()
    provider.chat.return_value = _fake_chat_response("hello back")
    reg.get.return_value = provider

    s = RAILLMSentinel(registry=reg)
    response = s.dispatch_with_history(
        model_address="groq/llama-3.3-70b-versatile",
        user_message="hi",
    )

    assert response.text == "hello back"
    assert len(s.thread.messages) == 2
    assert s.thread.messages[0].role == "user"
    assert s.thread.messages[1].role == "assistant"
    asst = s.thread.messages[1]
    assert asst.provider_hint == "groq"
    assert asst.model_hint == "llama-3.3-70b-versatile"


def test_dispatch_with_history_sends_compressed_prompt_when_over_budget():
    reg = _mock_registry()
    provider = MagicMock()
    provider.chat.return_value = _fake_chat_response("ok")
    reg.get.return_value = provider

    custom_map = {
        ModelTier.FRONTIER: DEFAULT_EQUIVALENCE_MAP[ModelTier.FRONTIER],
        ModelTier.FAST: ModelEquivalenceClass(
            tier=ModelTier.FAST,
            members=["groq/llama-3.3-70b-versatile"],
            context_window=50,
        ),
        ModelTier.FREE_TIER: DEFAULT_EQUIVALENCE_MAP[ModelTier.FREE_TIER],
        ModelTier.LOCAL: DEFAULT_EQUIVALENCE_MAP[ModelTier.LOCAL],
    }

    thread = ConversationThread()
    for _ in range(10):
        thread.append(Message(role="user", content="x" * 400))
        thread.append(Message(role="assistant", content="y" * 400))

    s = RAILLMSentinel(
        registry=reg,
        thread=thread,
        equivalence_map=custom_map,
    )
    s.dispatch_with_history(
        model_address="groq/llama-3.3-70b-versatile",
        user_message="new query",
    )

    assert provider.chat.call_count == 1
    call_kwargs = provider.chat.call_args.kwargs
    prompt_passed = call_kwargs["prompt"]
    assert "new query" in prompt_passed
    assert len(prompt_passed) < 10_000


# ---------------------------------------------------------------------------
# Network error failover — H_W1 proof (ConnectionError / TimeoutError / OSError)
# ---------------------------------------------------------------------------


def _make_two_provider_registry(bad_provider_name: str, bad_error, good_provider_name: str):
    """Build a registry where bad_provider raises bad_error and good_provider succeeds."""
    bad = MagicMock()
    bad.chat.side_effect = bad_error
    good = MagicMock()
    good.chat.return_value = _fake_chat_response("network fallback ok")

    reg = _mock_registry()
    reg.get.side_effect = lambda name: {
        bad_provider_name: bad, good_provider_name: good
    }.get(name)
    return reg, bad, good


def test_dispatch_connection_error_cascades_to_fallback():
    """ConnectionError from provider.chat() must trigger cascade, NOT escape."""
    reg, bad, good = _make_two_provider_registry(
        "groq", ConnectionError("network unreachable"), "openai"
    )
    sentinel = RAILLMSentinel(registry=reg)
    r = sentinel.dispatch_with_history(
        model_address="groq/llama-3.3-70b-versatile",
        user_message="hi",
        tier_fallback=["openai/gpt-4o"],
    )
    assert r.text == "network fallback ok"
    assert bad.chat.call_count >= 1
    assert good.chat.call_count >= 1


def test_dispatch_timeout_error_cascades_to_fallback():
    """TimeoutError from provider.chat() must trigger cascade."""
    reg, bad, good = _make_two_provider_registry(
        "groq", TimeoutError("request timed out"), "openai"
    )
    sentinel = RAILLMSentinel(registry=reg)
    r = sentinel.dispatch_with_history(
        model_address="groq/llama-3.3-70b-versatile",
        user_message="hi",
        tier_fallback=["openai/gpt-4o"],
    )
    assert r.text == "network fallback ok"


def test_dispatch_os_error_cascades_to_fallback():
    """OSError from provider.chat() must trigger cascade."""
    reg, bad, good = _make_two_provider_registry(
        "groq", OSError("socket error"), "openai"
    )
    sentinel = RAILLMSentinel(registry=reg)
    r = sentinel.dispatch_with_history(
        model_address="groq/llama-3.3-70b-versatile",
        user_message="hi",
        tier_fallback=["openai/gpt-4o"],
    )
    assert r.text == "network fallback ok"


def test_all_network_errors_raises_raillm_exhausted():
    """When every provider raises ConnectionError, RAILLMExhausted is raised."""
    bad = MagicMock()
    bad.chat.side_effect = ConnectionError("all dead")

    reg = _mock_registry()
    reg.get.return_value = bad

    # Constrained map so only one chain member exists
    from lib.hal.raillm import ModelEquivalenceClass, ModelTier, DEFAULT_EQUIVALENCE_MAP
    constrained = {
        t: ModelEquivalenceClass(tier=t, members=[], context_window=128_000)
        for t in ModelTier
    }
    constrained[ModelTier.FAST] = ModelEquivalenceClass(
        tier=ModelTier.FAST,
        members=["groq/llama-3.3-70b-versatile"],
        context_window=128_000,
    )
    sentinel = RAILLMSentinel(registry=reg, equivalence_map=constrained)
    with pytest.raises(RAILLMExhausted):
        sentinel.dispatch_with_history("groq/llama-3.3-70b-versatile", "hi")


def test_connection_error_type_name_in_exhausted_message():
    """ConnectionError type name must appear in RAILLMExhausted message for debugging."""
    bad = MagicMock()
    bad.chat.side_effect = ConnectionError("DNS resolution failed")

    reg = _mock_registry()
    reg.get.return_value = bad

    from lib.hal.raillm import ModelEquivalenceClass, ModelTier
    constrained = {
        t: ModelEquivalenceClass(tier=t, members=[], context_window=128_000)
        for t in ModelTier
    }
    constrained[ModelTier.FAST] = ModelEquivalenceClass(
        tier=ModelTier.FAST,
        members=["groq/llama-3.3-70b-versatile"],
        context_window=128_000,
    )
    sentinel = RAILLMSentinel(registry=reg, equivalence_map=constrained)
    with pytest.raises(RAILLMExhausted) as exc_info:
        sentinel.dispatch_with_history("groq/llama-3.3-70b-versatile", "hi")

    assert "ConnectionError" in str(exc_info.value)
    assert "groq" in str(exc_info.value)
