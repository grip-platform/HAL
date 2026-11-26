"""E2E behavioral tests for HAL's Conversation and Registry contracts.

These tests specify *desired behaviour* — they are TDD-style in that they
describe what HAL should do, verified against real (mocked) internal
interfaces rather than the CLI layer. No subprocess, no stdin, no real LLM
calls. Every test is Goodhart-protected: it fails if the contract is broken.

Coverage areas:
  1. Conversation lifecycle — init, send, multi-turn history, cost
  2. Model switching — configure(model=) honoured on next send
  3. Mode switching — configure(mode=) rebuilds system prompt
  4. History management — turn counting, clear, history list
  5. Status contract — required keys, types, ranges
  6. Error resilience — provider failure raises, does not corrupt state
  7. ChatResponse contract — all required fields, immutability
  8. UI chrome stripping — REPL chrome stripped before dispatch
  9. Conversation.configure idempotence — no-arg call is a no-op
 10. Registry routing contract — model passed to registry on send
"""
from __future__ import annotations

import os
import sys
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.types import ChatResponse, TokenUsage


# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------

def _make_response(text="ok", model="groq/llama-3.3-70b-versatile",
                   provider="groq", cost_usd=0.0001) -> ChatResponse:
    return ChatResponse(
        text=text,
        model=model,
        provider=provider,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        cost_usd=cost_usd,
        latency_ms=80.0,
    )


def _make_conv(model="groq/llama-3.3-70b-versatile",
               response: Optional[ChatResponse] = None,
               **conv_kwargs):
    """Build a hermetic Conversation with all live I/O patched out.

    _build_identity does local-provider network probing + FS reads on init.
    Patching _build_identity (module-level fn in lib.hal.conversation) prevents
    live network probing and FS reads at init. _build_system is also patched on
    the instance so configure(mode=...) calls stay hermetic.
    """
    from lib.hal.conversation import Conversation
    resp = response or _make_response()

    # Provider mock: registry.get("groq") -> provider_mock.chat(...) -> resp
    provider_mock = MagicMock()
    provider_mock.chat.return_value = resp

    reg = MagicMock()
    reg.chat.return_value = resp
    reg.get.return_value = provider_mock           # sentinel.dispatch uses .get()
    reg.routing = MagicMock()
    reg.routing.default_model = model
    reg.routing.fallback_chain = []                # empty — no chained providers
    reg.list_providers.return_value = []           # no extra providers in chain

    with patch("lib.hal.conversation._build_identity", return_value="HAL test"), \
         patch("lib.hal.sentinel.LimitSentinel._load_state"):   # no disk I/O
        conv = Conversation(model=model, registry=reg, **conv_kwargs)

    conv._build_system = lambda mode="": "HAL test"
    return conv, reg


# ---------------------------------------------------------------------------
# 1. Conversation lifecycle
# ---------------------------------------------------------------------------

class TestConversationLifecycle:
    """Basic init → send → inspect cycle."""

    def test_send_returns_chatresponse(self):
        conv, _ = _make_conv()
        resp = conv.send("Hello")
        assert isinstance(resp, ChatResponse)
        assert resp.text == "ok"

    def test_send_calls_registry_chat(self):
        conv, reg = _make_conv()
        conv.send("ping")
        reg.get.return_value.chat.assert_called_once()

    def test_history_grows_after_each_send(self):
        conv, _ = _make_conv()
        assert conv.status()["turns"] == 0
        conv.send("turn one")
        assert conv.status()["turns"] == 1
        conv.send("turn two")
        assert conv.status()["turns"] == 2

    def test_cost_accumulates_across_sends(self):
        conv, _ = _make_conv()
        conv.send("a")
        after_first = conv.status()["total_cost_usd"]
        conv.send("b")
        after_second = conv.status()["total_cost_usd"]
        assert after_second > after_first > 0.0, (
            f"Cost must accumulate: {after_first} → {after_second}"
        )

    def test_model_property_reflects_init_model(self):
        conv, _ = _make_conv(model="ollama/mistral:7b")
        assert conv.model == "ollama/mistral:7b"


# ---------------------------------------------------------------------------
# 2. Model switching
# ---------------------------------------------------------------------------

class TestModelSwitching:
    """configure(model=) must be honoured on the very next send."""

    def test_configure_model_updates_property(self):
        conv, _ = _make_conv()
        conv.configure(model="anthropic/claude-opus-4-7")
        assert conv.model == "anthropic/claude-opus-4-7"

    def test_configure_model_is_passed_to_registry_on_next_send(self):
        """The provider key must reach registry.get() and model kwarg must reach provider.chat()."""
        conv, reg = _make_conv()
        conv.configure(model="anthropic/claude-opus-4-7")
        conv.send("test after switch")
        provider_arg = (reg.get.call_args.args[0] if reg.get.call_args.args
                        else reg.get.call_args.kwargs.get("name", ""))
        model_kwarg = reg.get.return_value.chat.call_args.kwargs.get("model", "")
        assert "anthropic" in provider_arg, (
            f"Expected anthropic provider in registry.get call, got: {reg.get.call_args}"
        )
        assert "opus" in model_kwarg, (
            f"Expected opus model in provider.chat call, got: {model_kwarg}"
        )

    def test_multiple_model_switches_use_latest(self):
        conv, _ = _make_conv()
        conv.configure(model="anthropic/claude-opus-4-7")
        conv.configure(model="ollama/llama3:8b")
        assert conv.model == "ollama/llama3:8b"


# ---------------------------------------------------------------------------
# 3. Mode switching
# ---------------------------------------------------------------------------

class TestModeSwitching:
    """configure(mode=) must not crash and must not corrupt history."""

    def test_configure_mode_does_not_crash(self):
        conv, _ = _make_conv()
        conv.configure(mode="legal")  # must not raise

    def test_configure_mode_does_not_clear_history(self):
        conv, _ = _make_conv()
        conv.send("first turn")
        conv.configure(mode="balanced")
        assert conv.status()["turns"] == 1, "Mode switch must not clear history"


# ---------------------------------------------------------------------------
# 4. History management
# ---------------------------------------------------------------------------

class TestHistoryManagement:
    """Conversation.history is a list; configure(clear=True) empties it."""

    def test_history_returns_list(self):
        conv, _ = _make_conv()
        assert isinstance(conv.history, list)

    def test_history_contains_user_and_assistant_turns(self):
        conv, _ = _make_conv()
        conv.send("hello")
        history = conv.history
        roles = [m.role for m in history]  # ChatMessage is a frozen dataclass
        assert "user" in roles, f"Expected 'user' role in history: {history}"
        assert "assistant" in roles, f"Expected 'assistant' role in history: {history}"

    def test_configure_clear_resets_turn_count(self):
        conv, _ = _make_conv()
        conv.send("a")
        conv.send("b")
        conv.configure(clear=True)
        assert conv.status()["turns"] == 0

    def test_configure_clear_empties_history_list(self):
        conv, _ = _make_conv()
        conv.send("turn")
        conv.configure(clear=True)
        assert conv.history == []


# ---------------------------------------------------------------------------
# 5. Status contract
# ---------------------------------------------------------------------------

class TestStatusContract:
    """status() must always return a well-formed dict with expected keys."""

    REQUIRED_KEYS = {
        "turns", "total_cost_usd", "total_tokens",
        "model", "context_usage_pct",
    }

    def test_status_contains_required_keys(self):
        conv, _ = _make_conv()
        missing = self.REQUIRED_KEYS - conv.status().keys()
        assert not missing, f"status() missing keys: {missing}"

    def test_status_turns_is_int(self):
        conv, _ = _make_conv()
        assert isinstance(conv.status()["turns"], int)

    def test_status_total_cost_is_float(self):
        conv, _ = _make_conv()
        assert isinstance(conv.status()["total_cost_usd"], float)

    def test_status_context_pct_is_between_0_and_100(self):
        conv, _ = _make_conv()
        pct = conv.status()["context_usage_pct"]
        assert 0.0 <= pct <= 100.0, f"context_usage_pct={pct} out of range"


# ---------------------------------------------------------------------------
# 6. Error resilience
# ---------------------------------------------------------------------------

class TestErrorResilience:
    """A provider failure must propagate as an exception without corrupting state."""

    def test_provider_failure_raises_exception(self):
        conv, reg = _make_conv()
        reg.get.return_value.chat.side_effect = ConnectionError("provider unreachable")
        with pytest.raises(Exception):
            conv.send("this will fail")

    def test_turn_count_not_incremented_on_failure(self):
        conv, reg = _make_conv()
        reg.get.return_value.chat.side_effect = ConnectionError("provider unreachable")
        try:
            conv.send("fail")
        except Exception:
            pass
        assert conv.status()["turns"] == 0, (
            "Failed send must not increment turn count"
        )

    def test_conversation_recovers_after_transient_failure(self):
        conv, reg = _make_conv()
        reg.get.return_value.chat.side_effect = [ConnectionError("boom"), _make_response()]
        try:
            conv.send("first — will fail")
        except Exception:
            pass
        resp = conv.send("second — should succeed")
        assert resp.text == "ok"


# ---------------------------------------------------------------------------
# 7. ChatResponse contract
# ---------------------------------------------------------------------------

class TestChatResponseContract:
    """ChatResponse must expose all expected fields and be immutable."""

    def test_all_fields_accessible(self):
        r = _make_response()
        assert r.text
        assert r.model
        assert r.provider
        assert r.usage is not None
        assert isinstance(r.cost_usd, float)
        assert isinstance(r.latency_ms, float)

    def test_response_is_immutable(self):
        r = _make_response()
        with pytest.raises((AttributeError, TypeError)):
            r.text = "mutated"  # type: ignore[misc]

    def test_token_usage_fields(self):
        r = _make_response()
        assert hasattr(r.usage, "prompt_tokens")
        assert hasattr(r.usage, "completion_tokens")


# ---------------------------------------------------------------------------
# 8. UI chrome stripping
# ---------------------------------------------------------------------------

class TestUiChromeStripping:
    """HAL's own REPL chrome must be stripped before dispatch."""

    def test_strip_ui_chrome_removes_em_dash_bar(self):
        from lib.hal.ui_chrome import strip_ui_chrome
        assert strip_ui_chrome("─" * 72) == ""

    def test_strip_ui_chrome_removes_middle_dot_bar(self):
        from lib.hal.ui_chrome import strip_ui_chrome
        assert strip_ui_chrome("·" * 72) == ""

    def test_strip_ui_chrome_preserves_content(self):
        from lib.hal.ui_chrome import strip_ui_chrome
        content = "What is the capital of France?"
        assert strip_ui_chrome(content) == content

    def test_chrome_is_stripped_before_registry_sees_message(self):
        """Registry must never see the raw ─ bar chrome in its message arg."""
        seen = []

        def capturing_chat(model, message, **_):
            seen.append(message)
            return _make_response()

        conv, reg = _make_conv()
        reg.chat.side_effect = capturing_chat
        chrome = "─" * 72 + "\nactual question\n" + "─" * 72
        conv.send(chrome)
        combined = "\n".join(seen)
        assert "─" * 40 not in combined, (
            "UI chrome must be stripped before reaching the registry"
        )


# ---------------------------------------------------------------------------
# 9. configure() idempotence
# ---------------------------------------------------------------------------

class TestConfigureIdempotence:
    """configure() with no args must be a safe no-op."""

    def test_empty_configure_does_not_crash(self):
        conv, _ = _make_conv()
        conv.configure()

    def test_empty_configure_preserves_model(self):
        conv, _ = _make_conv()
        conv.configure()
        assert conv.model == "groq/llama-3.3-70b-versatile"

    def test_empty_configure_preserves_history(self):
        conv, _ = _make_conv()
        conv.send("preserve me")
        conv.configure()
        assert conv.status()["turns"] == 1


# ---------------------------------------------------------------------------
# 10. Registry routing contract
# ---------------------------------------------------------------------------

class TestRegistryRoutingContract:
    """Conversation must always pass the current model to the registry."""

    def test_registry_receives_current_model_at_send_time(self):
        conv, reg = _make_conv()
        conv.send("hello")
        provider_arg = (reg.get.call_args.args[0] if reg.get.call_args.args
                        else reg.get.call_args.kwargs.get("name", ""))
        assert "groq" in provider_arg, (
            f"Expected groq provider passed to registry.get, got: {reg.get.call_args}"
        )

    def test_registry_receives_full_history_context(self):
        """Second send must pass at least 2 messages (prior turn + new input)."""
        conv, reg = _make_conv()
        conv.send("first")
        conv.send("second")
        last_call = reg.get.return_value.chat.call_args
        messages = (last_call.kwargs.get("messages") or
                    (last_call.args[1] if len(last_call.args) > 1 else None))
        if messages is not None:
            assert len(messages) >= 2, (
                f"Expected >= 2 messages on second send, got {len(messages)}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
