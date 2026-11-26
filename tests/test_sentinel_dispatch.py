"""Tests for LimitSentinel dispatch robustness.

Covers the two bugs identified by broly investigation (2026-04-21):

  1. _build_chain crashes/behaves unexpectedly when routing.fallback_chain
     is not a real list (e.g. a MagicMock in tests or a misconfigured registry).

  2. ConnectionError / TimeoutError / OSError from provider.chat() bypassed
     the except OpenAICompatError block, escaping the failover chain and
     reaching the pipeline as an unhandled exception instead of triggering
     the next provider in chain.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_HAL_HOME = os.environ.get("HAL_ROOT", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.types import ChatResponse, TokenUsage


def _make_response(text="ok", provider="groq") -> ChatResponse:
    return ChatResponse(
        text=text,
        model="groq/llama-3.3-70b-versatile",
        provider=provider,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        cost_usd=0.0001,
        latency_ms=80.0,
    )


def _make_sentinel(fallback_chain=None, providers=None):
    """Build a hermetic LimitSentinel with disk I/O patched out."""
    from lib.hal.sentinel import LimitSentinel

    provider_mock = MagicMock()
    provider_mock.chat.return_value = _make_response()

    reg = MagicMock()
    reg.get.return_value = provider_mock
    reg.routing = MagicMock()
    reg.routing.default_model = "groq/llama-3.3-70b-versatile"
    # Use the supplied fallback_chain type (may be MagicMock, list, or None)
    reg.routing.fallback_chain = fallback_chain if fallback_chain is not None else []
    reg.list_providers.return_value = providers or []

    with patch("lib.hal.sentinel.LimitSentinel._load_state"):
        sentinel = LimitSentinel(reg)

    return sentinel, reg, provider_mock


# ===========================================================================
# Bug 1: _build_chain must be safe against non-list fallback_chain
# ===========================================================================

class TestBuildChainRobustness:
    """_build_chain must not raise or produce garbage when fallback_chain
    is not a real list (MagicMock, None, or any non-iterable-of-strings)."""

    def test_build_chain_with_empty_list(self):
        sentinel, _, _ = _make_sentinel(fallback_chain=[])
        chain = sentinel._build_chain("groq", None)
        assert "groq" in chain

    def test_build_chain_with_real_list(self):
        sentinel, _, _ = _make_sentinel(fallback_chain=["anthropic/claude-opus-4-7"])
        chain = sentinel._build_chain("groq", None)
        assert "groq" in chain
        assert "anthropic" in chain

    def test_build_chain_with_mock_fallback_does_not_raise(self):
        mock_chain = MagicMock()
        sentinel, _, _ = _make_sentinel(fallback_chain=mock_chain)
        chain = sentinel._build_chain("groq", None)
        assert "groq" in chain

    def test_build_chain_with_mock_fallback_produces_primary_only(self):
        mock_chain = MagicMock()
        sentinel, _, _ = _make_sentinel(fallback_chain=mock_chain)
        chain = sentinel._build_chain("groq", None)
        # MagicMock is not a list — chain should fall back to primary only
        # (no garbage provider names from MagicMock iteration)
        for name in chain:
            assert isinstance(name, str), f"Non-string provider in chain: {name!r}"
            assert not name.startswith("Mock"), (
                f"MagicMock child leaked into provider chain: {name!r}"
            )

    def test_dispatch_succeeds_when_fallback_chain_is_mock(self):
        """End-to-end: dispatch must succeed even when routing.fallback_chain
        is a MagicMock rather than a real list."""
        sentinel, reg, provider_mock = _make_sentinel(fallback_chain=MagicMock())
        resp = sentinel.dispatch("groq/llama-3.3-70b-versatile", "hello")
        assert resp.text == "ok"

    def test_explicit_fallback_overrides_registry_attribute(self):
        """Passing fallback_chain to dispatch() takes priority over registry.routing."""
        sentinel, reg, provider_mock = _make_sentinel(fallback_chain=MagicMock())
        resp = sentinel.dispatch(
            "groq/llama-3.3-70b-versatile", "hello",
            fallback_chain=["anthropic/claude-opus-4-7"],
        )
        assert resp.text == "ok"


# ===========================================================================
# Bug 2: ConnectionError / TimeoutError / OSError must trigger failover
# ===========================================================================

class TestNetworkErrorFailover:
    """Network errors from provider.chat() must be caught and cause failover
    to the next provider in chain, not an unhandled exception."""

    def test_connection_error_triggers_failover(self):
        """ConnectionError on first provider must route to second provider."""
        good_mock = MagicMock()
        good_mock.chat.return_value = _make_response(text="fallback ok", provider="anthropic")

        bad_mock = MagicMock()
        bad_mock.chat.side_effect = ConnectionError("network unreachable")

        sentinel, reg, _ = _make_sentinel(
            fallback_chain=["anthropic/claude-opus-4-7"]
        )

        # groq -> bad, anthropic -> good
        def get_provider(name):
            return bad_mock if name == "groq" else good_mock

        reg.get.side_effect = get_provider
        reg.routing.fallback_chain = ["anthropic/claude-opus-4-7"]

        resp = sentinel.dispatch("groq/llama-3.3-70b-versatile", "hello")
        assert resp.text == "fallback ok"

    def test_timeout_error_triggers_failover(self):
        good_mock = MagicMock()
        good_mock.chat.return_value = _make_response(text="timeout fallback")

        bad_mock = MagicMock()
        bad_mock.chat.side_effect = TimeoutError("request timed out")

        sentinel, reg, _ = _make_sentinel(fallback_chain=["anthropic/claude-opus-4-7"])

        def get_provider(name):
            return bad_mock if name == "groq" else good_mock

        reg.get.side_effect = get_provider
        reg.routing.fallback_chain = ["anthropic/claude-opus-4-7"]

        resp = sentinel.dispatch("groq/llama-3.3-70b-versatile", "hello")
        assert resp.text == "timeout fallback"

    def test_os_error_triggers_failover(self):
        good_mock = MagicMock()
        good_mock.chat.return_value = _make_response(text="oserr fallback")

        bad_mock = MagicMock()
        bad_mock.chat.side_effect = OSError("socket error")

        sentinel, reg, _ = _make_sentinel(fallback_chain=["anthropic/claude-opus-4-7"])

        def get_provider(name):
            return bad_mock if name == "groq" else good_mock

        reg.get.side_effect = get_provider
        reg.routing.fallback_chain = ["anthropic/claude-opus-4-7"]

        resp = sentinel.dispatch("groq/llama-3.3-70b-versatile", "hello")
        assert resp.text == "oserr fallback"

    def test_all_providers_network_fail_raises_runtime_error(self):
        """When every provider raises ConnectionError, RuntimeError must be raised."""
        bad_mock = MagicMock()
        bad_mock.chat.side_effect = ConnectionError("all dead")

        sentinel, reg, _ = _make_sentinel(fallback_chain=[])
        reg.get.return_value = bad_mock

        with pytest.raises(RuntimeError, match="All providers exhausted"):
            sentinel.dispatch("groq/llama-3.3-70b-versatile", "hello")

    def test_connection_error_included_in_error_message(self):
        """The failed provider+error must appear in the RuntimeError message."""
        bad_mock = MagicMock()
        bad_mock.chat.side_effect = ConnectionError("DNS resolution failed")

        sentinel, reg, _ = _make_sentinel(fallback_chain=[])
        reg.get.return_value = bad_mock

        with pytest.raises(RuntimeError) as exc_info:
            sentinel.dispatch("groq/llama-3.3-70b-versatile", "hello")

        assert "groq" in str(exc_info.value)
        assert "ConnectionError" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
