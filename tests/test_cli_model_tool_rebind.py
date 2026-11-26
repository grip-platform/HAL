"""Regression tests for /model switch with tools enabled.

Two bugs fixed:
  A. /model did not rebind tool_state["chat_fn"] — the old provider's URL
     (e.g. mlx_vlm -> localhost:8080) was used for every subsequent message
     even after switching to an incompatible model.
  B. Transport failures (ConnectionRefused / URLError) left tools "enabled"
     but permanently broken — every message failed until the session ended.

Mock-target note: _maybe_rebind_tools and _run_tool_turn use lazy imports
(inside the function body). patch() must target the SOURCE module, not
lib.hal.cli, otherwise the attribute is not found at patch time.

    detect_tool_chat_fn  → patch lib.hal.tool_cli_integration.detect_tool_chat_fn
    build_builtin_tool_stack → patch lib.hal.tool_cli_integration.build_builtin_tool_stack
    send_with_tools_to_cli   → patch lib.hal.tool_render.send_with_tools_to_cli
"""
from __future__ import annotations

import os
import sys
import urllib.error
from io import StringIO
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pytest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.cli import (
    _auto_disable_tools_transport,
    _is_transport_failure,
    _maybe_rebind_tools,
    _run_tool_turn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enabled_tool_state(chat_fn=None, auth="mlx_vlm") -> dict:
    from lib.hal.tool_circuit_breaker import CircuitBreaker
    return {
        "enabled": True,
        "chat_fn": chat_fn or MagicMock(return_value={}),
        "auth": auth,
        "breaker": CircuitBreaker(threshold=2),
    }


def _disabled_tool_state() -> dict:
    from lib.hal.tool_circuit_breaker import CircuitBreaker
    return {
        "enabled": False,
        "chat_fn": None,
        "auth": None,
        "breaker": CircuitBreaker(threshold=2),
    }


# ---------------------------------------------------------------------------
# _is_transport_failure
# ---------------------------------------------------------------------------

class TestIsTransportFailure:
    def test_connection_refused_is_transport(self):
        assert _is_transport_failure(ConnectionRefusedError("connection refused")) is True

    def test_url_error_wrapping_conn_refused_is_transport(self):
        inner = ConnectionRefusedError(61, "Connection refused")
        assert _is_transport_failure(urllib.error.URLError(inner)) is True

    def test_bare_url_error_is_transport(self):
        assert _is_transport_failure(urllib.error.URLError("unknown host")) is True

    def test_value_error_is_not_transport(self):
        assert _is_transport_failure(ValueError("bad json")) is False

    def test_runtime_error_is_not_transport(self):
        assert _is_transport_failure(RuntimeError("boom")) is False

    def test_os_error_with_reason_attr_is_transport(self):
        exc = Exception("wrapper")
        exc.reason = OSError("network unreachable")
        assert _is_transport_failure(exc) is True


# ---------------------------------------------------------------------------
# _maybe_rebind_tools
# ---------------------------------------------------------------------------

# Correct patch target: detect_tool_chat_fn is lazy-imported inside _maybe_rebind_tools
_DETECT = "lib.hal.tool_cli_integration.detect_tool_chat_fn"


class TestMaybeRebindTools:
    def test_noop_when_tools_disabled(self):
        ts = _disabled_tool_state()
        old_fn = ts["chat_fn"]
        with patch(_DETECT, side_effect=AssertionError("should not be called")):
            _maybe_rebind_tools("groq/llama", ts)
        assert ts["chat_fn"] is old_fn

    def test_noop_when_tool_state_is_none(self):
        _maybe_rebind_tools("groq/llama", None)  # must not raise

    def test_rebinds_chat_fn_on_model_switch(self):
        ts = _enabled_tool_state()
        old_fn = ts["chat_fn"]
        new_fn = MagicMock(return_value={})

        with patch(_DETECT, return_value=(new_fn, "groq")) as mock_detect:
            buf = StringIO()
            with redirect_stdout(buf):
                _maybe_rebind_tools("groq/llama-3.3", ts)

        mock_detect.assert_called_once_with("groq/llama-3.3")
        assert ts["chat_fn"] is new_fn, "chat_fn must be rebound to new provider"
        assert ts["chat_fn"] is not old_fn, "stale mlx_vlm fn must be replaced"
        assert ts["auth"] == "groq"
        assert ts["enabled"] is True

    def test_disables_tools_when_no_auth_for_new_model(self):
        ts = _enabled_tool_state()
        with patch(_DETECT, return_value=(None, None)):
            buf = StringIO()
            with redirect_stdout(buf):
                _maybe_rebind_tools("some/unprovisionable", ts)

        assert ts["enabled"] is False
        assert ts["chat_fn"] is None
        assert "disabled" in buf.getvalue()

    def test_prints_rebound_confirmation(self):
        ts = _enabled_tool_state()
        with patch(_DETECT, return_value=(MagicMock(), "gemini")):
            buf = StringIO()
            with redirect_stdout(buf):
                _maybe_rebind_tools("gemini/gemini-2.5-flash", ts)
        assert "rebound" in buf.getvalue()


# ---------------------------------------------------------------------------
# _auto_disable_tools_transport
# ---------------------------------------------------------------------------

class TestAutoDisableToolsTransport:
    def test_disables_tools(self):
        ts = _enabled_tool_state()
        exc = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        buf = StringIO()
        with redirect_stdout(buf):
            _auto_disable_tools_transport(ts, exc)
        assert ts["enabled"] is False
        assert ts["chat_fn"] is None
        assert ts["auth"] is None

    def test_prints_helpful_message(self):
        ts = _enabled_tool_state()
        exc = urllib.error.URLError("Connection refused")
        buf = StringIO()
        with redirect_stdout(buf):
            _auto_disable_tools_transport(ts, exc)
        output = buf.getvalue()
        assert "unreachable" in output.lower() or "auto-disabled" in output.lower()
        assert "/tools on" in output

    def test_does_not_raise(self):
        ts = _enabled_tool_state()
        buf = StringIO()
        with redirect_stdout(buf):
            _auto_disable_tools_transport(ts, urllib.error.URLError("whatever"))


# ---------------------------------------------------------------------------
# _run_tool_turn — transport-failure auto-disables tools (Bug B)
# ---------------------------------------------------------------------------

# Correct patch targets: all lazy-imported inside _run_tool_turn
_STACK = "lib.hal.tool_cli_integration.build_builtin_tool_stack"
_SEND  = "lib.hal.tool_render.send_with_tools_to_cli"


class TestRunToolTurnTransportFailure:
    def _make_conv(self):
        conv = MagicMock()
        conv.history = []
        conv._last_response_text = ""
        return conv

    def test_transport_failure_auto_disables_tools(self):
        """URLError from provider must disable tools — not loop forever."""
        ts = _enabled_tool_state()
        conv = self._make_conv()
        url_err = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))

        with patch(_STACK, return_value=([], MagicMock())), \
             patch(_SEND, side_effect=url_err):
            buf = StringIO()
            with redirect_stdout(buf):
                _run_tool_turn(conv, "hello", ts)

        assert ts["enabled"] is False, (
            "Transport failure must auto-disable tools so subsequent messages "
            "don't retry the dead endpoint."
        )
        assert ts["chat_fn"] is None
        assert "auto-disabled" in buf.getvalue() or "unreachable" in buf.getvalue()

    def test_transport_failure_does_not_reraise(self):
        """URLError must be caught inside _run_tool_turn, not propagate."""
        ts = _enabled_tool_state()
        conv = self._make_conv()
        url_err = urllib.error.URLError(ConnectionRefusedError(61, "refused"))

        with patch(_STACK, return_value=([], MagicMock())), \
             patch(_SEND, side_effect=url_err):
            buf = StringIO()
            with redirect_stdout(buf):
                _run_tool_turn(conv, "hello", ts)  # must not raise

    def test_other_exceptions_still_reraise(self):
        """Non-transport exceptions must still propagate to outer loop."""
        ts = _enabled_tool_state()
        conv = self._make_conv()

        with patch(_STACK, return_value=([], MagicMock())), \
             patch(_SEND, side_effect=TypeError("schema bug")):
            with pytest.raises(TypeError):
                _run_tool_turn(conv, "hello", ts)

        assert ts["enabled"] is True, "non-transport error must not disable tools"


# ---------------------------------------------------------------------------
# Integration: /model command rebinds tool_state
# ---------------------------------------------------------------------------

class TestHandleCommandModelRebind:
    def _make_deps(self):
        conv = MagicMock()
        conv.model = "mlx_vlm/auto"
        registry = MagicMock()
        council = MagicMock()
        return conv, registry, council

    def test_model_switch_rebinds_chat_fn_when_tools_enabled(self):
        """After /model, chat_fn must target the new provider, not the old localhost:8080."""
        from lib.hal.cli import _handle_command
        conv, registry, council = self._make_deps()
        ts = _enabled_tool_state(auth="mlx_vlm")
        old_fn = ts["chat_fn"]
        new_fn = MagicMock(return_value={})

        with patch(_DETECT, return_value=(new_fn, "groq")):
            buf = StringIO()
            with redirect_stdout(buf):
                _handle_command("/model groq/llama-3.3", conv, registry, council, ts)

        assert ts["chat_fn"] is new_fn, (
            "After /model groq/llama-3.3, chat_fn must target Groq, not mlx_vlm's localhost:8080"
        )
        assert ts["chat_fn"] is not old_fn
        assert ts["auth"] == "groq"

    def test_model_switch_noop_when_tools_disabled(self):
        """No tool re-probe when tools are already off."""
        from lib.hal.cli import _handle_command
        conv, registry, council = self._make_deps()
        ts = _disabled_tool_state()

        with patch(_DETECT, side_effect=AssertionError("must not be called")):
            buf = StringIO()
            with redirect_stdout(buf):
                _handle_command("/model groq/llama-3.3", conv, registry, council, ts)
        # No AssertionError means detect_tool_chat_fn was not called.
