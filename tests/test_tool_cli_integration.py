"""Tests for the CLI tool-loop auto-detection helper (W9).

Verifies:
- detect_tool_chat_fn picks CCH subscription first when available
- Falls back to HAL's API key only when its route is open
  (lib/hal/anthropic_key_route.py), and never reads the key when it is closed
- Returns (None, None) when neither is available
- The route asks the keychain for HAL's canonical "hal-anthropic" by name
- build_builtin_tool_stack returns all 8 built-in tools + a dispatcher
- CCH chat_fn wrapper converts messages and extracts raw response
"""
import os
import subprocess
import sys
import unittest
from unittest import mock

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.tool_cli_integration import (
    _wrap_cch_subscription,
    build_builtin_tool_stack,
    detect_tool_chat_fn,
)


def _route_open(calls=None):
    """Open HAL's API-key route for detection, with a ``security`` stand-in that
    holds ``hal-anthropic`` in place of tests/conftest.py's empty keychain. The
    permissive policy there already opts the key in with a cap, so this is the
    one condition left. Patched on ``subprocess.run``, the object every copy of a
    module shares. Reading the key (``-w``) fails the test: detection must not."""
    def run(argv, *_args, **_kwargs):
        argv = list(argv)
        if calls is not None:
            calls.append(argv)
        if argv[:1] != ["security"] or "-w" in argv:
            raise AssertionError(f"detection ran {argv[:2]} {'and read the key' if '-w' in argv else ''}")
        return subprocess.CompletedProcess(argv, 0 if argv[-1] == "hal-anthropic" else 44)

    return mock.patch.object(subprocess, "run", run)


class TestDetectNoAuth(unittest.TestCase):
    def test_returns_none_when_no_subscription_and_no_keychain(self):
        with mock.patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=None,
        ), mock.patch(
            "lib.hal.providers.anthropic_native._fetch_keychain_key",
            return_value=None,
        ):
            chat_fn, label = detect_tool_chat_fn("claude-opus-4-7")
        self.assertIsNone(chat_fn)
        self.assertIsNone(label)


class TestDetectCCH(unittest.TestCase):
    def test_cch_path_preferred_over_keychain(self):
        mock_cch = mock.MagicMock()
        mock_cch.health_check.return_value = True
        with mock.patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=mock_cch,
        ):
            chat_fn, label = detect_tool_chat_fn("claude-opus-4-7")
        self.assertIsNotNone(chat_fn)
        self.assertEqual(label, "cch")

    def test_unhealthy_cch_falls_through(self):
        """Unhealthy subscription should NOT short-circuit the probe: with HAL's
        API-key route open, the key serves instead."""
        mock_cch = mock.MagicMock()
        mock_cch.health_check.return_value = False
        with mock.patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=mock_cch,
        ), _route_open():
            chat_fn, label = detect_tool_chat_fn("claude-opus-4-7")
        self.assertIsNotNone(chat_fn)
        self.assertEqual(label, "api_key")

    def test_cch_exception_does_not_prevent_fallback(self):
        """Import errors (missing xxhash) must not crash the probe."""
        with mock.patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            side_effect=ImportError("xxhash missing"),
        ), _route_open():
            chat_fn, label = detect_tool_chat_fn("claude-opus-4-7")
        self.assertEqual(label, "api_key")


class TestDetectKeychain(unittest.TestCase):
    def test_keychain_service_is_grip_anthropic(self):
        """The canonical HAL service name, shared with AnthropicProvider.api_key
        (``anthropic_native.API_KEY_SERVICE``). Detection asks the keychain for
        it by name only: ``security`` never runs with ``-w``, so no key is read."""
        calls = []
        with mock.patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=None,
        ), _route_open(calls), mock.patch(
            "lib.hal.providers.anthropic_native._fetch_keychain_key",
        ) as read_key:
            _chat_fn, label = detect_tool_chat_fn("claude-opus-4-7")
        self.assertEqual(calls, [["security", "find-generic-password", "-s", "hal-anthropic"]])
        read_key.assert_not_called()
        self.assertEqual(label, "api_key")

    def test_closed_route_returns_none_and_never_reads_the_key(self):
        """Inverted from the old keychain fallback, which read hal-anthropic and
        billed it with no opt-in and no cap. With the route closed (the default in
        tests/conftest.py) there is no Anthropic transport, and the key is not read."""
        with mock.patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=None,
        ), mock.patch(
            "lib.hal.providers.anthropic_native._fetch_keychain_key",
            return_value="sk-ant-test",
        ) as read_key:
            chat_fn, label = detect_tool_chat_fn("claude-opus-4-7")
        self.assertIsNone(chat_fn)
        self.assertIsNone(label)
        read_key.assert_not_called()

    def test_empty_keychain_returns_none(self):
        with mock.patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=None,
        ), mock.patch(
            "lib.hal.providers.anthropic_native._fetch_keychain_key",
            return_value="",
        ):
            chat_fn, label = detect_tool_chat_fn("claude-opus-4-7")
        self.assertIsNone(chat_fn)
        self.assertIsNone(label)

    def test_keychain_returns_a_callable_chat_fn(self):
        """With no CCH token and HAL's API-key route open, the key serves through
        the route's transport, as a callable chat_fn."""
        with mock.patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=None,
        ), _route_open():
            chat_fn, label = detect_tool_chat_fn("claude-opus-4-7")
        self.assertEqual(label, "api_key")
        self.assertTrue(callable(chat_fn))


class TestBuildBuiltinToolStack(unittest.TestCase):
    def test_returns_all_tools_and_a_dispatcher(self):
        tools, dispatcher = build_builtin_tool_stack()
        # 8 core Claude-style tools + osa_run (added PR #226).
        self.assertEqual(len(tools), 9)
        self.assertTrue(hasattr(dispatcher, "execute"))

    def test_every_tool_has_anthropic_schema_fields(self):
        tools, _ = build_builtin_tool_stack()
        for tool in tools:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("input_schema", tool)


class TestCCHWrapper(unittest.TestCase):
    def test_wrapper_extracts_raw_from_chat_response(self):
        from lib.hal.types import ChatResponse

        raw_dict = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
        mock_sub = mock.MagicMock()
        mock_sub.chat.return_value = ChatResponse(
            text="ok", model="m", provider="cch", raw=raw_dict,
        )

        chat_fn = _wrap_cch_subscription(mock_sub, model="m", max_tokens=1024)
        result = chat_fn(messages=[{"role": "user", "content": "hi"}], tools=[])
        self.assertEqual(result, raw_dict)

    def test_wrapper_passes_tools_to_subscription(self):
        from lib.hal.types import ChatResponse

        mock_sub = mock.MagicMock()
        mock_sub.chat.return_value = ChatResponse(text="", model="m", provider="cch", raw={})
        chat_fn = _wrap_cch_subscription(mock_sub, model="m", max_tokens=512)
        tools = [{"name": "Read", "description": "", "input_schema": {}}]
        chat_fn(messages=[{"role": "user", "content": "hi"}], tools=tools)
        call_kwargs = mock_sub.chat.call_args.kwargs
        self.assertEqual(call_kwargs["tools"], tools)

    def test_wrapper_passes_none_when_tools_empty(self):
        from lib.hal.types import ChatResponse

        mock_sub = mock.MagicMock()
        mock_sub.chat.return_value = ChatResponse(text="", model="m", provider="cch", raw={})
        chat_fn = _wrap_cch_subscription(mock_sub, model="m", max_tokens=512)
        chat_fn(messages=[{"role": "user", "content": "hi"}], tools=[])
        self.assertIsNone(mock_sub.chat.call_args.kwargs["tools"])


class TestBareProviderRouting(unittest.TestCase):
    """Regression: bare provider names (mlx_vlm, vllm etc. with models:auto)
    must NOT fall through to _dispatch_anthropic and cause CCH HTTP 404."""

    def setUp(self):
        from lib.hal.tool_cli_integration import _match_openai_compat_route
        self._match = _match_openai_compat_route

    def test_mlx_vlm_bare_routes_to_local(self):
        result = self._match("mlx_vlm")
        self.assertIsNotNone(result, "bare 'mlx_vlm' must not fall through to Anthropic")
        stripped, base_url, _ = result
        self.assertEqual(stripped, "")
        self.assertIn("8080", base_url)

    def test_vllm_bare_routes_to_local(self):
        result = self._match("vllm")
        self.assertIsNotNone(result)
        stripped, base_url, _ = result
        self.assertEqual(stripped, "")
        self.assertIn("8000", base_url)

    def test_lmstudio_bare_routes_to_local(self):
        result = self._match("lmstudio")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "")

    def test_lemonade_bare_routes_to_local(self):
        result = self._match("lemonade")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "")

    def test_ollama_bare_routes_to_local(self):
        result = self._match("ollama")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "")

    def test_fully_qualified_still_works(self):
        result = self._match("mlx_vlm/gemma4:27b")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "gemma4:27b")

    def test_unknown_model_returns_none(self):
        self.assertIsNone(self._match("anthropic/claude-3"))

    def test_bare_anthropic_claude_returns_none(self):
        self.assertIsNone(self._match("claude-opus-4-7"))


class TestLocalProviderProbe(unittest.TestCase):
    """get_local_provider_info and probe_local_server correctness."""

    def setUp(self):
        from lib.hal.tool_cli_integration import get_local_provider_info, probe_local_server
        self._info = get_local_provider_info
        self._probe = probe_local_server

    def test_mlx_vlm_is_local(self):
        result = self._info("mlx_vlm")
        self.assertIsNotNone(result)
        name, base_url, hint = result
        self.assertEqual(name, "mlx_vlm")
        self.assertIn("8080", base_url)
        self.assertIn("mlx_vlm", hint)

    def test_ollama_is_local(self):
        name, _, hint = self._info("ollama")
        self.assertEqual(name, "ollama")
        self.assertIn("ollama serve", hint)

    def test_ollama_prefixed_is_local(self):
        result = self._info("ollama/gemma4:26b")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "ollama")

    def test_cloud_provider_not_local(self):
        self.assertIsNone(self._info("groq/llama-3.3-70b-versatile"))
        self.assertIsNone(self._info("deepseek/deepseek-v4-flash"))
        self.assertIsNone(self._info("claude-opus-4-7"))

    def test_probe_closed_port_returns_false(self):
        self.assertFalse(self._probe("http://127.0.0.1:19999", timeout=0.2))

    def test_probe_unknown_host_returns_false(self):
        self.assertFalse(self._probe("http://no-such-host-hal-test.local:8080", timeout=0.2))


class TestAutoStartLocalServer(unittest.TestCase):
    """auto_start_local_server: fires Popen, polls, returns correctly."""

    def setUp(self):
        from lib.hal.tool_cli_integration import auto_start_local_server
        self._start = auto_start_local_server

    def test_no_auto_start_command_returns_false(self):
        # vllm has no auto-start command (needs model arg)
        result = self._start("vllm", "http://localhost:8000", timeout=0.1)
        self.assertFalse(result)

    def test_bad_command_returns_false(self):
        with mock.patch(
            "lib.hal.tool_cli_integration._LOCAL_AUTO_START",
            {"fake": ["no-such-binary-xyz"]},
        ):
            result = self._start("fake", "http://127.0.0.1:19999", timeout=0.1)
        self.assertFalse(result)

    def test_server_comes_up_returns_true(self):
        mock_popen = mock.MagicMock()
        probe_calls = [False, True]  # fails once, then succeeds
        with mock.patch("subprocess.Popen", return_value=mock_popen) as popen_mock, \
             mock.patch(
                 "lib.hal.tool_cli_integration.probe_local_server",
                 side_effect=probe_calls,
             ), \
             mock.patch(
                 "lib.hal.tool_cli_integration._LOCAL_AUTO_START",
                 {"ollama": ["ollama", "serve"]},
             ):
            result = self._start("ollama", "http://localhost:11434/v1", timeout=5.0)
        self.assertTrue(result)
        popen_mock.assert_called_once()

    def test_timeout_returns_false(self):
        mock_popen = mock.MagicMock()
        with mock.patch("subprocess.Popen", return_value=mock_popen), \
             mock.patch(
                 "lib.hal.tool_cli_integration.probe_local_server",
                 return_value=False,
             ), \
             mock.patch(
                 "lib.hal.tool_cli_integration._LOCAL_AUTO_START",
                 {"ollama": ["ollama", "serve"]},
             ), \
             mock.patch("time.sleep"):
            result = self._start("ollama", "http://localhost:11434/v1", timeout=0.01)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
