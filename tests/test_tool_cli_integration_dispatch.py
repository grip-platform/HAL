"""Tests for provider-prefix dispatch in detect_tool_chat_fn (W3).

Verifies that:

- Bare ``claude-*`` / unprefixed model names route to CCH, or to HAL's
  API key when its route is open (those tests live in
  test_tool_cli_integration.py)
- ``ollama/<m>`` routes to OpenAI-compat at ``localhost:11434/v1`` with
  a literal bearer, stripping the prefix before the API call
- ``groq/<m>`` looks up ``grip-groq`` in Keychain and routes to the
  Groq OpenAI-compat endpoint
- ``gemini/<m>`` looks up ``grip-gemini`` and routes to Gemini's
  OpenAI-compat endpoint
- ``openai/<m>`` looks up ``hal-openai`` and routes to OpenAI
- Missing Keychain entries short-circuit to (None, None) without
  crashing the REPL
- Returned labels match the prefix (``ollama``, ``groq``, etc.) so the
  REPL can announce which path is live
- The Anthropic path is NOT probed when a non-Anthropic prefix is used
  (prevents spurious CCH calls when the operator explicitly picked
  another provider)
"""
import os
import subprocess
import sys
import unittest
from unittest import mock

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.tool_cli_integration import (  # noqa: E402
    _match_openai_compat_route,
    detect_tool_chat_fn,
)


class TestRouteMatching(unittest.TestCase):
    def test_ollama_prefix_matches(self):
        result = _match_openai_compat_route("ollama/gemma4:e4b")
        self.assertIsNotNone(result)
        stripped, base_url, _ = result
        self.assertEqual(stripped, "gemma4:e4b")
        self.assertEqual(base_url, "http://localhost:11434/v1")

    def test_groq_prefix_matches(self):
        result = _match_openai_compat_route("groq/llama-3.3-70b-versatile")
        self.assertIsNotNone(result)
        stripped, base_url, _ = result
        self.assertEqual(stripped, "llama-3.3-70b-versatile")
        self.assertEqual(base_url, "https://api.groq.com/openai/v1")

    def test_gemini_prefix_matches(self):
        result = _match_openai_compat_route("gemini/gemini-2.5-flash")
        self.assertIsNotNone(result)
        stripped, base_url, _ = result
        self.assertEqual(stripped, "gemini-2.5-flash")
        self.assertIn("generativelanguage.googleapis.com", base_url)

    def test_openai_prefix_matches(self):
        result = _match_openai_compat_route("openai/gpt-4o-mini")
        self.assertIsNotNone(result)
        stripped, base_url, _ = result
        self.assertEqual(stripped, "gpt-4o-mini")
        self.assertEqual(base_url, "https://api.openai.com/v1")

    def test_claude_prefix_does_not_match_openai_compat(self):
        self.assertIsNone(_match_openai_compat_route("claude-opus-4-7"))

    def test_unprefixed_does_not_match(self):
        self.assertIsNone(_match_openai_compat_route("some-model"))


class TestOllamaDispatch(unittest.TestCase):
    """Ollama uses a literal bearer — no Keychain lookup needed."""

    def test_ollama_returns_callable_with_label(self):
        chat_fn, label = detect_tool_chat_fn("ollama/gemma4:e4b")
        self.assertTrue(callable(chat_fn))
        self.assertEqual(label, "ollama")

    def test_ollama_dispatch_does_not_probe_anthropic(self):
        """An Ollama prefix must not trigger CCH or Anthropic keychain."""
        with mock.patch(
            "lib.hal.subscription.CCHSubscription.from_env",
        ) as cch_probe, mock.patch(
            "lib.hal.providers.anthropic_native._fetch_keychain_key",
        ) as kc_probe:
            detect_tool_chat_fn("ollama/gemma4:e4b")
            cch_probe.assert_not_called()
            kc_probe.assert_not_called()


class TestGroqDispatch(unittest.TestCase):
    def test_groq_keychain_hit_produces_chat_fn(self):
        with mock.patch(
            "lib.hal.providers.anthropic_native._fetch_keychain_key",
            return_value="gsk_test",
        ) as kc:
            chat_fn, label = detect_tool_chat_fn(
                "groq/llama-3.3-70b-versatile",
            )
        self.assertEqual(label, "groq")
        self.assertTrue(callable(chat_fn))
        kc.assert_called_with("grip-groq")

    def test_groq_keychain_miss_returns_none(self):
        with mock.patch(
            "lib.hal.providers.anthropic_native._fetch_keychain_key",
            return_value=None,
        ):
            chat_fn, label = detect_tool_chat_fn("groq/x")
        self.assertIsNone(chat_fn)
        self.assertIsNone(label)

    def test_groq_empty_keychain_value_returns_none(self):
        """Empty-string keychain value is treated as a miss."""
        with mock.patch(
            "lib.hal.providers.anthropic_native._fetch_keychain_key",
            return_value="",
        ):
            chat_fn, label = detect_tool_chat_fn("groq/x")
        self.assertIsNone(chat_fn)
        self.assertIsNone(label)


class TestGeminiDispatch(unittest.TestCase):
    def test_gemini_uses_grip_gemini_keychain(self):
        captured = {}

        def _capture(service):
            captured["service"] = service
            return "gemini_bearer"

        with mock.patch(
            "lib.hal.providers.anthropic_native._fetch_keychain_key",
            side_effect=_capture,
        ):
            chat_fn, label = detect_tool_chat_fn("gemini/gemini-2.5-flash")
        self.assertEqual(captured["service"], "grip-gemini")
        self.assertEqual(label, "gemini")
        self.assertTrue(callable(chat_fn))


class TestOpenAIDispatch(unittest.TestCase):
    def test_openai_uses_grip_openai_keychain(self):
        captured = {}

        def _capture(service):
            captured["service"] = service
            return "sk-openai"

        with mock.patch(
            "lib.hal.providers.anthropic_native._fetch_keychain_key",
            side_effect=_capture,
        ):
            chat_fn, label = detect_tool_chat_fn("openai/gpt-4o-mini")
        self.assertEqual(captured["service"], "hal-openai")
        self.assertEqual(label, "openai")
        self.assertTrue(callable(chat_fn))


class TestAnthropicPathUnchanged(unittest.TestCase):
    """Bare claude-* model strings must keep their pre-W3 routing intact."""

    def test_bare_claude_still_hits_cch(self):
        mock_cch = mock.MagicMock()
        mock_cch.health_check.return_value = True
        with mock.patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=mock_cch,
        ):
            chat_fn, label = detect_tool_chat_fn("claude-opus-4-7")
        self.assertTrue(callable(chat_fn))
        self.assertEqual(label, "cch")

    def test_bare_claude_falls_back_to_keychain(self):
        """With no CCH token, the key serves only through HAL's API-key route
        (lib/hal/anthropic_key_route.py); here it is open, and the policy from
        tests/conftest.py already opts the key in with a cap. The keychain stand-in
        holds ``hal-anthropic`` (patched on ``subprocess.run``, the object every
        copy of a module shares) and refuses to be read."""
        def holds_the_item(argv, *_args, **_kwargs):
            argv = list(argv)
            found = argv[-1] == "hal-anthropic" and "-w" not in argv
            return subprocess.CompletedProcess(argv, 0 if found else 44)

        with mock.patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=None,
        ), mock.patch.object(subprocess, "run", holds_the_item):
            chat_fn, label = detect_tool_chat_fn("claude-opus-4-7")
        self.assertTrue(callable(chat_fn))
        self.assertEqual(label, "api_key")


class TestStrippedModelReachesRuntime(unittest.TestCase):
    """The prefix must be stripped before being sent to the API."""

    def test_ollama_model_stripped_on_request(self):
        from lib.hal.tool_runtime import _normalize_openai_to_anthropic
        # Patch the transport seam used by make_openai_compat_chat_fn
        captured = {}

        def _capture(**kwargs):
            captured["body"] = kwargs.get("body")
            return {
                "choices": [{
                    "message": {"content": "ok"},
                    "finish_reason": "stop",
                }],
                "usage": {},
            }

        with mock.patch(
            "lib.hal.tool_runtime._post_openai_chat",
            side_effect=_capture,
        ):
            chat_fn, _ = detect_tool_chat_fn("ollama/gemma4:e4b")
            chat_fn(messages=[{"role": "user", "content": "hi"}], tools=[])
        self.assertEqual(captured["body"]["model"], "gemma4:e4b")
        # Normaliser still works when invoked directly — smoke check that
        # the chat_fn return path hasn't been broken by dispatch wiring.
        self.assertEqual(
            _normalize_openai_to_anthropic({
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {},
            })["stop_reason"],
            "end_turn",
        )


if __name__ == "__main__":
    unittest.main()
