"""Integration test — Issue #201.

Verify that :meth:`Conversation.send` strips UI chrome from user input
before anything else runs: history, KONO recall, provider dispatch, and
semantic cache all receive the cleaned message.

Failure mode guarded: the original bug (the LLM being fed HAL's own
status footer) can only recur if chrome bypasses `strip_ui_chrome` OR
`strip_ui_chrome` fails to remove the catalogued patterns. The unit
tests in ``test_ui_chrome_strip.py`` cover the latter; this test covers
the former.
"""
from __future__ import annotations

import importlib.util
import pathlib
import unittest
from unittest.mock import MagicMock, patch

_REPO = pathlib.Path(__file__).parent.parent


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_conv = _load("lib/hal/conversation.py", "_wt_conversation_ui_chrome")
Conversation = _conv.Conversation


def _make_registry(model_addr: str = "groq/llama-3.3-70b-versatile") -> MagicMock:
    provider = MagicMock()
    provider.name = model_addr.split("/")[0]
    provider.models = [model_addr.split("/", 1)[1]]
    provider.type.value = "cloud"

    routing = MagicMock()
    routing.default_model = model_addr

    registry = MagicMock()
    registry.list_providers.return_value = [provider]
    registry.routing = routing
    return registry


def _make_dispatch_response(
    model: str = "groq/llama-3.3-70b-versatile",
    text: str = "HAL does not auto-switch providers invisibly.",
) -> MagicMock:
    response = MagicMock()
    response.text = text
    response.model = model
    response.provider = model.split("/")[0]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    response.latency_ms = 42.0
    response.cost_usd = 0.0001
    return response


class ConversationUiChromeTests(unittest.TestCase):
    """End-to-end: chrome in user input never reaches the pipeline."""

    @patch.object(_conv.LimitSentinel, "dispatch")
    def test_chrome_stripped_before_history_append(self, mock_dispatch):
        mock_dispatch.return_value = _make_dispatch_response()
        conv = Conversation(registry=_make_registry())

        pasted = (
            "  ● 17 providers · groq/llama-3.3-70b-versatile · $0.000945 · tools: off\n"
            "What is a Knowledge Engine?"
        )
        conv.send(pasted)

        # History should carry the cleaned text only.
        history = conv.history
        user_msg = next(m for m in history if m.role == "user")
        self.assertNotIn("providers", str(user_msg.content))
        self.assertNotIn("●", str(user_msg.content))
        self.assertIn("Knowledge Engine", str(user_msg.content))

    @patch.object(_conv.LimitSentinel, "dispatch")
    def test_chrome_stripped_before_dispatch(self, mock_dispatch):
        mock_dispatch.return_value = _make_dispatch_response()
        conv = Conversation(registry=_make_registry())

        pasted = (
            "❯  explain /tools on\n"
            "  ● 1 provider · groq · $0.0 · tools: on"
        )
        conv.send(pasted)

        # The messages list passed into dispatch must have the cleaned
        # content as the last user turn.
        self.assertTrue(mock_dispatch.called)
        call_kwargs = mock_dispatch.call_args.kwargs
        messages = call_kwargs.get("messages") or (
            mock_dispatch.call_args.args[0] if mock_dispatch.call_args.args else []
        )
        # Find the last user message content.
        user_contents = [m.content for m in messages if getattr(m, "role", None) == "user"]
        self.assertTrue(user_contents, "no user message was dispatched")
        last_user = str(user_contents[-1])
        self.assertIn("explain /tools on", last_user)
        self.assertNotIn("providers", last_user)
        self.assertNotIn("●", last_user)

    @patch.object(_conv.LimitSentinel, "dispatch")
    def test_system_prompt_contains_ui_chrome_note(self, mock_dispatch):
        mock_dispatch.return_value = _make_dispatch_response()
        conv = Conversation(registry=_make_registry())
        # Force a send so the system prompt is exercised end-to-end.
        conv.send("any message")
        self.assertIn("UI NOTE", conv._system)
        self.assertIn("Limit Sentinel", conv._system)

    def test_pure_chrome_paste_raises_value_error(self):
        """Edge case: user pastes nothing but chrome — now raises ValueError.

        After the empty-message guard (PR fix for CCH 400), a message that
        strips to empty string is rejected before history is mutated. History
        must stay clean and a ValueError must propagate to the caller so the
        REPL can display a helpful 'empty input' message instead of sending a
        blank turn to the Anthropic API.
        """
        conv = Conversation(registry=_make_registry())
        with self.assertRaises(ValueError):
            conv.send("  ● 5 providers · ollama/mistral · $0.0 · tools: off")
        # History must not have been mutated.
        self.assertEqual(conv.history, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
