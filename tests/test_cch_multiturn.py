"""Tests for multi-turn body construction in lib/hal/utils/cch.py.

Hypothesis H-HAL-W4B-1: ``CCHSigner.build_body`` accepts an optional
``messages: list[ChatMessage]`` and ``system: str`` so ``CCHSubscription.chat``
can send structured multi-turn conversation history (with assistant
thinking-block signatures preserved verbatim) instead of flattening to
a single prompt string. The CCH fingerprint continues to derive from
the FIRST user message so turn N carries the same ``cc_version`` suffix
as turn 1 (W14 preserved).

Covers:
- build_body with messages=None falls back to single-turn (regression)
- build_body with messages=[...] and prompt="" builds multi-turn array
- First user message drives fingerprint (multi-turn path)
- Fingerprint stays stable across turns for the same conversation
- fingerprint_text kwarg override
- System blocks: billing header always first; user system appended second
- Assistant thinking-block list content round-trips verbatim (signature
  continuity for Haiku 4.5 extended thinking)
- Multimodal content-part lists serialise cleanly
- extract_first_user_text helper: empty, string content, list content,
  multi-message, no-user-msg, non-text first part
- _message_text + _first_text_part helpers
- CCHSubscription.chat passes messages/system through to signer (via
  the public API contract — no _flatten_messages intermediate)
- Regression: _flatten_messages no longer exists in subscription.py
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.types import ChatMessage
from lib.hal.utils.cch import (
    CCHSigner,
    extract_first_user_text,
    _first_text_part,
    _message_text,
    _serialise_chat_message,
)


class TestExtractFirstUserText(unittest.TestCase):

    def test_empty_messages(self):
        self.assertEqual(extract_first_user_text([]), "")

    def test_none_messages(self):
        self.assertEqual(extract_first_user_text(None), "")

    def test_single_user_string_content(self):
        msgs = [ChatMessage(role="user", content="hello there")]
        self.assertEqual(extract_first_user_text(msgs), "hello there")

    def test_first_user_ignores_assistant_prefix(self):
        msgs = [
            ChatMessage(role="assistant", content="unused"),
            ChatMessage(role="user", content="the real first user"),
        ]
        self.assertEqual(
            extract_first_user_text(msgs), "the real first user",
        )

    def test_first_user_ignores_later_users(self):
        msgs = [
            ChatMessage(role="user", content="turn 1"),
            ChatMessage(role="assistant", content="reply"),
            ChatMessage(role="user", content="turn 2"),
        ]
        self.assertEqual(extract_first_user_text(msgs), "turn 1")

    def test_list_content_first_text_part(self):
        msgs = [ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "describe this image"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
        )]
        self.assertEqual(
            extract_first_user_text(msgs), "describe this image",
        )

    def test_list_content_no_text_part(self):
        msgs = [ChatMessage(
            role="user",
            content=[{"type": "image_url", "image_url": {"url": "x"}}],
        )]
        self.assertEqual(extract_first_user_text(msgs), "")

    def test_no_user_in_history(self):
        msgs = [
            ChatMessage(role="assistant", content="a"),
            ChatMessage(role="system", content="b"),
        ]
        self.assertEqual(extract_first_user_text(msgs), "")


class TestHelpers(unittest.TestCase):

    def test_first_text_part_handles_non_list(self):
        self.assertEqual(_first_text_part("not a list"), "")
        self.assertEqual(_first_text_part(None), "")

    def test_first_text_part_skips_non_text_blocks(self):
        content = [
            {"type": "thinking", "thinking": "..."},
            {"type": "text", "text": "the answer"},
        ]
        self.assertEqual(_first_text_part(content), "the answer")

    def test_message_text_string(self):
        msg = ChatMessage(role="user", content="hello")
        self.assertEqual(_message_text(msg), "hello")

    def test_message_text_list(self):
        msg = ChatMessage(
            role="user",
            content=[{"type": "text", "text": "hi"}],
        )
        self.assertEqual(_message_text(msg), "hi")

    def test_serialise_preserves_content_verbatim(self):
        content = [
            {"type": "thinking", "thinking": "reasoning...",
             "signature": "sig123"},
            {"type": "text", "text": "the answer is 42"},
        ]
        msg = ChatMessage(role="assistant", content=content)
        serialised = _serialise_chat_message(msg)
        self.assertEqual(serialised["role"], "assistant")
        self.assertEqual(serialised["content"], content)
        # Signature must survive serialisation round-trip.
        self.assertEqual(serialised["content"][0]["signature"], "sig123")


class TestBuildBodySingleTurnRegression(unittest.TestCase):
    """Regression: existing single-turn callers keep working unchanged."""

    def setUp(self):
        self.signer = CCHSigner()

    def test_positional_prompt_still_works(self):
        body_str, _ = self.signer.build_body(
            "hello", model="claude-opus-4-7", session_id="s1",
        )
        payload = json.loads(body_str)
        self.assertEqual(payload["messages"],
                         [{"role": "user", "content": "hello"}])

    def test_single_turn_body_contains_cch_placeholder(self):
        body_str, _ = self.signer.build_body(
            "x", model="claude-opus-4-7",
        )
        self.assertIn("cch=00000", body_str)


class TestBuildBodyMultiTurn(unittest.TestCase):

    def setUp(self):
        self.signer = CCHSigner()

    def test_messages_list_serialised_in_order(self):
        msgs = [
            ChatMessage(role="user", content="q1"),
            ChatMessage(role="assistant", content="a1"),
            ChatMessage(role="user", content="q2"),
        ]
        body_str, _ = self.signer.build_body(
            messages=msgs, model="claude-opus-4-7",
        )
        payload = json.loads(body_str)
        self.assertEqual(len(payload["messages"]), 3)
        self.assertEqual(payload["messages"][0]["content"], "q1")
        self.assertEqual(payload["messages"][1]["role"], "assistant")
        self.assertEqual(payload["messages"][2]["content"], "q2")

    def test_prompt_ignored_when_messages_provided(self):
        """When messages is non-empty, prompt is NOT synthesised into
        the messages array — messages wins outright."""
        msgs = [ChatMessage(role="user", content="from messages")]
        body_str, _ = self.signer.build_body(
            "from prompt", messages=msgs, model="claude-opus-4-7",
        )
        payload = json.loads(body_str)
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(
            payload["messages"][0]["content"], "from messages",
        )

    def test_assistant_thinking_block_round_trips(self):
        """H-HAL-W4B-2: prior assistant thinking blocks with signatures
        must survive the serialise-then-JSON round trip unchanged so the
        Anthropic API can verify signature continuity on the next turn.
        """
        thinking_blocks = [
            {"type": "thinking", "thinking": "let me think...",
             "signature": "abc123def456"},
            {"type": "text", "text": "the answer is 42"},
        ]
        msgs = [
            ChatMessage(role="user", content="what is 6*7?"),
            ChatMessage(role="assistant", content=thinking_blocks),
            ChatMessage(role="user", content="thanks, now do 7*8"),
        ]
        body_str, _ = self.signer.build_body(
            messages=msgs, model="claude-opus-4-7",
        )
        payload = json.loads(body_str)
        assistant_msg = payload["messages"][1]
        self.assertEqual(assistant_msg["content"], thinking_blocks)
        self.assertEqual(
            assistant_msg["content"][0]["signature"], "abc123def456",
        )

    def test_fingerprint_derives_from_first_user_message(self):
        """CCH suffix must hash on the FIRST user message text, not
        the latest — so turn 2+ has the same suffix as turn 1."""
        msgs_turn1 = [ChatMessage(role="user", content="initial question")]
        msgs_turn2 = [
            ChatMessage(role="user", content="initial question"),
            ChatMessage(role="assistant", content="a1"),
            ChatMessage(role="user", content="follow-up"),
        ]
        _, suffix1 = self.signer.build_body(
            messages=msgs_turn1, model="claude-opus-4-7",
        )
        _, suffix2 = self.signer.build_body(
            messages=msgs_turn2, model="claude-opus-4-7",
        )
        self.assertEqual(
            suffix1, suffix2,
            "fingerprint must be stable across turns in the same conversation",
        )

    def test_fingerprint_text_override(self):
        """Caller can override the fingerprint source explicitly."""
        msgs = [ChatMessage(role="user", content="some user message")]
        _, suffix_default = self.signer.build_body(
            messages=msgs, model="claude-opus-4-7",
        )
        _, suffix_override = self.signer.build_body(
            messages=msgs, model="claude-opus-4-7",
            fingerprint_text="completely different",
        )
        self.assertNotEqual(suffix_default, suffix_override)


class TestBuildBodySystemBlocks(unittest.TestCase):

    def setUp(self):
        self.signer = CCHSigner()

    def test_billing_header_block_always_first(self):
        body_str, _ = self.signer.build_body(
            "hi", model="claude-opus-4-7",
        )
        payload = json.loads(body_str)
        self.assertEqual(len(payload["system"]), 1)
        self.assertIn(
            "x-anthropic-billing-header", payload["system"][0]["text"],
        )

    def test_user_system_appended_as_second_block(self):
        body_str, _ = self.signer.build_body(
            "hi", model="claude-opus-4-7",
            system="You are a helpful assistant.",
        )
        payload = json.loads(body_str)
        self.assertEqual(len(payload["system"]), 2)
        self.assertIn(
            "x-anthropic-billing-header", payload["system"][0]["text"],
        )
        self.assertEqual(
            payload["system"][1]["text"], "You are a helpful assistant.",
        )
        self.assertEqual(payload["system"][1]["type"], "text")

    def test_system_none_keeps_single_billing_block(self):
        body_str, _ = self.signer.build_body(
            "hi", model="claude-opus-4-7", system=None,
        )
        payload = json.loads(body_str)
        self.assertEqual(len(payload["system"]), 1)

    def test_empty_string_system_keeps_single_billing_block(self):
        """Empty string is falsy — should NOT append an empty second block."""
        body_str, _ = self.signer.build_body(
            "hi", model="claude-opus-4-7", system="",
        )
        payload = json.loads(body_str)
        self.assertEqual(len(payload["system"]), 1)


class TestBuildBodyContextManagementPreserved(unittest.TestCase):
    """W4-A regression: context_management field still injected."""

    def test_single_turn_still_has_context_management(self):
        body_str, _ = CCHSigner().build_body(
            "hi", model="claude-opus-4-7",
        )
        payload = json.loads(body_str)
        self.assertIn("context_management", payload)

    def test_multi_turn_still_has_context_management(self):
        msgs = [ChatMessage(role="user", content="hi")]
        body_str, _ = CCHSigner().build_body(
            messages=msgs, model="claude-opus-4-7",
        )
        payload = json.loads(body_str)
        self.assertIn("context_management", payload)


class TestSubscriptionChatThreads(unittest.TestCase):
    """CCHSubscription.chat must pass messages/system through to signer."""

    def test_chat_forwards_messages_and_system_to_signer(self):
        from lib.hal.subscription import CCHSubscription
        sub = CCHSubscription(oauth_token="fake-token")
        msgs = [ChatMessage(role="user", content="hello")]

        with patch.object(sub, "_signer") as mock_signer:
            mock_signer.call.return_value = MagicMock()
            sub.chat(
                model="claude-opus-4-7",
                messages=msgs,
                system="be concise",
            )
            _, kwargs = mock_signer.call.call_args
            self.assertEqual(kwargs["messages"], msgs)
            self.assertEqual(kwargs["system"], "be concise")
            self.assertEqual(kwargs["model"], "claude-opus-4-7")

    def test_chat_single_turn_form_still_works(self):
        from lib.hal.subscription import CCHSubscription
        sub = CCHSubscription(oauth_token="fake-token")
        with patch.object(sub, "_signer") as mock_signer:
            mock_signer.call.return_value = MagicMock()
            sub.chat("hello", model="claude-opus-4-7")
            args, kwargs = mock_signer.call.call_args
            # Single-turn: prompt is first positional, messages is None
            self.assertEqual(args[0], "hello")
            self.assertIsNone(kwargs.get("messages"))


class TestFlattenMessagesDeletion(unittest.TestCase):
    """Regression: _flatten_messages was removed from subscription.py."""

    def test_flatten_messages_not_importable(self):
        import lib.hal.subscription as sub_mod
        self.assertFalse(
            hasattr(sub_mod, "_flatten_messages"),
            "_flatten_messages should have been deleted by W4-B",
        )


if __name__ == "__main__":
    unittest.main()
