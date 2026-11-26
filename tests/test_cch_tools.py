"""Tests for CCH tool-use support (W5 integration sprint).

Verifies the CCH signing path accepts an Anthropic ``tools`` list and
serialises it into the request body correctly, so CCH-authenticated
(subscription-billed) sessions can drive :func:`run_tool_loop`.

Falsifiable claims:
- ``build_body(..., tools=[...])`` includes the tools verbatim in the
  payload under a top-level ``"tools"`` key
- Empty or ``None`` tools is treated as "no tools" and the field is
  omitted, preserving backward compat with existing tool-free callers
- ``CCHSubscription.chat(..., tools=[...])`` forwards the list through
  to the signer (important because dropping it silently would produce
  responses the model couldn't correlate with tool_use blocks)
- The signed body still round-trips through ``sign()`` without
  corrupting the tools payload
"""
import json
import os
import sys
import unittest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.subscription import CCHSubscription
from lib.hal.utils.cch import CCHSigner


_SAMPLE_TOOLS = [
    {
        "name": "Read",
        "description": "Read a file from disk.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "Grep",
        "description": "Search file contents with regex.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
]


class TestBuildBodyTools(unittest.TestCase):
    def setUp(self):
        self.signer = CCHSigner(cc_version="9.9.9")

    def _payload(self, **kwargs):
        """Return the parsed JSON payload (without the cch placeholder)."""
        body, _suffix = self.signer.build_body(
            "hi", model="claude-opus-4-7", **kwargs,
        )
        return json.loads(body)

    def test_tools_absent_when_not_provided(self):
        """Backward compat: callers not passing ``tools`` see no field."""
        payload = self._payload()
        self.assertNotIn("tools", payload)

    def test_tools_absent_when_none(self):
        payload = self._payload(tools=None)
        self.assertNotIn("tools", payload)

    def test_tools_absent_when_empty_list(self):
        """Empty list == "no tools" — matches make_anthropic_chat_fn."""
        payload = self._payload(tools=[])
        self.assertNotIn("tools", payload)

    def test_tools_present_when_non_empty(self):
        payload = self._payload(tools=_SAMPLE_TOOLS)
        self.assertIn("tools", payload)
        self.assertEqual(payload["tools"], _SAMPLE_TOOLS)

    def test_tools_preserve_shape_verbatim(self):
        """No field renaming, no schema coercion — verbatim forward."""
        payload = self._payload(tools=_SAMPLE_TOOLS)
        self.assertEqual(len(payload["tools"]), 2)
        self.assertEqual(payload["tools"][0]["name"], "Read")
        self.assertIn("input_schema", payload["tools"][0])
        self.assertIn("required", payload["tools"][0]["input_schema"])


class TestToolsRoundTripThroughSign(unittest.TestCase):
    def test_signed_body_still_parses_as_json_with_tools(self):
        """Sign replaces cch=00000 with a real hash; tools payload must survive."""
        signer = CCHSigner()
        # sign requires xxhash — skip if not installed rather than fail noisily
        try:
            signed_body, _suffix = signer.sign(
                "hi", model="claude-opus-4-7", tools=_SAMPLE_TOOLS,
            )
        except Exception as exc:  # pragma: no cover — env-dependent
            if "xxhash" in str(exc):
                self.skipTest("xxhash not installed in test env")
            raise
        parsed = json.loads(signed_body)
        self.assertEqual(parsed["tools"], _SAMPLE_TOOLS)
        # The cch placeholder should have been replaced with a real 5-hex hash
        header_text = parsed["system"][0]["text"]
        self.assertNotIn("cch=00000", header_text)
        self.assertIn("cch=", header_text)


class TestCCHSubscriptionForwardsTools(unittest.TestCase):
    def test_chat_passes_tools_to_signer_call(self):
        """Silent drop would cause the model to hallucinate instead of
        emitting tool_use blocks. We assert the kwarg is forwarded."""
        sub = CCHSubscription(oauth_token="test-token")
        captured_kwargs: dict = {}

        class _FakeSigner:
            def call(self, prompt, *, oauth_token, model, messages=None,
                     system=None, max_tokens=2048, tools=None):
                captured_kwargs.update(
                    tools=tools,
                    model=model,
                    oauth_token=oauth_token,
                )
                # Return a minimal ChatResponse-shaped stub
                from lib.hal.types import ChatResponse
                return ChatResponse(text="ok", model=model, provider="anthropic-cch")

        sub._signer = _FakeSigner()
        sub.chat("hi", model="claude-opus-4-7", tools=_SAMPLE_TOOLS)
        self.assertEqual(captured_kwargs["tools"], _SAMPLE_TOOLS)
        self.assertEqual(captured_kwargs["model"], "claude-opus-4-7")

    def test_chat_without_tools_passes_none_to_signer(self):
        sub = CCHSubscription(oauth_token="test-token")
        captured: dict = {}

        class _FakeSigner:
            def call(self, prompt, *, oauth_token, model, messages=None,
                     system=None, max_tokens=2048, tools=None):
                captured["tools"] = tools
                from lib.hal.types import ChatResponse
                return ChatResponse(text="ok", model=model, provider="anthropic-cch")

        sub._signer = _FakeSigner()
        sub.chat("hi", model="claude-opus-4-7")
        self.assertIsNone(captured["tools"])


class TestOtherPayloadFieldsUnchanged(unittest.TestCase):
    """Ensure adding tools didn't disturb existing payload structure."""

    def test_billing_header_still_first_system_block(self):
        signer = CCHSigner(cc_version="1.2.3")
        body, _ = signer.build_body(
            "hi", model="claude-opus-4-7", tools=_SAMPLE_TOOLS,
        )
        payload = json.loads(body)
        self.assertIn("system", payload)
        first_block = payload["system"][0]
        self.assertEqual(first_block["type"], "text")
        self.assertIn("x-anthropic-billing-header", first_block["text"])

    def test_thinking_adaptive_still_present(self):
        signer = CCHSigner()
        body, _ = signer.build_body(
            "hi", model="claude-opus-4-7", tools=_SAMPLE_TOOLS,
        )
        payload = json.loads(body)
        self.assertEqual(payload.get("thinking"), {"type": "adaptive"})

    def test_messages_still_correct_single_turn(self):
        signer = CCHSigner()
        body, _ = signer.build_body(
            "what is up", model="claude-opus-4-7", tools=_SAMPLE_TOOLS,
        )
        payload = json.loads(body)
        self.assertEqual(
            payload["messages"],
            [{"role": "user", "content": "what is up"}],
        )


if __name__ == "__main__":
    unittest.main()
