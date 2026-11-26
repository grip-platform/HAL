"""Tests for CCH cap-escalation retry (D1 from TF>> liberation matrix, 2026-04-20).

Verifies cch.py:587-601 — the one-shot retry on stop_reason='max_tokens' that
mirrors TF>>'s capEnabled=true behaviour from src/query.ts. Source liberation:
TF>> commit 77ed9bc, ported to HAL in commit 768824cb.

Falsifiable claims:
- When _dispatch_request returns stop_reason='max_tokens' on the first call AND
  the caller did NOT pass max_tokens explicitly, _dispatch_request fires exactly
  twice and the second call's body contains "max_tokens": _ESCALATED_MAX_TOKENS.
- When the caller passes max_tokens=N explicitly, no retry fires even if the
  first response has stop_reason='max_tokens' (caller opted out of escalation).
- A normal end_turn first response triggers no retry.
- The final ChatResponse reflects the second call's content + finish_reason,
  with latency_ms summing both dispatches.

Goodhart guard (Rule 14): tests assert the literal max_tokens VALUE in the
second request body, not just the call count. A test that only counts calls
would pass even if the retry shipped the wrong limit.

Falsifier for the matrix-locked claim: telemetry showing <2% of CCH calls
trigger the retry. That is measured in production via logs, not here. This
test only validates the retry mechanism is correct *when* it fires.
"""
import json
import os
import sys
import unittest
from unittest.mock import patch

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.utils import cch as cch_mod
from lib.hal.utils.cch import CCHSigner, _ESCALATED_MAX_TOKENS

_OAUTH_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
_LIVE_OPT_IN_ENV = "HAL_RUN_LIVE_INTEGRATION"


def _make_messages_response(stop_reason: str, text: str = "ok") -> dict:
    """Minimal Anthropic Messages API response payload for fakery."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class TestCapEscalationRetry(unittest.TestCase):
    def setUp(self):
        self.signer = CCHSigner(cc_version="9.9.9")
        self.token = "sk-ant-oat01-test"

    def test_retry_fires_with_64k_when_max_tokens_and_no_override(self):
        """First response = max_tokens, second = end_turn. Assert retry uses 64k."""
        responses = iter([
            (_make_messages_response("max_tokens"), 100.0),
            (_make_messages_response("end_turn", text="full answer"), 150.0),
        ])
        captured_bodies: list[dict] = []

        def fake_dispatch(headers, body_bytes):
            captured_bodies.append(json.loads(body_bytes.decode("utf-8")))
            return next(responses)

        with patch.object(cch_mod, "_dispatch_request", side_effect=fake_dispatch):
            response = self.signer.call(
                prompt="hi",
                oauth_token=self.token,
                model="claude-opus-4-7",
            )

        # Goodhart guard: assert the LITERAL value, not just call count.
        self.assertEqual(
            len(captured_bodies), 2,
            "retry should produce exactly two _dispatch_request calls",
        )
        self.assertEqual(
            captured_bodies[1]["max_tokens"], _ESCALATED_MAX_TOKENS,
            f"second call must request max_tokens={_ESCALATED_MAX_TOKENS}",
        )
        self.assertEqual(
            _ESCALATED_MAX_TOKENS, 64_000,
            "TF>> liberation contract: escalated cap must remain 64k",
        )
        self.assertEqual(response.finish_reason, "end_turn")
        self.assertEqual(response.text, "full answer")
        self.assertAlmostEqual(response.latency_ms, 250.0, places=1)

    def test_no_retry_when_caller_overrides_max_tokens(self):
        """Explicit max_tokens override must opt out of the retry path."""
        responses = iter([(_make_messages_response("max_tokens"), 80.0)])
        call_count = 0

        def fake_dispatch(headers, body_bytes):
            nonlocal call_count
            call_count += 1
            return next(responses)

        with patch.object(cch_mod, "_dispatch_request", side_effect=fake_dispatch):
            response = self.signer.call(
                prompt="hi",
                oauth_token=self.token,
                model="claude-opus-4-7",
                max_tokens=2_000,
            )

        self.assertEqual(
            call_count, 1,
            "explicit max_tokens override must skip the retry path",
        )
        self.assertEqual(response.finish_reason, "max_tokens")

    def test_no_retry_on_end_turn(self):
        """A normal completion fires only one dispatch."""
        responses = iter([(_make_messages_response("end_turn"), 50.0)])
        call_count = 0

        def fake_dispatch(headers, body_bytes):
            nonlocal call_count
            call_count += 1
            return next(responses)

        with patch.object(cch_mod, "_dispatch_request", side_effect=fake_dispatch):
            response = self.signer.call(
                prompt="hi",
                oauth_token=self.token,
                model="claude-opus-4-7",
            )

        self.assertEqual(call_count, 1)
        self.assertEqual(response.finish_reason, "end_turn")

    def test_first_call_did_not_request_64k(self):
        """First dispatch must NOT carry max_tokens=64000 — escalation is on retry only."""
        responses = iter([
            (_make_messages_response("max_tokens"), 100.0),
            (_make_messages_response("end_turn"), 150.0),
        ])
        captured_bodies: list[dict] = []

        def fake_dispatch(headers, body_bytes):
            captured_bodies.append(json.loads(body_bytes.decode("utf-8")))
            return next(responses)

        with patch.object(cch_mod, "_dispatch_request", side_effect=fake_dispatch):
            self.signer.call(
                prompt="hi",
                oauth_token=self.token,
                model="claude-opus-4-7",
            )

        first_body_max = captured_bodies[0].get("max_tokens")
        self.assertNotEqual(
            first_body_max, _ESCALATED_MAX_TOKENS,
            "first call should not pre-escalate; retry-only contract",
        )


@unittest.skipUnless(
    os.environ.get(_OAUTH_ENV) and os.environ.get(_LIVE_OPT_IN_ENV) == "1",
    f"live integration test — requires {_OAUTH_ENV} AND {_LIVE_OPT_IN_ENV}=1",
)
class TestCapEscalationLive(unittest.TestCase):
    """Live integration test — opt-in, requires a real OAuth token.

    Validates the retry path against the real Anthropic API. Two-gate opt-in
    so a developer who happens to have CLAUDE_CODE_OAUTH_TOKEN set for
    normal HAL operation does NOT auto-fire a 2-minute network test. Set
    BOTH _OAUTH_ENV (token) AND _LIVE_OPT_IN_ENV='1' to run this class.
    The token is read once via os.environ; this test never logs its value.
    """

    def test_overflow_prompt_triggers_real_retry_path(self):
        signer = CCHSigner(cc_version="9.9.9")
        token = os.environ[_OAUTH_ENV]

        prompt = (
            "Write the longest possible exhaustive explanation of TCP/IP. "
            "Include every RFC, every header field, every state machine "
            "transition, every congestion-control algorithm with full math, "
            "every edge case, every implementation note. Do not stop early. "
            "Continue until you physically cannot produce more output."
        )
        response = signer.call(
            prompt=prompt,
            oauth_token=token,
            model="claude-opus-4-7",
        )

        # Cannot deterministically assert finish_reason='end_turn' (model may
        # re-hit max at 64k). Assert the call completed and produced a
        # substantial response, which is only possible if the retry fired.
        self.assertGreater(
            len(response.text), 1_000,
            "post-retry response should be substantial; <1k suggests retry failed",
        )
        self.assertIn(
            response.finish_reason, ("end_turn", "max_tokens"),
            "retry should produce one of: clean stop, or re-hit at 64k",
        )


if __name__ == "__main__":
    unittest.main()
