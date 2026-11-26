"""#386 — a billing `insufficient_quota` 429 must be classified apart from a
transient rate-limit 429.

Before this fix, openai_compat._request retried every 429 three times then
flattened it into RateLimitError("rate limited after 3 retries") — dropping the
body — so a billing cap (never retryable) wasted 3 retries AND was mis-classified
by the Limit-Sentinel as a transient rate-limit (exponential-backoff retry)
instead of a rejected provider. Per rate-limit-never-blocks.md these are
different classes.

Goodhart-proof: the short-circuit test asserts the provider is called EXACTLY
ONCE on insufficient_quota (reverting the short-circuit → 1+retries calls, fails);
the transient test asserts it IS retried (reverting the retry path → 1 call, fails);
the sentinel test asserts insufficient_quota → rejected+max-cooldown while a plain
rate-limit does NOT reject (collapsing the two → fails).
"""
import io
import json
import sys
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.hal.providers.openai_compat import (  # noqa: E402
    OpenAICompatProvider,
    RateLimitError,
    _parse_error_code,
)


def _make_429(body_dict: dict) -> urllib.error.HTTPError:
    raw = json.dumps(body_dict).encode()
    return urllib.error.HTTPError(
        "http://x/chat/completions", 429, "Too Many Requests", {}, io.BytesIO(raw)
    )


_QUOTA_BODY = {"error": {"code": "insufficient_quota", "type": "insufficient_quota",
                         "message": "You exceeded your current quota"}}
_RATE_BODY = {"error": {"code": "rate_limit_exceeded", "type": "requests",
                        "message": "Rate limit reached"}}


class TestInsufficientQuotaShortCircuit(unittest.TestCase):
    def _provider(self, max_retries=3):
        return OpenAICompatProvider(name="openai", base_url="http://x",
                                    api_key="k", max_retries=max_retries)

    def test_insufficient_quota_does_not_retry(self):
        """A billing 429 is never retryable: exactly ONE call, immediate raise."""
        p = self._provider(max_retries=3)
        calls = {"n": 0}

        def fake(*a, **k):
            calls["n"] += 1
            raise _make_429(_QUOTA_BODY)

        with patch.object(p, "_do_request", side_effect=fake), patch("time.sleep"):
            with self.assertRaises(RateLimitError) as ctx:
                p._request("POST", "/chat/completions", {})
        self.assertEqual(calls["n"], 1, "insufficient_quota must NOT be retried")
        self.assertEqual(ctx.exception.error_code, "insufficient_quota")
        self.assertEqual(ctx.exception.status_code, 429)

    def test_transient_429_is_retried_and_carries_metadata(self):
        """A non-quota 429 still retries to exhaustion, then raises WITH the code."""
        p = self._provider(max_retries=2)
        calls = {"n": 0}

        def fake(*a, **k):
            calls["n"] += 1
            raise _make_429(_RATE_BODY)

        with patch.object(p, "_do_request", side_effect=fake), patch("time.sleep"):
            with self.assertRaises(RateLimitError) as ctx:
                p._request("POST", "/chat/completions", {})
        self.assertEqual(calls["n"], 3, "transient 429 must retry 1 + max_retries times")
        self.assertEqual(ctx.exception.error_code, "rate_limit_exceeded")

    def test_ratelimit_error_is_backward_compatible(self):
        """Existing RateLimitError('msg') call sites keep working (no kwargs)."""
        e = RateLimitError("openai: rate limited")
        self.assertEqual(str(e), "openai: rate limited")
        self.assertIsNone(e.error_code)
        self.assertIsNone(e.status_code)


class TestParseErrorCode(unittest.TestCase):
    def test_json_body(self):
        self.assertEqual(
            _parse_error_code(json.dumps(_QUOTA_BODY)),
            ("insufficient_quota", "insufficient_quota"),
        )

    def test_json_code_wins_over_substring_in_message(self):
        """#402: a well-formed body whose CODE is not insufficient_quota but whose
        MESSAGE happens to contain the substring must classify by the code field —
        otherwise a transient 429 (code=rate_limit_exceeded) would be wrongly
        short-circuited as a billing cap and never retried."""
        body = {"error": {"code": "rate_limit_exceeded", "type": "requests",
                          "message": "slow down — this is not insufficient_quota"}}
        self.assertEqual(
            _parse_error_code(json.dumps(body)),
            ("rate_limit_exceeded", "requests"),
        )

    def test_substring_fallback_on_garbage(self):
        self.assertEqual(
            _parse_error_code("<html>insufficient_quota buried in noise</html>"),
            ("insufficient_quota", None),
        )

    def test_unparseable_returns_none(self):
        self.assertEqual(_parse_error_code("nothing here"), (None, None))


class TestSentinelClassifiesQuota(unittest.TestCase):
    def _sentinel(self):
        from lib.hal.sentinel import LimitSentinel
        with patch.object(LimitSentinel, "_load_state", lambda self: None), \
             patch.object(LimitSentinel, "_save_state", lambda self: None):
            return LimitSentinel(registry=MagicMock())

    def test_insufficient_quota_rejects_provider_with_max_cooldown(self):
        from lib.hal.sentinel import _COOLDOWN_MAX_S
        s = self._sentinel()
        err = RateLimitError(
            "openai: insufficient_quota — billing quota exhausted",
            status_code=429, error_code="insufficient_quota",
        )
        t0 = time.time()
        with patch.object(type(s), "_save_state", lambda self: None):
            s._on_error("openai", err)
        state = s._get_or_create("openai")
        self.assertEqual(state.status, "rejected")
        self.assertGreaterEqual(state.cooldown_until, t0 + _COOLDOWN_MAX_S - 10)

    def test_plain_rate_limit_does_not_reject(self):
        """Control: a transient rate-limit must NOT be rejected (backoff path),
        proving the two 429 classes are handled differently."""
        from lib.hal.sentinel import _COOLDOWN_MAX_S
        s = self._sentinel()
        err = RateLimitError("openai: rate limited after 3 retries (429)",
                             error_code="rate_limit_exceeded")
        with patch.object(type(s), "_save_state", lambda self: None):
            s._on_error("openai", err)
        state = s._get_or_create("openai")
        self.assertNotEqual(state.status, "rejected")
        self.assertLess(state.cooldown_until, time.time() + _COOLDOWN_MAX_S)


class TestAgora485ConvergeFixes(unittest.TestCase):
    """Forward-anchors for the AGORA #485 review concerns."""

    def test_json_code_authoritative_over_substring_in_message(self):
        """AGORA falsifier: a transient 429 whose PROSE mentions
        'insufficient_quota' must still be classified by its JSON code
        (rate_limit_exceeded) and RETRIED, not short-circuited as a billing cap.
        Proves the JSON-first parse wins over the substring fallback."""
        body = {"error": {"code": "rate_limit_exceeded", "type": "requests",
                          "message": "slow down; this is not an insufficient_quota problem"}}
        self.assertEqual(_parse_error_code(json.dumps(body))[0], "rate_limit_exceeded")

        p = OpenAICompatProvider(name="openai", base_url="http://x",
                                 api_key="k", max_retries=2)
        calls = {"n": 0}

        def fake(*a, **k):
            calls["n"] += 1
            raise _make_429(body)

        with patch.object(p, "_do_request", side_effect=fake), patch("time.sleep"):
            with self.assertRaises(RateLimitError) as ctx:
                p._request("POST", "/chat/completions", {})
        self.assertEqual(calls["n"], 3, "must retry, NOT short-circuit as quota")
        self.assertEqual(ctx.exception.error_code, "rate_limit_exceeded")

    def test_max_tokens_400_retries_without_backoff_sleep(self):
        """AGORA concern #1: the max_tokens graceful-degradation retry is a
        parameter fix, not a rate signal — it must retry IMMEDIATELY, never
        sleeping the exponential backoff (which the earlier unified-sleep
        refactor accidentally introduced)."""
        p = OpenAICompatProvider(name="openai", base_url="http://x",
                                 api_key="k", max_retries=3)
        calls = {"n": 0}

        def fake(method, url, body):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(
                    url, 400, "Bad Request", {},
                    io.BytesIO(b'{"error":{"message":"max_tokens exceeds the model limit"}}'),
                )
            return {"choices": [{"message": {"content": "ok"}}], "model": "m", "usage": {}}

        sent = {"max_tokens": 16384}
        with patch.object(p, "_do_request", side_effect=fake), \
             patch("time.sleep") as slept:
            result = p._request("POST", "/chat/completions", sent)
        self.assertEqual(calls["n"], 2, "failed once on max_tokens, retried, succeeded")
        slept.assert_not_called()  # max_tokens degradation must NOT back off
        self.assertEqual(sent["max_tokens"], 8192, "max_tokens halved for the retry")
        self.assertIn("choices", result)


if __name__ == "__main__":
    unittest.main()
