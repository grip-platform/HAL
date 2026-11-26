"""Regression anchor for the max_tokens=16384 default bump.

Low-output-cap providers (deepseek 8192, moonshot-8k, GLM-4, small groq/qwen)
reject a max_tokens larger than the model's output cap with a 400 instead of
clamping. OpenAICompatProvider._request halves max_tokens and retries so a high
default (16384) degrades gracefully rather than hard-failing fixed-routing or
silently dropping HAL Council members. High-cap providers (gemini/xai/gpt-oss)
accept 16384 and never reach the retry branch.

Mutation guard: delete the retry branch and test 1 fails (the first 400
propagates as OpenAICompatError). test 2 keeps the retry NARROW — a 400 that is
not about max_tokens must still raise, not be swallowed by a retry loop.
"""
import io
import os
import sys
import unittest
import urllib.error
from unittest import mock

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.providers.openai_compat import (  # noqa: E402
    OpenAICompatError,
    OpenAICompatProvider,
)


def _openai_text_response(text: str = "ok", model: str = "deepseek-chat") -> dict:
    return {
        "id": "chatcmpl-1",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }


def _http_400(url: str, message: str) -> urllib.error.HTTPError:
    body = ('{"error":{"message":"' + message + '"}}').encode()
    return urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(body))


class TestMaxTokensRetry(unittest.TestCase):
    def test_max_tokens_400_halves_and_retries(self):
        provider = OpenAICompatProvider(
            name="deepseek", base_url="https://api.deepseek.com/v1", api_key="sk-test",
        )
        seen = []

        def fake_do_request(_self, method, url, body):
            seen.append(dict(body) if body else None)
            if len(seen) == 1:
                raise _http_400(url, "max_tokens must be <= 8192")
            return _openai_text_response()

        with mock.patch.object(OpenAICompatProvider, "_do_request", fake_do_request):
            resp = provider.chat("hi", model="deepseek-chat", max_tokens=16384)

        # Retried once with a halved max_tokens, then succeeded (no hard-fail).
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0]["max_tokens"], 16384)
        self.assertEqual(seen[1]["max_tokens"], 8192)
        self.assertEqual(resp.text, "ok")

    def test_non_max_tokens_400_still_raises(self):
        """A 400 unrelated to max_tokens is NOT retried — it raises (narrow retry)."""
        provider = OpenAICompatProvider(
            name="deepseek", base_url="https://api.deepseek.com/v1", api_key="sk-test",
        )

        def fake_do_request(_self, method, url, body):
            raise _http_400(url, "invalid api key")

        with mock.patch.object(OpenAICompatProvider, "_do_request", fake_do_request):
            with self.assertRaises(OpenAICompatError):
                provider.chat("hi", model="deepseek-chat", max_tokens=16384)


if __name__ == "__main__":
    unittest.main()
