"""Goodhart-resistant tests for the Gemini thought_signature round-trip cache.

Each test resets the module-global cache and asserts EXACT dict/byte states, so a
mutation (drop the gemini scoping, overwrite an existing signature, skip the SSE
merge, break the LRU bound, or rewrite non-gemini bytes) fails a test rather than
silently regressing. Mirrors the live-proven mechanism (turn 2 WITH the
re-injected signature = HTTP 200; WITHOUT = HTTP 400)."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.hal import gemini_signature_cache as gsc  # noqa: E402

SIG = "Eo0DCooDAQw51scO53pkiP7YQEYh0aGmJ2iIuOjK"  # representative base64-ish signature


def _tool_call(tid, *, sig=None, args="{}"):
    tc = {"id": tid, "type": "function", "function": {"name": "get_time", "arguments": args}}
    if sig is not None:
        tc["extra_content"] = {"google": {"thought_signature": sig}}
    return tc


def _gemini_request(tool_calls):
    return {"model": "gemini-3.1-pro-preview", "messages": [
        {"role": "user", "content": "what time is it"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "12:00"},
    ]}


class _Base(unittest.TestCase):
    def setUp(self):
        gsc._cache.clear()


class TestCaptureNonStream(_Base):
    def test_captures_id_to_signature(self):
        resp = {"choices": [{"message": {"role": "assistant", "tool_calls": [
            _tool_call("call_a", sig=SIG)]}}]}
        n = gsc.capture_from_bytes(json.dumps(resp).encode())
        self.assertEqual(n, 1)
        self.assertEqual(gsc._cache.get("call_a"), SIG)

    def test_tool_call_without_signature_not_cached(self):
        resp = {"choices": [{"message": {"tool_calls": [_tool_call("call_b")]}}]}
        self.assertEqual(gsc.capture_from_bytes(json.dumps(resp).encode()), 0)
        self.assertNotIn("call_b", gsc._cache)


class TestCaptureSSE(_Base):
    def test_captures_from_streamed_delta(self):
        sse = (
            'data: {"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":'
            '[{"index":0,"id":"call_s","type":"function","function":{"name":"get_time",'
            '"arguments":""},"extra_content":{"google":{"thought_signature":"' + SIG + '"}}}]}}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"{}"}}]}}]}\n\n'
            'data: [DONE]\n\n'
        )
        n = gsc.capture_from_bytes(sse.encode())
        self.assertEqual(n, 1)
        self.assertEqual(gsc._cache.get("call_s"), SIG)

    def test_id_and_signature_in_separate_deltas_merge_by_index(self):
        sse = (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_m"}]}}]}\n\n'
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"extra_content":{"google":{"thought_signature":"' + SIG + '"}}}]}}]}\n\n'
        )
        self.assertEqual(gsc.capture_from_bytes(sse.encode()), 1)
        self.assertEqual(gsc._cache.get("call_m"), SIG)


class TestReinject(_Base):
    def test_injects_cached_signature_by_id_keeps_tools(self):
        gsc.remember("call_a", SIG)
        req = _gemini_request([_tool_call("call_a")])  # assistant tool_call lacks sig
        gsc.reinject_request(req)
        tc = req["messages"][1]["tool_calls"][0]
        self.assertEqual(tc["extra_content"]["google"]["thought_signature"], SIG)
        self.assertEqual(tc["function"]["name"], "get_time")  # tool preserved

    def test_never_overwrites_existing_signature(self):
        gsc.remember("call_a", "CACHED")
        req = _gemini_request([_tool_call("call_a", sig="ORIGINAL")])
        gsc.reinject_request(req)
        self.assertEqual(
            req["messages"][1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
            "ORIGINAL")

    def test_noop_when_id_not_cached(self):
        req = _gemini_request([_tool_call("call_unknown")])
        gsc.reinject_request(req)
        self.assertNotIn("extra_content", req["messages"][1]["tool_calls"][0])

    def test_noop_for_non_gemini_model(self):
        gsc.remember("call_a", SIG)
        req = _gemini_request([_tool_call("call_a")])
        req["model"] = "gpt-4o"  # not gemini
        gsc.reinject_request(req)
        self.assertNotIn("extra_content", req["messages"][1]["tool_calls"][0])


class TestReinjectRaw(_Base):
    def test_non_gemini_bytes_are_verbatim(self):
        raw = json.dumps({"model": "gpt-4o", "messages": [
            {"role": "assistant", "tool_calls": [_tool_call("call_a")]}]}).encode()
        gsc.remember("call_a", SIG)
        self.assertEqual(gsc.reinject_raw(raw), raw)  # byte-identical

    def test_unparseable_bytes_are_verbatim(self):
        raw = b"{not valid json"
        self.assertEqual(gsc.reinject_raw(raw), raw)

    def test_gemini_bytes_get_signature(self):
        gsc.remember("call_a", SIG)
        raw = json.dumps(_gemini_request([_tool_call("call_a")])).encode()
        out = json.loads(gsc.reinject_raw(raw))
        self.assertEqual(
            out["messages"][1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
            SIG)


class TestCacheBounded(_Base):
    def test_lru_eviction_at_cap(self):
        for i in range(gsc._MAX_ENTRIES + 50):
            gsc.remember(f"id_{i}", f"sig_{i}")
        self.assertLessEqual(len(gsc._cache), gsc._MAX_ENTRIES)
        self.assertNotIn("id_0", gsc._cache)              # oldest evicted
        self.assertIn(f"id_{gsc._MAX_ENTRIES + 49}", gsc._cache)  # newest kept


class TestRoundTrip(_Base):
    def test_capture_then_reinject_full_cycle(self):
        # the live-proven mechanism, unit-level: response -> cache -> next request
        resp = {"choices": [{"message": {"tool_calls": [_tool_call("call_rt", sig=SIG)]}}]}
        gsc.capture_from_bytes(json.dumps(resp).encode())
        req = _gemini_request([_tool_call("call_rt")])
        gsc.reinject_request(req)
        self.assertEqual(
            req["messages"][1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
            SIG)


if __name__ == "__main__":
    unittest.main()
