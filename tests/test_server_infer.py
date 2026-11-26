"""GRIP #2722 transport B — hal-server POST /api/infer (wraps grip_infer).

Goodhart-proof: asserts the EXACT response key set + values + HTTP codes,
not call counts. Each test fails under the obvious mutation:

  - response shape or usage drifts             -> test_success_shape_parity
  - fail-safe becomes a 5xx                    -> test_*_failsafe_200
  - missing-prompt stops being 400             -> test_missing_prompt_400
  - audit not threaded to grip_infer           -> test_audit_threads_idr_ref
  - route unwired                              -> test_route_registered
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.server import _make_handler

# Exact HTTP result contract consumed by GRIP's lib/hal_llm_adapter.py
# (_grip_infer_call and _record_grip_infer_usage). HAL #632 added usage and
# HAL #648 finish_reason; GRIP reads both by name.
_BRIDGE_KEYS = {"ok", "text", "provider", "model", "idr_ref",
                "error_code", "error_message", "media_attached", "usage",
                "finish_reason"}
# Import/runtime exceptions have no result object, so they carry neither
# usage nor finish_reason.
_FAILSAFE_KEYS = _BRIDGE_KEYS - {"usage", "finish_reason"}


def _handler():
    """Real Handler class from the factory; instance without socket init."""
    H = _make_handler(MagicMock(), MagicMock(), MagicMock(),
                      MagicMock(), MagicMock())
    h = H.__new__(H)
    h._captured = []
    h._json = lambda data, code=200: h._captured.append((data, code))
    return h


def _fake_grip_runtime(grip_infer):
    mod = types.ModuleType("lib.hal.grip_runtime")
    mod.grip_infer = grip_infer
    return {"lib.hal.grip_runtime": mod}


class TestApiInfer(unittest.TestCase):

    def test_success_shape_parity(self):
        """Response keeps the exact HTTP result keys and reported token counts."""
        r = types.SimpleNamespace(
            ok=True, text="HELLO", provider="cch", model="claude",
            idr_ref={"id": "idr-9"}, error_code=None, error_message=None,
            usage={"in": 12, "out": 3},
            media_attached=False)  # current-result semantics: text turn, no media
        h = _handler()
        h._read_body = lambda: {"prompt": "hi"}
        with patch.dict(sys.modules, _fake_grip_runtime(lambda *a, **k: r)):
            h._api_infer()
        data, code = h._captured[-1]
        self.assertEqual(code, 200)
        self.assertEqual(set(data.keys()), _BRIDGE_KEYS)   # shape parity
        self.assertEqual(data["ok"], True)
        self.assertEqual(data["text"], "HELLO")
        self.assertEqual(data["provider"], "cch")
        self.assertEqual(data["idr_ref"], {"id": "idr-9"})
        self.assertFalse(data["media_attached"])           # no media in the body
        self.assertIs(type(data["usage"]), dict)
        self.assertEqual(set(data["usage"]), {"input_tokens", "output_tokens"})
        self.assertEqual(data["usage"], {"input_tokens": 12, "output_tokens": 3})
        for count in data["usage"].values():
            self.assertIs(type(count), int)  # bool is not a token count

    def test_unreported_usage_is_empty(self):
        """Unreported counts stay absent, never fabricated as zero."""
        for usage in (None, {}):
            with self.subTest(usage=usage):
                r = types.SimpleNamespace(
                    ok=True, text="hi", provider="p", model="m",
                    idr_ref=None, error_code=None, error_message=None,
                    media_attached=False, usage=usage)
                h = _handler()
                h._read_body = lambda: {"prompt": "hi"}
                with patch.dict(sys.modules, _fake_grip_runtime(lambda *a, **k: r)):
                    h._api_infer()
                data, code = h._captured[-1]
                self.assertEqual(code, 200)
                self.assertEqual(set(data), _BRIDGE_KEYS)
                self.assertIs(type(data["usage"]), dict)
                self.assertEqual(data["usage"], {})

    def test_missing_prompt_400(self):
        h = _handler()
        h._read_body = lambda: {}
        h._api_infer()
        data, code = h._captured[-1]
        self.assertEqual(code, 400)
        self.assertIn("error", data)

    def test_grip_infer_raises_is_failsafe_200_not_5xx(self):
        def boom(*a, **k):
            raise RuntimeError("provider exploded")
        h = _handler()
        h._read_body = lambda: {"prompt": "hi"}
        with patch.dict(sys.modules, _fake_grip_runtime(boom)):
            h._api_infer()
        data, code = h._captured[-1]
        self.assertEqual(code, 200)                        # NEVER 5xx
        self.assertEqual(data["ok"], False)
        self.assertEqual(data["error_code"], "grip_infer_raised")
        self.assertEqual(set(data.keys()), _FAILSAFE_KEYS)

    def test_grip_runtime_unavailable_is_failsafe_200(self):
        h = _handler()
        h._read_body = lambda: {"prompt": "hi"}
        # sys.modules[name]=None makes `from name import x` raise ImportError
        with patch.dict(sys.modules, {"lib.hal.grip_runtime": None}):
            h._api_infer()
        data, code = h._captured[-1]
        self.assertEqual(code, 200)
        self.assertEqual(data["ok"], False)
        self.assertEqual(data["error_code"], "grip_runtime_unavailable")
        self.assertEqual(set(data.keys()), _FAILSAFE_KEYS)

    def test_audit_threads_idr_ref(self):
        seen = {}

        def gi(prompt, *, model=None, audit=False, media=None, content_parts=None, no_failover=False, max_tokens=None):
            seen["model"] = model
            seen["audit"] = audit
            return types.SimpleNamespace(
                ok=True, text="A", provider="p", model=model,
                idr_ref={"chain": "x"}, error_code=None, error_message=None)
        h = _handler()
        h._read_body = lambda: {"prompt": "q", "model": "m1", "audit": True}
        with patch.dict(sys.modules, _fake_grip_runtime(gi)):
            h._api_infer()
        self.assertEqual(seen, {"model": "m1", "audit": True})
        self.assertEqual(h._captured[-1][0]["idr_ref"], {"chain": "x"})

    def test_media_body_threads_and_relaxes_prompt(self):
        """A media (or content_parts) body routes through grip_infer, and the
        empty-prompt 400 guard is relaxed because the parts carry the turn —
        the server seam that makes grip-hear routable through HAL."""
        seen = {}

        def gi(prompt, *, model=None, audit=False, media=None, content_parts=None, no_failover=False, max_tokens=None):
            seen["prompt"] = prompt
            seen["media"] = media
            seen["content_parts"] = content_parts
            return types.SimpleNamespace(
                ok=True, text="heard", provider="gemini", model="g",
                idr_ref=None, error_code=None, error_message=None,
                media_attached=True)  # response-derived: the turn carried audio
        h = _handler()
        h._read_body = lambda: {"media": ["/tmp/song.wav"]}    # NO prompt
        with patch.dict(sys.modules, _fake_grip_runtime(gi)):
            h._api_infer()
        data, code = h._captured[-1]
        self.assertEqual(code, 200)                            # not the 400 guard
        self.assertEqual(seen["media"], ["/tmp/song.wav"])     # threaded through
        self.assertEqual(seen["prompt"], "")                   # empty prompt accepted
        self.assertEqual(data["text"], "heard")
        self.assertEqual(set(data.keys()), _BRIDGE_KEYS)       # shape parity held
        self.assertIs(data["media_attached"], True)            # attested: media threaded
        self.assertEqual(data["usage"], {})                   # older result: no usage field

    def test_media_attached_true_even_when_turn_failed(self):
        """Dispatch attached the media; a provider error later does not un-attach it."""
        r = types.SimpleNamespace(
            ok=False, text="", provider=None, model=None,
            idr_ref=None, error_code="x", error_message="boom",
            media_attached=True)
        h = _handler()
        h._read_body = lambda: {"media": ["/tmp/song.wav"]}
        with patch.dict(sys.modules, _fake_grip_runtime(lambda *a, **k: r)):
            h._api_infer()
        data, code = h._captured[-1]
        self.assertEqual(code, 200)
        self.assertIs(data["media_attached"], True)            # dispatch-time truth survives

    def test_text_only_parts_never_attest(self):
        """A text-only content_parts body is a text turn — no attachment."""
        r = types.SimpleNamespace(
            ok=True, text="hi", provider="p", model="m",
            idr_ref=None, error_code=None, error_message=None,
            media_attached=False)
        h = _handler()
        h._read_body = lambda: {"content_parts": [{"type": "text", "text": "hi"}]}
        with patch.dict(sys.modules, _fake_grip_runtime(lambda *a, **k: r)):
            h._api_infer()
        data, code = h._captured[-1]
        self.assertEqual(code, 200)
        self.assertIs(data["media_attached"], False)           # text is not media

    def test_version_skew_reports_unknown_not_false(self):
        """An older result object without the field -> None (UNKNOWN), never False."""
        r = types.SimpleNamespace(
            ok=True, text="hi", provider="p", model="m",
            idr_ref=None, error_code=None, error_message=None)
        h = _handler()
        h._read_body = lambda: {"prompt": "hi"}
        with patch.dict(sys.modules, _fake_grip_runtime(lambda *a, **k: r)):
            h._api_infer()
        data, code = h._captured[-1]
        self.assertEqual(code, 200)
        self.assertIsNone(data["media_attached"])              # skew -> unknown

    def test_route_registered(self):
        src = open(os.path.join(os.path.dirname(__file__), "..",
                                "lib", "hal", "server.py"),
                   encoding="utf-8").read()
        self.assertIn('"/api/infer": self._api_infer', src)

    def test_max_tokens_threads_to_grip_infer(self):
        """HAL#648 ask 2: a positive int max_tokens reaches grip_infer."""
        seen = {}

        def gi(prompt, *, model=None, audit=False, media=None, content_parts=None, no_failover=False, max_tokens=None):
            seen["max_tokens"] = max_tokens
            return types.SimpleNamespace(
                ok=True, text="A", provider="p", model=model,
                idr_ref=None, error_code=None, error_message=None)
        h = _handler()
        h._read_body = lambda: {"prompt": "q", "max_tokens": 64000}
        with patch.dict(sys.modules, _fake_grip_runtime(gi)):
            h._api_infer()
        self.assertEqual(seen["max_tokens"], 64000)

    def test_bogus_max_tokens_does_not_thread(self):
        """A bool/zero/negative max_tokens is dropped, never a fabricated budget."""
        for bogus in (True, 0, -5, "64000", 3.5):
            seen = {}

            def gi(prompt, *, model=None, audit=False, media=None, content_parts=None, no_failover=False, max_tokens=None):
                seen["max_tokens"] = max_tokens
                return types.SimpleNamespace(
                    ok=True, text="A", provider="p", model=model,
                    idr_ref=None, error_code=None, error_message=None)
            h = _handler()
            h._read_body = lambda: {"prompt": "q", "max_tokens": bogus}
            with patch.dict(sys.modules, _fake_grip_runtime(gi)):
                h._api_infer()
            self.assertIsNone(seen["max_tokens"], f"bogus max_tokens={bogus!r}")

    def test_finish_reason_passes_through(self):
        """HAL#648 ask 1: the provider's finish_reason reaches the response."""
        r = types.SimpleNamespace(
            ok=False, text="", provider="deepseek", model="deepseek-v4-pro",
            idr_ref=None, error_code="answer_budget_exhausted",
            error_message="budget ran out", media_attached=False,
            finish_reason="length")
        h = _handler()
        h._read_body = lambda: {"prompt": "hi"}
        with patch.dict(sys.modules, _fake_grip_runtime(lambda *a, **k: r)):
            h._api_infer()
        data, code = h._captured[-1]
        self.assertEqual(code, 200)
        self.assertEqual(data["ok"], False)
        self.assertEqual(data["error_code"], "answer_budget_exhausted")
        self.assertEqual(data["finish_reason"], "length")
        self.assertEqual(set(data.keys()), _BRIDGE_KEYS)


if __name__ == "__main__":
    unittest.main()
