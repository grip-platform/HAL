"""HAL ``/api/infer-with-tools`` route — W3 server wiring of W1+W2.

Goodhart-proof: asserts the EXACT response shape, capability-lock
behaviour, fail-safe degradation, and route registration. Each test
fails under one obvious code mutation; the docstring names which.

Test register:

| ID  | Mutation it survives                                            |
|-----|-----------------------------------------------------------------|
| T17 | Drop route from routes dict → /api/infer-with-tools 404         |
| T18 | Drop missing-prompt 400 → silent ok=false                       |
| T19 | Drop missing-sandbox-root 400 → silent ok=false                 |
| T20 | Drop SandboxRootRefused handling → 5xx instead of fail-safe     |
| T21 | Drop CCH None-check → AttributeError instead of fail-safe       |
| T22 | Drop tools-allowlist filter → Bash schema reaches the model     |
| T23 | Drop final-text extraction → empty text on happy path           |
| T24 | Drop tool_calls summary projection → loop's raw dict leaks      |
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.hal.server import _make_handler  # noqa: E402

# Exact public response shape callers depend on. A diff here = breaking
# contract change.
_RESPONSE_KEYS = {
    "ok", "text", "tool_calls", "iterations",
    "hit_iteration_cap", "error_code", "error_message",
}


def _handler():
    """Real Handler class from the factory; instance without socket init."""
    H = _make_handler(MagicMock(), MagicMock(), MagicMock(),
                      MagicMock(), MagicMock())
    h = H.__new__(H)
    h._captured = []
    h._json = lambda data, code=200: h._captured.append((data, code))
    return h


class _TmpRoot(unittest.TestCase):
    """Per-test temp sandbox root, hermetic."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


def _fake_cch(text="ok", tool_use_blocks=None, stop_reason="end_turn"):
    """Return a fake CCHSubscription whose .chat() yields scripted responses.

    If ``tool_use_blocks`` is non-empty, the FIRST call returns them with
    stop_reason="tool_use"; subsequent calls return text+end_turn.
    """
    sub = MagicMock()
    call_count = {"n": 0}

    def chat_fn(prompt, *, model, messages=None, system=None,
                temperature=None, max_tokens=2048, tools=None):
        call_count["n"] += 1
        if tool_use_blocks and call_count["n"] == 1:
            raw = {"content": list(tool_use_blocks),
                   "stop_reason": "tool_use"}
        else:
            raw = {"content": [{"type": "text", "text": text}],
                   "stop_reason": stop_reason}
        return types.SimpleNamespace(
            text=text, raw=raw, model=model, provider="anthropic",
            finish_reason=stop_reason,
        )

    sub.chat = MagicMock(side_effect=chat_fn)
    return sub


# ---------------------------------------------------------------------------
# T17 — Route registered
# ---------------------------------------------------------------------------


class TestRouteRegistered(unittest.TestCase):

    def test_route_in_routes_dict(self):
        """T17 — Mutation: drop the route entry from the routes dict
        → /api/infer-with-tools 404. Survives by grepping the source."""
        src = open(_REPO_ROOT / "lib" / "hal" / "server.py",
                   encoding="utf-8").read()
        self.assertIn(
            '"/api/infer-with-tools": self._api_infer_with_tools', src,
        )

    def test_method_exists(self):
        """T17b — Mutation: rename the method without updating the
        route → AttributeError at request time. Survives via attr check."""
        H = _make_handler(MagicMock(), MagicMock(), MagicMock(),
                          MagicMock(), MagicMock())
        self.assertTrue(hasattr(H, "_api_infer_with_tools"))


# ---------------------------------------------------------------------------
# T18/T19 — Input validation (HTTP 400 on missing required args)
# ---------------------------------------------------------------------------


class TestInputValidation(_TmpRoot):

    def test_missing_prompt_400(self):
        """T18 — Mutation: drop the empty-prompt 400 → falls through to
        ok=false on a downstream call. Survives by asserting HTTP 400."""
        h = _handler()
        h._read_body = lambda: {"sandbox_root": self.root}
        h._api_infer_with_tools()
        data, code = h._captured[-1]
        self.assertEqual(code, 400)
        self.assertIn("error", data)

    def test_missing_sandbox_root_400(self):
        """T19 — Mutation: drop the empty-sandbox-root 400 → silent
        ok=false. Survives by asserting HTTP 400."""
        h = _handler()
        h._read_body = lambda: {"prompt": "hi"}
        h._api_infer_with_tools()
        data, code = h._captured[-1]
        self.assertEqual(code, 400)
        self.assertIn("error", data)


# ---------------------------------------------------------------------------
# T20 — Bad sandbox_root → ok=false (not 5xx)
# ---------------------------------------------------------------------------


class TestSandboxRefusal(unittest.TestCase):

    def test_refused_root_is_failsafe_200(self):
        """T20 — Mutation: drop SandboxRootRefused handling → 5xx on
        ``sandbox_root=/etc``. Survives by asserting HTTP 200 +
        ok=false + error_code='sandbox_root_refused'."""
        if not os.path.isdir("/etc"):
            self.skipTest("/etc not present")
        h = _handler()
        h._read_body = lambda: {"prompt": "hi", "sandbox_root": "/etc"}
        h._api_infer_with_tools()
        data, code = h._captured[-1]
        self.assertEqual(code, 200)
        self.assertEqual(data["ok"], False)
        self.assertEqual(data["error_code"], "sandbox_root_refused")
        self.assertEqual(set(data.keys()), _RESPONSE_KEYS)

    def test_nonexistent_root_is_failsafe(self):
        """T20b — Same mutation, different bad-root shape."""
        h = _handler()
        h._read_body = lambda: {
            "prompt": "hi",
            "sandbox_root": "/nonexistent/abc/xyz123",
        }
        h._api_infer_with_tools()
        data, code = h._captured[-1]
        self.assertEqual(code, 200)
        self.assertEqual(data["ok"], False)
        self.assertEqual(data["error_code"], "sandbox_root_refused")


# ---------------------------------------------------------------------------
# T21 — CCH unavailable → ok=false
# ---------------------------------------------------------------------------


class TestCCHUnavailable(_TmpRoot):

    def test_cch_returns_none_is_failsafe(self):
        """T21 — Mutation: drop the None-check on CCHSubscription.from_env
        → AttributeError on .chat(). Survives by asserting ok=false +
        error_code='cch_subscription_unavailable'."""
        h = _handler()
        h._read_body = lambda: {
            "prompt": "hi", "sandbox_root": self.root,
        }
        with patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=None,
        ):
            h._api_infer_with_tools()
        data, code = h._captured[-1]
        self.assertEqual(code, 200)
        self.assertEqual(data["ok"], False)
        self.assertEqual(
            data["error_code"], "cch_subscription_unavailable",
        )
        self.assertEqual(set(data.keys()), _RESPONSE_KEYS)


# ---------------------------------------------------------------------------
# T22 — Tools list filtered to allowlist BEFORE schemas reach the model
# ---------------------------------------------------------------------------


class TestToolsAllowlistFilter(_TmpRoot):

    def test_bash_in_tools_filtered_out(self):
        """T22 — Mutation: drop the allowlist intersection → Bash schema
        is sent to the model. Survives by asserting the schemas the
        model sees only include Read."""
        h = _handler()
        h._read_body = lambda: {
            "prompt": "hi", "sandbox_root": self.root,
            "tools": ["Read", "Bash", "Write", "Edit", "WebFetch"],
        }
        sub = _fake_cch(text="done")
        with patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=sub,
        ):
            h._api_infer_with_tools()
        self.assertGreater(sub.chat.call_count, 0)
        first_kwargs = sub.chat.call_args_list[0].kwargs
        names = {s.get("name") for s in (first_kwargs.get("tools") or [])}
        self.assertEqual(names, {"Read"})

    def test_no_allowed_tools_is_failsafe(self):
        """T22b — Mutation: drop the no-allowed-tools guard → empty
        tool list reaches the loop. Survives by asserting ok=false +
        error_code='no_allowed_tools'."""
        h = _handler()
        h._read_body = lambda: {
            "prompt": "hi", "sandbox_root": self.root,
            "tools": ["Bash", "Write", "Edit"],
        }
        h._api_infer_with_tools()
        data, code = h._captured[-1]
        self.assertEqual(code, 200)
        self.assertEqual(data["ok"], False)
        self.assertEqual(data["error_code"], "no_allowed_tools")

    def test_empty_tools_list_rejected_400(self):
        """T22d (R0 forward-anchor, AGORA Council #373 concern 3) —
        Mutation: ``body.get("tools") or default`` silently widens an
        explicitly empty list to the full allowlist. Survives by
        asserting HTTP 400 on ``tools=[]`` — caller intent honoured."""
        h = _handler()
        h._read_body = lambda: {
            "prompt": "hi", "sandbox_root": self.root,
            "tools": [],
        }
        h._api_infer_with_tools()
        data, code = h._captured[-1]
        self.assertEqual(code, 400)
        self.assertIn(
            "tools must be a non-empty list",
            data.get("error", ""),
        )

    def test_tools_must_be_list_400(self):
        """T22e — Mutation: skip the isinstance(list) check → a string
        ``"Read,Grep"`` reaches the iteration and explodes. Survives by
        asserting 400 on non-list types."""
        h = _handler()
        h._read_body = lambda: {
            "prompt": "hi", "sandbox_root": self.root,
            "tools": "Read",  # string, not list
        }
        h._api_infer_with_tools()
        data, code = h._captured[-1]
        self.assertEqual(code, 400)

    def test_default_tools_is_full_allowlist(self):
        """T22c — Mutation: change default tools → operator behaviour
        silently shifts. Survives by asserting all three allowlist
        members in schemas when caller omits ``tools``."""
        h = _handler()
        h._read_body = lambda: {
            "prompt": "hi", "sandbox_root": self.root,
        }
        sub = _fake_cch(text="done")
        with patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=sub,
        ):
            h._api_infer_with_tools()
        first_kwargs = sub.chat.call_args_list[0].kwargs
        names = {s.get("name") for s in (first_kwargs.get("tools") or [])}
        self.assertEqual(names, {"Read", "Grep", "Glob"})


# ---------------------------------------------------------------------------
# T23 — Happy path round-trip
# ---------------------------------------------------------------------------


class TestHappyPath(_TmpRoot):

    def test_text_only_response_round_trips(self):
        """T23 — Mutation: drop the final-text extraction → empty text
        on happy path. Survives by asserting model text appears."""
        h = _handler()
        h._read_body = lambda: {
            "prompt": "say hi", "sandbox_root": self.root,
        }
        sub = _fake_cch(text="HI THERE")
        with patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=sub,
        ):
            h._api_infer_with_tools()
        data, code = h._captured[-1]
        self.assertEqual(code, 200)
        self.assertEqual(data["ok"], True)
        self.assertEqual(data["text"], "HI THERE")
        self.assertEqual(set(data.keys()), _RESPONSE_KEYS)

    def test_tool_use_round_trip(self):
        """T23b — Tool-use → tool-result → final-text via the real
        SandboxedDispatcher reading a real file under sandbox_root."""
        (Path(self.root) / "hello.txt").write_text("WORLD\n",
                                                    encoding="utf-8")
        h = _handler()
        h._read_body = lambda: {
            "prompt": "read hello.txt",
            "sandbox_root": self.root,
            "tools": ["Read"],
        }
        tool_use_blocks = [{
            "type": "tool_use", "id": "tu_1", "name": "Read",
            "input": {"file_path": "hello.txt"},
        }]
        sub = _fake_cch(text="the file says WORLD",
                        tool_use_blocks=tool_use_blocks)
        with patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=sub,
        ):
            h._api_infer_with_tools()
        data, code = h._captured[-1]
        self.assertEqual(code, 200)
        self.assertEqual(data["ok"], True)
        self.assertEqual(len(data["tool_calls"]), 1)
        self.assertEqual(data["tool_calls"][0]["tool"], "Read")
        self.assertFalse(data["tool_calls"][0]["is_error"])
        self.assertEqual(data["iterations"], 2)

    def test_capability_lock_blocks_bash_tool_use(self):
        """T24 — Even if the model emits a tool_use for ``Bash``, the
        SandboxedDispatcher blocks it. Survives the mutation 'remove
        dispatcher allowlist check' by asserting is_error=True and the
        loop didn't crash."""
        h = _handler()
        h._read_body = lambda: {
            "prompt": "do something",
            "sandbox_root": self.root,
        }
        evil = [{
            "type": "tool_use", "id": "tu_bash", "name": "Bash",
            "input": {"command": "rm -rf /"},
        }]
        sub = _fake_cch(text="ok stopping", tool_use_blocks=evil)
        with patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=sub,
        ):
            h._api_infer_with_tools()
        data, _ = h._captured[-1]
        self.assertEqual(data["ok"], True)
        self.assertEqual(len(data["tool_calls"]), 1)
        self.assertEqual(data["tool_calls"][0]["tool"], "Bash")
        self.assertTrue(data["tool_calls"][0]["is_error"])


# ---------------------------------------------------------------------------
# T24 — tool_calls summary shape (no internal leakage)
# ---------------------------------------------------------------------------


class TestToolCallsSummary(_TmpRoot):

    def test_summary_has_only_expected_keys(self):
        """T24 — Mutation: drop the tool_calls projection → leaks loop
        internals (raw tool_use/tool_result dicts). Survives by asserting
        the per-call keys are exactly the public set."""
        (Path(self.root) / "a.txt").write_text("x\n", encoding="utf-8")
        h = _handler()
        h._read_body = lambda: {
            "prompt": "read a.txt", "sandbox_root": self.root,
            "tools": ["Read"],
        }
        tool_use_blocks = [{
            "type": "tool_use", "id": "tu", "name": "Read",
            "input": {"file_path": "a.txt"},
        }]
        sub = _fake_cch(text="done", tool_use_blocks=tool_use_blocks)
        with patch(
            "lib.hal.subscription.CCHSubscription.from_env",
            return_value=sub,
        ):
            h._api_infer_with_tools()
        data, _ = h._captured[-1]
        self.assertEqual(len(data["tool_calls"]), 1)
        self.assertEqual(
            set(data["tool_calls"][0].keys()),
            {"tool", "input", "is_error", "iteration"},
        )


# ---------------------------------------------------------------------------
# Helper-level direct tests
# ---------------------------------------------------------------------------


class TestHelpersDirect(unittest.TestCase):

    def test_parse_body_defaults(self):
        from lib.hal.server import _iwt_parse_body
        out = _iwt_parse_body({"prompt": "x", "sandbox_root": "/tmp"})
        self.assertEqual(out["requested_tools"], ["Read", "Grep", "Glob"])
        self.assertEqual(out["max_iterations"], 8)
        self.assertEqual(out["max_tokens"], 16384)
        self.assertIsNone(out["system"])

    def test_parse_body_missing_prompt(self):
        from lib.hal.server import _iwt_parse_body
        out = _iwt_parse_body({"sandbox_root": "/tmp"})
        self.assertIn("error", out)

    def test_parse_body_missing_sandbox_root(self):
        from lib.hal.server import _iwt_parse_body
        out = _iwt_parse_body({"prompt": "x"})
        self.assertIn("error", out)

    def test_failsafe_shape(self):
        from lib.hal.server import _iwt_failsafe
        r = _iwt_failsafe("code_x", "msg")
        self.assertEqual(r["ok"], False)
        self.assertEqual(r["text"], "")
        self.assertEqual(r["tool_calls"], [])
        self.assertEqual(r["error_code"], "code_x")
        self.assertEqual(r["error_message"], "msg")
        self.assertEqual(set(r.keys()), _RESPONSE_KEYS)

    def test_failsafe_error_message_truncated_to_300(self):
        from lib.hal.server import _iwt_failsafe
        r = _iwt_failsafe("code_x", "z" * 500)
        self.assertEqual(len(r["error_message"]), 300)

    def test_extract_text_concats_text_blocks(self):
        from lib.hal.server import _iwt_extract_text
        r = {"content": [
            {"type": "text", "text": "hello "},
            {"type": "tool_use", "id": "tu", "name": "Read"},
            {"type": "text", "text": "world"},
        ]}
        self.assertEqual(_iwt_extract_text(r), "hello world")

    def test_extract_text_malformed_returns_empty(self):
        from lib.hal.server import _iwt_extract_text
        self.assertEqual(_iwt_extract_text(None), "")
        self.assertEqual(_iwt_extract_text({}), "")
        self.assertEqual(_iwt_extract_text({"content": "not a list"}), "")

    def test_summarise_tool_calls_keys_exact(self):
        from lib.hal.server import _iwt_summarise_tool_calls
        loop_log = [{
            "iteration": 0,
            "tool_use": {"id": "tu", "name": "Read",
                          "input": {"file_path": "x"}},
            "tool_result": {"type": "tool_result", "tool_use_id": "tu",
                             "content": "hi", "is_error": False},
        }]
        out = _iwt_summarise_tool_calls(loop_log)
        self.assertEqual(
            set(out[0].keys()),
            {"tool", "input", "is_error", "iteration"},
        )
        self.assertEqual(out[0]["tool"], "Read")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
