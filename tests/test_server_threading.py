"""Multi-threaded hal-server: ThreadingHTTPServer + thread-safe lazy gateway-proxy.

Re-extracted from PR #386 (superseded — main already shipped its throttle /
SessionRegistry / per-session _api_chat via other PRs; this carries only the
unique multi-threading nugget). Goodhart-proof: the concurrency test fails if
the double-checked _GATEWAY_LOCK is removed (the race would build the proxy
more than once).
"""
from __future__ import annotations

import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer
from socketserver import ThreadingMixIn

_HAL = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL not in sys.path:
    sys.path.insert(0, _HAL)

from lib.hal import server  # noqa: E402
from lib.hal.types import ChatResponse, FallbackInfo  # noqa: E402


class TestThreadingHTTPServerClass(unittest.TestCase):
    def test_is_threaded_daemon_subclass(self):
        """ThreadingHTTPServer mixes ThreadingMixIn into HTTPServer with daemon
        threads — so each request is served concurrently and Ctrl+C exits clean."""
        self.assertTrue(issubclass(server.ThreadingHTTPServer, ThreadingMixIn))
        self.assertTrue(issubclass(server.ThreadingHTTPServer, HTTPServer))
        self.assertIs(server.ThreadingHTTPServer.daemon_threads, True)


class TestGatewayProxyThreadSafety(unittest.TestCase):
    def setUp(self):
        self._orig_proxy = server._GATEWAY_PROXY
        self._orig_build = server.build_failover_proxy

    def tearDown(self):
        server._GATEWAY_PROXY = self._orig_proxy
        server.build_failover_proxy = self._orig_build

    def test_concurrent_first_calls_build_proxy_exactly_once(self):
        """N threads racing the lazy init must all receive the SAME proxy,
        built exactly once. This FAILS if the double-checked _GATEWAY_LOCK is
        removed: without it every thread observes None and builds its own.

        A 10ms build widens the race window so a missing lock reliably
        double-builds rather than getting saved by GIL scheduling luck."""
        N = 16
        build_count = {"n": 0}
        count_lock = threading.Lock()
        sentinel = object()

        def fake_build():
            with count_lock:
                build_count["n"] += 1
            time.sleep(0.01)  # widen the race window
            return sentinel

        server.build_failover_proxy = fake_build
        server._GATEWAY_PROXY = None

        barrier = threading.Barrier(N)
        results: list[object] = [None] * N

        def worker(i: int) -> None:
            barrier.wait()  # release all threads at once → force the race
            results[i] = server._get_gateway_proxy()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(
            build_count["n"], 1,
            f"gateway proxy built {build_count['n']}x — _GATEWAY_LOCK missing?")
        self.assertTrue(
            all(r is sentinel for r in results),
            "concurrent callers observed different proxy instances")

    def test_cached_after_first_build(self):
        """Once built, subsequent calls return the cached instance without
        rebuilding (the outer lock-free check)."""
        calls = {"n": 0}
        cached = object()

        def fake_build():
            calls["n"] += 1
            return cached

        server.build_failover_proxy = fake_build
        server._GATEWAY_PROXY = None

        first = server._get_gateway_proxy()
        second = server._get_gateway_proxy()
        self.assertIs(first, cached)
        self.assertIs(second, cached)
        self.assertEqual(calls["n"], 1, "proxy rebuilt on second call")

    def test_build_exception_leaves_none_and_releases_lock(self):
        """Exception safety (refutes #390 review concerns 1 + 4): if
        build_failover_proxy raises, the global stays None (the next caller
        retries — correct, not a leak) AND _GATEWAY_LOCK is released, because
        the `with` statement releases on exception. Goodhart-proof: fails if a
        later build is blocked by a still-held lock or a poisoned global."""
        def failing_build():
            raise RuntimeError("upstream config missing")

        server.build_failover_proxy = failing_build
        server._GATEWAY_PROXY = None
        with self.assertRaises(RuntimeError):
            server._get_gateway_proxy()
        # Global left safe for retry, not a half-built object.
        self.assertIsNone(server._GATEWAY_PROXY)
        # Lock was released despite the exception — a later good build succeeds.
        good = object()
        server.build_failover_proxy = lambda: good
        self.assertIs(server._get_gateway_proxy(), good)


def _fire_overlapping_chats(port: int, n_requests: int) -> list[int]:
    """Fire n_requests concurrent POST /api/chat (distinct session_ids + IPs),
    barrier-synced for maximal overlap. Returns the list of HTTP status codes."""
    results: list[int] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(n_requests)

    def fire(i: int) -> None:
        barrier.wait()  # release all at once → force maximal overlap
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            payload = json.dumps(
                {"message": f"hi {i}", "session_id": f"sess-{i}", "model": "test-model"}
            )
            conn.request(
                "POST", "/api/chat", body=payload,
                headers={"Content-Type": "application/json",
                         "CF-Connecting-IP": f"10.1.{i // 256}.{i % 256}"},
            )
            status = conn.getresponse().status
        finally:
            conn.close()
        with results_lock:
            results.append(status)

    with ThreadPoolExecutor(max_workers=n_requests) as pool:
        list(pool.map(fire, range(n_requests)))
    return results


class TestConcurrentChatEndToEnd(unittest.TestCase):
    """Full-stack assertion (AGORA #391 C4): a REAL ThreadingHTTPServer serving
    OVERLAPPING /api/chat requests with distinct session_ids. The classes above
    prove the threading + gateway-lock invariants in isolation; this proves the
    handler path under genuine concurrency — (a) the daily budget gates the FULL
    /api/chat path (reserve-then-settle, not just ChatThrottle.check) and (b) each
    admitted request gets its OWN session.

    Goodhart-proof: remove the throttle gating and all N return 200 (ok > budget);
    revert /api/chat to the shared `conversation` singleton and the registry holds
    one session, not `ok`."""

    def setUp(self):
        import lib.hal.api_throttle as api_throttle
        import lib.hal.conversation as conversation_mod
        import lib.hal.session_registry as session_registry
        from unittest.mock import patch

        # Real conversations build an identity prompt by probing local model
        # servers. These concurrency tests use fake providers and need no discovery.
        discovery = patch("lib.hal.discovery.discover_local_providers", return_value=[])
        discovery.start()
        self.addCleanup(discovery.stop)
        self._api_throttle = api_throttle
        self._conversation_mod = conversation_mod
        self._session_registry = session_registry
        self._tmp = tempfile.mkdtemp()
        self._orig_dir = session_registry._SESSIONS_DIR
        self._orig_send = conversation_mod.Conversation.send
        self._orig_instance = api_throttle._INSTANCE
        session_registry._SESSIONS_DIR = os.path.join(self._tmp, "sessions")

    def tearDown(self):
        self._session_registry._SESSIONS_DIR = self._orig_dir
        self._conversation_mod.Conversation.send = self._orig_send
        self._api_throttle._INSTANCE = self._orig_instance
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_overlapping_chat_gates_budget_and_isolates_sessions(self):
        from unittest.mock import MagicMock

        from lib.hal.api_throttle import ChatThrottle
        from lib.hal.session_registry import SessionRegistry

        def _fake_send(conv_self, message, model=None):
            # Append a real turn so per-session history (message_count) is
            # observable; return a fixed small-cost response. provider=None skips
            # _mark_active's catalogue check (unrelated here).
            conv_self._history.append(types.SimpleNamespace(role="user", content=message))
            conv_self._history.append(types.SimpleNamespace(role="assistant", content="ok"))
            return types.SimpleNamespace(
                text="ok", model="m", provider=None, latency_ms=1, cost_usd=0.0,
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )
        self._conversation_mod.Conversation.send = _fake_send

        budget, n_requests = 15, 60
        throttle = ChatThrottle(per_min=10_000, daily_budget=float(budget))
        self._api_throttle._INSTANCE = throttle

        registry = MagicMock()
        sessions = SessionRegistry(registry=registry)
        handler_cls = server._make_handler(
            registry, MagicMock(), MagicMock(), sessions, MagicMock(),
        )
        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        try:
            results = _fire_overlapping_chats(httpd.server_address[1], n_requests)
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=5)

        ok = sum(1 for s in results if s == 200)
        throttled = sum(1 for s in results if s == 429)
        self.assertEqual(ok + throttled, n_requests, f"unexpected statuses: {results}")
        self.assertGreaterEqual(ok, 1, "at least one concurrent request must succeed")
        self.assertGreaterEqual(throttled, 1, "throttle must reject the surplus under concurrency")
        self.assertLessEqual(
            ok, budget,
            f"{ok} admitted on a budget of {budget} — breaker overshot the full /api/chat path")
        live = sessions.list_sessions()
        self.assertEqual(
            len(live), ok,
            f"{len(live)} sessions for {ok} successful requests — per-session routing collapsed")
        self.assertTrue(
            all(s.message_count == 2 for s in live),
            "each session must hold exactly its own single turn (no cross-session bleed)")

    def test_fully_throttled_requests_leave_no_sessions(self):
        # AGORA #393 forward-anchor (the judge's "what I would falsify"): with
        # daily_budget=0 every /api/chat is rejected at the throttle gate BEFORE
        # _serve_chat, so no session is ever created — the throttled path cannot
        # leak empty sessions. Goodhart-proof: if the throttle moved AFTER session
        # creation, sessions would accumulate and both assertions fail.
        from unittest.mock import MagicMock

        from lib.hal.api_throttle import ChatThrottle
        from lib.hal.session_registry import SessionRegistry

        def _boom_send(conv_self, message, model=None):
            raise AssertionError("send reached despite a budget=0 throttle")
        self._conversation_mod.Conversation.send = _boom_send

        n_requests = 30
        self._api_throttle._INSTANCE = ChatThrottle(per_min=10_000, daily_budget=0.0)
        registry = MagicMock()
        sessions = SessionRegistry(registry=registry)
        handler_cls = server._make_handler(
            registry, MagicMock(), MagicMock(), sessions, MagicMock(),
        )
        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        try:
            results = _fire_overlapping_chats(httpd.server_address[1], n_requests)
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=5)

        self.assertTrue(
            all(s == 429 for s in results), f"budget=0 must 429 every request: {results}")
        self.assertEqual(
            len(sessions.list_sessions()), 0,
            "throttled requests must create no sessions (no empty-session leak)")

    def test_prepare_session_failure_on_api_chat_emits_500(self):
        # AGORA #393 r2 falsifier: if _prepare_session raises after _read_body
        # succeeds, /api/chat must surface a 500 via _serve_chat's try/except — not
        # crash the worker thread or return a silent None. Proves the per-handler
        # _json error contract holds on the core (throttled) path. The reservation
        # is released (settled stays False) so a backend blip can't leak budget.
        from unittest import mock
        from unittest.mock import MagicMock

        from lib.hal.api_throttle import ChatThrottle
        from lib.hal.session_registry import SessionRegistry

        def _boom_prepare(*a, **k):
            raise RuntimeError("session backend down")
        patcher = mock.patch.object(server, "_prepare_session", _boom_prepare)
        patcher.start()
        self.addCleanup(patcher.stop)

        self._api_throttle._INSTANCE = ChatThrottle(per_min=10_000, daily_budget=100.0)
        registry = MagicMock()
        sessions = SessionRegistry(registry=registry)
        handler_cls = server._make_handler(
            registry, MagicMock(), MagicMock(), sessions, MagicMock(),
        )
        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        try:
            results = _fire_overlapping_chats(httpd.server_address[1], 1)
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=5)

        self.assertEqual(
            results, [500],
            "a _prepare_session failure on /api/chat must surface as a 500, not a crash")

    def test_read_chat_request_emits_500_on_prepare_failure(self):
        # Fix 1: _read_chat_request must CATCH a _prepare_session failure, emit a
        # 500 via the handler's _json contract, and return None — so the streaming
        # handlers (which call it OUTSIDE their own try) get a clean error instead
        # of an uncaught exception crashing the worker thread. Unit-level so it
        # covers every caller, route-independent.
        from unittest import mock
        from unittest.mock import MagicMock

        def _boom_prepare(*a, **k):
            raise RuntimeError("session backend down")
        patcher = mock.patch.object(server, "_prepare_session", _boom_prepare)
        patcher.start()
        self.addCleanup(patcher.stop)

        captured: list = []
        handler = MagicMock()
        handler._read_body.return_value = {"message": "hi", "session_id": "s1"}
        handler._json = lambda data, code=200: captured.append((code, data))

        result = server._read_chat_request(handler, MagicMock())
        self.assertIsNone(result, "must return None after emitting the error")
        self.assertEqual(len(captured), 1, "exactly one error response emitted")
        self.assertEqual(captured[0][0], 500, "prepare failure must emit a 500")

    def test_same_session_concurrent_turns_serialize(self):
        # Fix 2: two requests sharing one session_id must serialise on the
        # per-session lock so their turns can't interleave writes to the one
        # Conversation. fake_send sleeps between the user+assistant appends to
        # widen the window; WITHOUT the lock the history interleaves
        # (user,user,...,assistant,assistant) and the strict pairing check fails.
        from unittest.mock import MagicMock

        from lib.hal.api_throttle import ChatThrottle
        from lib.hal.session_registry import SessionRegistry

        def _fake_send(conv_self, message, model=None):
            conv_self._history.append(types.SimpleNamespace(role="user", content=message))
            time.sleep(0.02)  # widen the interleave window
            conv_self._history.append(types.SimpleNamespace(role="assistant", content="ok"))
            return types.SimpleNamespace(
                text="ok", model="m", provider=None, latency_ms=1, cost_usd=0.0,
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )
        self._conversation_mod.Conversation.send = _fake_send

        self._api_throttle._INSTANCE = ChatThrottle(per_min=10_000, daily_budget=1000.0)
        registry = MagicMock()
        sessions = SessionRegistry(registry=registry)
        sid = sessions.create(model="test-model")  # one existing session (real uuid)
        handler_cls = server._make_handler(
            registry, MagicMock(), MagicMock(), sessions, MagicMock(),
        )
        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()

        n = 8
        port = httpd.server_address[1]
        barrier = threading.Barrier(n)
        statuses: list[int] = []
        statuses_lock = threading.Lock()

        def fire(i: int) -> None:
            barrier.wait()
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                conn.request(
                    "POST", "/api/chat",
                    body=json.dumps({"message": f"hi {i}", "session_id": sid, "model": "test-model"}),
                    headers={"Content-Type": "application/json", "CF-Connecting-IP": f"10.2.0.{i}"},
                )
                st = conn.getresponse().status
            finally:
                conn.close()
            with statuses_lock:
                statuses.append(st)

        try:
            with ThreadPoolExecutor(max_workers=n) as pool:
                list(pool.map(fire, range(n)))
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=5)

        self.assertTrue(all(s == 200 for s in statuses), f"same-session turns should all 200: {statuses}")
        roles = [m.role for m in sessions.get(sid).conversation.history]
        self.assertEqual(len(roles), 2 * n, f"expected {2 * n} history entries, got {len(roles)}")
        for i in range(0, len(roles), 2):
            self.assertEqual(roles[i], "user", f"position {i} must be 'user' (turns interleaved?)")
            self.assertEqual(roles[i + 1], "assistant", f"position {i + 1} must be 'assistant'")

    def test_uncatalogued_model_does_not_crash_chat(self):
        # Fix 3 + AGORA #394 N1: _mark_active is cosmetic — it must NOT crash the
        # request thread for ANY exception class. set_active is stubbed to raise a
        # custom BaseException subclass (NOT Exception, NOT SystemExit), which
        # anchors the broad `except BaseException`; reverting to a narrower except
        # lets _Boom escape and crash the worker. With the fix the turn returns 200.
        from unittest import mock
        from unittest.mock import MagicMock

        from lib.hal.api_throttle import ChatThrottle
        from lib.hal.session_registry import SessionRegistry

        class _Boom(BaseException):
            pass

        class _BoomActive:
            def set_active(self, address, reason=None):
                raise _Boom(f"uncatalogued model: {address}")
        patcher = mock.patch.object(server, "_grip_active_llm", lambda: _BoomActive())
        patcher.start()
        self.addCleanup(patcher.stop)

        def _fake_send(conv_self, message, model=None):
            conv_self._history.append(types.SimpleNamespace(role="user", content=message))
            conv_self._history.append(types.SimpleNamespace(role="assistant", content="ok"))
            return types.SimpleNamespace(
                text="ok", model="m", provider="p", latency_ms=1, cost_usd=0.0,
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )
        self._conversation_mod.Conversation.send = _fake_send

        self._api_throttle._INSTANCE = ChatThrottle(per_min=10_000, daily_budget=1000.0)
        registry = MagicMock()
        sessions = SessionRegistry(registry=registry)
        handler_cls = server._make_handler(
            registry, MagicMock(), MagicMock(), sessions, MagicMock(),
        )
        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        try:
            results = _fire_overlapping_chats(httpd.server_address[1], 1)
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=5)

        self.assertEqual(
            results, [200],
            "uncatalogued model must not crash the request (BaseException from _mark_active swallowed)")

    def test_session_lock_shared_across_copy_and_replace(self):
        # AGORA #394 N3 refutation: default_factory fires ONLY on fresh
        # construction. dataclasses.replace passes existing init-field values and
        # copy.copy is shallow, so both SHARE the original per-session lock — they
        # do NOT mint a fresh one, and the serialisation invariant survives a
        # clone. (No code currently clones a session; this anchors the behaviour
        # against the reviewer's "fresh lock" concern.)
        import copy
        import dataclasses
        from unittest.mock import MagicMock

        from lib.hal.session_registry import SessionRegistry

        sessions = SessionRegistry(registry=MagicMock())
        session = sessions.get(sessions.create(model="test-model"))

        self.assertIs(
            copy.copy(session).lock, session.lock,
            "copy.copy must share the per-session lock (shallow copy)")
        self.assertIs(
            dataclasses.replace(session, title="renamed").lock, session.lock,
            "dataclasses.replace must carry the existing lock, not re-default it")

    def test_mark_active_reraises_interrupts_but_swallows_others(self):
        # AGORA #394 r2 N1: _mark_active must swallow ordinary errors AND the
        # catalogue SystemExit (cosmetic write — never crash a request), but must
        # NOT eat a genuine KeyboardInterrupt/GeneratorExit. Anchors the re-raise
        # guard: remove it and the KeyboardInterrupt case is swallowed (assertRaises
        # fails); narrow to `except Exception` and the SystemExit case propagates
        # (the swallow loop fails).
        from unittest import mock

        class _Active:
            def __init__(self, exc):
                self._exc = exc

            def set_active(self, address, reason=None):
                raise self._exc

        for exc in (RuntimeError("boom"), SystemExit("catalogue")):
            with mock.patch.object(server, "_grip_active_llm", lambda e=exc: _Active(e)):
                server._mark_active("p/m")  # swallowed — must not raise

        with mock.patch.object(server, "_grip_active_llm", lambda: _Active(KeyboardInterrupt())):
            with self.assertRaises(KeyboardInterrupt):
                server._mark_active("p/m")


class TestFallbackReason(unittest.TestCase):
    """Anchor the status-line "why" consumption: response.reason (Router
    strategy) wins over fallback_info (mid-dispatch swap), and _mark_active
    passes it through to set_active. Added for the AGORA round-2 review of
    PR #763 (the consumer path had no direct test)."""

    def test_reason_wins_when_stamped(self):
        resp = ChatResponse(
            text="ok", model="m", provider="p", reason="quality-cost:simple",
        )
        self.assertEqual(server._fallback_reason(resp), "quality-cost:simple")

    def test_falls_back_to_fallback_info_when_no_reason(self):
        fi = FallbackInfo(
            requested_provider="p1", requested_model="m1",
            actual_provider="p2", actual_model="m2",
        )
        resp = ChatResponse(text="ok", model="m2", provider="p2", fallback_info=fi)
        self.assertEqual(server._fallback_reason(resp), "fallback: p1/m1→p2/m2")

    def test_empty_for_clean_direct_response(self):
        resp = ChatResponse(text="ok", model="m", provider="p")
        self.assertEqual(server._fallback_reason(resp), "")

    def test_mark_active_passes_reason_to_set_active(self):
        from unittest import mock

        calls = {}

        class _Active:
            def set_active(self, address, reason=None):
                calls["address"] = address
                calls["reason"] = reason

        with mock.patch.object(server, "_grip_active_llm", lambda: _Active()):
            server._mark_active("p/m", reason="quality-cost:simple")

        self.assertEqual(calls["address"], "p/m")
        self.assertEqual(calls["reason"], "quality-cost:simple")


if __name__ == "__main__":
    unittest.main()
