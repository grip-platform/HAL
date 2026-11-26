"""Default-deny-remote access gate for the native /api/* surface.

Closes the gap where /v1/* was guarded by enforce_gateway_access but the
native /api/* routes (tool-execution, fleet control, metered inference) were
reachable UNAUTHENTICATED through the public tunnel.

Goodhart-proof: asserts the EXACT public allowlist + each forbidden path
INDIVIDUALLY + the real gateway_guard composition + that the gate is wired
into every dispatch method. Each test fails under the obvious mutation:

  - a control/spend route added to a public set   -> test_*_exact_allowlist
  - a forbidden route flips to public             -> test_forbidden_paths_gated
  - the demo route stops being public             -> test_public_paths_allowed
  - remote deny semantics break                   -> test_remote_* composition
  - a gate removed from do_GET/do_POST/do_DELETE  -> test_gate_wired_into_*
"""
import os
import unittest
from types import SimpleNamespace

from lib.hal.gateway_guard import gateway_access_allowed
from lib.hal.server import (
    _PUBLIC_GET_PATHS,
    _PUBLIC_POST_PATHS,
    _is_public_path,
)

# The dangerous native surface — must ALL be gated (default-deny-remote).
# Hardcoded here, not derived from the route table, so a mutation that widens
# the public set is caught by an independent source of truth.
_FORBIDDEN = [
    ("POST", "/api/tools/execute"),   # P0 — tool execution / RCE
    ("POST", "/api/query"),           # metered provider spend
    ("POST", "/api/council"),         # 5x metered fan-out
    ("POST", "/api/infer"),           # CCH subscription
    ("POST", "/api/infer-with-tools"),
    ("POST", "/api/agents"),          # fleet: create agent
    ("POST", "/api/tasks"),           # fleet: submit task
    ("POST", "/api/environments"),
    ("POST", "/api/sessions"),
    ("POST", "/api/conversation"),
    ("POST", "/api/tools/stream"),
    ("GET", "/api/fleet"),            # info disclosure: spend telemetry
    ("GET", "/api/status"),
    ("GET", "/api/agents"),           # info disclosure: system prompts
    ("GET", "/api/providers"),
    ("GET", "/api/tools"),
    ("GET", "/api/sessions"),
    ("GET", "/api/chat"),             # /api/chat is POST-only-public; GET gated
    ("DELETE", "/api/sessions/abc-123"),
    ("DELETE", "/api/agents/deadbeef"),
    ("DELETE", "/api/environments/x1"),
]

_PUBLIC = [
    ("POST", "/api/chat"),            # the public chat demo
    ("POST", "/api/chat?ts=1"),       # query string stripped before match
    ("GET", "/"),
    ("GET", "/index.html"),
    ("GET", "/api/health"),           # liveness probe
    ("GET", "/assets/app.123.js"),    # static UI dist asset
    ("GET", "/favicon.ico"),
]


class TestPublicAllowlistExact(unittest.TestCase):
    """The allowlists are frozen — widening them must be a deliberate edit
    that fails these equality assertions first."""

    def test_post_allowlist_is_exactly_api_chat(self):
        self.assertEqual(_PUBLIC_POST_PATHS, frozenset({"/api/chat"}))

    def test_get_allowlist_is_exactly_ui_and_health(self):
        self.assertEqual(
            _PUBLIC_GET_PATHS,
            frozenset({"/", "/index.html", "/api/health"}),
        )

    def test_no_control_route_in_post_allowlist(self):
        for danger in ("/api/tools/execute", "/api/query", "/api/council",
                       "/api/agents", "/api/tasks", "/api/infer"):
            self.assertNotIn(danger, _PUBLIC_POST_PATHS)


class TestIsPublicPath(unittest.TestCase):

    def test_public_paths_allowed(self):
        for method, path in _PUBLIC:
            with self.subTest(method=method, path=path):
                self.assertTrue(
                    _is_public_path(method, path),
                    f"{method} {path} should be remote-public",
                )

    def test_forbidden_paths_gated(self):
        # Each forbidden path asserted individually — a mutation that opens
        # any ONE of them fails here.
        for method, path in _FORBIDDEN:
            with self.subTest(method=method, path=path):
                self.assertFalse(
                    _is_public_path(method, path),
                    f"{method} {path} MUST be gated (default-deny-remote)",
                )

    def test_query_string_does_not_smuggle_a_gated_path(self):
        # Stripping the query string must not turn a gated path public, nor
        # a public path gated.
        self.assertFalse(_is_public_path("POST", "/api/tools/execute?x=1"))
        self.assertTrue(_is_public_path("POST", "/api/chat?model=opus"))

    def test_unknown_method_is_gated(self):
        # PUT/PATCH/etc. are not served and must never be treated as public.
        self.assertFalse(_is_public_path("PUT", "/api/chat"))
        self.assertFalse(_is_public_path("PATCH", "/"))


class TestRemoteAccessComposition(unittest.TestCase):
    """The handler decision is: public OR gateway_access_allowed. Exercise the
    REAL gateway_guard so the composition (not a re-implementation) is tested."""

    # Both connect from loopback; the tunnel also supplies a proxy header.
    REMOTE = {"cf-ray": "fake-proxy-ray"}
    LOCAL = {}

    def _decide(self, method, path, headers, token=None):
        handler = SimpleNamespace(client_address=("127.0.0.1", 12345))
        return _is_public_path(method, path) or gateway_access_allowed(
            headers, token=token, handler=handler
        )

    def test_remote_unauth_tools_execute_denied(self):
        # The P0: remote caller, no token -> DENIED.
        self.assertFalse(
            self._decide("POST", "/api/tools/execute", self.REMOTE, token="")
        )

    def test_remote_unauth_query_denied(self):
        self.assertFalse(
            self._decide("POST", "/api/query", self.REMOTE, token="fake-account-A-token")
        )

    def test_remote_chat_demo_allowed(self):
        # The public demo stays reachable remotely with no token.
        self.assertTrue(
            self._decide("POST", "/api/chat", self.REMOTE, token="")
        )

    def test_loopback_tools_execute_allowed(self):
        # Local flows (GRIP/HAL on the host) are unaffected.
        self.assertTrue(
            self._decide("POST", "/api/tools/execute", self.LOCAL, token="")
        )

    def test_remote_with_correct_token_allowed(self):
        headers = {**self.REMOTE, "x-hal-gateway-token": "fake-account-A-token"}
        self.assertTrue(
            self._decide("POST", "/api/tools/execute", headers, token="fake-account-A-token")
        )

    def test_remote_with_wrong_token_denied(self):
        headers = {**self.REMOTE, "x-hal-gateway-token": "fake-account-B-token"}
        self.assertFalse(
            self._decide("POST", "/api/tools/execute", headers, token="fake-account-A-token")
        )


class TestGateWiredIntoDispatch(unittest.TestCase):
    """Source-level wiring anchors. If a gate is deleted from a dispatch
    method, the corresponding assertion fails (the policy function existing
    is useless if it is never called)."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(
            os.path.dirname(__file__), "..", "lib", "hal", "server.py"
        )
        with open(path, encoding="utf-8") as fh:
            cls.src = fh.read()

    def test_gate_wired_into_do_get(self):
        self.assertIn('_is_public_path("GET", self.path)', self.src)

    def test_gate_wired_into_do_post(self):
        self.assertIn('_is_public_path("POST", self.path)', self.src)

    def test_gate_wired_into_do_delete(self):
        # do_DELETE has no public route -> a bare enforce_gateway_access guard.
        idx = self.src.index("def do_DELETE(self):")
        body = self.src[idx:idx + 300]
        self.assertIn("enforce_gateway_access(self)", body)

    def test_api_health_redacts_path_for_non_loopback_callers(self):
        # /api/health is public to REMOTE callers (see _PUBLIC_GET_PATHS
        # above), so unlike every other route here it must narrow its body
        # rather than gate the request outright. If a future edit drops the
        # reveal_path wiring, this anchor fails even though the endpoint
        # keeps answering 200 — the leak would otherwise be invisible to
        # every test in this file, which only asserts on reachability.
        idx = self.src.index("def _api_health(self):")
        body = self.src[idx:idx + 900]
        self.assertIn("is_loopback_caller(self)", body)
        self.assertIn("reveal_path=", body)


if __name__ == "__main__":
    unittest.main()
