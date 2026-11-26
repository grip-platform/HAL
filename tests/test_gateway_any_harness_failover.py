"""Any-harness failover regression test.

Scenario: an external harness (Codex, Aider, opencode, relay ``claude -p``)
points ``ANTHROPIC_BASE_URL=http://localhost:4010`` at the HAL gateway.
Anthropic throttles (HTTP 429). The gateway must serve the response from the
next upstream in the same-format failover chain — transparently, without the
harness seeing a 429.

Goodhart-proof mutations each test defends against:
  - primary 429 NOT triggering failover on /v1/messages    -> test_messages_429_fails_over
  - failover NOT transparent to handle_proxy_request       -> test_handle_proxy_request_429_fails_over
  - primary 529 (Anthropic overload) NOT triggering        -> test_messages_529_overload_fails_over
  - secondary also 429, harness sees the real status       -> test_all_upstreams_exhausted_relays_last_status
  - primary 200 incorrectly engaging failover              -> test_primary_200_skips_secondary
  - failover wired into the standalone serve() entry-point -> test_gateway_serve_uses_failover_proxy

Zero network: opener injected via DIP throughout.
"""
import io
import unittest

from lib.hal.gateway import GatewayUpstreamError, UpstreamResponse, handle_proxy_request
from lib.hal.gateway_failover import FailoverProxy, GatewayTelemetry, UpstreamSpec


# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------

class _FakeResp:
    """Minimal UpstreamResponse stand-in."""
    def __init__(self, status: int, body: bytes = b'{"id":"msg_fake"}'):
        self.status = status
        self._body = body
        self.closed = False
        self.headers = []          # gateway.response_headers expects an iterable

    def iter_chunks(self, size=65536):
        if self._body:
            yield self._body

    def read_all(self) -> bytes:
        return self._body

    def close(self):
        self.closed = True


class _FakeProxy:
    """Scripted GatewayProxy duck-type; records every forward() call."""
    def __init__(self, result):
        self.result = result   # _FakeResp OR Exception instance
        self.calls = []

    def forward(self, path, body, client_headers, *, method="POST", max_retries=None):
        self.calls.append(path)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _FakeHandler:
    """Minimal BaseHTTPRequestHandler duck-type for handle_proxy_request."""
    def __init__(self, body: bytes = b"{}"):
        self.command = "POST"
        self.headers = {"Content-Length": str(len(body))}
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self._response_code = None
        self._sent_headers = {}

    def send_response(self, code):
        self._response_code = code

    def send_header(self, name, value):
        self._sent_headers[name] = value

    def end_headers(self):
        pass


def _make_fp(specs_and_fakes, telemetry=None):
    """Build a FailoverProxy injecting fake sub-proxies by spec name."""
    fakes = {name: proxy for name, proxy in specs_and_fakes}
    specs = [UpstreamSpec(name) for name, _ in specs_and_fakes]
    return FailoverProxy(specs, proxy_factory=lambda s: fakes[s.name],
                         telemetry=telemetry)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnyHarnessFailover(unittest.TestCase):
    """Primary-429 → secondary-serves path through FailoverProxy."""

    def test_messages_429_fails_over(self):
        """A 429 from the primary (Anthropic throttle) must be served by the secondary."""
        primary = _FakeProxy(_FakeResp(429))
        secondary = _FakeProxy(_FakeResp(200, b'{"id":"from_secondary"}'))
        fp = _make_fp([("primary", primary), ("secondary", secondary)])

        resp = fp.forward("/v1/messages", b"{}", [("x-api-key", "tok")])

        self.assertEqual(resp.status, 200, "harness must see 200, not 429")
        self.assertEqual(resp.read_all(), b'{"id":"from_secondary"}')
        self.assertEqual(len(primary.calls), 1, "primary must have been tried")
        self.assertEqual(len(secondary.calls), 1, "secondary must have served")

    def test_messages_529_overload_fails_over(self):
        """529 (Anthropic overload) is also in _FAILOVER_STATUS — must trigger failover."""
        primary = _FakeProxy(_FakeResp(529))
        secondary = _FakeProxy(_FakeResp(200))
        fp = _make_fp([("primary", primary), ("secondary", secondary)])

        resp = fp.forward("/v1/messages", b"{}", [])

        self.assertEqual(resp.status, 200)
        self.assertEqual(len(primary.calls), 1)

    def test_handle_proxy_request_429_fails_over(self):
        """handle_proxy_request (the per-request gateway dispatch) must relay the
        secondary's 200 when the failover proxy serves it — the harness sees 200."""
        primary = _FakeProxy(_FakeResp(429))
        secondary = _FakeProxy(_FakeResp(200, b'{"ok":1}'))
        fp = _make_fp([("primary", primary), ("secondary", secondary)])

        handler = _FakeHandler(b'{"model":"claude-opus-4-5","messages":[]}')
        handle_proxy_request(handler, fp, "/v1/messages")

        self.assertEqual(handler._response_code, 200,
                         "harness handler must see 200, not 429")

    def test_all_upstreams_exhausted_relays_last_status(self):
        """When every upstream returns 429 the harness sees the last real status
        (not a gateway-synthesised error) — per FailoverProxy spec."""
        primary = _FakeProxy(_FakeResp(429))
        secondary = _FakeProxy(_FakeResp(429))
        fp = _make_fp([("primary", primary), ("secondary", secondary)])

        resp = fp.forward("/v1/messages", b"{}", [])

        self.assertEqual(resp.status, 429,
                         "last upstream's real 429 must be relayed, not 502")
        self.assertTrue(primary.result.closed, "primary response must be closed")

    def test_primary_200_skips_secondary(self):
        """A successful primary must never engage the secondary (regression: no spurious calls)."""
        primary = _FakeProxy(_FakeResp(200))
        secondary = _FakeProxy(_FakeResp(200))
        fp = _make_fp([("primary", primary), ("secondary", secondary)])

        fp.forward("/v1/messages", b"{}", [])

        self.assertEqual(secondary.calls, [], "secondary must not be called on primary success")

    def test_primary_unreachable_fails_over(self):
        """Transport failure (connection refused) triggers failover to secondary."""
        primary = _FakeProxy(GatewayUpstreamError("connection refused"))
        secondary = _FakeProxy(_FakeResp(200))
        fp = _make_fp([("primary", primary), ("secondary", secondary)])

        resp = fp.forward("/v1/messages", b"{}", [])

        self.assertEqual(resp.status, 200)
        self.assertEqual(len(secondary.calls), 1)

    def test_failover_telemetry_records_failover(self):
        """Telemetry must record failed_over=True for the secondary-served event."""
        events = []
        primary = _FakeProxy(_FakeResp(429))
        secondary = _FakeProxy(_FakeResp(200))
        fp = _make_fp([("primary", primary), ("secondary", secondary)],
                      telemetry=GatewayTelemetry(sink=events.append))

        fp.forward("/v1/messages", b"{}", [])

        served = [e for e in events if e.get("served")]
        self.assertTrue(served, "at least one served event must be recorded")
        self.assertTrue(served[-1]["failed_over"],
                        "served event must carry failed_over=True")
        self.assertEqual(served[-1]["upstream"], "secondary")


class TestGatewayEntryPointWiring(unittest.TestCase):
    """Verify the standalone serve() entry-point wires the failover proxy."""

    def test_gateway_serve_uses_failover_proxy(self):
        """gateway.py's serve() must call build_failover_proxy() — not a bare GatewayProxy."""
        import ast, os
        src_path = os.path.join(os.path.dirname(__file__), "..", "lib", "hal", "gateway.py")
        src = open(src_path, encoding="utf-8").read()
        self.assertIn("build_failover_proxy", src,
                      "serve() must use build_failover_proxy for any-harness failover")
        tree = ast.parse(src)
        # Verify the call is actually INSIDE the serve() function body
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "serve":
                body_src = ast.unparse(node)
                self.assertIn("build_failover_proxy", body_src)
                return
        self.fail("serve() function not found in gateway.py")


if __name__ == "__main__":
    unittest.main()
