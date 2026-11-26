"""HAL Gateway P3 — failover + audit tests. Goodhart-proof.

Each test fails under the obvious mutation:

  - failover not triggered on 429             -> test_failover_primary_429_to_secondary
  - failover not triggered on unreachable     -> test_failover_on_primary_unreachable
  - single upstream silently swallows 429      -> test_single_upstream_relays_429
  - last upstream error synthesised not relayed-> test_last_upstream_status_relayed
  - secondary called when primary succeeds     -> test_primary_success_skips_secondary
  - fallback credential not applied, or sent
    beside the client's own                    -> test_fallback_credential_override
  - failover-class response not closed          -> test_failed_over_response_is_closed
  - telemetry failure breaks the request        -> test_telemetry_failure_never_breaks
  - proxy not failover-aware                    -> test_failover_proxy_wired_*

Failover is same-format only by spec design (a /v1/messages fallback is another
Anthropic credential, never a cross-provider swap that would break P1's
thinking-block signature guarantee). Pure logic; fake proxies injected via
proxy_factory (DIP) — zero network.
"""
import os
import unittest

from lib.hal.gateway import DEFAULT_UPSTREAM, GatewayUpstreamError
from lib.hal.gateway_failover import (
    FailoverProxy,
    GatewayTelemetry,
    UpstreamSpec,
    _merge_headers,
    _specs_from_env,
    build_failover_proxy,
)

_HERE = os.path.dirname(__file__)
_SERVER_SRC = os.path.join(_HERE, "..", "lib", "hal", "server.py")
_GATEWAY_SRC = os.path.join(_HERE, "..", "lib", "hal", "gateway.py")


class FakeResp:
    """Stand-in for UpstreamResponse — status + close tracking."""

    def __init__(self, status, body=b"{}"):
        self.status = status
        self._body = body
        self.closed = False

    def read_all(self):
        return self._body

    def iter_chunks(self, size=65536):
        if self._body:
            yield self._body

    def close(self):
        self.closed = True


class FakeProxy:
    """Returns a scripted result (FakeResp or raises an Exception); records calls."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def forward(self, path, body, client_headers, *, method="POST", max_retries=None):
        self.calls.append({"path": path, "headers": list(client_headers), "method": method})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _fp(specs, fakes, telemetry=None):
    return FailoverProxy(specs, proxy_factory=lambda s: fakes[s.name], telemetry=telemetry)


class TestFailover(unittest.TestCase):

    def test_failover_primary_429_to_secondary(self):
        fakes = {"a": FakeProxy(FakeResp(429)), "b": FakeProxy(FakeResp(200, b'{"ok":1}'))}
        events = []
        fp = _fp([UpstreamSpec("a"), UpstreamSpec("b")], fakes,
                 GatewayTelemetry(sink=events.append))
        resp = fp.forward("/v1/messages", b"{}", [("x-api-key", "k")])
        self.assertEqual(resp.status, 200)               # served by secondary
        self.assertEqual(resp.read_all(), b'{"ok":1}')
        self.assertEqual(len(fakes["a"].calls), 1)       # primary tried
        self.assertEqual(len(fakes["b"].calls), 1)       # then secondary
        served = [e for e in events if e["served"]]
        self.assertEqual(served[-1]["upstream"], "b")
        self.assertTrue(served[-1]["failed_over"])

    def test_failover_on_primary_unreachable(self):
        fakes = {"a": FakeProxy(GatewayUpstreamError("down")), "b": FakeProxy(FakeResp(200))}
        fp = _fp([UpstreamSpec("a"), UpstreamSpec("b")], fakes)
        self.assertEqual(fp.forward("/v1/messages", b"{}", []).status, 200)

    def test_single_upstream_relays_429(self):
        # Backward compat: one upstream behaves like a bare GatewayProxy — the
        # 429 is relayed (harness sees the real rate-limit), not failed over.
        fakes = {"a": FakeProxy(FakeResp(429))}
        fp = _fp([UpstreamSpec("a")], fakes)
        resp = fp.forward("/v1/messages", b"{}", [])
        self.assertEqual(resp.status, 429)
        self.assertFalse(fakes["a"].result.closed)       # returned, not failover-closed

    def test_last_upstream_status_relayed(self):
        # Chain exhausted on status: the LAST upstream's 429 is relayed, never
        # synthesised into a different error.
        fakes = {"a": FakeProxy(FakeResp(429)), "b": FakeProxy(FakeResp(429))}
        fp = _fp([UpstreamSpec("a"), UpstreamSpec("b")], fakes)
        resp = fp.forward("/v1/messages", b"{}", [])
        self.assertEqual(resp.status, 429)
        self.assertTrue(fakes["a"].result.closed)        # primary closed on failover

    def test_all_upstreams_unreachable_raises(self):
        fakes = {"a": FakeProxy(GatewayUpstreamError("d1")),
                 "b": FakeProxy(GatewayUpstreamError("d2"))}
        fp = _fp([UpstreamSpec("a"), UpstreamSpec("b")], fakes)
        with self.assertRaises(GatewayUpstreamError):
            fp.forward("/v1/messages", b"{}", [])

    def test_primary_success_skips_secondary(self):
        fakes = {"a": FakeProxy(FakeResp(200)), "b": FakeProxy(FakeResp(200))}
        fp = _fp([UpstreamSpec("a"), UpstreamSpec("b")], fakes)
        fp.forward("/v1/messages", b"{}", [])
        self.assertEqual(len(fakes["a"].calls), 1)
        self.assertEqual(fakes["b"].calls, [], "secondary must not be called on primary success")

    def test_fallback_credential_override(self):
        # Inverted 2026-10-05 (Step 5b, one auth scheme): this test used to inject
        # the fallback as x-api-key, which replaced only a same-named client
        # header, so a client's own Authorization rode along beside it. The
        # fallback's credential now travels alone, as Authorization, and the
        # client's credential -- either scheme -- never reaches the fallback.
        fakes = {"a": FakeProxy(FakeResp(429)), "b": FakeProxy(FakeResp(200))}
        fp = _fp([UpstreamSpec("a"),
                  UpstreamSpec("b", header_overrides=(("authorization", "Bearer fallback-key"),))],
                 fakes)
        fp.forward("/v1/messages", b"{}", [("x-api-key", "primary-key")])
        self.assertEqual(dict(fakes["a"].calls[0]["headers"])["x-api-key"], "primary-key")
        sent = {k.lower(): v for k, v in fakes["b"].calls[0]["headers"]}
        self.assertEqual(sent.get("authorization"), "Bearer fallback-key")
        self.assertNotIn("x-api-key", sent, "the client's key must not reach the fallback")

    def test_failed_over_response_is_closed(self):
        fakes = {"a": FakeProxy(FakeResp(503)), "b": FakeProxy(FakeResp(200))}
        fp = _fp([UpstreamSpec("a"), UpstreamSpec("b")], fakes)
        fp.forward("/v1/messages", b"{}", [])
        self.assertTrue(fakes["a"].result.closed, "failed-over response must be closed (no leak)")

    def test_telemetry_failure_never_breaks(self):
        def boom(_event):
            raise IOError("disk full")
        fakes = {"a": FakeProxy(FakeResp(200))}
        fp = _fp([UpstreamSpec("a")], fakes, GatewayTelemetry(sink=boom))
        self.assertEqual(fp.forward("/v1/messages", b"{}", []).status, 200)


class TestHelpers(unittest.TestCase):

    def test_merge_headers_overrides_win_dedup(self):
        merged = _merge_headers(
            [("X-Api-Key", "old"), ("Content-Type", "application/json")],
            [("x-api-key", "new")])
        names = [k.lower() for k, _ in merged]
        self.assertEqual(names.count("x-api-key"), 1, "must de-dup case-insensitively")
        d = {k.lower(): v for k, v in merged}
        self.assertEqual(d["x-api-key"], "new")
        self.assertEqual(d["content-type"], "application/json")

    def test_specs_from_env_ignores_the_removed_fallback_chain(self):
        """Inverted 2026-10-05 (spend guard, Step 5h). This test used to require
        that HAL_GATEWAY_FALLBACK_KEY appended a second upstream carrying that
        key as ``x-api-key``: a paid key chosen by an env var, never asking the
        spend guard. The rung is removed, so setting both variables must leave
        the one primary upstream, with no credential override."""
        keys = ("HAL_GATEWAY_FALLBACK_UPSTREAM", "HAL_GATEWAY_FALLBACK_KEY")
        saved = {k: os.environ.pop(k, None) for k in keys}
        try:
            self.assertEqual(len(_specs_from_env()), 1)   # default: single upstream
            os.environ["HAL_GATEWAY_FALLBACK_KEY"] = "fake-fallback-key"
            os.environ["HAL_GATEWAY_FALLBACK_UPSTREAM"] = "https://mirror.example"
            specs = _specs_from_env()
            self.assertEqual([s.name for s in specs], ["anthropic-primary"])
            self.assertEqual(specs[0].header_overrides, ())
            self.assertEqual(specs[0].base_url, DEFAULT_UPSTREAM)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)

    def test_empty_specs_rejected(self):
        with self.assertRaises(ValueError):
            FailoverProxy([])

    def test_build_failover_proxy_default_single(self):
        keys = ("HAL_GATEWAY_FALLBACK_UPSTREAM", "HAL_GATEWAY_FALLBACK_KEY")
        saved = {k: os.environ.pop(k, None) for k in keys}
        try:
            fp = build_failover_proxy()
            self.assertIsInstance(fp, FailoverProxy)
            self.assertEqual(len(fp._specs), 1)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


class TestWiring(unittest.TestCase):

    def test_failover_proxy_wired_in_server(self):
        src = open(_SERVER_SRC, encoding="utf-8").read()
        self.assertIn("build_failover_proxy", src)

    def test_failover_proxy_wired_in_standalone_gateway(self):
        src = open(_GATEWAY_SRC, encoding="utf-8").read()
        self.assertIn("build_failover_proxy", src)


if __name__ == "__main__":
    unittest.main()
