"""HAL Gateway Proxy (P1) — Goodhart-proof tests. Closes GRIP #3274.

Every test fails under the obvious mutation:

  - reserialise/drop a thinking signature  -> test_signatures_preserved_*,
                                               test_mutant_reserialising_*,
                                               test_audit_detects_dropped_*
  - parse-failure blocks the forward        -> test_failopen_malformed_*
  - upstream outage crashes the gateway     -> test_failopen_unreachable_*
  - leak the upstream key to client/logs    -> test_key_never_*
  - forward Host/hop-by-hop upstream        -> test_request_headers_filtered
  - relay gzip/length/conn to the client    -> test_response_headers_filtered
  - retry a real 4xx / not retry a 5xx      -> test_upstream_4xx_*, test_retry_*
  - mangle the SSE stream                   -> test_streaming_relayed_unmodified
  - bind a routable interface               -> test_bind_localhost_only
  - route unwired                           -> test_route_registered_in_server

The proxy logic is exercised through an injected fake opener (DIP) so there
is zero network — the retry / preservation / fail-open behaviour is tested
deterministically.
"""
import io
import json
import os
import unittest
import urllib.error
from unittest import mock

from lib.hal.gateway import (
    DEFAULT_UPSTREAM,
    GatewayProxy,
    GatewayUpstreamError,
    UpstreamResponse,
    _resp_status,
    build_default_opener,
    extract_thinking_signatures,
    handle_proxy_request,
    redact,
    relay_response,
    response_headers,
)

_HERE = os.path.dirname(__file__)
_SERVER_SRC = os.path.join(_HERE, "..", "lib", "hal", "server.py")
_GATEWAY_SRC = os.path.join(_HERE, "..", "lib", "hal", "gateway.py")

# A realistic multi-turn body with two thinking blocks (signatures) and one
# redacted_thinking block (opaque data) — the exact shape #3274 corrupts.
_THINKING_BODY = {
    "model": "claude-opus-4-7",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "reasoning one",
             "signature": "ErcBCkYIA-SIGNATURE-ONE-abc123=="},
            {"type": "text", "text": "Answer one"},
        ]},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": [
            {"type": "redacted_thinking",
             "data": "EkwhREDACTED-DATA-BLOB-xyz789=="},
            {"type": "thinking", "thinking": "reasoning two",
             "signature": "ErcBCkYIB-SIGNATURE-TWO-def456=="},
            {"type": "text", "text": "Answer two"},
        ]},
        {"role": "user", "content": "Third question"},
    ],
}
_EXPECTED_SIGS = [
    "ErcBCkYIA-SIGNATURE-ONE-abc123==",
    "EkwhREDACTED-DATA-BLOB-xyz789==",
    "ErcBCkYIB-SIGNATURE-TWO-def456==",
]


class FakeResponse:
    """Minimal stand-in for an http response: status + headers + readable body."""

    def __init__(self, status, body=b"", headers=None):
        self.status = status
        self.headers = headers if headers is not None else []
        self._buf = io.BytesIO(body if isinstance(body, bytes) else body.encode())
        self.read_count = 0

    def read(self, n=-1):
        self.read_count += 1
        return self._buf.read() if n is None or n < 0 else self._buf.read(n)


class RecordingOpener:
    """Captures every (method, url, headers, body) and returns a scripted seq.

    ``script`` is a list of FakeResponse or Exception instances, consumed one
    per attempt. The recorded ``bodies`` list is the test's window onto exactly
    what bytes were sent upstream on each attempt — the byte-equality anchor.
    """

    def __init__(self, script):
        self._script = list(script)
        self.calls = []
        self.bodies = []
        self.header_sets = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url))
        self.bodies.append(body)
        self.header_sets.append(list(headers))
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeHandler:
    """Captures everything a BaseHTTPRequestHandler would write to the client."""

    def __init__(self, command="POST", headers=None, body=b""):
        self.command = command
        self.headers = headers if headers is not None else {}
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.status = None
        self.sent_headers = []
        self.ended = False

    def send_response(self, code):
        self.status = code

    def send_header(self, k, v):
        self.sent_headers.append((k, v))

    def end_headers(self):
        self.ended = True

    def client_facing_text(self):
        """All bytes/strings the client could observe — for leak assertions."""
        parts = [str(self.status)]
        parts += [f"{k}: {v}" for k, v in self.sent_headers]
        parts.append(self.wfile.getvalue().decode("utf-8", "replace"))
        return "\n".join(parts)


class RaisingOpener:
    """Opener that RAISES (not returns) per its script — models a non-conformant
    opener that throws urllib.error.HTTPError instead of returning it."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url))
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _httperror(code, body=b"{}"):
    return urllib.error.HTTPError(
        "https://api.anthropic.com/v1/messages", code, "err", {}, io.BytesIO(body))


def _no_sleep(_seconds):
    pass


class TestDefaultOpenerHTTPError(unittest.TestCase):
    """The REAL build_default_opener() must relay an upstream HTTPError verbatim.

    Regression for the Python 3.14 crash that broke Codex's first live call
    through the gateway. ``urllib.error.HTTPError.status`` is a read-only
    property on 3.9+ (it aliases ``.code``); assigning it raises
    ``property 'status' of 'HTTPError' object has no setter``. The pre-fix
    opener did ``http_err.status = http_err.code``, so any upstream 4xx/5xx
    surfaced to the client as ``gateway translation error: ... no setter``
    (gateway_responses.py wraps the AttributeError). The existing
    ``RaisingOpener`` is a TEST DOUBLE — it never exercised the production
    opener, so this path shipped uncovered. These tests drive the real opener
    with a patched ``urlopen`` that raises. (Detection fires on HAL's 3.14
    runtime, where the read-only-property assignment raises.)
    """

    # The default opener opens through its own timed urllib handlers (Step 5a,
    # fleet finding f83), so these patch that seam, ``_open_timed``, where they
    # used to patch ``urllib.request.urlopen``. Patching urlopen no longer
    # reaches the default path, and a test that misses its patch makes a real
    # network call. The explicit-``timeout`` opener still calls urlopen: the
    # last test here covers it.
    _SEAM = "lib.hal.gateway._open_timed"

    def test_default_opener_relays_raised_httperror_without_assigning_status(self):
        err = _httperror(429, b'{"type":"error","error":{"type":"rate_limit"}}')
        opener = build_default_opener()
        with mock.patch(self._SEAM, side_effect=err):
            resp = opener(
                "POST", "https://upstream.invalid/v1/messages",
                [("x-api-key", "test-key-not-real")], b"{}",
            )
        # Must NOT raise (pre-fix: AttributeError 'no setter' on 3.14). The
        # HTTPError is relayed unchanged so forward()/_resp_status pass it on.
        self.assertIs(resp, err)
        self.assertEqual(_resp_status(resp), 429)
        self.assertEqual(
            resp.read(), b'{"type":"error","error":{"type":"rate_limit"}}')

    def test_default_opener_status_matches_each_upstream_code(self):
        # Goodhart guard: relayed status must equal the upstream code for every
        # error class — a mutant that hardcodes/zeros the status fails here.
        for code in (400, 401, 404, 429, 500, 503, 529):
            err = _httperror(code, b'{"x":1}')
            opener = build_default_opener()
            with mock.patch(self._SEAM, side_effect=err):
                resp = opener("POST", "https://u.invalid/v1/messages", [], b"{}")
            self.assertIs(resp, err)
            self.assertEqual(_resp_status(resp), code)

    def test_default_opener_returns_2xx_response_directly(self):
        # Success path unchanged: a 200 urllib response is returned as-is.
        ok = FakeResponse(200, b'{"id":"msg_ok"}')
        opener = build_default_opener()
        with mock.patch(self._SEAM, return_value=ok):
            resp = opener("POST", "https://u.invalid/v1/messages", [], b"{}")
        self.assertIs(resp, ok)
        self.assertEqual(_resp_status(resp), 200)

    def test_an_explicit_timeout_keeps_the_single_timeout_urlopen_path(self):
        err = _httperror(503, b'{"x":1}')
        with mock.patch("urllib.request.urlopen", side_effect=err) as urlopen:
            resp = build_default_opener(timeout=7)("GET", "https://u.invalid/v1/x", [], None)
        self.assertIs(resp, err)
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 7)


class TestSignaturePreservation(unittest.TestCase):
    """The #3274 core: signatures survive the proxy + retries byte-identical."""

    def test_signatures_preserved_byte_identical_across_retry(self):
        original = json.dumps(_THINKING_BODY).encode()
        # First attempt 529 (retryable), second 200 — both must send same bytes.
        opener = RecordingOpener([
            FakeResponse(529, b'{"error":"overloaded"}'),
            FakeResponse(200, b'{"id":"msg_ok"}'),
        ])
        proxy = GatewayProxy(opener=opener, max_retries=2, sleep=_no_sleep)

        resp = proxy.forward("/v1/messages", original, [("content-type", "application/json")])

        self.assertEqual(resp.status, 200)
        self.assertEqual(len(opener.bodies), 2, "expected one retry")
        # Verbatim: every attempt sent the byte-identical original body.
        self.assertEqual(opener.bodies[0], original)
        self.assertEqual(opener.bodies[1], original)
        self.assertEqual(opener.bodies[0], opener.bodies[1])
        # Semantic: signatures recovered from sent bytes equal the originals.
        for sent in opener.bodies:
            self.assertEqual(extract_thinking_signatures(sent), _EXPECTED_SIGS)

    def test_extract_returns_thinking_and_redacted_in_order(self):
        original = json.dumps(_THINKING_BODY).encode()
        self.assertEqual(extract_thinking_signatures(original), _EXPECTED_SIGS)

    def test_audit_detects_dropped_signature_so_equality_is_nonvacuous(self):
        intact = json.dumps({"messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "x", "signature": "SIG-A"}]}]}).encode()
        dropped = json.dumps({"messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "x"}]}]}).encode()  # signature gone
        self.assertEqual(extract_thinking_signatures(intact), ["SIG-A"])
        self.assertEqual(extract_thinking_signatures(dropped), [])
        self.assertNotEqual(
            extract_thinking_signatures(intact),
            extract_thinking_signatures(dropped),
            "the byte/semantic equality assertions must be able to FAIL",
        )

    def test_mutant_reserialising_proxy_fails_byte_equality(self):
        """A build that reserialises the body (the #3274 bug class) MUST break
        the verbatim guarantee — proves the byte-equality test is non-vacuous."""

        class _ReserialisingProxy(GatewayProxy):
            def forward(self, path, body, client_headers, *, method="POST", max_retries=None):
                # Reorders keys on reserialise — a #3274 corruption mode ("must
                # remain as they were"). ANY byte drift breaks the verbatim rule.
                reserialised = json.dumps(json.loads(body), sort_keys=True).encode()
                return super().forward(path, reserialised, client_headers,
                                       method=method, max_retries=max_retries)

        original = json.dumps(_THINKING_BODY).encode()
        opener = RecordingOpener([FakeResponse(200, b"{}")])
        mutant = _ReserialisingProxy(opener=opener, max_retries=0, sleep=_no_sleep)

        mutant.forward("/v1/messages", original, [])
        self.assertNotEqual(opener.bodies[0], original,
                            "mutant must corrupt the verbatim bytes")


class TestFailOpen(unittest.TestCase):
    """Gateway value-adds never turn a would-succeed request into a failure."""

    def test_failopen_malformed_body_forwarded_verbatim(self):
        # Not JSON at all — the forward path must NOT parse it; pass bytes through.
        garbage = b"\x00not-json-at-all\xff"
        opener = RecordingOpener([FakeResponse(200, b'{"id":"ok"}')])
        proxy = GatewayProxy(opener=opener, max_retries=2, sleep=_no_sleep)

        resp = proxy.forward("/v1/messages", garbage, [])

        self.assertEqual(resp.status, 200)
        self.assertEqual(opener.bodies[0], garbage)  # verbatim, never parsed

    def test_failopen_unreachable_returns_clean_503_not_crash(self):
        # Every attempt is a transport failure → graceful 503, never an exception.
        # Inverted 2026-10-05 (InfiniteMachine v2, Step 5a): this pinned a 502.
        # A transport fault is a 503, the status every harness retries (fleet
        # findings f48, f68); a gateway-named failure still gets the 502.
        opener = RecordingOpener([
            ConnectionError("connection refused"),
            ConnectionError("connection refused"),
            ConnectionError("connection refused"),
        ])
        proxy = GatewayProxy(opener=opener, max_retries=2, sleep=_no_sleep)
        handler = FakeHandler(
            command="POST",
            headers={"Content-Length": "2"},
            body=b"{}",
        )
        handle_proxy_request(handler, proxy, "/v1/messages")

        self.assertEqual(handler.status, 503)
        envelope = json.loads(handler.wfile.getvalue())
        self.assertEqual(envelope["type"], "error")
        self.assertEqual(envelope["error"]["type"], "gateway_error")

    def test_forward_raises_gatewayupstreamerror_on_exhaustion(self):
        opener = RecordingOpener([ConnectionError("dns fail")])
        proxy = GatewayProxy(opener=opener, max_retries=0, sleep=_no_sleep)
        with self.assertRaises(GatewayUpstreamError):
            proxy.forward("/v1/messages", b"{}", [])


class TestCustody(unittest.TestCase):
    """The upstream credential is relayed upstream but never leaked."""

    def test_redact_strips_key_oauth_and_bearer(self):
        raw = ("x-api-key: sk-ant-secret-XYZ123 and "
               "authorization: Bearer sk-ant-oat-TOKEN456 and sk-proj-abcdef99")
        out = redact(raw)
        self.assertNotIn("sk-ant-secret-XYZ123", out)
        self.assertNotIn("sk-ant-oat-TOKEN456", out)
        self.assertNotIn("sk-proj-abcdef99", out)
        self.assertIn("[REDACTED]", out)

    def test_key_relayed_upstream_but_never_in_client_response(self):
        secret = "sk-ant-secret-XYZ123"
        token = "sk-ant-oat-TOKEN456"
        opener = RecordingOpener([FakeResponse(
            200, b'{"id":"msg_ok"}',
            headers=[("content-type", "application/json")],
        )])
        proxy = GatewayProxy(opener=opener, max_retries=0, sleep=_no_sleep)
        handler = FakeHandler(
            command="POST",
            headers={
                "Content-Length": "2",
                "x-api-key": secret,
                "authorization": f"Bearer {token}",
                "content-type": "application/json",
            },
            body=b"{}",
        )
        handle_proxy_request(handler, proxy, "/v1/messages")

        # The key MUST reach Anthropic (relayed upstream) ...
        upstream_headers = dict(opener.header_sets[0])
        self.assertEqual(upstream_headers.get("x-api-key"), secret)
        self.assertEqual(upstream_headers.get("authorization"), f"Bearer {token}")
        # ... but MUST NOT appear in anything the client can observe.
        observed = handler.client_facing_text()
        self.assertNotIn(secret, observed)
        self.assertNotIn(token, observed)

    def test_error_envelope_redacts_a_leaked_key(self):
        # Upstream failure whose message embeds a key → the envelope is redacted.
        # A transport failure, so a 503 since Step 5a (fleet findings f48, f68).
        opener = RecordingOpener([
            ConnectionError("auth failed for sk-ant-secret-LEAK999"),
        ])
        proxy = GatewayProxy(opener=opener, max_retries=0, sleep=_no_sleep)
        handler = FakeHandler(command="POST", headers={"Content-Length": "2"}, body=b"{}")
        handle_proxy_request(handler, proxy, "/v1/messages")

        self.assertEqual(handler.status, 503)
        self.assertNotIn("sk-ant-secret-LEAK999", handler.client_facing_text())


class TestHeaderFiltering(unittest.TestCase):

    def test_request_headers_filtered(self):
        proxy = GatewayProxy(opener=RecordingOpener([]), max_retries=0)
        client = [
            ("Host", "127.0.0.1:4010"),
            ("Content-Length", "123"),
            ("Connection", "keep-alive"),
            ("Accept-Encoding", "gzip, br"),
            ("x-api-key", "sk-ant-x"),
            ("anthropic-version", "2023-06-01"),
            ("anthropic-beta", "context-1m-2025-08-07"),
            ("content-type", "application/json"),
        ]
        out = dict((k.lower(), v) for k, v in proxy.upstream_headers(client))
        # Stripped (hop-by-hop / managed / forced-identity):
        for stripped in ("host", "content-length", "connection", "accept-encoding"):
            self.assertNotIn(stripped, out)
        # Preserved verbatim (auth + anthropic-* + content-type):
        self.assertEqual(out["x-api-key"], "sk-ant-x")
        self.assertEqual(out["anthropic-version"], "2023-06-01")
        self.assertEqual(out["anthropic-beta"], "context-1m-2025-08-07")
        self.assertEqual(out["content-type"], "application/json")

    def test_response_headers_filtered(self):
        upstream = [
            ("Content-Type", "text/event-stream"),
            ("Content-Encoding", "gzip"),
            ("Transfer-Encoding", "chunked"),
            ("Connection", "keep-alive"),
            ("Content-Length", "999"),
            ("request-id", "req_abc"),
            ("anthropic-ratelimit-requests-remaining", "42"),
        ]
        out = dict((k.lower(), v) for k, v in response_headers(upstream))
        for stripped in ("content-encoding", "transfer-encoding", "connection", "content-length"):
            self.assertNotIn(stripped, out)
        self.assertEqual(out["content-type"], "text/event-stream")
        self.assertEqual(out["request-id"], "req_abc")
        self.assertEqual(out["anthropic-ratelimit-requests-remaining"], "42")


class TestStreamingAndStatusRelay(unittest.TestCase):

    def test_streaming_relayed_unmodified(self):
        sse = (b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
               b"event: content_block_delta\ndata: {\"delta\":\"hi\"}\n\n")
        opener = RecordingOpener([FakeResponse(
            200, sse, headers=[("content-type", "text/event-stream")])])
        proxy = GatewayProxy(opener=opener, max_retries=0, sleep=_no_sleep)
        handler = FakeHandler(command="POST", headers={"Content-Length": "2"}, body=b"{}")

        handle_proxy_request(handler, proxy, "/v1/messages")

        self.assertEqual(handler.status, 200)
        self.assertEqual(handler.wfile.getvalue(), sse)  # byte-identical stream
        relayed = dict((k.lower(), v) for k, v in handler.sent_headers)
        self.assertEqual(relayed["content-type"], "text/event-stream")

    def test_relay_response_closes_upstream_socket(self):
        # Refutes the AGORA C4 leak concern at the P1 layer too: relay_response
        # releases the upstream reader even after streaming the body.
        class _Reader:
            def __init__(self, data):
                self._b = io.BytesIO(data)
                self.closed = False

            def read(self, n=-1):
                return self._b.read() if n is None or n < 0 else self._b.read(n)

            def close(self):
                self.closed = True

        reader = _Reader(b'{"ok":true}')
        resp = UpstreamResponse(
            status=200, headers=[("content-type", "application/json")], _reader=reader)
        handler = FakeHandler(command="POST", headers={}, body=b"")
        relay_response(handler, resp)
        self.assertTrue(reader.closed, "relay_response must close the upstream reader")

    def test_upstream_4xx_relayed_not_retried(self):
        # The literal #3274 error: a 400 must pass through transparently, once.
        body = (b'{"type":"error","error":{"type":"invalid_request_error",'
                b'"message":"thinking blocks cannot be modified"}}')
        opener = RecordingOpener([FakeResponse(400, body)])
        proxy = GatewayProxy(opener=opener, max_retries=2, sleep=_no_sleep)

        resp = proxy.forward("/v1/messages", b"{}", [])

        self.assertEqual(resp.status, 400)
        self.assertEqual(len(opener.calls), 1, "4xx must NOT be retried")
        self.assertEqual(resp.read_all(), body)

    def test_non_retryable_4xx_returns_immediately(self):
        opener = RecordingOpener([FakeResponse(401, b'{"error":"unauthorized"}')])
        proxy = GatewayProxy(opener=opener, max_retries=2, sleep=_no_sleep)
        resp = proxy.forward("/v1/messages", b"{}", [])
        self.assertEqual(resp.status, 401)
        self.assertEqual(len(opener.calls), 1)

    def test_retryable_500_then_200_relays_fresh_body_not_drained(self):
        # Refutes the HAL Council "drained response relayed" concern: each
        # attempt fetches a FRESH response; the relayed body is the second
        # (undrained) 200, never the drained 500.
        opener = RecordingOpener([
            FakeResponse(500, b'{"error":"server"}'),
            FakeResponse(200, b'{"id":"ok-fresh"}'),
        ])
        proxy = GatewayProxy(opener=opener, max_retries=2, sleep=_no_sleep)
        resp = proxy.forward("/v1/messages", b"{}", [])
        self.assertEqual(resp.status, 200)
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(resp.read_all(), b'{"id":"ok-fresh"}')

    def test_raised_httperror_retryable_then_relayed(self):
        # Reviewer falsifier: an opener that RAISES (not returns) a 503 must be
        # retried and the eventual 200 relayed — never crash, never leak as 500.
        opener = RaisingOpener([
            _httperror(503, b'{"error":"overloaded"}'),
            FakeResponse(200, b'{"id":"ok"}'),
        ])
        proxy = GatewayProxy(opener=opener, max_retries=2, sleep=_no_sleep)
        resp = proxy.forward("/v1/messages", b"{}", [])
        self.assertEqual(resp.status, 200)
        self.assertEqual(len(opener.calls), 2)

    def test_raised_httperror_non_retryable_relayed_not_502(self):
        # A RAISED 400 must be relayed verbatim (status 400), NOT mis-retried
        # into a 502 — returned-vs-raised HTTPError behave identically.
        opener = RaisingOpener([_httperror(400, b'{"error":"bad_request"}')])
        proxy = GatewayProxy(opener=opener, max_retries=2, sleep=_no_sleep)
        resp = proxy.forward("/v1/messages", b"{}", [])
        self.assertEqual(resp.status, 400)
        self.assertEqual(len(opener.calls), 1, "400 must not be retried")
        self.assertEqual(resp.read_all(), b'{"error":"bad_request"}')


class TestRawRelayAuthTranslation(unittest.TestCase):
    """The raw relay converts an OpenAI API-key Bearer to x-api-key so OpenAI
    clients (Codex) work on GET /v1/models + POST /v1/messages. OAuth Bearer
    (sk-ant-oat...) and an explicit x-api-key are left untouched. Regression:
    Codex's /v1/models probe 401'd 'Invalid bearer token' through the relay.
    """

    def _hdrs(self, client_headers):
        proxy = GatewayProxy(opener=RecordingOpener([]), max_retries=0, sleep=_no_sleep)
        return {n.lower(): v for n, v in proxy.upstream_headers(client_headers)}

    def test_apikey_bearer_translated_to_xapikey(self):
        out = self._hdrs([("Authorization", "Bearer sk-ant-api03-XYZ"),
                          ("content-type", "application/json")])
        self.assertEqual(out.get("x-api-key"), "sk-ant-api03-XYZ")
        self.assertNotIn("authorization", out)  # Bearer dropped, not sent alongside
        self.assertEqual(out.get("anthropic-version"), "2023-06-01")  # required header added

    def test_oauth_bearer_left_as_bearer(self):
        # OAuth/CCH tokens authenticate via Bearer at Anthropic — must NOT convert.
        out = self._hdrs([("Authorization", "Bearer sk-ant-oat01-TOK")])
        self.assertEqual(out.get("authorization"), "Bearer sk-ant-oat01-TOK")
        self.assertNotIn("x-api-key", out)

    def test_explicit_xapikey_untouched(self):
        out = self._hdrs([("x-api-key", "sk-ant-api03-ABC"),
                          ("anthropic-version", "2099-01-01")])
        self.assertEqual(out.get("x-api-key"), "sk-ant-api03-ABC")
        self.assertEqual(out.get("anthropic-version"), "2099-01-01")  # caller's version kept

    def test_no_auth_left_unchanged(self):
        out = self._hdrs([("content-type", "application/json")])
        self.assertNotIn("x-api-key", out)
        self.assertNotIn("authorization", out)

    def test_anthropic_version_not_duplicated(self):
        pairs = GatewayProxy(opener=RecordingOpener([]), max_retries=0, sleep=_no_sleep)\
            .upstream_headers([("Authorization", "Bearer sk-ant-api03-XYZ"),
                               ("anthropic-version", "2023-06-01")])
        versions = [v for n, v in pairs if n.lower() == "anthropic-version"]
        self.assertEqual(len(versions), 1)


class TestWiringAndBind(unittest.TestCase):

    def test_bind_localhost_only(self):
        src = open(_GATEWAY_SRC, encoding="utf-8").read()
        self.assertIn('("127.0.0.1", port)', src)
        self.assertNotIn('"0.0.0.0"', src)

    def test_route_registered_in_server(self):
        src = open(_SERVER_SRC, encoding="utf-8").read()
        self.assertIn("from lib.hal.gateway import GatewayProxy, handle_proxy_request", src)
        self.assertIn("handle_proxy_request(self, _get_gateway_proxy(), self.path)", src)
        self.assertIn('self.path.split("?", 1)[0].startswith("/v1/")', src)

    def test_default_upstream_is_anthropic(self):
        self.assertEqual(DEFAULT_UPSTREAM, "https://api.anthropic.com")


if __name__ == "__main__":
    unittest.main()
