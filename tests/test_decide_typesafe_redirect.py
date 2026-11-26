"""HAL's TypeSafe transport refuses redirects, so the bearer key never follows one.

urllib copies ordinary request headers onto a 301, 302 or 303 to ANY host. A
sovereign council found this on GRASP#24, whose example mirrors this transport,
and it was reproduced here on loopback: before the fix, a 302 from the TypeSafe
URL delivered ``Authorization: Bearer <key>`` to a second server. Both servers
listen on 127.0.0.1, so nothing in this file leaves the machine.

Mutation that turns the transport test red: drop ``_RefuseRedirects()`` from the
opener in ``_no_proxy_transport``. The second server then receives the canary
key on 301, 302 and 303. (urllib already refuses to replay a POST on 307 and
308, so those two cases pin the returned status rather than the fix.)
"""
from __future__ import annotations

import http.server
import json
import threading
import urllib.error
import urllib.request
from decimal import Decimal

import pytest

from lib.hal import spend_guard
from lib.hal.decide.contract import DecisionModel, ProviderSpec, noul
from lib.hal.decide.typesafe import TypeSafeDecisionProvider, _Breaker, _no_proxy_transport

CANARY = "Bearer CANARY-NOT-A-KEY"
SOURCE = "typesafe-api-key"
SPEC = ProviderSpec(name="typesafe", endpoint="https://api.typesafe.ai/v1/systemone",
                    key_source="keychain:grip-typesafe", spend_source=SOURCE, billing="prepaid",
                    usd_per_million_input=Decimal("0.042"), usd_per_million_output=Decimal("0"),
                    max_request_bytes=524288)
MODEL = DecisionModel(id="jev-latest", provider="typesafe", pinned="jev-1.13.0")
REDIRECTS = [301, 302, 303, 307, 308]


def _serve(handler: type) -> http.server.HTTPServer:
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _stop(*servers: http.server.HTTPServer) -> None:
    for server in servers:
        server.shutdown()
        server.server_close()


def _recorder(seen: list) -> type:
    """A server that records every request it receives, whatever its method, as
    (method, Authorization header). It records in ``parse_request``, which runs
    before method dispatch, so even a method it does not serve is counted."""

    class Elsewhere(http.server.BaseHTTPRequestHandler):
        def parse_request(self):
            parsed = super().parse_request()
            if parsed:
                seen.append((self.command, self.headers.get("Authorization")))
            return parsed

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        do_POST = do_GET

        def log_message(self, *args):
            pass

    return Elsewhere


def _redirector(code: int, location: str) -> type:
    """A stand-in for the TypeSafe endpoint that answers every POST with a redirect."""

    class Redirector(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(code)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    return Redirector


@pytest.mark.parametrize("code", REDIRECTS)
def test_a_redirect_is_refused_and_the_key_never_reaches_the_new_host(code):
    """``seen`` records every request the second host receives, with or without the
    header, so an empty list means no request reached it at all."""
    seen: list = []
    elsewhere = _serve(_recorder(seen))
    redirector = _serve(_redirector(code, f"http://127.0.0.1:{elsewhere.server_port}/elsewhere"))
    try:
        status, _, _ = _no_proxy_transport(f"http://127.0.0.1:{redirector.server_port}/v1/systemone",
                                           b"{}", {"Authorization": CANARY}, 5.0)
    finally:
        _stop(redirector, elsewhere)
    assert status == code, "the redirect must come back as an answer, not be followed"
    assert seen == [], "the second host must receive nothing, least of all the key"


@pytest.mark.parametrize("code", REDIRECTS)
def test_a_refused_redirect_ends_as_bad_response_with_one_attempt_and_no_spend(code, tmp_path, monkeypatch):
    """What the provider makes of the refused redirect the transport returns: reason
    ``bad_response``, one attempt (a 3xx is never retried), nothing recorded against
    the daily cap. AGORA asked for this on HAL#729, because the module docstring
    now claims all three.

    Mutations: make a 3xx retryable in ``_retryable`` (attempts rise above 1), or
    map a non-5xx status to another reason in ``_reason`` (the reason changes).
    """
    policy = tmp_path / "spend-policy.json"
    policy.write_text(json.dumps({"paid_opt_in": [SOURCE], "daily_caps_usd": {SOURCE: 1}}))
    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(policy))
    sent: list = []

    def wire(url, data, headers, timeout):
        sent.append(url)
        return code, {"Location": "https://elsewhere.example/"}, b""

    provider = TypeSafeDecisionProvider(
        SPEC, transport=wire, shield=lambda body, *, to: body,
        key_resolver=lambda source: "apikey_FAKECANARY_notarealkey", sleep=lambda seconds: None,
        clock=lambda: 100.0, jitter=lambda: 0.0, breaker=_Breaker())
    result = provider.decide(MODEL, "A synthetic ticket.", {"urgent": noul("Is it urgent?")})
    assert (result.ok, result.reason, result.attempts) == (False, "bad_response", 1)
    assert sent == [SPEC.endpoint]
    entry = spend_guard._today_entry(SOURCE, spend_guard.today_utc().isoformat())
    assert (entry["calls"], entry["input_tokens"], entry["spent_usd"]) == (0, 0, "0.00000000")


def test_the_recorder_sees_a_request_of_any_method():
    """The control for ``seen == []`` above: the recorder must see requests by methods
    it does not serve, or an empty list would prove nothing about them.

    Mutation: record in ``do_GET`` instead of ``parse_request`` (HEAD and DELETE go
    unseen and this test turns red).
    """
    seen: list = []
    server = _serve(_recorder(seen))
    try:
        for method in ("HEAD", "DELETE"):
            request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/x", method=method,
                                             headers={"Authorization": CANARY})
            try:
                urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=5.0).close()
            except urllib.error.HTTPError:
                pass  # 501 for an unserved method: the request still arrived
    finally:
        _stop(server)
    assert seen == [("HEAD", CANARY), ("DELETE", CANARY)]
