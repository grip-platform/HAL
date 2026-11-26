"""A provider the router KNOWS but cannot reach must fail loud, never fall back.

THE DEFECT (live, 2026-08-17 — the SECOND instance of one shape). ``route_model``
answers "which provider owns this model"; ``PROVIDER_ENDPOINTS`` answers "can this
gateway reach that provider". The two disagreed about ``mistral``: it is in
``model_route.KNOWN_PROVIDERS`` and has two family routes (``mistral``,
``mixtral``), but had no endpoint. ``same_wire_provider`` therefore returned
``None`` — and all three request surfaces read ``None`` as "use the Anthropic
path", so the caller's key went to api.anthropic.com as ``x-api-key``:

    POST 127.0.0.1:4010/v1/chat/completions {"model":"mistral/mistral-small-latest"}
    -> {"error":{"message":"invalid x-api-key","type":"gateway_error"}}
    POST 127.0.0.1:4010/v1/chat/completions {"model":"mistral-small-latest"}
    -> {"error":{"message":"invalid x-api-key","type":"gateway_error"}}

That is the same silent misroute the ``<provider>/`` prefix fix closed for
``groq/…`` (``test_gateway_provider_prefix_routing``), reached by a different
door. A wrong-provider credential error is the worst possible diagnostic: it
blames the caller's key for a gateway configuration gap.

TWO HALVES, because either alone leaves the hole open:

  1. THE INSTANCE — mistral is a declared provider (``config/providers.yaml``:
     ``base_url: https://api.mistral.ai/v1``, ``keychain:grip-mistral``) whose
     mirror entry in ``PROVIDER_ENDPOINTS`` was simply missing. Live-verified
     2026-08-17: that credential + base_url answer ``GET /v1/models`` with HTTP
     200 and 55 ids including ``mistral-small-latest``. So the honest fix is to
     restore the endpoint, not to refuse the request.
  2. THE CLASS — the silent ``None`` must stop being reachable at all. Any FUTURE
     name added to ``KNOWN_PROVIDERS`` without an endpoint must raise a clear,
     actionable error instead of borrowing the Anthropic upstream. ``route_model``
     has exactly one production consumer (``same_wire_provider``), which is why
     the guard lives there: it covers ``/v1/chat/completions``, ``/v1/responses``
     and ``/v1/messages`` at once, and covers the fourth surface nobody has
     written yet.

Goodhart notes. ``test_every_known_provider_has_an_endpoint`` derives its inputs
FROM ``KNOWN_PROVIDERS``, so it cannot be satisfied by listing today's providers
in the test. The guard tests SYNTHESISE the missing-endpoint condition by
removing a real endpoint rather than relying on mistral staying broken, so they
keep their teeth after half 1 lands and after every provider is wired.

Hermetic: no network, no keychain, env pinned, opener + key_resolver injected.

MEASUREMENT RECORD (corrects commit 62c0d43a).
----------------------------------------------
The commit message of 62c0d43a states "Proven red before the fix: 14 of 20
failed". That figure does not reproduce and is CORRECTED here to **17 of 20**
(10 failures + 7 errors, 3 passed). The stale 14 appears to have been carried
over from an earlier 16-case draft of this file; the committed file has 20
tests. The direction is benign — the pre-fix evidence is STRONGER than the
commit claimed, not weaker — but a repository whose product is replayable
provenance does not get to leave an unreproducible number in its own record.

Reproduce it (isolated, never touching the working tree that launchd runs):

    git worktree add --detach /tmp/hal-prefix 62c0d43a^
    git show 62c0d43a:tests/test_gateway_unroutable_provider.py \\
        > /tmp/hal-prefix/tests/test_gateway_unroutable_provider.py
    cd /tmp/hal-prefix && python3 -m unittest tests.test_gateway_unroutable_provider
    # => Ran 20 tests ... FAILED (failures=10, errors=7)

The 7 errors are the not-yet-written ``unroutable_provider`` predicate
(AttributeError); the 10 failures include "Lists differ: ['mistral'] != []" for
the class invariant and, on all three surfaces, the spy proxy firing with "the
Anthropic upstream was reached for an unroutable provider". Green after the fix:
``Ran 20 tests ... OK``.
"""
import io
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from lib.hal import gateway_egress as egress
from lib.hal.gateway import GatewayUpstreamError
from lib.hal.model_route import ANTHROPIC, KNOWN_PROVIDERS, route_model

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROVIDERS_YAML = _REPO_ROOT / "config" / "providers.yaml"

# The gateway's kill-switch is env-driven; pin it ON so these assertions are
# about routing, never about ambient environment.
_FLAG_ON = {"HAL_GATEWAY_MULTIPROVIDER": "1"}


def _routable(provider: str) -> bool:
    return (provider in egress.PROVIDER_ENDPOINTS
            or provider in egress.LOCAL_PROVIDER_ENDPOINTS)


def _without_endpoint(provider: str):
    """Context manager: the endpoint tables minus ``provider``.

    Synthesises the known-but-unreachable condition from a provider that IS
    wired, so the guard tests survive every provider being correctly wired.
    """
    reduced = {k: v for k, v in egress.PROVIDER_ENDPOINTS.items() if k != provider}
    return mock.patch.dict(egress.PROVIDER_ENDPOINTS, reduced, clear=True)


def _forwarded_model(provider: str, model: str) -> str:
    """The ``model`` value that actually reaches ``provider`` upstream."""
    sink = {}

    def opener(method, url, headers, body):
        sink["url"], sink["body"] = url, body
        return _FakeResp()

    raw = json.dumps({"model": model, "messages": [{"role": "user", "content": "x"}]})
    egress.forward_same_wire(provider, "/v1/chat/completions", raw.encode("utf-8"),
                             opener=opener, key_resolver=lambda s: "k")
    return json.loads(sink["body"])["model"]


class _FakeResp:
    """Minimal opener response — mirrors test_gateway_egress/_prefix_routing."""

    def __init__(self, status=200):
        self.status = status
        self.headers = [("Content-Type", "application/json")]
        self._read = False

    def read(self, size=-1):
        if self._read:
            return b""
        self._read = True
        return b"{}"


class TestEveryKnownProviderIsReachable(unittest.TestCase):
    """Half 2, the class invariant: routing knowledge and reach must agree."""

    def test_every_known_provider_has_an_endpoint(self):
        # Inputs derived FROM KNOWN_PROVIDERS: a provider added there tomorrow
        # without an endpoint fails here, which is the whole point.
        orphans = sorted(p for p in KNOWN_PROVIDERS
                         if p != ANTHROPIC and not _routable(p))
        self.assertEqual(
            orphans, [],
            "known to model_route but unreachable by gateway_egress -> "
            "same_wire_provider would answer None and the request would be "
            "misrouted to the Anthropic upstream: " + repr(orphans))

    def test_no_known_provider_silently_falls_through(self):
        # The behavioural restatement: for every known non-Anthropic provider,
        # asking about one of its models must never answer None.
        for provider in sorted(KNOWN_PROVIDERS - {ANTHROPIC}):
            with mock.patch.dict(os.environ, _FLAG_ON):
                self.assertEqual(
                    egress.same_wire_provider(f"{provider}/some-model"), provider,
                    f"{provider}/some-model must route to {provider}")


class TestUnroutableKnownProviderFailsLoud(unittest.TestCase):
    """The guard: a known provider with no endpoint raises, never returns None."""

    def test_missing_endpoint_raises_instead_of_returning_none(self):
        with mock.patch.dict(os.environ, _FLAG_ON), _without_endpoint("groq"):
            with self.assertRaises(GatewayUpstreamError):
                egress.same_wire_provider("groq/openai/gpt-oss-120b")

    def test_the_error_is_actionable(self):
        # An operator reading this line must learn: which provider, which model,
        # which table to edit, and that Anthropic was deliberately NOT used.
        with mock.patch.dict(os.environ, _FLAG_ON), _without_endpoint("groq"):
            with self.assertRaises(GatewayUpstreamError) as ctx:
                egress.same_wire_provider("groq/llama-3.3-70b-versatile")
        message = str(ctx.exception)
        for token in ("groq", "llama-3.3-70b-versatile",
                      "PROVIDER_ENDPOINTS", "anthropic"):
            self.assertIn(token, message.lower() if token.islower() else message,
                          f"error must name {token!r}: {message!r}")

    def test_bare_family_id_is_guarded_too(self):
        # The live defect hit BOTH address forms; a guard on only the explicit
        # `<provider>/` form would have left `mistral-small-latest` misrouting.
        with mock.patch.dict(os.environ, _FLAG_ON), _without_endpoint("groq"):
            with self.assertRaises(GatewayUpstreamError):
                egress.same_wire_provider("llama-3.3-70b-versatile")

    def test_predicate_names_the_unreachable_provider(self):
        with mock.patch.dict(os.environ, _FLAG_ON), _without_endpoint("groq"):
            self.assertEqual(
                egress.unroutable_provider("groq/openai/gpt-oss-120b"), "groq")

    def test_predicate_and_router_are_exact_complements(self):
        # At most one of the two ever answers a name, for every model shape.
        models = ["claude-opus-4-8", "anthropic/claude-opus-4-8", "gpt-4o",
                  "groq/openai/gpt-oss-120b", "mistral-small-latest",
                  "totally-made-up-9000", "", "notaprovider/x"]
        models += [f"{p}/some-model" for p in sorted(KNOWN_PROVIDERS)]
        for model in models:
            with mock.patch.dict(os.environ, _FLAG_ON):
                unroutable = egress.unroutable_provider(model)
                if unroutable is not None:
                    with self.assertRaises(GatewayUpstreamError, msg=model):
                        egress.same_wire_provider(model)
                else:
                    egress.same_wire_provider(model)  # must not raise


class TestTheFallThroughPathIsPreserved(unittest.TestCase):
    """What must KEEP returning None — over-guarding would break the gateway."""

    def test_claude_is_still_verbatim(self):
        with mock.patch.dict(os.environ, _FLAG_ON):
            self.assertIsNone(egress.same_wire_provider("claude-opus-4-8"))
            self.assertIsNone(egress.same_wire_provider("anthropic/claude-opus-4-8"))

    def test_unknown_models_still_fall_through_silently(self):
        # An id HAL does not recognise is NOT this defect: the gateway made no
        # routing decision to contradict, and Anthropic answers for its own ids.
        with mock.patch.dict(os.environ, _FLAG_ON):
            for model in ("totally-made-up-9000", "notaprovider/x", "", "groq/"):
                self.assertIsNone(egress.same_wire_provider(model), model)
                self.assertIsNone(egress.unroutable_provider(model), model)

    def test_kill_switch_still_disables_everything(self):
        # Flag OFF must be total: no routing AND no raising, or the documented
        # kill-switch would itself become a new failure mode.
        with mock.patch.dict(os.environ, {"HAL_GATEWAY_MULTIPROVIDER": "0"}), \
                _without_endpoint("groq"):
            self.assertIsNone(egress.same_wire_provider("groq/openai/gpt-oss-120b"))
            self.assertIsNone(egress.unroutable_provider("mistral-small-latest"))


class TestMistralIsReachable(unittest.TestCase):
    """Half 1, the instance: the declared provider that had no mirror entry."""

    def test_explicit_address_routes_to_mistral(self):
        with mock.patch.dict(os.environ, _FLAG_ON):
            self.assertEqual(
                egress.same_wire_provider("mistral/mistral-small-latest"), "mistral")

    def test_bare_family_ids_route_to_mistral(self):
        with mock.patch.dict(os.environ, _FLAG_ON):
            for model in ("mistral-small-latest", "mistral-large-latest",
                          "mixtral-8x7b-instruct"):
                self.assertEqual(egress.same_wire_provider(model), "mistral", model)
                self.assertFalse(route_model(model).verbatim, model)

    def test_egress_url_is_the_declared_endpoint(self):
        # Live-verified 2026-08-17: GET this host /v1/models with the
        # keychain:grip-mistral credential -> HTTP 200, 55 ids.
        self.assertEqual(
            egress.build_egress_url("mistral", "/v1/chat/completions"),
            "https://api.mistral.ai/v1/chat/completions")

    def test_mistral_receives_the_unprefixed_model_id(self):
        self.assertEqual(
            _forwarded_model("mistral", "mistral/mistral-small-latest"),
            "mistral-small-latest")

    def test_key_comes_from_the_declared_keychain_service(self):
        self.assertEqual(egress.PROVIDER_ENDPOINTS["mistral"][1], "grip-mistral")


class _FakeHeaders(dict):
    """Case-insensitive ``get`` — read_request_body wants ``content-length``."""

    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default

    def items(self):
        return list(super().items())


class _FakeHandler:
    """BaseHTTPRequestHandler stand-in capturing the client-facing write."""

    def __init__(self, payload: dict, path: str, command: str = "POST"):
        body = json.dumps(payload).encode("utf-8")
        self.command = command
        self.path = path
        self.headers = _FakeHeaders({"Content-Length": str(len(body))})
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.status = None

    def send_response(self, code):
        self.status = code

    def send_header(self, *_a):
        pass

    def end_headers(self):
        pass

    def body_text(self) -> str:
        return self.wfile.getvalue().decode("utf-8")


class _SpyProxy:
    """Records every Anthropic-upstream forward. It must stay empty here."""

    def __init__(self):
        self.calls = []

    def forward(self, path, body, headers, *, method="POST", max_retries=None):
        self.calls.append(path)
        raise AssertionError(
            "the Anthropic upstream was reached for an unroutable provider — "
            "this is exactly the silent misroute the guard exists to stop")


class TestKimiCodingIsReachable(unittest.TestCase):
    """HAL#550 rows A+B landed together: the Kimi Code membership provider
    (``kimi-coding``, host api.kimi.com/coding, GRIP Pool D, council seal
    e49c00c214c500be) is known to the router AND reachable by the gateway.
    Red before the rows: ``kimi-coding`` was outside KNOWN_PROVIDERS, so the
    address form was UNKNOWN and the endpoint lookup raised KeyError."""

    def test_explicit_address_routes_to_kimi_coding(self):
        for m in ("k3", "k3-256k", "kimi-for-coding", "kimi-for-coding-highspeed"):
            r = route_model(f"kimi-coding/{m}")
            self.assertTrue(r.known, m)
            self.assertEqual(r.provider, "kimi-coding", m)

    def test_bare_kimi_code_ids_still_refuse_the_metered_host(self):
        # The bare-id guard stays: without the address form these would hit
        # the ``kimi`` family row and go to the pay-per-token host.
        for m in ("k3", "kimi-for-coding"):
            r = route_model(m)
            self.assertFalse(r.known, m)
            self.assertNotEqual(r.provider, "moonshot", m)

    def test_egress_url_is_the_declared_endpoint(self):
        self.assertEqual(egress.build_egress_url("kimi-coding", "/v1/chat/completions"),
                         "https://api.kimi.com/coding/v1/chat/completions")

    def test_kimi_coding_receives_the_unprefixed_model_id(self):
        from lib.hal.model_route import strip_provider_prefix
        self.assertEqual(strip_provider_prefix("kimi-coding/k3-256k", "kimi-coding"), "k3-256k")

    def test_key_comes_from_the_kimi_code_keychain_service_never_moonshot(self):
        base, service = egress.PROVIDER_ENDPOINTS["kimi-coding"]
        self.assertEqual(service, "grip-kimi-coding")
        self.assertNotIn("moonshot", base)

    def test_same_wire_provider_answers_kimi_coding_when_enabled(self):
        with mock.patch.dict(os.environ, _FLAG_ON):
            self.assertEqual(egress.same_wire_provider("kimi-coding/k3-256k"), "kimi-coding")


class TestEverySurfaceRefusesInsteadOfMisrouting(unittest.TestCase):
    """Cross-surface coverage: all three request wires, not just the router.

    The bug was never in ``route_model`` alone — it was that every surface reads
    ``None`` as "use Anthropic". A guard proven only at the pure-function level
    would leave open the question these tests answer: does the client actually
    see a clean 502, and is the Anthropic upstream actually left untouched?
    ``_SpyProxy`` raises if it is ever called, so a regression is unmissable.
    """

    UNROUTABLE = {"model": "groq/openai/gpt-oss-120b",
                  "messages": [{"role": "user", "content": "x"}]}

    def _assert_clean_502(self, handler):
        self.assertEqual(handler.status, 502, handler.body_text())
        text = handler.body_text()
        self.assertIn("groq", text)
        self.assertIn("PROVIDER_ENDPOINTS", text)
        self.assertIn("error", json.loads(text))

    def test_chat_completions_returns_502_and_never_calls_anthropic(self):
        from lib.hal import gateway_openai
        proxy = _SpyProxy()
        handler = _FakeHandler(self.UNROUTABLE, "/v1/chat/completions")
        with mock.patch.dict(os.environ, _FLAG_ON), _without_endpoint("groq"):
            gateway_openai.handle_chat_completions(handler, proxy)
        self._assert_clean_502(handler)
        self.assertEqual(proxy.calls, [])

    def test_responses_returns_502_and_never_calls_anthropic(self):
        from lib.hal import gateway_responses
        proxy = _SpyProxy()
        handler = _FakeHandler(dict(self.UNROUTABLE, input="x"), "/v1/responses")
        with mock.patch.dict(os.environ, _FLAG_ON), _without_endpoint("groq"):
            gateway_responses.handle_responses(handler, proxy)
        self._assert_clean_502(handler)
        self.assertEqual(proxy.calls, [])

    def test_messages_rail_returns_502_and_never_calls_anthropic(self):
        from lib.hal.gateway import handle_proxy_request
        proxy = _SpyProxy()
        handler = _FakeHandler(self.UNROUTABLE, "/v1/messages")
        with mock.patch.dict(os.environ, _FLAG_ON), _without_endpoint("groq"):
            handle_proxy_request(handler, proxy, "/v1/messages")
        self._assert_clean_502(handler)
        self.assertEqual(proxy.calls, [])

    def test_a_routable_model_still_reaches_its_provider(self):
        # The over-guarding falsifier: with groq's endpoint present, the same
        # request must route normally rather than 502.
        from lib.hal.gateway import _maybe_cross_wire
        sentinel = object()
        with mock.patch.dict(os.environ, _FLAG_ON), \
                mock.patch("lib.hal.gateway_egress.cross_wire_forward",
                           return_value=sentinel):
            self.assertIs(
                _maybe_cross_wire("/v1/messages", "POST",
                                  json.dumps(self.UNROUTABLE).encode("utf-8")),
                sentinel)


class TestEndpointTableMirrorsProvidersYaml(unittest.TestCase):
    """``PROVIDER_ENDPOINTS`` claims to mirror ``config/providers.yaml``.

    Line-scanned rather than yaml-parsed: the interpreter that RUNS the gateway
    (/opt/homebrew/bin/python3) has no pyyaml, and a test that can only run on a
    different interpreter than production is a test that silently skips.
    """

    @staticmethod
    def _declared_base_urls() -> dict:
        declared, name = {}, None
        for line in _PROVIDERS_YAML.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("- name:"):
                name = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("base_url:") and name:
                declared[name] = stripped.split(":", 1)[1].strip()
        return declared

    def test_every_cloud_endpoint_matches_its_declaration(self):
        declared = self._declared_base_urls()
        self.assertIn("mistral", declared, "providers.yaml must declare mistral")
        for provider, (base_url, _service) in sorted(egress.PROVIDER_ENDPOINTS.items()):
            self.assertEqual(declared.get(provider), base_url,
                             f"{provider} endpoint disagrees with providers.yaml")


if __name__ == "__main__":
    unittest.main()
