"""Regression anchor: the ``<provider>/<model>`` address form must route.

THE DEFECT (live, 2026-08-17). ``config/model_catalog.yaml`` addresses every
model as ``<provider>/<model>`` (``groq/openai/gpt-oss-120b`` at line 452), and
``docs/provider-routing.md`` names ``provider/model`` as the "Fixed" strategy's
address form. ``model_route.route_model`` did NOT understand it: it matched only
a FAMILY PREFIX at position 0, so ``groq/openai/gpt-oss-120b`` matched nothing,
``Route.known`` was False, ``same_wire_provider`` returned ``None`` — and the
caller (``gateway_openai._dispatch_openai_chat``) treats ``None`` as "use the
Anthropic path". A Groq key was therefore relayed to api.anthropic.com as
``x-api-key`` and correctly rejected:

    POST 127.0.0.1:4010/v1/chat/completions {"model":"groq/openai/gpt-oss-120b"}
    -> HTTP 401 {"error":{"message":"invalid x-api-key","type":"gateway_error"}}

That silent fall-through contradicts ``model_route``'s own stated contract
("the caller must surface a clear error, never silently misroute to the default
upstream"). These tests pin BOTH halves of the fix, because either alone still
fails end to end:

  1. the ROUTING decision  — ``groq/...`` must resolve to provider ``groq``;
  2. the OUTBOUND BODY     — the ``groq/`` prefix must be stripped before the
     request reaches Groq, which knows the model as ``openai/gpt-oss-120b``
     (verified live: GET api.groq.com/openai/v1/models lists exactly that id).

Goodhart notes. The strip is PROVIDER-SCOPED — only a head segment equal to the
provider being forwarded to is removed. That is what stops it eating the leading
segment of a model id that legitimately contains a slash: ``gateway_fanout``
sends the rung ``("groq", "openai/gpt-oss-120b")`` with the bare model in the
body, and a naive "strip any known-provider head" would mangle it to
``gpt-oss-120b`` and 404. ``test_cross_provider_head_is_never_stripped`` is that
mutation's executioner.

Hermetic: opener + key_resolver injected, no network, no keychain.
"""
import json
import unittest

from lib.hal.gateway_egress import forward_same_wire, same_wire_provider
from lib.hal.model_route import _split_provider_head, route_model, strip_provider_prefix


class _FakeResp:
    """Minimal opener response (status/headers/read) — mirrors test_gateway_egress."""

    def __init__(self, status=200, headers=None, body=b"{}"):
        self.status = status
        self.headers = headers or [("Content-Type", "application/json")]
        self._body = body
        self._read = False

    def read(self, size=-1):
        if self._read:
            return b""
        self._read = True
        return self._body


def _capture_opener(sink):
    def opener(method, url, headers, body):
        sink["method"], sink["url"], sink["headers"], sink["body"] = (
            method, url, headers, body)
        return _FakeResp(status=200)
    return opener


def _forwarded_model(provider: str, model: str) -> str:
    """The ``model`` value that actually reaches ``provider`` upstream."""
    sink = {}
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "x"}]})
    forward_same_wire(provider, "/v1/chat/completions", body.encode("utf-8"),
                      opener=_capture_opener(sink), key_resolver=lambda s: "k")
    return json.loads(sink["body"])["model"]


class TestExplicitProviderPrefixRoutes(unittest.TestCase):
    """Half 1 — the routing decision."""

    def test_catalog_address_form_routes_to_groq(self):
        # The exact address in config/model_catalog.yaml:452.
        self.assertEqual(same_wire_provider("groq/openai/gpt-oss-120b"), "groq")

    def test_catalog_address_is_known_and_not_verbatim(self):
        r = route_model("groq/openai/gpt-oss-120b")
        self.assertTrue(r.known, "unknown => silent fall-through to Anthropic")
        self.assertFalse(r.verbatim)
        self.assertEqual(r.provider, "groq")

    def test_prefix_beats_the_family_table(self):
        # Bare 'gpt-oss-120b' family-matches 'gpt-' -> openai (OpenAI 404s it).
        # The explicit groq/ prefix is the operator's stated intent and wins.
        self.assertEqual(route_model("gpt-oss-120b").provider, "openai")
        self.assertEqual(route_model("groq/openai/gpt-oss-120b").provider, "groq")

    def test_other_providers_address_the_same_way(self):
        for address, provider in (
            ("groq/llama-3.3-70b-versatile", "groq"),
            ("openai/gpt-4o", "openai"),
            ("gemini/gemini-2.5-flash", "gemini"),
            ("deepseek/deepseek-chat", "deepseek"),
            ("xai/grok-4", "xai"),
            ("moonshot/kimi-k2", "moonshot"),
        ):
            self.assertEqual(same_wire_provider(address), provider, address)

    def test_unknown_head_is_still_unknown(self):
        # 'notaprovider' is not a provider, so this is NOT an address — it must
        # stay unknown rather than being invented into a route.
        self.assertFalse(route_model("notaprovider/some-model").known)

    def test_bare_prefix_with_no_model_is_not_an_address(self):
        self.assertFalse(route_model("groq/").known)


class TestThinking3274InvariantUnderPrefixes(unittest.TestCase):
    """The load-bearing #3274 guard, restated for the address form.

    Over-routing to the Anthropic verbatim relay is safe; under-routing (a
    Claude model sent through a deserialising translator) is the regression that
    drops thinking-block signatures.
    """

    def test_anthropic_prefixed_claude_is_verbatim(self):
        r = route_model("anthropic/claude-opus-4-8")
        self.assertEqual(r.provider, "anthropic")
        self.assertTrue(r.verbatim)

    def test_anthropic_prefixed_never_same_wires(self):
        # Must fall through to the verbatim Anthropic path, never an egress provider.
        self.assertIsNone(same_wire_provider("anthropic/claude-opus-4-8"))

    def test_no_non_anthropic_prefix_is_ever_verbatim(self):
        for address in ("groq/openai/gpt-oss-120b", "openai/gpt-4o",
                        "gemini/gemini-2.5-flash", "groq/claude-lookalike"):
            self.assertFalse(route_model(address).verbatim, address)


class TestEndpointTableIntegrity(unittest.TestCase):
    """Duplicate keys in PROVIDER_ENDPOINTS silently overwrite — the AGORA #534
    follow-up anchor. A re-introduced dup (merge artifact) turns this red."""

    def test_provider_endpoint_keys_are_unique(self):
        from lib.hal.gateway_egress import PROVIDER_ENDPOINTS, LOCAL_PROVIDER_ENDPOINTS
        self.assertEqual(len(PROVIDER_ENDPOINTS), len(set(PROVIDER_ENDPOINTS)))
        self.assertEqual(len(LOCAL_PROVIDER_ENDPOINTS),
                         len(set(LOCAL_PROVIDER_ENDPOINTS)))


class TestOutboundBodyCarriesTheUpstreamModelId(unittest.TestCase):
    """Half 2 — what actually reaches the provider."""

    def test_groq_receives_the_id_groq_publishes(self):
        # api.groq.com publishes 'openai/gpt-oss-120b' (live-verified 2026-08-17).
        self.assertEqual(
            _forwarded_model("groq", "groq/openai/gpt-oss-120b"),
            "openai/gpt-oss-120b")

    def test_single_segment_model_loses_its_prefix(self):
        self.assertEqual(
            _forwarded_model("deepseek", "deepseek/deepseek-chat"), "deepseek-chat")

    def test_cross_provider_head_is_never_stripped(self):
        # gateway_fanout rung ("groq", "openai/gpt-oss-120b"): the head names a
        # KNOWN provider but NOT the one being forwarded to, so it is part of the
        # model id and must survive byte-identical.
        self.assertEqual(
            _forwarded_model("groq", "openai/gpt-oss-120b"), "openai/gpt-oss-120b")

    def test_only_one_prefix_is_removed(self):
        # Groq really serves an id called 'groq/compound', addressed groq/groq/compound.
        self.assertEqual(
            _forwarded_model("groq", "groq/groq/compound"), "groq/compound")

    def test_unprefixed_body_is_byte_identical(self):
        # Fail-open/verbatim contract of _fix_cloud_body: a no-op transform must
        # not reserialise (key order + whitespace preserved).
        sink = {}
        body = b'{"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":"x"}]}'
        forward_same_wire("groq", "/v1/chat/completions", body,
                          opener=_capture_opener(sink), key_resolver=lambda s: "k")
        self.assertEqual(sink["body"], body)


class TestStripProviderPrefixHelper(unittest.TestCase):
    """The pure address-grammar helper, exercised directly."""

    def test_strips_only_the_matching_head(self):
        self.assertEqual(
            strip_provider_prefix("groq/openai/gpt-oss-120b", "groq"),
            "openai/gpt-oss-120b")
        self.assertEqual(
            strip_provider_prefix("openai/gpt-oss-120b", "groq"),
            "openai/gpt-oss-120b")

    def test_case_insensitive_head_match(self):
        self.assertEqual(strip_provider_prefix("GROQ/Llama-Whatever", "groq"),
                         "Llama-Whatever")

    def test_degenerate_inputs_pass_through(self):
        self.assertEqual(strip_provider_prefix("", "groq"), "")
        self.assertEqual(strip_provider_prefix("groq/", "groq"), "groq/")
        self.assertEqual(strip_provider_prefix("plain-model", "groq"), "plain-model")
        self.assertEqual(strip_provider_prefix(None, "groq"), "")


class TestSplitProviderHeadGrammar(unittest.TestCase):
    """The shared partition helper both prefix sites use (AGORA #534 C2)."""

    def test_plain_model_has_no_head(self):
        self.assertEqual(_split_provider_head("gpt-5.5"), ("", "gpt-5.5"))

    def test_head_is_lowercased_tail_keeps_case(self):
        self.assertEqual(_split_provider_head("GROQ/Llama-Whatever"),
                         ("groq", "Llama-Whatever"))

    def test_bare_prefix_has_empty_tail(self):
        self.assertEqual(_split_provider_head("groq/"), ("groq", ""))

    def test_empty_and_nonstring(self):
        self.assertEqual(_split_provider_head(""), ("", ""))
        self.assertEqual(_split_provider_head(None), ("", ""))

    def test_only_first_segment_is_the_head(self):
        self.assertEqual(_split_provider_head("mistral/groq/llama-x"),
                         ("mistral", "groq/llama-x"))


class TestExplicitHeadBeatsFamilyAndStripIsProviderScoped(unittest.TestCase):
    """The falsifier AGORA #534 asked for: a family-prefixed id whose explicit
    head names a DIFFERENT known provider."""

    def test_mistral_head_wins_over_groq_family_content(self):
        # "groq/llama-x" inside would family-route to groq, but the EXPLICIT
        # head is the operator's stated intent — it wins.
        route = route_model("mistral/groq/llama-x")
        self.assertEqual(route.provider, "mistral")

    def test_strip_removes_only_the_forwarding_provider_segment(self):
        self.assertEqual(
            strip_provider_prefix("mistral/groq/llama-x", "mistral"),
            "groq/llama-x")
        # a DIFFERENT forwarding provider leaves the whole id untouched
        self.assertEqual(
            strip_provider_prefix("mistral/groq/llama-x", "groq"),
            "mistral/groq/llama-x")


if __name__ == "__main__":
    unittest.main()
