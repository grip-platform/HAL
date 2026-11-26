"""Tests for the HAL model→provider routing chokepoint.

Goodhart-proof, with the #3274 verbatim invariant asserted TWO-SIDED: every
claude-* must be verbatim AND no non-claude model may be — so a mutation that
sends Claude through a translator OR sends GPT/Gemini through the verbatim
Anthropic relay fails here.
"""
import unittest

from lib.hal.model_route import (
    _FAMILY_ROUTES,
    KNOWN_PROVIDERS,
    Route,
    route_model,
    strip_provider_prefix,
)

_CLAUDE = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5",
           "claude-3-5-sonnet-20241022", "claude-3-opus"]
_NON_CLAUDE = ["gpt-5.5", "gpt-4o", "chatgpt-4o", "o3-mini",
               "gemini-3.1-pro-preview", "gemini-2.5-flash", "gemma-4-27b",
               "grok-4", "deepseek-v4-pro", "qwen3.5-plus", "llama-3.3-70b-versatile"]


class TestFamilyRouting(unittest.TestCase):

    def test_openai_family(self):
        for m in ["gpt-5.5", "gpt-4o", "chatgpt-4o", "o3-mini"]:
            self.assertEqual(route_model(m).provider, "openai", m)

    def test_gemini_family(self):
        for m in ["gemini-3.1-pro-preview", "gemini-2.5-flash", "gemma-4-27b"]:
            self.assertEqual(route_model(m).provider, "gemini", m)

    def test_other_families(self):
        self.assertEqual(route_model("grok-4-0709").provider, "xai")
        self.assertEqual(route_model("deepseek-v4-pro").provider, "deepseek")
        self.assertEqual(route_model("qwen3.5-plus").provider, "qwen")
        self.assertEqual(route_model("llama-3.3-70b-versatile").provider, "groq")
        self.assertEqual(route_model("mistral-large-latest").provider, "mistral")
        self.assertEqual(route_model("mixtral-8x7b").provider, "mistral")

    def test_case_insensitive(self):
        self.assertEqual(route_model("GPT-5.5").provider, "openai")
        self.assertEqual(route_model("Gemini-3.1-Pro").provider, "gemini")


class TestThinking3274Invariant(unittest.TestCase):
    """The load-bearing #3274 guard — claude verbatim, nothing else verbatim."""

    def test_every_claude_is_verbatim_anthropic(self):
        for m in _CLAUDE:
            r = route_model(m)
            self.assertEqual(r.provider, "anthropic", m)
            self.assertTrue(r.verbatim, f"{m} MUST be verbatim (#3274)")
        # Defensive: a hyphen-less / future claude id must STILL be verbatim.
        self.assertTrue(route_model("claude3opus").verbatim)
        self.assertTrue(route_model("claudenext").verbatim)

    def test_no_non_claude_is_ever_verbatim(self):
        for m in _NON_CLAUDE:
            r = route_model(m)
            self.assertFalse(
                r.verbatim,
                f"{m} routed verbatim — would send a non-Anthropic model through "
                f"the byte-identical Anthropic relay (breaks #3274 / the provider)",
            )

    def test_case_insensitive_claude_still_verbatim(self):
        self.assertTrue(route_model("CLAUDE-OPUS-4-8").verbatim)


class TestUnknownFailsClosed(unittest.TestCase):

    def test_unknown_model_is_not_known(self):
        r = route_model("totally-made-up-9000")
        self.assertFalse(r.known)
        self.assertEqual(r.provider, "")

    def test_empty_and_none(self):
        self.assertFalse(route_model("").known)
        self.assertFalse(route_model(None).known)
        self.assertFalse(route_model("   ").known)

    def test_unknown_carries_a_reason(self):
        self.assertIn("no family match", route_model("xyz-1").reason)


class TestProviderNameContract(unittest.TestCase):
    """Every emitted provider must be a name the gateway/registry knows."""

    def test_all_routed_providers_are_known(self):
        for m in _CLAUDE + _NON_CLAUDE:
            provider = route_model(m).provider
            self.assertIn(provider, KNOWN_PROVIDERS, f"{m} → unknown provider {provider!r}")

    def test_every_family_table_provider_is_declared(self):
        """Derived from the table itself, so it CAN fail.

        The hand-written model list above happened to contain no moonshot/kimi
        id, so ``moonshot`` was emitted by ``_FAMILY_ROUTES`` while missing from
        ``KNOWN_PROVIDERS`` and this contract stayed green for both. Iterating
        the table means a name added there without being declared here fails —
        which also matters because ``KNOWN_PROVIDERS`` is now the allowlist for
        the explicit ``<provider>/`` address form.
        """
        for prefix, provider in _FAMILY_ROUTES:
            self.assertIn(provider, KNOWN_PROVIDERS,
                          f"family {prefix!r} → undeclared provider {provider!r}")

    def test_every_family_prefix_actually_routes(self):
        """A prefix that routes nowhere is a dead table row (spelling drift)."""
        for prefix, provider in _FAMILY_ROUTES:
            self.assertEqual(route_model(prefix + "x").provider, provider, prefix)

    def test_route_is_frozen(self):
        r = route_model("gpt-5.5")
        self.assertIsInstance(r, Route)
        with self.assertRaises(Exception):
            r.provider = "evil"  # frozen dataclass — immutable decision


class TestKimiCodeIdsNeverRouteBare(unittest.TestCase):
    """Kimi Code SUBSCRIPTION ids (provider ``kimi-coding``, host
    api.kimi.com/coding) carry no family marker the metered host knows, and two
    of them start with ``kimi`` — so the family table would have sent a bare
    ``kimi-for-coding`` to the pay-per-token ``moonshot`` upstream: wrong
    product, wrong host, a 404 with the metered key attached. A bare Kimi Code
    id therefore stays UNKNOWN and the reason tells the caller to write
    ``kimi-coding/<id>``. The address form routes at the gateway once
    ``kimi-coding`` joins ``KNOWN_PROVIDERS`` together with its
    ``gateway_egress`` endpoint (tests/test_gateway_unroutable_provider.py
    binds the two) — the Pool D seam ships unarmed (council seal
    e49c00c214c500be, 2026-09-07)."""

    _IDS = ("k3", "k3-256k", "kimi-for-coding", "kimi-for-coding-highspeed")

    def test_bare_kimi_code_ids_are_unknown_not_moonshot(self):
        for m in self._IDS + ("K3", "Kimi-For-Coding"):
            r = route_model(m)
            self.assertFalse(r.known, m)
            self.assertNotEqual(r.provider, "moonshot", m)
            self.assertFalse(r.verbatim, m)

    def test_reason_names_the_mandatory_address_form(self):
        for m in self._IDS:
            self.assertIn(f"kimi-coding/{m}", route_model(m).reason, m)

    def test_address_form_routes_to_kimi_coding_never_moonshot(self):
        # INVERTED 2026-09-07 (tests-must-be-able-to-fail.md mandate 4) when the
        # gateway_egress endpoint row landed and ``kimi-coding`` joined
        # KNOWN_PROVIDERS. Before the guard, all three address forms routed to
        # moonshot, known=True, with the 'kimi-coding/' prefix still on; while
        # the row was pending they were UNKNOWN with a reason naming
        # KNOWN_PROVIDERS. Now the explicit-prefix branch claims them: known,
        # provider kimi-coding, upstream id the bare vendor id, never moonshot.
        for m in self._IDS + ("K3",):
            r = route_model(f"kimi-coding/{m}")
            self.assertTrue(r.known, m)
            self.assertEqual(r.provider, "kimi-coding", m)
            self.assertFalse(r.verbatim, m)
            # The prefix is stripped; the id keeps the caller's spelling (the vendor
            # says to spell ids exactly, so the router never re-cases them).
            self.assertEqual(strip_provider_prefix(f"kimi-coding/{m}", r.provider), m, m)

    def test_metered_kimi_ids_still_route_to_moonshot(self):
        for m in ("kimi-k3", "kimi-k2.6", "kimi-k2.7-code", "moonshot/kimi-k3"):
            self.assertEqual(route_model(m).provider, "moonshot", m)

    def test_address_form_strips_to_the_bare_vendor_id(self):
        self.assertEqual(strip_provider_prefix("kimi-coding/k3", "kimi-coding"), "k3")
        self.assertEqual(
            strip_provider_prefix("kimi-coding/kimi-for-coding-highspeed", "kimi-coding"),
            "kimi-for-coding-highspeed")


if __name__ == "__main__":
    unittest.main()
