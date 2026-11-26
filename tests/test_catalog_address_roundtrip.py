"""Round-trip every catalog address through the routing chokepoint (AGORA #535 S1).

Parses ``config/model_catalog.yaml`` — the live registry every picker/roster
reads — and asserts, for every address a CLOUD provider serves, that:

  1. ``route_model`` resolves it to a KNOWN provider (never the silent
     Anthropic fall-through);
  2. ``strip_provider_prefix`` yields the upstream model id the provider
     actually knows (a non-empty tail);
  3. the resulting (provider, upstream-id) pair is reachable through
     ``same_wire_provider`` when multiprovider is on (an endpoint exists).

``tier: local`` rows are held to a DIFFERENT contract (see
``test_every_local_address_head_is_a_declared_local_provider``), because the
gateway deliberately keeps local providers out of the implicit router: no
``_FAMILY_ROUTES`` entry maps to them, so ``same_wire_provider`` never selects
one and only an EXPLICIT (provider, model) rung forwards there
(``gateway_egress.LOCAL_PROVIDER_ENDPOINTS``, and its comment on ``ollama``).
Asserting implicit routability of a local address measures a property the
design says it must not have.

Goodhart note: inputs are DERIVED from the catalog file — and the local-provider
set from ``config/providers.yaml`` — not hand-listed, so adding a bad address to
either file turns this test red. Hermetic: no network, no keychain. Skipped
(with reason) only when the catalog is unreadable — a corrupt catalog fails the
yaml-load tests elsewhere, not this one.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from lib.hal.gateway_egress import PROVIDER_ENDPOINTS, LOCAL_PROVIDER_ENDPOINTS, same_wire_provider
from lib.hal.model_route import KNOWN_PROVIDERS, route_model, strip_provider_prefix

_CATALOG = Path(__file__).resolve().parents[1] / "config" / "model_catalog.yaml"
_PROVIDERS = Path(__file__).resolve().parents[1] / "config" / "providers.yaml"

# Catalogue rows whose head is not a declared provider, kept OUT of the
# hard-fail set until the address itself is decided (HAL#554 follow-up).
# Asserted as a SUPERSET, never an equality: fixing one of these keeps the test
# green (it must not require the defect — tests-must-be-able-to-fail shape 4),
# while any NEW malformed local address turns it red.
#
#   microsoft/bitnet-b1.58-2b-4t — head "microsoft" is the model's VENDOR, not a
#   provider; the provider that serves it is "bitnet" (type: local,
#   127.0.0.1:8114). Its tail also disagrees with the id providers.yaml declares
#   for that provider ("bitnet-b1.58-2b4t"). Renaming it touches 5 call sites
#   under examples/gridlm/ and wants a live probe of the llama-server, so the
#   address decision is deliberately not made here.
_UNDECIDED_LOCAL_ADDRESSES = frozenset({"microsoft/bitnet-b1.58-2b-4t"})


@unittest.skipUnless(_CATALOG.exists(), "model_catalog.yaml missing — nothing to round-trip")
class TestCatalogAddressRoundTrip(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with _CATALOG.open() as fh:
            raw = yaml.safe_load(fh) or {}
        # The catalogue is a MAPPING whose one top-level key, models:, holds the
        # list of entries. Iterating the mapping yields the string "models",
        # which is not a dict — so a direct iteration collects nothing and every
        # assertion below runs over an empty set and cannot fail
        # (rules/tests-must-be-able-to-fail.md shape 3, HAL#554). The anchor
        # test_addresses_were_actually_collected keeps that honest.
        entries = [e for e in (raw.get("models") or [])
                   if isinstance(e, dict) and e.get("address")]
        cls._addresses = [e["address"] for e in entries]
        cls._local_addresses = [e["address"] for e in entries if e.get("tier") == "local"]
        cls._cloud_addresses = [e["address"] for e in entries if e.get("tier") != "local"]

        # Deliberately NOT guarded by a skipUnless(_PROVIDERS.exists()) the way
        # the catalogue is (AGORA #565 concern 4, whose suggested fix is the
        # regression): a skip would let the local-address contract below stop
        # running SILENTLY, which is the same vacuity this file exists to remove.
        # providers.yaml is tracked config every provider path already needs, so
        # its absence is a broken checkout and a loud FileNotFoundError naming the
        # path is the honest outcome.
        with _PROVIDERS.open() as fh:
            pconf = yaml.safe_load(fh) or {}
        cls._local_providers = frozenset(
            p["name"] for p in (pconf.get("providers") or [])
            if isinstance(p, dict) and p.get("name") and p.get("type") == "local"
        )

    def test_addresses_were_actually_collected(self):
        # Anti-vacuity anchor. Every other assertion here is a statement about a
        # set; if a set is empty the statement is true for free.
        self.assertGreater(len(self._cloud_addresses), 0, "no cloud addresses collected")
        self.assertGreater(len(self._local_addresses), 0, "no local addresses collected")
        self.assertGreater(len(self._local_providers), 0,
                           "no type: local providers read from providers.yaml")
        self.assertEqual(len(self._addresses),
                         len(self._cloud_addresses) + len(self._local_addresses),
                         "tier partition lost or double-counted an address")

    def test_every_catalog_address_routes_to_a_known_provider(self):
        unknown = [(a, route_model(a).reason) for a in self._cloud_addresses
                   if not route_model(a).known]
        self.assertEqual(unknown, [], "catalog addresses that route nowhere (silent-misroute risk)")

    def test_every_catalog_address_strips_to_a_nonempty_upstream_id(self):
        broken = []
        for a in self._cloud_addresses:
            route = route_model(a)
            upstream = strip_provider_prefix(a, route.provider)
            if not upstream or upstream == a or "/" in upstream and upstream.count("/") > 1:
                # an upstream id MAY contain one slash (groq serves openai/gpt-oss-120b);
                # more than one means a prefix was not stripped
                broken.append((a, upstream))
        self.assertEqual(broken, [], "catalog addresses whose upstream id is empty/unstripped")

    def test_every_catalog_address_provider_has_a_gateway_endpoint(self):
        from lib.hal.gateway_egress import multiprovider_enabled
        if not multiprovider_enabled():
            self.skipTest("multiprovider disabled — endpoint reachability is opt-in")
        unroutable = []
        for a in self._cloud_addresses:
            route = route_model(a)
            if not route.known or route.provider == "anthropic":
                continue  # verbatim relay has no PROVIDER_ENDPOINTS entry by design
            provider = same_wire_provider(a)
            if provider is None:
                unroutable.append((a, route.provider))
        self.assertEqual(unroutable, [], "catalog addresses a known provider cannot serve")

    def test_every_local_address_head_is_a_declared_local_provider(self):
        """The contract a ``tier: local`` row IS held to.

        Implicit routability is the wrong question for these (see module
        docstring), but the address must still name the provider that serves it,
        or an explicit rung has nothing to resolve. Deriving the permitted heads
        from providers.yaml also closes the dodge of relabelling a cloud row
        ``tier: local`` to escape the round-trip above: ``openai`` is not a
        declared local provider, so the relabelled row lands here instead.
        """
        offenders = {a for a in self._local_addresses
                     if a.split("/", 1)[0] not in self._local_providers}
        self.assertLessEqual(
            offenders, _UNDECIDED_LOCAL_ADDRESSES,
            "local catalog addresses whose head is not a declared type: local provider "
            f"(known local providers: {sorted(self._local_providers)})")
