"""The four ``kimi-coding/*`` catalogue rows, proven against the REAL consumer.

The rows below are the exact text destined for ``config/model_catalog.yaml``.
They are NOT in that file yet: ``tests/test_catalog_address_roundtrip.py``
binds every catalogue address to a ``gateway_egress`` endpoint, and the
``kimi-coding`` endpoint row belongs to the gateway lane (W2). Rather than ship
prose that says "these rows will parse", this test feeds them through
``lib.hal.models_command`` -- the hand-rolled parser and renderer behind
``hal --models`` -- so the follow-up lands verified text, not an invented shape
(``tests-must-be-able-to-fail.md`` mandate 5).

Vendor facts quoted from kimi.com/code/docs/en/kimi-code/models and
/kimi-code/membership. ``paid: true`` although the bill is a membership quota:
``rules/adding-a-new-llm-to-hal.md`` section 0 records that the catalogue's
header comment ("false if free or flat-fee") disagrees with every LIVE
``tier: subscription`` row -- all six ``anthropic/*`` subscription rows carry
``paid: true`` (verified 2026-09-07) -- and rules "Follow the live entries, not
the comment". ``k3`` is recorded at 262144, the plan-independent floor: the
vendor's table says "Up to 1M (for higher-tier members)" / "1M context for
Allegretto and above", and no credential exists to know the plan. Hermetic:
temp file, stub registry, no network.
"""
from __future__ import annotations

import os
import tempfile
import unittest

from lib.hal.models_command import (
    annotate_availability,
    filter_entries,
    handle_models_command,
    load_catalogue,
)

KIMI_CODING_CATALOG_ROWS = """\
  # Kimi Code -- Moonshot's Kimi membership coding service (GRIP Pool D,
  # sovereign-council seal e49c00c214c500be, 2026-09-07). A SEPARATE product
  # from the pay-per-token moonshot/* rows: "the Kimi Code API is independent
  # from the API service on this platform". Host api.kimi.com/coding,
  # credential keychain grip-kimi-coding (env KIMI_CODING_API_KEY), ids
  # k3 / k3-256k / kimi-for-coding / kimi-for-coding-highspeed. Billed as a
  # membership quota, not per token: "refreshes automatically every 7 days
  # ... a rolling 5-hour rate window ... Requests from the CLI, VS Code, and
  # third-party tools all count toward that quota". UNARMED until the
  # credential exists; lands together with the gateway_egress endpoint row.
  - address: kimi-coding/k3
    tier: subscription
    paid: true
    modality: text+vision
    # 262144 is the floor every K3-capable plan has (Moderato). Allegretto and
    # above get 1048576 -- "manually set the context-window field to 1048576
    # to use K3's full up-to-1M context" -- raise this only once the plan is
    # known; a 1M value on a Moderato key would over-fill the window.
    context_window: 262144
    tags: [subscription, coding, pool-d, frontier, reasoning, long-context, vision]
    strengths:
      - Kimi K3 (2.8T parameters) on the Kimi Code membership quota, not per token
      - 1M context on Allegretto and above; 262144 on Moderato
      - reasoning_effort low / high / max (default high on this endpoint)
      - Image and video input
    description: >
      The Kimi Code flagship. "k3 (1M) consumes about twice as much quota as
      k3-256k" -- prefer k3-256k unless the 1M window is needed. A 401 here
      can mean "the requested capability exceeds your plan's entitlements",
      not a bad key.

  - address: kimi-coding/k3-256k
    tier: subscription
    paid: true
    modality: text+vision
    context_window: 262144
    tags: [subscription, coding, pool-d, reasoning, long-context, vision]
    strengths:
      - Same Kimi K3 within 256k context, about half the quota of k3
      - reasoning_effort low / high / max
      - Image input only (no video)
    description: >
      The quota-friendly K3: "Within 256k context, it delivers the same
      results." The GRIP ladder's Pool D rung. Moderato and above.

  - address: kimi-coding/kimi-for-coding
    tier: subscription
    paid: true
    modality: text+vision
    context_window: 262144
    tags: [subscription, coding, pool-d, agentic, vision]
    strengths:
      - Kimi K2.7 Code, thinking always on; output up to 32768 tokens
      - Available on every membership tier (Andante and above)
      - Image and video input
    description: >
      The Kimi Code K2.7 Code model -- "good at code completion and routine
      development tasks".

  - address: kimi-coding/kimi-for-coding-highspeed
    tier: subscription
    paid: true
    modality: text+vision
    context_window: 262144
    tags: [subscription, coding, pool-d, fast, agentic, vision]
    strengths:
      - K2.7 Code served at about six times the output speed
      - '"6× speed, 3× quota usage" per the vendor -- three times the quota of kimi-for-coding'
      - Allegretto plan or above
    description: >
      The high-speed K2.7 Code. A mistyped id "silently falls back to the
      standard kimi-for-coding -- no error, no speedup", so spell it exactly.
"""

_EXPECTED_WINDOWS = {
    "kimi-coding/k3": 262_144,      # plan-independent floor; 1_048_576 on Allegretto+
    "kimi-coding/k3-256k": 262_144,
    "kimi-coding/kimi-for-coding": 262_144,
    "kimi-coding/kimi-for-coding-highspeed": 262_144,
}


class _StubRegistry:
    def list_providers(self):
        return []


class TestKimiCodingCatalogRows(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fd, cls.path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("models:\n\n" + KIMI_CODING_CATALOG_ROWS)
        cls.entries = load_catalogue(cls.path)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.path)

    def test_rows_are_valid_yaml_for_pyyaml_too(self):
        # models_command's hand-rolled parser accepted a strengths bullet that
        # PyYAML rejects (a quoted string followed by more text), and
        # tests/test_catalog_address_roundtrip.py reads the live catalogue with
        # PyYAML -- the first paste broke it at setup. Both parsers must agree.
        import yaml
        doc = yaml.safe_load("models:\n\n" + KIMI_CODING_CATALOG_ROWS)
        self.assertEqual({e["address"] for e in doc["models"]}, set(_EXPECTED_WINDOWS))

    def test_exactly_the_four_vendor_ids_parse(self):
        self.assertEqual({e.address for e in self.entries}, set(_EXPECTED_WINDOWS))
        self.assertEqual(len(self.entries), 4)

    def test_every_row_is_a_quota_billed_subscription(self):
        for e in self.entries:
            with self.subTest(address=e.address):
                self.assertEqual(e.tier, "subscription")
                # Matches the six live anthropic/* subscription rows (all
                # paid: true), per rules/adding-a-new-llm-to-hal.md section 0.
                self.assertTrue(e.paid, "subscription rows are paid: true in the live catalogue")
                self.assertEqual(e.modality, "text+vision")
                self.assertEqual(e.context_window, _EXPECTED_WINDOWS[e.address])
                self.assertTrue({"subscription", "coding", "pool-d"} <= set(e.tags))
                self.assertTrue(e.strengths and e.description)

    def test_hal_models_kimi_coding_lists_all_four(self):
        out = handle_models_command(_StubRegistry(), "kimi-coding",
                                    catalogue_path=self.path, subscriptions={})
        for address in _EXPECTED_WINDOWS:
            self.assertIn(address, out)

    def test_status_follows_the_subscription_presence_gate(self):
        unarmed = annotate_availability(self.entries, _StubRegistry(), {})
        self.assertEqual({a.status for a in unarmed}, {"setup_needed"})
        armed = annotate_availability(self.entries, _StubRegistry(), {"kimi-coding": object()})
        self.assertEqual({a.status for a in armed}, {"active"})
        # A resolved METERED moonshot key must not light these rows up.
        metered_only = annotate_availability(self.entries, _StubRegistry(), {"moonshot": object()})
        self.assertEqual({a.status for a in metered_only}, {"setup_needed"})

    def test_pool_d_tag_filters_to_exactly_these_rows(self):
        annotated = annotate_availability(self.entries, _StubRegistry(), {})
        self.assertEqual(len(filter_entries(annotated, "pool-d")), 4)


if __name__ == "__main__":
    unittest.main()
