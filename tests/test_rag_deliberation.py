"""Tests for Phase β W4 — RAG-aware phase prompt composer (H-RAG_DELIB)."""
from __future__ import annotations

import unittest

from lib.hal.rag_deliberation import (
    Phase,
    RagContext,
    all_phases,
    compose_phase_prompt,
    injection_spec,
)


class TestInjectionSpec(unittest.TestCase):
    def test_gut_feel_receives_rag(self):
        """The H-RAG_DELIB invariant: only phase 0 gets RAG."""
        self.assertTrue(injection_spec()[Phase.GUT_FEEL])

    def test_synthesis_excluded(self):
        self.assertFalse(injection_spec()[Phase.SYNTHESIS])

    def test_debate_excluded(self):
        self.assertFalse(injection_spec()[Phase.DEBATE])

    def test_verdict_excluded(self):
        self.assertFalse(injection_spec()[Phase.VERDICT])

    def test_exactly_one_phase_enabled(self):
        """If someone ever flips another phase to True, this test fails loudly."""
        enabled = [p for p, on in injection_spec().items() if on]
        self.assertEqual(enabled, [Phase.GUT_FEEL])

    def test_covers_all_phases(self):
        self.assertEqual(set(injection_spec().keys()), set(all_phases()))


class TestComposePhasePrompt(unittest.TestCase):
    def setUp(self):
        self.base = "Deliberate on claim X."
        self.rag = RagContext(
            recalled=["fact A", "fact B", "fact C"],
        )

    def test_gut_feel_contains_recall_block(self):
        out = compose_phase_prompt(self.base, Phase.GUT_FEEL, rag=self.rag)
        self.assertIn("fact A", out)
        self.assertIn("fact B", out)
        self.assertIn("initial take", out.lower())
        self.assertIn(self.base, out)

    def test_synthesis_omits_recall_even_with_rag(self):
        out = compose_phase_prompt(self.base, Phase.SYNTHESIS, rag=self.rag)
        self.assertNotIn("fact A", out)
        self.assertNotIn("Relevant memory", out)
        self.assertEqual(out, self.base)

    def test_debate_omits_recall(self):
        out = compose_phase_prompt(self.base, Phase.DEBATE, rag=self.rag)
        self.assertNotIn("fact A", out)
        self.assertEqual(out, self.base)

    def test_verdict_omits_recall(self):
        out = compose_phase_prompt(self.base, Phase.VERDICT, rag=self.rag)
        self.assertNotIn("fact A", out)
        self.assertEqual(out, self.base)

    def test_unknown_phase_defaults_to_no_injection(self):
        """Fail-closed: unknown phases get the plain base prompt."""
        out = compose_phase_prompt(self.base, "unknown_phase", rag=self.rag)
        self.assertEqual(out, self.base)

    def test_no_rag_argument_is_plain_base(self):
        self.assertEqual(compose_phase_prompt(self.base, Phase.GUT_FEEL), self.base)

    def test_empty_recall_produces_plain_base(self):
        empty_rag = RagContext(recalled=[])
        out = compose_phase_prompt(self.base, Phase.GUT_FEEL, rag=empty_rag)
        self.assertEqual(out, self.base)

    def test_max_entries_truncates(self):
        crowded = RagContext(
            recalled=[f"entry-{i}" for i in range(20)],
            max_entries=3,
        )
        out = compose_phase_prompt(self.base, Phase.GUT_FEEL, rag=crowded)
        self.assertIn("entry-0", out)
        self.assertIn("entry-1", out)
        self.assertIn("entry-2", out)
        self.assertNotIn("entry-3", out)
        # Truncation marker renders the drop count.
        self.assertIn("17 more recall entries truncated", out)

    def test_header_and_footer_customisable(self):
        custom = RagContext(
            recalled=["only one"],
            header="## CUSTOM HEADER",
            footer="CUSTOM FOOTER LINE",
        )
        out = compose_phase_prompt(self.base, Phase.GUT_FEEL, rag=custom)
        self.assertIn("## CUSTOM HEADER", out)
        self.assertIn("CUSTOM FOOTER LINE", out)

    def test_all_phases_sequence_only_gut_feel_differs(self):
        """Running all four phases with the same RAG shows exactly one differs from base."""
        outs = {
            p: compose_phase_prompt(self.base, p, rag=self.rag)
            for p in all_phases()
        }
        differs = [p for p, txt in outs.items() if txt != self.base]
        self.assertEqual(differs, [Phase.GUT_FEEL])


class TestRagContextDefaults(unittest.TestCase):
    def test_default_values(self):
        c = RagContext()
        self.assertEqual(c.recalled, [])
        self.assertEqual(c.max_entries, 5)
        self.assertTrue(c.header)
        self.assertTrue(c.footer)


if __name__ == "__main__":
    unittest.main()
