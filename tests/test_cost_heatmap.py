"""Tests for Phase α W6 cost heatmap + citation chip."""
import unittest

from lib.hal.cost_heatmap import (
    _BLOCK_FULL,
    _BLOCK_PARTS,
    _render_bar,
    format_citation_chip,
    render_heatmap,
)


class TestRenderBar(unittest.TestCase):
    def test_zero_fraction_is_all_space(self):
        self.assertEqual(_render_bar(0.0, 10), " " * 10)

    def test_full_fraction_is_all_full_blocks(self):
        self.assertEqual(_render_bar(1.0, 5), _BLOCK_FULL * 5)

    def test_above_one_clamps_to_full(self):
        self.assertEqual(_render_bar(1.5, 4), _BLOCK_FULL * 4)

    def test_half_fraction_half_bar(self):
        # 0.5 * 8 = 4 eighths, width 8 chars -> 4 full + 0 partial + 4 space.
        self.assertEqual(_render_bar(0.5, 8), _BLOCK_FULL * 4 + " " * 4)

    def test_small_fraction_produces_partial_block(self):
        bar = _render_bar(0.1, 10)  # 8 eighths -> 1 full block + 9 space
        self.assertEqual(bar[0], _BLOCK_FULL)
        self.assertEqual(bar[1:], " " * 9)

    def test_tiny_fraction_produces_partial_eighth(self):
        # 0.01 of 10 width = 0.8 eighths -> rounds to 1 -> 1/8 partial.
        bar = _render_bar(0.01, 10)
        self.assertEqual(bar[0], _BLOCK_PARTS[0])
        self.assertEqual(bar[1:], " " * 9)

    def test_exact_bar_width(self):
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertEqual(len(_render_bar(fraction, 20)), 20,
                             f"width mismatch at fraction={fraction}")

    def test_zero_width_returns_empty(self):
        self.assertEqual(_render_bar(0.5, 0), "")


class TestRenderHeatmap(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        self.assertEqual(render_heatmap({}), [])

    def test_single_provider(self):
        lines = render_heatmap({"openai": 0.042}, bar_width=10)
        self.assertEqual(len(lines), 1)
        self.assertIn("openai", lines[0])
        self.assertIn("$0.042000", lines[0])
        # Peak = 0.042 -> bar is full (width 10).
        self.assertIn(_BLOCK_FULL * 10, lines[0])
        # 100% of total.
        self.assertIn("100.0%", lines[0])

    def test_sorted_by_cost_desc(self):
        lines = render_heatmap({
            "groq": 0.01,
            "openai": 0.05,
            "ollama": 0.005,
        }, bar_width=10)
        self.assertEqual(len(lines), 3)
        # openai (0.05) should be first, then groq, then ollama.
        self.assertIn("openai", lines[0])
        self.assertIn("groq", lines[1])
        self.assertIn("ollama", lines[2])

    def test_bar_scaled_to_peak(self):
        lines = render_heatmap({
            "a": 0.10,
            "b": 0.05,
        }, bar_width=10)
        # Peak (a) -> 10 full blocks. b at 50% -> 5 full + 5 space.
        self.assertIn(_BLOCK_FULL * 10, lines[0])
        self.assertIn(_BLOCK_FULL * 5 + " " * 5, lines[1])

    def test_min_display_cost_filters(self):
        lines = render_heatmap({
            "free-tier": 0.0,
            "openai": 0.05,
        }, min_display_cost=0.001)
        self.assertEqual(len(lines), 1)
        self.assertIn("openai", lines[0])
        self.assertNotIn("free-tier", "".join(lines))

    def test_all_zero_costs_with_min_zero_returns_all_providers(self):
        lines = render_heatmap({"a": 0.0, "b": 0.0}, bar_width=10)
        # With min_display_cost=0 default, both pass the filter.
        # Both have zero cost -> 0.0% each (total = 0 -> protected).
        self.assertEqual(len(lines), 2)

    def test_percentage_sums_to_total(self):
        # The percentages in the rendered lines should sum close to 100%.
        lines = render_heatmap({
            "a": 0.10,
            "b": 0.05,
            "c": 0.03,
        }, bar_width=10)
        import re
        pcts = []
        for line in lines:
            match = re.search(r"(\d+\.\d+)%", line)
            if match:
                pcts.append(float(match.group(1)))
        self.assertAlmostEqual(sum(pcts), 100.0, places=0)


class TestCitationChip(unittest.TestCase):
    def test_zero_entries_returns_empty_string(self):
        self.assertEqual(format_citation_chip(0), "")
        self.assertEqual(format_citation_chip(-1), "")

    def test_singular_grammar(self):
        chip = format_citation_chip(1)
        self.assertIn("1 entry", chip)
        self.assertNotIn("1 entries", chip)
        self.assertIn("/provenance", chip)

    def test_plural_grammar(self):
        chip = format_citation_chip(5)
        self.assertIn("5 entries", chip)

    def test_custom_command(self):
        chip = format_citation_chip(3, command="/cite")
        self.assertIn("/cite for details", chip)
        self.assertNotIn("/provenance", chip)

    def test_large_entry_count(self):
        # Reasonable upper bound; shouldn't format-panic on large values.
        chip = format_citation_chip(1_000_000)
        self.assertIn("1000000 entries", chip)


if __name__ == "__main__":
    unittest.main()
