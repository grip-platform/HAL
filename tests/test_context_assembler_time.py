"""Tests for HAL real-time awareness in the context assembler.

GRIP's time-awareness hook injects a `[TIME]` reminder into every prompt; HAL
had no equivalent, so the model HAL routes to reasoned from its training-cutoff
sense of "now". This wires the parity: ContextAssembler now stamps the real
wall clock into the system prompt as the first section.

Goodhart-proof: the tests assert the actual stamped value (independently
recomputed), not call counts — they fail if the injection is removed or the
formatter drifts.
"""
import unittest
from datetime import datetime, timezone

from lib.hal.context_assembler import _build_system_prompt, _time_awareness_section

# 2026-05-30 16:26 UTC — a Saturday (epoch captured from a real `date` run).
_FIXED_EPOCH = 1780158385.0


class TestContextAssemblerTimeAwareness(unittest.TestCase):

    def test_section_contains_iso_date_utc_and_weekday(self):
        section = _time_awareness_section(_FIXED_EPOCH)
        self.assertIn("2026-05-30", section)
        self.assertIn("UTC", section)
        self.assertIn("Saturday", section)
        self.assertIn("Current time", section)

    def test_section_matches_independent_utc_recomputation(self):
        expect = datetime.fromtimestamp(_FIXED_EPOCH, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC (%A)"
        )
        self.assertIn(expect, _time_awareness_section(_FIXED_EPOCH))

    def test_build_system_prompt_injects_time_first(self):
        out = _build_system_prompt([], "", "", [], "", {}, now=_FIXED_EPOCH)
        self.assertTrue(
            out.startswith("## Current time (real wall clock)"),
            f"time section must lead the system prompt; got: {out[:80]!r}",
        )
        self.assertIn("2026-05-30", out)

    def test_time_present_alongside_other_sections(self):
        out = _build_system_prompt(
            ["# Instructions"], "code", "do X", ["skillA"], "mem",
            {"branch": "main"}, now=_FIXED_EPOCH,
        )
        self.assertIn("2026-05-30", out)
        self.assertIn("# Instructions", out)  # other sources still assembled

    def test_default_now_uses_live_clock(self):
        # No now= -> time.time(); still produces a valid current stamp.
        out = _build_system_prompt([], "", "", [], "", {})
        self.assertIn("Current time", out)
        self.assertIn("UTC", out)


if __name__ == "__main__":
    unittest.main()
