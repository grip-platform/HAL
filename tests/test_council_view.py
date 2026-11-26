"""Tests for Phase α W5 council view.

Verifies the four concern-addressing invariants from
``plans/council-view-verdict.md``:

  1. Narrow terminals / piped output / CI → SingleLineRenderer
  2. Cognitive overload → per-member box content stays minimal
  3. Low/ambiguous data → None agreement renders neutral (never
     painted), 0 tokens render dim
  4. rich unavailable → SingleLineRenderer
"""
from __future__ import annotations

import io
import os
import unittest
from unittest import mock

from lib.hal.council_view import (
    CouncilMemberState,
    CouncilViewConfig,
    RichLiveRenderer,
    SingleLineRenderer,
    agreement_colour,
    build_renderer,
    rich_available,
)


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def _members(*names: str) -> list[CouncilMemberState]:
    return [
        CouncilMemberState(name=n, role=f"role-{n}", model=f"model-{n}")
        for n in names
    ]


# ---------------------------------------------------------------------------
# agreement_colour — concern #3 invariant
# ---------------------------------------------------------------------------
class TestAgreementColour(unittest.TestCase):
    def test_none_returns_neutral(self):
        """None agreement MUST NOT pick a colour — invariant from concern #3."""
        self.assertEqual(agreement_colour(None), "neutral")

    def test_green_threshold(self):
        self.assertEqual(agreement_colour(0.85), "green")
        self.assertEqual(agreement_colour(0.99), "green")

    def test_amber_threshold(self):
        self.assertEqual(agreement_colour(0.60), "amber")
        self.assertEqual(agreement_colour(0.70), "amber")
        self.assertEqual(agreement_colour(0.849), "amber")

    def test_red_threshold(self):
        self.assertEqual(agreement_colour(0.59), "red")
        self.assertEqual(agreement_colour(0.0), "red")

    def test_custom_thresholds(self):
        self.assertEqual(agreement_colour(0.8, green=0.9, amber=0.5), "amber")
        self.assertEqual(agreement_colour(0.4, green=0.9, amber=0.5), "red")


# ---------------------------------------------------------------------------
# SingleLineRenderer — baseline
# ---------------------------------------------------------------------------
class TestSingleLineRenderer(unittest.TestCase):
    def test_start_writes_member_list(self):
        sink = io.StringIO()
        r = SingleLineRenderer(stream=sink)
        r.start(_members("analyst", "explorer"))
        output = sink.getvalue()
        self.assertIn("analyst", output)
        self.assertIn("explorer", output)

    def test_update_writes_status_line(self):
        sink = io.StringIO()
        r = SingleLineRenderer(stream=sink)
        r.start(_members("analyst"))
        sink.truncate(0); sink.seek(0)
        r.update("analyst", phase="debate", tokens_out=42, agreement=0.9, status="active")
        output = sink.getvalue()
        self.assertIn("analyst", output)
        self.assertIn("phase=debate", output)
        self.assertIn("tokens=42", output)
        self.assertIn("agreement=0.90", output)
        self.assertIn("[green]", output)
        self.assertIn("status=active", output)

    def test_update_with_none_agreement_shows_dash(self):
        """Concern #3: None agreement MUST render as '-', NOT 0.00."""
        sink = io.StringIO()
        r = SingleLineRenderer(stream=sink)
        r.start(_members("a"))
        sink.truncate(0); sink.seek(0)
        r.update("a", agreement=None)
        output = sink.getvalue()
        self.assertIn("agreement=-", output)
        self.assertIn("[neutral]", output)
        self.assertNotIn("[red]", output)

    def test_update_unknown_member_is_noop(self):
        sink = io.StringIO()
        r = SingleLineRenderer(stream=sink)
        r.start(_members("a"))
        sink.truncate(0); sink.seek(0)
        r.update("nonexistent", phase="x")
        self.assertEqual(sink.getvalue(), "")

    def test_update_before_start_is_noop(self):
        sink = io.StringIO()
        r = SingleLineRenderer(stream=sink)
        r.update("a", phase="x")
        self.assertEqual(sink.getvalue(), "")

    def test_finalize_writes_verdict(self):
        sink = io.StringIO()
        r = SingleLineRenderer(stream=sink)
        r.start(_members("a"))
        sink.truncate(0); sink.seek(0)
        r.finalize(verdict="PROCEED")
        self.assertIn("verdict: PROCEED", sink.getvalue())

    def test_finalize_without_verdict(self):
        sink = io.StringIO()
        r = SingleLineRenderer(stream=sink)
        r.start(_members("a"))
        sink.truncate(0); sink.seek(0)
        r.finalize()
        self.assertEqual(sink.getvalue(), "")

    def test_finalize_before_start_is_noop(self):
        sink = io.StringIO()
        r = SingleLineRenderer(stream=sink)
        r.finalize(verdict="PROCEED")
        self.assertEqual(sink.getvalue(), "")


# ---------------------------------------------------------------------------
# RichLiveRenderer — delight path (inspected without running Live)
# ---------------------------------------------------------------------------
@unittest.skipUnless(rich_available(), "rich library not installed")
class TestRichLiveRenderer(unittest.TestCase):
    def test_build_renderable_returns_columns_with_panels(self):
        from rich.columns import Columns
        r = RichLiveRenderer(stream=_FakeTTY())
        r._members = {m.name: m for m in _members("a", "b", "c")}
        renderable = r._build_renderable()
        self.assertIsInstance(renderable, Columns)

    def test_none_agreement_panel_renders_dim_dash(self):
        """Concern #3 in the rich path: None agreement never coloured."""
        from rich.console import Console
        r = RichLiveRenderer(stream=_FakeTTY())
        r._members = {
            "a": CouncilMemberState(name="a", role="r", model="m", agreement=None),
        }
        renderable = r._build_renderable()
        console = Console(file=io.StringIO(), width=80, record=True)
        console.print(renderable)
        text = console.export_text()
        self.assertIn("-", text)
        # Don't accidentally spell out a fake score.
        self.assertNotIn("0.00", text)
        self.assertNotIn("0.50", text)


# ---------------------------------------------------------------------------
# build_renderer factory — the safety envelope
# ---------------------------------------------------------------------------
class TestBuildRendererFactory(unittest.TestCase):
    def setUp(self):
        os.environ.pop("HAL_NO_COUNCIL_SPLIT", None)

    def test_non_tty_falls_back_to_single_line(self):
        """Concern #1: piped / CI → SingleLineRenderer."""
        r = build_renderer(stream=io.StringIO())  # not a TTY
        self.assertIsInstance(r, SingleLineRenderer)

    def test_hal_no_council_split_forces_single_line(self):
        """Operator escape hatch."""
        with mock.patch.dict(os.environ, {"HAL_NO_COUNCIL_SPLIT": "1"}):
            r = build_renderer(stream=_FakeTTY())
            self.assertIsInstance(r, SingleLineRenderer)

    def test_config_force_single_line_flag(self):
        cfg = CouncilViewConfig(force_single_line=True)
        r = build_renderer(stream=_FakeTTY(), config=cfg)
        self.assertIsInstance(r, SingleLineRenderer)

    def test_narrow_terminal_falls_back(self):
        """Concern #1 narrow-terminal path."""
        with mock.patch("lib.hal.council_view._terminal_width", return_value=40):
            r = build_renderer(stream=_FakeTTY())
            self.assertIsInstance(r, SingleLineRenderer)

    def test_rich_unavailable_falls_back(self):
        """Concern #4 — rich.live ImportError → SingleLineRenderer."""
        with mock.patch("lib.hal.council_view.rich_available", return_value=False):
            with mock.patch("lib.hal.council_view._terminal_width", return_value=120):
                r = build_renderer(stream=_FakeTTY())
                self.assertIsInstance(r, SingleLineRenderer)

    @unittest.skipUnless(rich_available(), "rich library not installed")
    def test_happy_path_returns_rich_live_renderer(self):
        with mock.patch("lib.hal.council_view._terminal_width", return_value=120):
            r = build_renderer(stream=_FakeTTY())
            self.assertIsInstance(r, RichLiveRenderer)

    def test_custom_min_cols_honoured(self):
        cfg = CouncilViewConfig(min_cols=200)  # arbitrarily strict
        with mock.patch("lib.hal.council_view._terminal_width", return_value=120):
            r = build_renderer(stream=_FakeTTY(), config=cfg)
            self.assertIsInstance(r, SingleLineRenderer)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
class TestCouncilMemberStateDefaults(unittest.TestCase):
    def test_default_values(self):
        m = CouncilMemberState(name="a", role="r", model="m")
        self.assertEqual(m.phase, "pending")
        self.assertEqual(m.tokens_out, 0)
        self.assertIsNone(m.agreement)
        self.assertEqual(m.status, "pending")


class TestConfigDefaults(unittest.TestCase):
    def test_config_defaults(self):
        c = CouncilViewConfig()
        self.assertEqual(c.min_cols, 80)
        self.assertEqual(c.agreement_green, 0.85)
        self.assertEqual(c.agreement_amber, 0.60)
        self.assertFalse(c.force_single_line)


if __name__ == "__main__":
    unittest.main()
