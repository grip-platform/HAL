"""Tests for the HAL Phase α W2 status footer."""
from __future__ import annotations

import io
import os
import signal
import unittest
from unittest import mock

from lib.hal.status_footer import FooterState, StatusFooter


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def _make_state(**kwargs) -> FooterState:
    defaults = dict(providers=3, model="groq/llama-3.3-70b-versatile",
                    cost_usd=0.012345, tools_enabled=True)
    defaults.update(kwargs)
    return FooterState(**defaults)


class TestFooterEnablement(unittest.TestCase):
    def test_non_tty_is_disabled(self):
        footer = StatusFooter(stream=io.StringIO())
        self.assertFalse(footer.enabled())
        self.assertEqual(footer.render(_make_state()), 0)

    def test_no_color_env_disables_even_on_tty(self):
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            footer = StatusFooter(stream=_FakeTTY())
            self.assertFalse(footer.enabled())
            self.assertEqual(footer.render(_make_state()), 0)

    def test_enabled_on_tty_without_no_color(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NO_COLOR", None)
            footer = StatusFooter(stream=_FakeTTY())
            self.assertTrue(footer.enabled())


class TestFooterRendering(unittest.TestCase):
    def setUp(self):
        os.environ.pop("NO_COLOR", None)

    def test_renders_two_lines_with_content(self):
        sink = _FakeTTY()
        footer = StatusFooter(stream=sink)
        lines_written = footer.render(_make_state(
            providers=5, model="opus-4-7", cost_usd=0.5, tools_enabled=True,
        ))
        self.assertEqual(lines_written, 2)
        output = sink.getvalue()
        self.assertIn("5 providers", output)
        self.assertIn("opus-4-7", output)
        self.assertIn("$0.5000", output)
        # Tools on -> "on" is present.
        self.assertIn("on", output)

    def test_tools_off_state_rendered(self):
        sink = _FakeTTY()
        footer = StatusFooter(stream=sink)
        footer.render(_make_state(tools_enabled=False))
        self.assertIn("off", sink.getvalue())

    def test_singular_provider_pluralisation(self):
        sink = _FakeTTY()
        footer = StatusFooter(stream=sink)
        footer.render(_make_state(providers=1))
        self.assertIn("1 provider ", sink.getvalue())  # trailing space, NOT 'providers'
        self.assertNotIn("1 providers", sink.getvalue())

    def test_no_ansi_cursor_save_restore(self):
        """Swiss Nihilism fix #1: footer must NOT use cursor save/restore."""
        sink = _FakeTTY()
        footer = StatusFooter(stream=sink)
        footer.render(_make_state())
        output = sink.getvalue()
        self.assertNotIn("\033[s", output)
        self.assertNotIn("\033[u", output)

    def test_render_marks_two_last_lines(self):
        sink = _FakeTTY()
        footer = StatusFooter(stream=sink)
        footer.render(_make_state())
        self.assertEqual(footer._last_lines, 2)

    def test_clear_wipes_last_lines(self):
        sink = _FakeTTY()
        footer = StatusFooter(stream=sink)
        footer.render(_make_state())
        sink.truncate(0)
        sink.seek(0)
        cleared = footer.clear()
        self.assertEqual(cleared, 2)
        self.assertEqual(footer._last_lines, 0)
        output = sink.getvalue()
        # Cursor-up + clear-line sequence, twice.
        self.assertEqual(output.count("\033[1A"), 2)

    def test_clear_noop_when_nothing_rendered(self):
        sink = _FakeTTY()
        footer = StatusFooter(stream=sink)
        self.assertEqual(footer.clear(), 0)
        self.assertEqual(sink.getvalue(), "")

    def test_render_respects_narrow_terminal_via_get_terminal_size(self):
        sink = _FakeTTY()
        footer = StatusFooter(stream=sink)
        with mock.patch("lib.hal.status_footer.shutil.get_terminal_size",
                        return_value=os.terminal_size((30, 24))):
            footer.render(_make_state())
        # Minimum width floor is 40 -> 40 separator chars for the bar.
        # Swiss Nihilism fix #3: bar uses ─ (em dash), not · (middle dot).
        bar_line = sink.getvalue().splitlines()[0]
        self.assertGreaterEqual(bar_line.count("─"), 40)


@unittest.skipUnless(hasattr(signal, "SIGWINCH"), "SIGWINCH only available on POSIX")
class TestResizeHandler(unittest.TestCase):
    def test_install_then_uninstall(self):
        footer = StatusFooter(stream=_FakeTTY())
        self.assertFalse(footer.installed)
        installed = footer.install_resize_handler()
        self.assertTrue(installed)
        self.assertTrue(footer.installed)
        footer.uninstall_resize_handler()
        self.assertFalse(footer.installed)

    def test_double_install_is_idempotent(self):
        footer = StatusFooter(stream=_FakeTTY())
        footer.install_resize_handler()
        # second call should not replace the previously-stored handler.
        current = signal.getsignal(signal.SIGWINCH)
        footer.install_resize_handler()
        self.assertIs(signal.getsignal(signal.SIGWINCH), current)
        footer.uninstall_resize_handler()

    def test_sigwinch_sets_resize_pending(self):
        footer = StatusFooter(stream=_FakeTTY())
        footer.install_resize_handler()
        try:
            self.assertFalse(footer.resize_pending)
            # Fire the signal handler directly — no real SIGWINCH needed.
            handler = signal.getsignal(signal.SIGWINCH)
            self.assertIsNotNone(handler)
            handler(signal.SIGWINCH, None)
            self.assertTrue(footer.resize_pending)
        finally:
            footer.uninstall_resize_handler()

    def test_render_clears_resize_pending(self):
        os.environ.pop("NO_COLOR", None)
        footer = StatusFooter(stream=_FakeTTY())
        footer.install_resize_handler()
        try:
            handler = signal.getsignal(signal.SIGWINCH)
            handler(signal.SIGWINCH, None)
            self.assertTrue(footer.resize_pending)
            footer.render(_make_state())
            self.assertFalse(footer.resize_pending)
        finally:
            footer.uninstall_resize_handler()


class TestFooterStateDefaults(unittest.TestCase):
    def test_default_values(self):
        state = FooterState()
        self.assertEqual(state.providers, 0)
        self.assertEqual(state.model, "(none)")
        self.assertEqual(state.cost_usd, 0.0)
        self.assertFalse(state.tools_enabled)


if __name__ == "__main__":
    unittest.main()
