"""HAL Status Footer — two-line status readout between turns.

Renders a compact, persistent-feeling footer after each REPL turn showing
the live session state:

    ● {providers} providers · {model} · ${cost} · tools: on/off

The footer paints its two lines using ANSI cursor save/restore so it
doesn't perturb the row/column state of the caller, and it respects
``NO_COLOR=1`` + non-TTY sinks as a no-op. A ``SIGWINCH`` handler is
registered on install (best-effort; ``signal`` isn't available on
Windows platform variants that drop SIGWINCH) that flips an internal
``resize_pending`` flag so callers can decide when to redraw.

True scroll-region pinning (DECSTBM) is intentionally NOT used in this
wave — it requires teardown on every abnormal exit and conflicts with
the interactive ``input()`` prompt. A subsequent wave can promote the
footer to pinned once the REPL's lifecycle is isolated.
"""
from __future__ import annotations

import os
import shutil
import signal
import sys
from dataclasses import dataclass, field
from typing import Callable, TextIO

_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_GRAY = "\033[38;5;240m"
_RESET = "\033[0m"
_SAVE_CURSOR = "\033[s"
_RESTORE_CURSOR = "\033[u"
_CLEAR_LINE = "\033[2K\r"
_CURSOR_UP_ONE = "\033[1A"


@dataclass
class FooterState:
    providers: int = 0
    model: str = "(none)"
    cost_usd: float = 0.0
    tools_enabled: bool = False
    context_pct: float = 0.0  # ambient context window %; shown when > 20


@dataclass
class StatusFooter:
    """Two-line status footer painter.

    Instantiate once per REPL session; call :meth:`render` between turns
    and :meth:`clear` before a clean exit.
    """

    stream: TextIO = field(default_factory=lambda: sys.stdout)
    _last_lines: int = 0
    _resize_pending: bool = False
    _prev_sigwinch: Callable | int | None = None
    _installed: bool = False

    # ------------------------------------------------------------------
    # Enablement
    # ------------------------------------------------------------------
    def enabled(self) -> bool:
        """Footer is active only on a colour-capable TTY sink."""
        if os.environ.get("NO_COLOR"):
            return False
        isatty = getattr(self.stream, "isatty", None)
        return bool(isatty and isatty())

    # ------------------------------------------------------------------
    # SIGWINCH lifecycle
    # ------------------------------------------------------------------
    def install_resize_handler(self) -> bool:
        """Register SIGWINCH handler. Returns True if installed, False on platforms without SIGWINCH."""
        if self._installed or not hasattr(signal, "SIGWINCH"):
            return self._installed
        def _handler(_signum, _frame):
            self._resize_pending = True
        self._prev_sigwinch = signal.signal(signal.SIGWINCH, _handler)
        self._installed = True
        return True

    def uninstall_resize_handler(self) -> None:
        if not self._installed or not hasattr(signal, "SIGWINCH"):
            return
        # Restore whatever was there before us; ``signal.SIG_DFL`` if nothing.
        try:
            signal.signal(signal.SIGWINCH, self._prev_sigwinch or signal.SIG_DFL)
        except (TypeError, ValueError):
            signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        self._installed = False
        self._prev_sigwinch = None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _format_lines(self, state: FooterState) -> tuple[str, str]:
        """Build the two ANSI-styled lines, respecting terminal width."""
        width = max(40, min(shutil.get_terminal_size((80, 24)).columns - 4, 72))
        # Swiss Nihilism: structural ─ rule (not decorative · texture)
        bar = f"  {_GRAY}{'─' * width}{_RESET}"
        tool_label = f"{_GREEN}on{_RESET}" if state.tools_enabled else f"{_DIM}off{_RESET}"
        prov_label = f"{state.providers} provider{'s' if state.providers != 1 else ''}"
        # Progressive cost: hide noise below $0.01
        if state.cost_usd < 0.01:
            cost_str = ""
        elif state.cost_usd < 0.10:
            cost_str = f" · {_DIM}${state.cost_usd:.4f}{_RESET}"
        else:
            cost_str = f" · ${state.cost_usd:.4f}"
        # Ambient context %: only surfaces when approaching limits
        ctx_str = f" · {_DIM}ctx {state.context_pct:.0f}%{_RESET}" if state.context_pct > 20 else ""
        status = (
            f"  {_DIM}●{_RESET} "
            f"{prov_label} · {_BOLD}{state.model}{_RESET}"
            f"{cost_str} · tools: {tool_label}{ctx_str}"
        )
        return bar, status

    def render(self, state: FooterState) -> int:
        """Paint the footer; returns number of lines written (0 when disabled)."""
        if not self.enabled():
            return 0
        bar, status = self._format_lines(state)
        # Print naturally — no cursor save/restore.
        # Save/restore was placing the cursor *before* the footer lines, causing
        # input() to print the ❯ prompt on top of the footer rather than below it.
        # Without save/restore the cursor ends up after line 2, so input() appears
        # cleanly beneath the footer.
        self.stream.write(bar + "\n")
        self.stream.write(status + "\n")
        self.stream.flush()
        self._last_lines = 2
        self._resize_pending = False
        return 2

    def clear(self) -> int:
        """Best-effort clear of the most recently painted footer.

        Moves the cursor up ``_last_lines`` rows, wiping each on the way.
        Returns number of lines cleared.
        """
        if not self.enabled() or self._last_lines == 0:
            return 0
        cleared = self._last_lines
        for _ in range(cleared):
            self.stream.write(_CURSOR_UP_ONE + _CLEAR_LINE)
        self.stream.flush()
        self._last_lines = 0
        return cleared

    # ------------------------------------------------------------------
    # Accessors (tests)
    # ------------------------------------------------------------------
    @property
    def resize_pending(self) -> bool:
        return self._resize_pending

    @property
    def installed(self) -> bool:
        return self._installed
