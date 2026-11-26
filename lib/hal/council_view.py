"""HAL Council View — split-view live render for multi-model deliberation.

Two concrete renderers + a factory:

* ``SingleLineRenderer`` — baseline parity with the pre-W5 output.
  One status line per update, written to the caller-provided stream.
  This is what gets selected when the environment is unsafe for live
  rendering (non-TTY, narrow terminal, missing rich, or escape hatch).
* ``RichLiveRenderer`` — the delight path. Per-member boxed columns
  with phase + token counter + colour-coded agreement bar, driven by
  ``rich.live.Live``. Only selected when every pre-condition holds.

The factory ``build_renderer`` encapsulates the selection logic so
every caller gets the same safety envelope. See
``plans/council-view-verdict.md`` for the pre-coding council
deliberation and the concerns that shaped the invariants below.

Invariants (all unit-tested):
  * ``None`` agreement never paints a colour — renders as neutral dash.
  * Token counter defaults to 0 and renders dim until a measurement.
  * Factory returns ``SingleLineRenderer`` on non-TTY, on narrow
    terminal (<80 cols by default), on ``HAL_NO_COUNCIL_SPLIT=1``, and
    when ``rich`` cannot be imported.
  * Agreement colour thresholds: green ``>=0.85``, amber ``>=0.60``,
    red below.
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, TextIO

DEFAULT_MIN_COLS = 80
DEFAULT_AGREEMENT_GREEN = 0.85
DEFAULT_AGREEMENT_AMBER = 0.60


@dataclass
class CouncilMemberState:
    """Mutable per-member state the renderer paints each frame."""

    name: str
    role: str
    model: str
    phase: str = "pending"
    tokens_out: int = 0
    agreement: float | None = None
    status: str = "pending"  # pending | active | done | error


@dataclass(frozen=True)
class CouncilViewConfig:
    min_cols: int = DEFAULT_MIN_COLS
    agreement_green: float = DEFAULT_AGREEMENT_GREEN
    agreement_amber: float = DEFAULT_AGREEMENT_AMBER
    force_single_line: bool = False


def agreement_colour(
    score: float | None,
    *,
    green: float = DEFAULT_AGREEMENT_GREEN,
    amber: float = DEFAULT_AGREEMENT_AMBER,
) -> str:
    """Bucket a score into ``"green"`` / ``"amber"`` / ``"red"`` / ``"neutral"``.

    ``None`` → ``"neutral"``. The function never fabricates a colour
    for missing data — this is the concern #3 invariant from the
    council verdict.
    """
    if score is None:
        return "neutral"
    if score >= green:
        return "green"
    if score >= amber:
        return "amber"
    return "red"


# ---------------------------------------------------------------------------
# Renderer protocol
# ---------------------------------------------------------------------------
class CouncilRenderer(Protocol):
    """Minimal contract every renderer honours."""

    def start(self, members: Iterable[CouncilMemberState]) -> None: ...
    def update(self, name: str, **changes: Any) -> None: ...
    def finalize(self, verdict: str | None = None) -> None: ...


# ---------------------------------------------------------------------------
# SingleLineRenderer — the baseline / fallback
# ---------------------------------------------------------------------------
class SingleLineRenderer:
    """One line per update. Safe everywhere."""

    def __init__(self, *, stream: TextIO | None = None):
        self._stream = stream if stream is not None else sys.stdout
        self._members: dict[str, CouncilMemberState] = {}
        self._active = False

    def start(self, members: Iterable[CouncilMemberState]) -> None:
        self._members = {m.name: m for m in members}
        self._active = True
        names = ", ".join(m.name for m in self._members.values())
        self._stream.write(f"[council] members: {names}\n")
        self._stream.flush()

    def update(self, name: str, **changes: Any) -> None:
        if not self._active:
            return
        member = self._members.get(name)
        if member is None:
            return
        for k, v in changes.items():
            if hasattr(member, k):
                setattr(member, k, v)
        colour = agreement_colour(member.agreement)
        agreement_str = (
            f"{member.agreement:.2f}" if member.agreement is not None else "-"
        )
        self._stream.write(
            f"[council] {member.name} ({member.role}/{member.model}) "
            f"phase={member.phase} tokens={member.tokens_out} "
            f"agreement={agreement_str} [{colour}] status={member.status}\n"
        )
        self._stream.flush()

    def finalize(self, verdict: str | None = None) -> None:
        if not self._active:
            return
        self._active = False
        if verdict:
            self._stream.write(f"[council] verdict: {verdict}\n")
            self._stream.flush()


# ---------------------------------------------------------------------------
# RichLiveRenderer — the delight path
# ---------------------------------------------------------------------------
class RichLiveRenderer:
    """Boxed-column live render via ``rich.live``.

    Kept deliberately thin: builds a panel per member on every update
    and re-renders the whole frame. For council sizes <=10 the cost is
    negligible. Heavier optimisations (diff-only updates, etag-guarded
    panel rebuild) are YAGNI at this scale.
    """

    def __init__(self, *, stream: TextIO | None = None, config: CouncilViewConfig | None = None):
        self._stream = stream if stream is not None else sys.stdout
        self._config = config or CouncilViewConfig()
        self._members: dict[str, CouncilMemberState] = {}
        self._live: Any = None  # rich.live.Live instance, lazily constructed

    # Visible for tests: exposes the current layout renderable.
    def _build_renderable(self) -> Any:
        from rich.columns import Columns
        from rich.panel import Panel
        from rich.text import Text

        panels = []
        for m in self._members.values():
            body = Text()
            body.append(f"{m.role}\n", style="bold")
            body.append(f"{m.model}\n", style="dim")
            body.append(f"phase: {m.phase}\n")
            body.append(f"tokens: {m.tokens_out}\n", style="" if m.tokens_out else "dim")
            agreement_colour_name = agreement_colour(
                m.agreement,
                green=self._config.agreement_green,
                amber=self._config.agreement_amber,
            )
            if m.agreement is None:
                body.append("agreement: -", style="dim")
            else:
                body.append(f"agreement: {m.agreement:.2f}",
                            style=_rich_colour_for(agreement_colour_name))
            border_style = _rich_colour_for(agreement_colour_name)
            panels.append(Panel(body, title=m.name, border_style=border_style))
        return Columns(panels, equal=True, expand=True)

    def start(self, members: Iterable[CouncilMemberState]) -> None:
        from rich.console import Console
        from rich.live import Live

        self._members = {m.name: m for m in members}
        console = Console(file=self._stream, force_terminal=True)
        self._live = Live(self._build_renderable(), console=console, refresh_per_second=8)
        self._live.start()

    def update(self, name: str, **changes: Any) -> None:
        if self._live is None:
            return
        member = self._members.get(name)
        if member is None:
            return
        for k, v in changes.items():
            if hasattr(member, k):
                setattr(member, k, v)
        self._live.update(self._build_renderable())

    def finalize(self, verdict: str | None = None) -> None:
        if self._live is None:
            return
        self._live.stop()
        self._live = None
        if verdict:
            self._stream.write(f"[council] verdict: {verdict}\n")
            self._stream.flush()


def _rich_colour_for(bucket: str) -> str:
    return {"green": "green", "amber": "yellow", "red": "red", "neutral": "dim"}[bucket]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def rich_available() -> bool:
    try:
        import rich  # noqa: F401
        import rich.live  # noqa: F401
    except ImportError:
        return False
    return True


def _stream_is_tty(stream: TextIO) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


def _terminal_width(fallback: int = 80) -> int:
    try:
        return shutil.get_terminal_size((fallback, 24)).columns
    except OSError:
        return fallback


def build_renderer(
    *,
    stream: TextIO | None = None,
    config: CouncilViewConfig | None = None,
) -> CouncilRenderer:
    """Pick the safest renderer that can actually run right now.

    Decision table (first match wins):

    | Condition                                  | Result                |
    |--------------------------------------------|-----------------------|
    | ``config.force_single_line``               | ``SingleLineRenderer`` |
    | ``HAL_NO_COUNCIL_SPLIT`` env set           | ``SingleLineRenderer`` |
    | ``stream`` is not a TTY                    | ``SingleLineRenderer`` |
    | ``shutil.get_terminal_size < min_cols``    | ``SingleLineRenderer`` |
    | ``rich`` not importable                    | ``SingleLineRenderer`` |
    | otherwise                                  | ``RichLiveRenderer``  |
    """
    cfg = config or CouncilViewConfig()
    out = stream if stream is not None else sys.stdout
    if cfg.force_single_line:
        return SingleLineRenderer(stream=out)
    if os.environ.get("HAL_NO_COUNCIL_SPLIT"):
        return SingleLineRenderer(stream=out)
    if not _stream_is_tty(out):
        return SingleLineRenderer(stream=out)
    if _terminal_width() < cfg.min_cols:
        return SingleLineRenderer(stream=out)
    if not rich_available():
        return SingleLineRenderer(stream=out)
    return RichLiveRenderer(stream=out, config=cfg)
