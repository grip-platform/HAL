"""HAL Cost Heatmap — per-provider cost bar charts.

Renders the cumulative ``cost_by_provider`` dict as Unicode block bars,
one line per provider, sorted by spend descending. Intended for display
between REPL turns so the operator sees where the session's budget is
going at a glance.

No runtime cost — pure string construction. Callers control rendering.

Example output (stripped of ANSI dim/reset)::

    openai       $0.042100 ██████████████████████████████████  68.5%
    groq         $0.012500 ██████████                          20.3%
    ollama/local $0.006900 █████                               11.2%
"""
from __future__ import annotations

from typing import Mapping

_BLOCK_FULL = "\u2588"  # FULL BLOCK
_BLOCK_PARTS = (
    "\u258f",  # LEFT ONE EIGHTH BLOCK    1/8
    "\u258e",  # LEFT ONE QUARTER BLOCK   2/8
    "\u258d",  # LEFT THREE EIGHTHS BLOCK 3/8
    "\u258c",  # LEFT HALF BLOCK          4/8
    "\u258b",  # LEFT FIVE EIGHTHS BLOCK  5/8
    "\u258a",  # LEFT THREE QUARTERS BLOCK 6/8
    "\u2589",  # LEFT SEVEN EIGHTHS BLOCK 7/8
)


def _render_bar(fraction: float, width_chars: int) -> str:
    """Render ``fraction in [0, 1]`` as a width_chars-wide unicode block bar.

    Uses fractional-block code points for sub-character resolution, so
    a very small cost still produces a visible sliver and the bar total
    width stays constant regardless of value.
    """
    if width_chars <= 0 or fraction <= 0.0:
        return " " * max(width_chars, 0)
    clamped = min(fraction, 1.0)
    total_eighths = int(round(clamped * width_chars * 8))
    full, rem = divmod(total_eighths, 8)
    bar = _BLOCK_FULL * full
    if rem and len(bar) < width_chars:
        bar += _BLOCK_PARTS[rem - 1]
    if len(bar) < width_chars:
        bar += " " * (width_chars - len(bar))
    return bar[:width_chars]


def render_heatmap(
    cost_by_provider: Mapping[str, float],
    *,
    bar_width: int = 34,
    name_width: int = 14,
    min_display_cost: float = 0.0,
) -> list[str]:
    """Return one line per provider, sorted by cost desc.

    Args:
        cost_by_provider: dict of provider -> cumulative usd cost.
        bar_width: width of the bar column in characters (default 34).
        name_width: padding width for the provider column.
        min_display_cost: providers whose cost is strictly less than this
            are omitted (set > 0 to hide zero-cost providers).

    Returns:
        Rendered lines. Empty list when no provider qualifies.
    """
    if not cost_by_provider:
        return []
    filtered = {k: v for k, v in cost_by_provider.items() if v >= min_display_cost}
    if not filtered:
        return []
    total = sum(v for v in filtered.values() if v > 0) or 1.0
    peak = max(filtered.values()) or 1.0
    lines: list[str] = []
    for name, cost in sorted(filtered.items(), key=lambda kv: kv[1], reverse=True):
        fraction = (cost / peak) if peak > 0 else 0.0
        bar = _render_bar(fraction, bar_width)
        pct = 100.0 * cost / total if total > 0 else 0.0
        lines.append(
            f"  {name:<{name_width}} ${cost:.6f} {bar} {pct:5.1f}%"
        )
    return lines


def format_citation_chip(entries: int, *, command: str = "/provenance") -> str:
    """Return the standard citation chip string.

    Used when KONO memory is injected into a turn to signal provenance
    is available. Phase β W2 will wire ``/provenance`` to the chip.

    >>> format_citation_chip(3)
    '[memory: 3 entries · /provenance for details]'
    >>> format_citation_chip(1)
    '[memory: 1 entry · /provenance for details]'
    >>> format_citation_chip(0)
    ''
    """
    if entries <= 0:
        return ""
    noun = "entry" if entries == 1 else "entries"
    return f"[memory: {entries} {noun} · {command} for details]"
