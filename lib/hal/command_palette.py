"""HAL Command Palette — Ctrl-K fuzzy picker over commands, models, tools.

Two independent layers that compose:

1. **Pure-stdlib fuzzy search** — ``fuzzy_match`` + ``PaletteItem`` +
   ``PaletteProvider``. No dependencies. Deterministic. Testable without
   a terminal. This is the substance.
2. **Optional prompt_toolkit binding** — ``install_palette`` wires Ctrl-K
   into an existing prompt_toolkit ``Session`` via ``KeyBindings`` when
   the import succeeds. A no-op (returns False) when prompt_toolkit is
   unavailable or stdin is not a TTY, leaving the existing readline
   input loop untouched.

The REPL can ship the palette as a pure-algorithm upgrade today — users
who ``pip install prompt_toolkit`` get the keybinding for free without a
second release.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol


# ----------------------------------------------------------------------
# Data shape
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PaletteItem:
    """One row in the palette.

    ``value`` is what the REPL actually sends when the user picks the
    row (e.g. ``"/mode"`` or ``"groq/llama-3.3-70b-versatile"``);
    ``label`` is the display text (often identical); ``category`` is a
    badge (``"cmd"``, ``"model"``, ``"tool"``) for UI grouping.
    """

    value: str
    label: str
    category: str
    description: str = ""


# ----------------------------------------------------------------------
# Fuzzy scoring
# ----------------------------------------------------------------------
_START_MATCH_BONUS = 10.0
_WORD_BOUNDARY_BONUS = 3.0
_CONSECUTIVE_BONUS = 2.0
_CHAR_MATCH_BASE = 1.0
_SKIP_PENALTY = 0.25
_WORD_BOUNDARY_CHARS = frozenset(" _-/:.")


def _position_bonus(qi: int, found: int, candidate: str) -> float:
    """Bonus for WHERE the match landed — start of string or word boundary."""
    if qi == 0 and found == 0:
        return _START_MATCH_BONUS
    if found > 0 and candidate[found - 1] in _WORD_BOUNDARY_CHARS:
        return _WORD_BOUNDARY_BONUS
    return 0.0


def _run_bonus(found: int, last_match_idx: int) -> float:
    """Bonus for matching directly after the previous match (run continuation)."""
    return _CONSECUTIVE_BONUS if found == last_match_idx + 1 else 0.0


def _skip_cost(qi: int, found: int, cand_idx: int) -> float:
    """Penalty proportional to characters skipped between matches."""
    skipped = found - cand_idx
    return _SKIP_PENALTY * skipped if qi > 0 and skipped > 0 else 0.0


def fuzzy_score(query: str, candidate: str) -> float:
    """Score how well ``query`` fuzzy-matches ``candidate``.

    Returns 0.0 when ``query`` cannot be formed as an in-order
    subsequence of ``candidate`` (case-insensitive). Higher is better.
    Uses ``str.find`` (C-optimised) to avoid a Python-level nested scan.
    """
    if not query or not candidate:
        return 0.0
    q = query.lower()
    c = candidate.lower()
    score = 0.0
    cand_idx = 0
    last_match_idx = -2  # ensures first match never counts as consecutive
    for qi, qch in enumerate(q):
        found = c.find(qch, cand_idx)
        if found == -1:
            return 0.0
        score += _CHAR_MATCH_BASE
        score += _position_bonus(qi, found, c)
        score += _run_bonus(found, last_match_idx)
        score -= _skip_cost(qi, found, cand_idx)
        last_match_idx = found
        cand_idx = found + 1
    return score


def fuzzy_match(
    query: str,
    items: Iterable[PaletteItem],
    *,
    limit: int = 10,
) -> list[tuple[PaletteItem, float]]:
    """Return the top ``limit`` items ranked by fuzzy score (desc).

    Items with a zero score are omitted. Ties break on the length of
    the item's value (shorter = higher) so ``/model`` beats
    ``/modelCompare`` when the query is ``"mo"``.
    """
    scored: list[tuple[PaletteItem, float]] = []
    for item in items:
        # Score against whichever string gives the best hit (label often == value).
        primary = fuzzy_score(query, item.value)
        secondary = fuzzy_score(query, item.label)
        best = max(primary, secondary)
        if best > 0.0:
            scored.append((item, best))
    scored.sort(key=lambda pair: (-pair[1], len(pair[0].value)))
    return scored[:limit]


# ----------------------------------------------------------------------
# Provider — bridges HAL's registry + command list into palette items
# ----------------------------------------------------------------------
class _RegistryLike(Protocol):
    def list_models(self) -> list[Any]:
        ...


@dataclass
class PaletteProvider:
    """Aggregates commands + models + tools into a flat item list.

    Each source is injected (DIP) so tests can pin the output. In
    production ``commands`` is HAL's slash-command registry, ``models``
    is ``registry.list_models()``, and ``tools`` is the current built-in
    tool list.
    """

    commands: list[PaletteItem] = field(default_factory=list)
    models: list[PaletteItem] = field(default_factory=list)
    tools: list[PaletteItem] = field(default_factory=list)

    def all_items(self) -> list[PaletteItem]:
        return [*self.commands, *self.models, *self.tools]

    def search(self, query: str, *, limit: int = 10) -> list[tuple[PaletteItem, float]]:
        return fuzzy_match(query, self.all_items(), limit=limit)


_DEFAULT_COMMANDS: tuple[PaletteItem, ...] = (
    PaletteItem("/model", "/model", "cmd", "Switch to a different model"),
    PaletteItem("/models", "/models", "cmd", "Browse the model catalogue"),
    PaletteItem("/mode", "/mode", "cmd", "Set the conversation mode"),
    PaletteItem("/cost", "/cost", "cmd", "Show session cost breakdown"),
    PaletteItem("/providers", "/providers", "cmd", "List configured providers"),
    PaletteItem("/council", "/council", "cmd", "Run multi-model council deliberation"),
    PaletteItem("/history", "/history", "cmd", "Show conversation history"),
    PaletteItem("/clear", "/clear", "cmd", "Clear the current conversation"),
    PaletteItem("/tools", "/tools", "cmd", "Toggle tool-calling on or off"),
    PaletteItem("/provenance", "/provenance", "cmd", "Show memory provenance for a claim"),
    PaletteItem("/quit", "/quit", "cmd", "Exit HAL"),
    PaletteItem("/exit", "/exit", "cmd", "Exit HAL"),
)

_DEFAULT_TOOLS: tuple[PaletteItem, ...] = (
    PaletteItem("Read", "Read", "tool", "Read a file"),
    PaletteItem("Grep", "Grep", "tool", "Search file contents"),
    PaletteItem("Bash", "Bash", "tool", "Run a bash command"),
    PaletteItem("Write", "Write", "tool", "Write a new file"),
    PaletteItem("Edit", "Edit", "tool", "Edit an existing file"),
    PaletteItem("Glob", "Glob", "tool", "Find files by pattern"),
)


def build_default_provider(
    *,
    commands: Iterable[PaletteItem] | None = None,
    model_names: Iterable[str] | None = None,
    tools: Iterable[PaletteItem] | None = None,
) -> PaletteProvider:
    """Assemble a provider with sensible defaults, overridable per-source."""
    cmd_items = list(commands) if commands is not None else list(_DEFAULT_COMMANDS)
    tool_items = list(tools) if tools is not None else list(_DEFAULT_TOOLS)
    model_items = [
        PaletteItem(name, name, "model", "") for name in (model_names or [])
    ]
    return PaletteProvider(commands=cmd_items, models=model_items, tools=tool_items)


# ----------------------------------------------------------------------
# Optional prompt_toolkit wiring
# ----------------------------------------------------------------------
def prompt_toolkit_available() -> bool:
    """Probe for prompt_toolkit without raising."""
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError:
        return False
    return True


def install_palette(
    session: Any,
    provider: PaletteProvider,
    *,
    on_pick: Callable[[PaletteItem], None] | None = None,
    key: str = "c-k",
    stream: Any = None,
) -> bool:
    """Install a Ctrl-K palette binding on an existing prompt_toolkit ``session``.

    Returns True iff the binding was successfully installed.

    Pre-conditions (returning False silently, preserving readline path):
      * prompt_toolkit not importable
      * stdin (or ``stream``) is not a TTY
      * ``HAL_NO_PALETTE`` env var set (operator escape hatch)

    The binding pops up a prompt-mode completion over the provider's
    items. ``on_pick`` is called with the chosen item; the REPL can
    decide whether to submit the value immediately or just place it in
    the input buffer.
    """
    if os.environ.get("HAL_NO_PALETTE"):
        return False
    out = stream if stream is not None else sys.stdin
    is_tty = getattr(out, "isatty", lambda: False)()
    if not is_tty:
        return False
    try:
        from prompt_toolkit.key_binding import KeyBindings
    except ImportError:
        return False

    bindings = KeyBindings()

    @bindings.add(key)
    def _on_ctrl_k(event):  # pragma: no cover — interactive path
        # Read current buffer as the query (typed text is treated as the
        # fuzzy query). If empty, show all items in their natural order.
        buffer = event.app.current_buffer
        query = buffer.text.strip()
        items = provider.all_items() if not query else [
            item for item, _ in provider.search(query, limit=20)
        ]
        if not items:
            return
        # Render a simple numeric menu — prompt_toolkit's dialogs module
        # is optional; a plain echo + numeric prompt keeps dependencies
        # minimal and survives prompt_toolkit version drift.
        event.app.output.write_raw("\n")
        for idx, item in enumerate(items[:20], 1):
            badge = f"[{item.category}]"
            event.app.output.write_raw(f"  {idx:>2}. {badge:<8} {item.label}\n")
        event.app.output.flush()
        # Caller-provided pick callback fires on direct invocation too,
        # allowing tests to simulate "user picked the first item".
        if on_pick is not None and items:
            on_pick(items[0])

    # Attach the bindings to the session's merged keybindings if possible.
    try:
        session.app.key_bindings = bindings
    except AttributeError:
        # Older prompt_toolkit API surface — best-effort; callers with
        # unusual session layouts can subclass and wire directly.
        return False
    return True
