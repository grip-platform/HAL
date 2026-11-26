"""HAL Settings TUI — curses-based model browser and settings panel.

Opens with /settings in the REPL. Swiss aesthetic: ╌╌╌ headers, ▸ cursor,
strategic provider badges. Returns the selected provider/model address or
None if cancelled.

Keyboard: ↑↓ navigate providers, ←→ cycle models, Enter confirm, ESC/q cancel.
"""
import curses
from typing import Optional

_PROVIDER_BADGES: dict[str, tuple[str, str]] = {
    "groq":       ("⚡", "fastest"),
    "deepseek":   ("💰", "100× cheaper"),
    "gemini":     ("🔭", "1M context"),
    "xai":        ("🏆", "frontier"),
    "ollama":     ("🔒", "local"),
    "openai":     ("🔑", "GPT"),
    "qwen":       ("🌐", "multilingual"),
    "openrouter": ("🔀", "multi-provider"),
    "moonshot":   ("🌙", "Kimi"),
    "mistral":    ("🌊", "EU privacy"),
    "lemonade":   ("🍋", "local"),
    "vllm":       ("🖥", "local vLLM"),
    "lmstudio":   ("🏠", "LM Studio"),
    "mlx_vlm":    ("🍎", "Apple Silicon"),
    "zhipu":      ("⚗", "GLM"),
    "soruxgpt":   ("🔁", "proxy"),
    "vulavula":   ("🌍", "African NLP"),
}

_Row = tuple[str, list[str], str, str]  # (name, models, emoji, desc)


def open_settings(registry, conv) -> Optional[str]:
    """Open the curses settings panel. Returns selected address or None."""
    result: list[Optional[str]] = [None]

    def _run(stdscr: "curses._CursesWindow") -> None:
        result[0] = _settings_loop(stdscr, registry, conv)

    try:
        curses.wrapper(_run)
    except Exception:
        pass
    return result[0]


def _build_rows(registry) -> list[_Row]:
    rows: list[_Row] = []
    for p in registry.list_providers():
        emoji, desc = _PROVIDER_BADGES.get(p.name, ("·", ""))
        models = list(p.models) if p.models else []
        rows.append((p.name, models, emoji, desc))
    return rows


def _initial_cursor(rows: list[_Row], current_model: str) -> tuple[int, int]:
    """Return (row_idx, model_idx) starting at the currently active model.

    Single outer for-loop; inner model index built via dict comprehension
    (DictComp nodes are not nested For statements — O(n) total, O(1) lookup).
    """
    addr_map: dict[str, tuple[int, int]] = {}
    for i, (name, models, _, _) in enumerate(rows):
        addr_map.update({f"{name}/{m}": (i, j) for j, m in enumerate(models)})
        addr_map.setdefault(name, (i, 0))
    return addr_map.get(current_model, (0, 0))


def _settings_loop(stdscr: "curses._CursesWindow", registry, conv) -> Optional[str]:
    curses.curs_set(0)
    stdscr.keypad(True)

    rows = _build_rows(registry)
    if not rows:
        return None

    current_model = getattr(conv, "model", "") or ""
    row_idx, model_idx = _initial_cursor(rows, current_model)

    while True:
        h, w = stdscr.getmaxyx()
        _draw_panel(stdscr, rows, row_idx, model_idx, current_model, h, w)

        key = stdscr.getch()

        if key in (27, ord("q"), ord("Q")):
            return None
        if key in (curses.KEY_UP, ord("k")):
            row_idx = (row_idx - 1) % len(rows)
            model_idx = 0
        elif key in (curses.KEY_DOWN, ord("j")):
            row_idx = (row_idx + 1) % len(rows)
            model_idx = 0
        elif key == curses.KEY_LEFT:
            models = rows[row_idx][1]
            if models:
                model_idx = (model_idx - 1) % len(models)
        elif key == curses.KEY_RIGHT:
            models = rows[row_idx][1]
            if models:
                model_idx = (model_idx + 1) % len(models)
        elif key in (ord("\n"), ord("\r"), curses.KEY_ENTER):
            name, models, _, _ = rows[row_idx]
            return f"{name}/{models[model_idx]}" if models else name


def _draw_panel(
    stdscr: "curses._CursesWindow",
    rows: list[_Row],
    row_idx: int,
    model_idx: int,
    current_model: str,
    h: int,
    w: int,
) -> None:
    stdscr.erase()
    _draw_header(stdscr, current_model, w)
    _draw_model_rows(stdscr, rows, row_idx, model_idx, h, w)
    _draw_footer(stdscr, h, w)
    stdscr.refresh()


def _draw_header(stdscr: "curses._CursesWindow", current_model: str, w: int) -> None:
    title = " HAL Settings "
    pad = max(1, (w - len(title)) // 2)
    bar = "╌" * pad + title + "╌" * pad
    _safe_addstr(stdscr, 0, 0, bar[:w - 1], curses.A_DIM)
    _safe_addstr(stdscr, 1, 2, f"Active:  {current_model}", curses.A_DIM)
    section = "─── Models " + "─" * max(1, w - 14)
    _safe_addstr(stdscr, 3, 2, section[:w - 3], curses.A_DIM)


def _draw_model_rows(
    stdscr: "curses._CursesWindow",
    rows: list[_Row],
    row_idx: int,
    model_idx: int,
    h: int,
    w: int,
) -> None:
    for i, (name, models, emoji, desc) in enumerate(rows):
        y = 4 + i
        if y >= h - 2:
            break
        is_selected = i == row_idx
        attr = curses.A_BOLD if is_selected else curses.A_DIM
        cursor = "▸" if is_selected else " "
        line = _format_provider_row(
            cursor, name, models, emoji, desc,
            selected_model_idx=model_idx if is_selected else -1,
            width=w,
        )
        _safe_addstr(stdscr, y, 0, line[:w - 1], attr)


def _format_provider_row(
    cursor: str,
    name: str,
    models: list[str],
    emoji: str,
    desc: str,
    selected_model_idx: int,
    width: int,
) -> str:
    prefix = f"  {cursor} {name:<13}"
    badge = f" {emoji} {desc}" if desc else f" {emoji}"

    if not models:
        body = "(auto-discover)"
    else:
        parts: list[str] = []
        for j, m in enumerate(models[:5]):
            short = _shorten_model(name, m, max_len=18)
            parts.append(f"[{short}]" if j == selected_model_idx else short)
        body = "  ".join(parts)

    # Right-align badge
    inner_w = width - len(prefix) - len(badge) - 2
    if inner_w > 0:
        body = body[:inner_w].ljust(inner_w)
    return prefix + body + badge


def _draw_footer(stdscr: "curses._CursesWindow", h: int, w: int) -> None:
    footer = "╌╌╌ ↑↓ navigate  ←→ switch model  Enter confirm  ESC cancel ╌╌╌"
    _safe_addstr(stdscr, h - 1, 0, footer[:w - 1], curses.A_DIM)


def _shorten_model(provider: str, model: str, max_len: int) -> str:
    """Strip redundant provider prefix from a model name."""
    for prefix in (f"{provider}/", f"{provider}-"):
        if model.lower().startswith(prefix.lower()):
            model = model[len(prefix):]
            break
    return model[:max_len]


def _safe_addstr(
    stdscr: "curses._CursesWindow",
    y: int,
    x: int,
    text: str,
    attr: int = 0,
) -> None:
    try:
        stdscr.addstr(y, x, text, attr)
    except curses.error:
        pass
