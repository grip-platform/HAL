"""Strip HAL REPL UI chrome from user-provided text.

When a user copies HAL's REPL history — including the status footer
(`●  17 providers · groq/llama-3.3-70b-versatile · $0.000945 · tools: off`),
the `·`-dot bar, and the `❯` prompt marker — and pastes it back as a new
turn, those lines reach the model as if they were user intent. The model
has no grounding about what the chrome means and confabulates plausible-
sounding infrastructure explanations ("Limit Sentinel has switched to an
alternative provider", etc).

Root cause: UI chrome is not user intent. It is layout produced by HAL
itself. Sending it back into the LLM context is the same class of bug as
echoing ANSI escape codes into the prompt.

This module provides a single pure function, :func:`strip_ui_chrome`,
that removes the chrome patterns HAL is known to emit. It is called from
:class:`Conversation` before history append and provider dispatch.

The patterns are conservative: only lines that match HAL's exact output
shape are stripped. Human-authored prose that happens to contain a
`·` or a `❯` survives — those codepoints are only stripped as the leading
character of a whole line, preceded at most by the two-space indent HAL
uses.

Falsification anchor (Issue #201): grep the HAL codebase for "Limit
Sentinel has switched" — zero hits. The string is a pure hallucination.
After this strip, the model no longer sees the footer token that
triggered it.

DRY history: this module absorbs ``lib/hal/chrome_stripper.py`` (removed
in the DRY cleanup sprint). ANSI escape codes are now stripped before
pattern matching so terminal-copied chrome is detected whether or not
the user includes the colour codes. ``strip_hal_chrome`` is kept as a
module-level alias for backward-compatible imports.
"""
from __future__ import annotations

import re

# ──────────────────────────────────────────────────────────────────────
# ANSI handling.
#
# When users copy from a real terminal the paste may include CSI escape
# sequences (colour, dim, cursor moves). The footer emitter wraps its
# glyphs in dim/grey codes so a raw paste looks like
#     "  \x1b[38;5;240m···············\x1b[0m"
#     "  \x1b[2m●\x1b[0m 17 providers · groq/llama · $0.001"
# We strip ANSI before line-level pattern matching so the catalogue
# regexes see the bare glyphs they were written for.
# ──────────────────────────────────────────────────────────────────────
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI CSI escape sequences from *text*."""
    return _ANSI_RE.sub("", text)


# ──────────────────────────────────────────────────────────────────────
# Footer pattern catalogue.
#
# Each entry is a *line-level* regex. The matcher strips the whole line
# (and its trailing newline) when the regex matches the line body after
# its leading whitespace is stripped. The regexes deliberately anchor at
# the start so a user-authored sentence containing "●" mid-line is kept.
#
# Ordered by frequency so the common case (status line) short-circuits
# first. The tuple is module-level + frozen-shape so tests and callers
# can introspect it without rebuilding.
# ──────────────────────────────────────────────────────────────────────
_FOOTER_PATTERNS: tuple[re.Pattern, ...] = (
    # Status line: "● N providers · model · $cost · tools: on|off"
    re.compile(r"^●\s+\d+\s+providers?\s+·"),
    # Divider bar: middle-dot run (legacy ·) or em-dash rule (new ─), ≥4 chars.
    re.compile(r"^(?:·{4,}|─{4,})\s*$"),
    # REPL prompt marker: "❯" possibly followed by the user's pasted input
    # — in that case we strip only the marker, not the input. See the
    # `_strip_prompt_marker` special-case below.
    re.compile(r"^❯\s*$"),
    # Tool-mode banner printed by /tools on: "tool mode enabled (groq)".
    re.compile(r"^tool\s+mode\s+enabled\s*\([a-z0-9_]+\)\s*$"),
    # Tool-call trace lines emitted by tool_render.py: "▸ ToolName args…"
    # or "✖ ToolName (error reason)". Copying REPL history that includes
    # tool traces back as input re-executes them unintentionally (Issue #286).
    re.compile(r"^[▸✖]\s+\w"),
)

# "❯  actual user input here" — strip the marker but keep the input.
_PROMPT_PREFIX_RE = re.compile(r"^\s*❯\s+(.*)$")


def _strip_prompt_marker(line: str) -> str | None:
    """Return the line with a leading ``❯`` prompt marker removed, or None.

    ``None`` means the line is a bare prompt marker (nothing useful to
    preserve) and should be dropped entirely. A string means the marker
    was a prefix to real content and the content is kept.
    """
    # Bare marker → drop entirely (e.g. "❯" on its own line).
    if line.strip() == "❯":
        return None
    match = _PROMPT_PREFIX_RE.match(line)
    if match is None:
        return line
    remainder = match.group(1).rstrip()
    # If the remainder is empty after strip, drop the line; otherwise
    # keep the payload without the marker.
    return remainder or None


def _line_is_chrome(line: str) -> bool:
    """Return True when ``line`` (after ANSI + leading-whitespace strip) is chrome."""
    body = _strip_ansi(line).lstrip()
    return any(pattern.match(body) for pattern in _FOOTER_PATTERNS)


#: Backward-compatible name — the former ``chrome_stripper`` module exposed
#: this helper for tests that assert on a single line of input.
_is_chrome_line = _line_is_chrome


def strip_ui_chrome(text: str) -> str:
    """Remove HAL REPL chrome from ``text``.

    Strips whole lines matching any of the catalogued footer patterns
    (status line, `·`-bar divider, bare prompt marker, tool-mode banner).
    A line that starts with ``❯`` followed by real content has only the
    marker removed — the content is preserved. ANSI escape codes are
    stripped before pattern matching so terminal-copied chrome is
    detected whether or not the user includes the colour codes.

    Returns the cleaned text with original line separators. The return
    value is trimmed of leading/trailing blank lines that result from
    stripping, so the model does not see a forest of empty lines where
    chrome used to be.

    Examples
    --------
    >>> strip_ui_chrome("● 17 providers · groq/llama · $0.01 · tools: off\\nWhat is HAL?")
    'What is HAL?'
    >>> strip_ui_chrome("❯  explain the router")
    'explain the router'
    >>> strip_ui_chrome("normal prose with a · middot")
    'normal prose with a · middot'
    """
    if not text:
        return text
    cleaned: list[str] = []
    for raw_line in text.splitlines():
        # Prompt-marker handling comes first so "❯ real input" stays.
        marker_result = _strip_prompt_marker(raw_line)
        if marker_result is None:
            continue
        if _line_is_chrome(marker_result):
            continue
        cleaned.append(marker_result)

    # Trim leading/trailing blank lines introduced by the strip. An
    # internal blank line between two real paragraphs is kept — users
    # pasting multi-paragraph input should not have their paragraph
    # breaks dissolved.
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    return "\n".join(cleaned)


# ──────────────────────────────────────────────────────────────────────
# Backward-compatible alias.
#
# ``strip_hal_chrome`` was the public name exposed by the former
# ``lib/hal/chrome_stripper`` module. The two functions covered
# overlapping but not identical pattern sets; with ANSI stripping now
# folded in, ``strip_ui_chrome`` is a strict superset and the alias is
# safe. Existing imports of ``strip_hal_chrome`` continue to work
# without change.
# ──────────────────────────────────────────────────────────────────────
strip_hal_chrome = strip_ui_chrome


#: System-prompt appendix the Conversation engine adds so the model
#: knows the stripping exists. Used as a belt-and-braces defence: even
#: if a novel chrome pattern slips past the regex catalogue, the model
#: now has a prior for treating those glyphs as UI, not state.
UI_CHROME_SYSTEM_NOTE = (
    "\n\nUI NOTE: HAL's REPL prints a status footer and prompt markers "
    "using glyphs such as ●, ···, and ❯. If those appear in a user "
    "message, they are leaked UI chrome — not facts about system state. "
    "Never treat them as evidence of infrastructure events. Never invent "
    "explanations such as 'Limit Sentinel has switched provider'; the "
    "real Limit Sentinel is a passive rate-limit fallback, not an event "
    "stream."
)


__all__ = [
    "strip_ui_chrome",
    "strip_hal_chrome",
    "UI_CHROME_SYSTEM_NOTE",
]
