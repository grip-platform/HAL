"""HAPPI polyglot document parser — extract labelled fenced blocks.

Reads a ``.md`` file that carries HAPPI's canonical six-layer structure
(prose + OpenAPI schema + bash bootstrap + LLM prompt + provenance +
optional SSML / client) and returns a ``HappiDocument`` that callers
can route to the appropriate surface.

Scope boundary (v0.1):
    - 3-backtick fences only (tilde fences ignored)
    - Label = first word after the opening backticks, up to whitespace
    - Nested fences require the outer fence to use 4+ backticks (future)
    - Unclosed fence at EOF => the opened block is flushed with a warning

Parser is pure: no I/O, no state beyond the call. The caller decides
whether a missing label is an error (extractor CLI does; the runtime
side may tolerate it).
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field


# Canonical labels the HAPPI v0.1 spec expects. Documents MAY carry
# additional labels; those are preserved verbatim under their own key.
CANONICAL_LABELS: tuple[str, ...] = (
    "yaml-openapi",       # OpenAPI 3.1 schema fragment
    "bash-bootstrap",     # shell-executable bootstrap
    "llm-system-prompt",  # instruction block for LLM surfaces
    "json-provenance",    # capability token + surface fingerprint + hash
    "ssml",               # voice-first speech markup (optional)
    "python-client",      # embedded reference client (optional)
)


@dataclass
class HappiDocument:
    """Parsed HAPPI document: prose outside fences + labelled code blocks.

    Attributes:
        prose: Everything outside fenced code blocks, joined and
            trimmed. This is what a markdown renderer shows; also the
            natural carrier of V>>-facing explanation text.
        blocks: Mapping ``label -> [content, ...]``. A document MAY
            have multiple blocks with the same label (e.g. two
            bash-bootstrap fragments); parser preserves document order.
            Unlabelled fences accumulate under the empty-string key.
    """

    prose: str
    blocks: dict[str, list[str]] = field(default_factory=dict)

    def get_first(self, label: str) -> str | None:
        """First block with ``label``, or None if absent."""
        items = self.blocks.get(label, [])
        return items[0] if items else None

    def get_all(self, label: str) -> list[str]:
        """All blocks with ``label`` in document order."""
        return list(self.blocks.get(label, []))

    def has(self, label: str) -> bool:
        """True iff at least one block carries ``label``."""
        return bool(self.blocks.get(label))

    def labels_present(self) -> tuple[str, ...]:
        """Every label that appears at least once, in first-seen order."""
        return tuple(self.blocks.keys())


# --- fence matchers --------------------------------------------------------

# Opening fence: `^```<label?>[anything...]$`. The label is the first
# bareword; trailing metadata (e.g. ``title="x"``) is tolerated but
# discarded. Closing fence is just ``` with nothing after it.
_FENCE_OPEN_RE = re.compile(
    r"^\s*```[ \t]*(?P<label>[a-zA-Z0-9_.\-+]*)[ \t]*[^\n]*$"
)
_FENCE_CLOSE_RE = re.compile(r"^\s*```\s*$")


# --- parser ----------------------------------------------------------------

def _consume_fenced_block(
    lines: list[str], start: int
) -> tuple[str, int, bool]:
    """Walk from ``start`` to the next closing fence.

    ``start`` is the first line *after* the opening fence. Returns
    ``(body, next_index, closed)``. Closing fence, when found, is
    consumed but not included in the body.
    """
    end = len(lines)
    i = start
    while i < end and not _FENCE_CLOSE_RE.match(lines[i].rstrip("\n")):
        i += 1
    body = "".join(lines[start:i]).rstrip("\n")
    closed = i < end
    return body, (i + 1 if closed else i), closed


def parse_happi(text: str) -> HappiDocument:
    """Parse a HAPPI markdown document into a ``HappiDocument``.

    Single top-level loop over lines; body-collection is delegated to
    ``_consume_fenced_block`` so this function stays KISS. Unclosed
    fences at EOF emit one stderr warning and still commit the partial
    block so the caller never silently loses content.
    """
    blocks: dict[str, list[str]] = {}
    prose: list[str] = []
    lines = text.splitlines(keepends=True)
    i, n = 0, len(lines)
    while i < n:
        match = _FENCE_OPEN_RE.match(lines[i].rstrip("\n"))
        if match is None:
            prose.append(lines[i])
            i += 1
            continue
        label = match.group("label").strip()
        body, i, closed = _consume_fenced_block(lines, i + 1)
        if not closed:
            sys.stderr.write(
                f"[happi.parser] warning: unclosed fence (label={label!r}) "
                "at EOF; committing partial block\n"
            )
        blocks.setdefault(label, []).append(body)
    return HappiDocument(prose="".join(prose).strip(), blocks=blocks)


def parse_happi_file(path: str) -> HappiDocument:
    """Convenience: read ``path`` as UTF-8 and parse it."""
    with open(path, encoding="utf-8") as fh:
        return parse_happi(fh.read())


__all__ = [
    "CANONICAL_LABELS",
    "HappiDocument",
    "parse_happi",
    "parse_happi_file",
]
