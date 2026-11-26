"""String-literal escapers for AppleScript and JavaScript-for-Automation.

``applescript_escape`` is ported verbatim from
``lib.grip_notify.MacOSLocalBackend._q`` and has been battle-tested across
notification, iMessage, and tab-manager call sites for multiple generations.

``js_escape`` is new for α-3b. osa supports two languages — AppleScript and
JavaScript-for-Automation (JXA) — and the escaping rules differ:

*   AppleScript only needs ``\\`` and ``"`` escaped. Newlines, tabs, and
    unicode pass through unchanged.
*   JXA (ES5-compatible) additionally needs the control characters
    ``\\n``, ``\\r``, and ``\\t`` escaped, because the JavaScript parser
    treats a literal newline inside a string literal as a syntax error.

Keeping the two escapers separate (not a single "generic" escape) is
deliberate: a shared escape function would either over-escape AppleScript
(breaking legitimate newlines inside literals) or under-escape JavaScript
(letting newlines break the source). Language dispatch happens at the
call site via the ``language`` field on :class:`lib.hal.osa.Script`.
"""
from __future__ import annotations

__all__ = ["applescript_escape", "js_escape"]


def applescript_escape(s: str) -> str:
    """Quote ``s`` as an AppleScript string literal.

    Order of substitutions matters: the backslash escape MUST run first so
    that the escape character emitted for a quote is not itself doubled.
    """
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def js_escape(s: str) -> str:
    """Quote ``s`` as a JavaScript double-quoted string literal.

    Escapes, in order: backslash (must be first to avoid doubling later
    escape sequences), double-quote, then the three parser-breaking control
    characters (``\\n``, ``\\r``, ``\\t``). Other control characters are
    left through — callers running JXA should avoid embedding raw binary.
    """
    return (
        '"'
        + s.replace("\\", "\\\\")
           .replace('"', '\\"')
           .replace("\n", "\\n")
           .replace("\r", "\\r")
           .replace("\t", "\\t")
        + '"'
    )
