"""HAL Web Sanitiser — strip injection attacks from web content.

Defends against AI Agent Trap Class 1 (Content Injection): hidden instructions
in HTML comments, CSS, image metadata, accessibility tags.

Usage:
    from lib.hal.sanitise import sanitise_html

    clean = sanitise_html(raw_html)  # Safe for LLM consumption
"""
import re


def sanitise_html(html: str) -> str:
    """Strip potentially malicious content from HTML before LLM processing."""
    text = _remove_scripts(html)
    text = _remove_css_class_hidden(text)
    text = _remove_styles(text)
    text = _remove_comments(text)
    text = _remove_hidden_elements(text)
    text = _strip_tags(text)
    return _normalise_whitespace(text)


def _remove_scripts(html: str) -> str:
    return re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)


# CSS-class-based hiding (AI Agent Trap Class 1, A6). An attacker can do
# ``<style>.hide{display:none}</style><div class="hide">LEAK</div>`` —
# the ``<style>`` block is stripped by ``_remove_styles`` but without any
# CSS parsing the ``<div class="hide">`` is left intact, so its text
# content survives to the LLM. This pre-pass runs BEFORE ``_remove_styles``
# so we can still see the CSS: we scan every ``<style>`` block for bare
# class selectors with hiding declarations, then strip elements whose
# ``class`` attribute contains any matched class name.
#
# Scope is deliberately narrow — bare class selectors only
# (``.X{display:none}``). Compound selectors (``.a.b``, ``div.hide``,
# ``.a > .b``) and pseudo-classes are NOT handled. They require a real
# CSS parser and are deferred to a follow-up PR.
_STYLE_BLOCK_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.DOTALL | re.IGNORECASE)

# Rule body containing a hiding declaration. The first group is the
# selector list that precedes the ``{...}`` rule body. Selector lists
# are comma-separated (``.a, .b, .c``) so a later pass extracts bare
# class selectors from this string.
_HIDING_RULE_RE = re.compile(
    r"([^{}]+?)\s*\{[^}]*?"
    r"(?:display\s*:\s*none|visibility\s*:\s*hidden)"
    r"[^}]*?\}",
    re.DOTALL | re.IGNORECASE,
)

# A bare class selector inside a selector list. Matches ``.name`` only
# when it is the whole selector (not ``div.hide``, not ``.a.b``, not
# ``.a > .b``) — compound selectors are documented as v1 out of scope.
_BARE_CLASS_IN_SELECTOR_RE = re.compile(
    r"(?:^|[,\s])\.([a-zA-Z_][\w-]*)(?=[\s,{]|$)",
)


def _classes_in_selector(selector_list: str) -> list[str]:
    """Extract every bare class selector from a CSS selector list."""
    return [
        match.group(1)
        for match in _BARE_CLASS_IN_SELECTOR_RE.finditer(selector_list)
    ]


def _hidden_classes_in_css(css: str) -> list[str]:
    """Return every class name carrying a hiding declaration inside one CSS
    block. Two linear passes: rule bodies with hiding declarations, then
    bare class selectors inside each selector list.
    """
    classes: list[str] = []
    for rule in _HIDING_RULE_RE.finditer(css):
        classes.extend(_classes_in_selector(rule.group(1)))
    return classes


def _extract_hidden_classes(html: str) -> set[str]:
    """Return the set of CSS class names marked display:none or visibility:hidden
    via a bare class selector inside any ``<style>`` block.
    """
    hidden: set[str] = set()
    for block in _STYLE_BLOCK_RE.finditer(html):
        hidden.update(_hidden_classes_in_css(block.group(1)))
    return hidden


def _strip_element_with_class(html: str, class_name: str) -> str:
    """Remove elements whose ``class`` attribute contains ``class_name``.

    Handles double-quoted, single-quoted, and unquoted class values, and
    multi-class attributes via a word boundary so ``class="foo hide bar"``
    matches on ``hide`` but ``class="hiden"`` does not. Non-greedy closing
    tag match — does not descend into nested same-class elements, which
    is an accepted v1 limitation.
    """
    escaped = re.escape(class_name)
    pattern = re.compile(
        r'<[^>]+?\sclass\s*=\s*'
        r'(?:'
            r'"[^"]*\b' + escaped + r'\b[^"]*"'
            r"|"
            r"'[^']*\b" + escaped + r"\b[^']*'"
            r"|"
            + escaped +
            r'(?=[\s>])'
        r')'
        r'[^>]*>.*?</[^>]+>',
        re.DOTALL | re.IGNORECASE,
    )
    return pattern.sub("", html)


def _remove_css_class_hidden(html: str) -> str:
    """Strip elements hidden via a bare-class-selector CSS rule."""
    hidden = _extract_hidden_classes(html)
    for class_name in hidden:
        html = _strip_element_with_class(html, class_name)
    return html


def _remove_styles(html: str) -> str:
    return re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)


def _remove_comments(html: str) -> str:
    """Remove HTML comments — common injection vector."""
    return re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)


# Inline-hiding attribute value patterns. Detecting ``display:none`` or
# ``visibility:hidden`` in a style attribute under:
#   * double-quoted values:  style="display:none"
#   * single-quoted values:  style='display:none'
#   * unquoted values:       style=display:none (HTML5 permits this)
#
# CSS-class-based hiding — ``<style>.hide{display:none}</style>`` with
# ``<div class="hide">`` — is NOT covered. That requires a CSS parser and
# is tracked as a separate W3 item.
_HIDING_DECLARATION = r"display\s*:\s*none|visibility\s*:\s*hidden"

# Quoted style attribute. Uses a backreference so the closing quote matches
# the opening one, and ``[^"']*`` excludes both quote chars from the value
# body so we do not walk off the end of the attribute.
_HIDDEN_QUOTED_RE = re.compile(
    r'<[^>]+?\sstyle\s*=\s*(["\'])[^"\']*'
    rf"(?:{_HIDING_DECLARATION})"
    r'[^"\']*\1[^>]*>.*?</[^>]+>',
    re.DOTALL | re.IGNORECASE,
)

# Unquoted style attribute. HTML5 permits unquoted values when they contain
# no spaces, quotes, ``=``, ``>``, or backticks. Match the hiding declaration
# immediately after ``=`` and terminate on a char that ends the attribute.
_HIDDEN_UNQUOTED_RE = re.compile(
    r"<[^>]+?\sstyle\s*=\s*"
    rf"(?:{_HIDING_DECLARATION})"
    r"(?:[;\s][^>]*)?>.*?</[^>]+>",
    re.DOTALL | re.IGNORECASE,
)


def _remove_hidden_elements(html: str) -> str:
    """Remove elements hidden via inline ``display:none`` / ``visibility:hidden``.

    Handles double-quoted, single-quoted, and unquoted style attribute values.
    The quoted pass runs first, then the unquoted pass operates on the result
    so the filters compose correctly and neither pass discards the other's work.
    """
    text = _HIDDEN_QUOTED_RE.sub("", html)
    return _HIDDEN_UNQUOTED_RE.sub("", text)


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html)


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
