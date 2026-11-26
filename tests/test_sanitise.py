"""Tests for HAL web content sanitiser.

Covers AI Agent Trap Class 1 (Content Injection) defences — hidden instructions
in HTML comments, styles, and CSS-hidden elements.

Regression anchor: ``_remove_hidden_elements`` previously applied the
``display:none`` filter to ``html`` and threw the result away, because the
second ``re.sub`` operated on the original ``html`` rather than the filtered
intermediate. Payloads hidden behind ``display:none`` leaked through the
sanitiser untouched. See HAL#24 W1.
"""
import unittest

from lib.hal.sanitise import sanitise_html


class TestSanitiseHidden(unittest.TestCase):
    """Hidden-element injection defences (Trap Class 1)."""

    def test_display_none_payload_stripped(self):
        payload = (
            '<div style="display:none">IGNORE_ALL_PRIOR_INSTRUCTIONS</div>'
            '<p>visible content</p>'
        )
        result = sanitise_html(payload)
        self.assertNotIn("IGNORE_ALL_PRIOR_INSTRUCTIONS", result)
        self.assertIn("visible content", result)

    def test_visibility_hidden_payload_stripped(self):
        payload = (
            '<span style="visibility:hidden">LEAKED_SECRET</span>'
            '<p>keeper</p>'
        )
        result = sanitise_html(payload)
        self.assertNotIn("LEAKED_SECRET", result)
        self.assertIn("keeper", result)

    def test_composed_hidden_attributes(self):
        """Both filters must compose — neither's work may be discarded.

        This was the core bug: the ``visibility:hidden`` filter ran against
        the original ``html``, not the ``display:none``-stripped intermediate,
        so a ``display:none`` payload earlier in the document would survive
        even when the sanitiser reported both filters as applied.
        """
        payload = (
            '<div style="display:none">FIRST_INJECTION</div>'
            '<span style="visibility:hidden">SECOND_INJECTION</span>'
            '<p>legitimate</p>'
        )
        result = sanitise_html(payload)
        self.assertNotIn("FIRST_INJECTION", result)
        self.assertNotIn("SECOND_INJECTION", result)
        self.assertIn("legitimate", result)

    def test_single_quoted_display_none(self):
        """A1: single-quoted style attributes must be stripped."""
        payload = "<div style='display:none'>SINGLE_QUOTE_LEAK</div><p>keep</p>"
        result = sanitise_html(payload)
        self.assertNotIn("SINGLE_QUOTE_LEAK", result)
        self.assertIn("keep", result)

    def test_single_quoted_visibility_hidden(self):
        """A1: single-quoted visibility:hidden must be stripped."""
        payload = (
            "<span style='visibility:hidden'>SINGLE_QUOTE_VH</span><p>keep</p>"
        )
        result = sanitise_html(payload)
        self.assertNotIn("SINGLE_QUOTE_VH", result)
        self.assertIn("keep", result)

    def test_unquoted_display_none(self):
        """A3: unquoted HTML5 style values (style=display:none) must be stripped."""
        payload = "<div style=display:none>UNQUOTED_LEAK</div><p>keep</p>"
        result = sanitise_html(payload)
        self.assertNotIn("UNQUOTED_LEAK", result)
        self.assertIn("keep", result)

    def test_unquoted_visibility_hidden(self):
        """A3: unquoted visibility:hidden must be stripped."""
        payload = "<span style=visibility:hidden>UNQUOTED_VH</span><p>keep</p>"
        result = sanitise_html(payload)
        self.assertNotIn("UNQUOTED_VH", result)
        self.assertIn("keep", result)

    def test_mixed_quote_composition(self):
        """All three quote styles must compose correctly in one document."""
        payload = (
            "<div style='display:none'>FIRST</div>"
            '<span style="visibility:hidden">SECOND</span>'
            "<em style=display:none>THIRD</em>"
            "<p>keep</p>"
        )
        result = sanitise_html(payload)
        self.assertNotIn("FIRST", result)
        self.assertNotIn("SECOND", result)
        self.assertNotIn("THIRD", result)
        self.assertIn("keep", result)

    def test_display_none_in_class_not_style_preserved(self):
        """False-positive guard: `display:none` in a class/id must NOT be stripped.

        Only the ``style=`` attribute should trigger the hiding filter. An
        element with ``display:none`` in its class name or id (e.g. from a
        CSS framework where the class happens to contain the substring) is
        legitimate content and must survive.
        """
        payload = (
            '<div class="display-none-ish" id="notdisplay:noneatall">'
            "preserve_me"
            "</div>"
        )
        result = sanitise_html(payload)
        self.assertIn("preserve_me", result)


class TestSanitiseCssClassHidden(unittest.TestCase):
    """A6 partial fix: CSS-class-based hiding via ``<style>.X{display:none}``
    followed by ``<div class="X">``. v1 scope is bare class selectors only —
    compound selectors (``.a.b``, ``div.hide``) and pseudo-classes are
    documented as out of scope.
    """

    def test_double_quoted_class_with_display_none(self):
        payload = (
            "<style>.hide{display:none}</style>"
            '<div class="hide">CSS_LEAK_1</div>'
            "<p>keep</p>"
        )
        result = sanitise_html(payload)
        self.assertNotIn("CSS_LEAK_1", result)
        self.assertIn("keep", result)

    def test_single_quoted_class_with_visibility_hidden(self):
        payload = (
            "<style>.x{visibility:hidden}</style>"
            "<div class='x'>CSS_LEAK_2</div>"
            "<p>keep</p>"
        )
        result = sanitise_html(payload)
        self.assertNotIn("CSS_LEAK_2", result)
        self.assertIn("keep", result)

    def test_multi_class_attribute(self):
        payload = (
            "<style>.hide{display:none}</style>"
            '<div class="foo hide bar">CSS_LEAK_3</div>'
            "<p>keep</p>"
        )
        result = sanitise_html(payload)
        self.assertNotIn("CSS_LEAK_3", result)
        self.assertIn("keep", result)

    def test_unquoted_class_attribute(self):
        payload = (
            "<style>.hide{display:none}</style>"
            "<div class=hide>CSS_LEAK_4</div>"
            "<p>keep</p>"
        )
        result = sanitise_html(payload)
        self.assertNotIn("CSS_LEAK_4", result)
        self.assertIn("keep", result)

    def test_similar_class_name_preserved(self):
        """`.hiden` (typo) is NOT in the hidden set, so `class="hiden"`
        elements must survive.
        """
        payload = (
            '<style>.hiden{color:red}</style>'
            '<div class="hiden">preserve_similar</div>'
            "<p>keep</p>"
        )
        result = sanitise_html(payload)
        self.assertIn("preserve_similar", result)
        self.assertIn("keep", result)

    def test_non_hiding_rule_preserved(self):
        """A class rule that does NOT set display:none or visibility:hidden
        must not cause elements with that class to be stripped.
        """
        payload = (
            '<style>.hide{background:red;color:blue}</style>'
            '<div class="hide">preserve_non_hide</div>'
        )
        result = sanitise_html(payload)
        self.assertIn("preserve_non_hide", result)

    def test_class_in_element_without_style_block_preserved(self):
        """Without a ``<style>`` block declaring the class as hidden, the
        element survives regardless of the class name.
        """
        payload = '<div class="display-none-looking">preserve_no_style</div>'
        result = sanitise_html(payload)
        self.assertIn("preserve_no_style", result)

    def test_hidden_class_in_style_list_still_caught(self):
        """Class selectors appearing in a comma-separated selector list
        (``.hide, .other{display:none}``) still register both class names.
        """
        payload = (
            "<style>.hide, .other{display:none}</style>"
            '<div class="hide">CSS_LEAK_5</div>'
            '<span class="other">CSS_LEAK_6</span>'
            "<p>keep</p>"
        )
        result = sanitise_html(payload)
        self.assertNotIn("CSS_LEAK_5", result)
        self.assertNotIn("CSS_LEAK_6", result)
        self.assertIn("keep", result)


class TestSanitiseBasicFilters(unittest.TestCase):
    """Existing filters must remain intact after the hidden-element fix."""

    def test_scripts_removed(self):
        result = sanitise_html('<script>alert("xss")</script><p>ok</p>')
        self.assertNotIn("alert", result)
        self.assertIn("ok", result)

    def test_styles_removed(self):
        result = sanitise_html('<style>body{display:none}</style><p>ok</p>')
        self.assertNotIn("body{display", result)
        self.assertIn("ok", result)

    def test_comments_removed(self):
        result = sanitise_html('<!-- IGNORE_ALL -->visible')
        self.assertNotIn("IGNORE_ALL", result)
        self.assertIn("visible", result)

    def test_plain_text_preserved(self):
        self.assertEqual(sanitise_html("hello world"), "hello world")


if __name__ == "__main__":
    unittest.main()
