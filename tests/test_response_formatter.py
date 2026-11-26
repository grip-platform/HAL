"""Tests for HAL response formatter."""
import os
import sys
import unittest

from lib.hal.response_formatter import ResponseFormatter


class TestForTerminal(unittest.TestCase):

    def setUp(self):
        self.fmt = ResponseFormatter()

    def test_preserves_plain_text(self):
        result = self.fmt.for_terminal("Hello world")
        self.assertIn("Hello world", result)

    def test_bold_gets_ansi(self):
        result = self.fmt.for_terminal("This is **bold** text")
        self.assertIn("\033[1m", result)
        self.assertIn("bold", result)

    def test_code_block_gets_colour(self):
        result = self.fmt.for_terminal("```python\nx = 1\n```")
        self.assertIn("\033[36m", result)  # Cyan
        self.assertIn("x = 1", result)

    def test_inline_code(self):
        result = self.fmt.for_terminal("Use `sorted()` function")
        self.assertIn("sorted()", result)


class TestForPlain(unittest.TestCase):

    def setUp(self):
        self.fmt = ResponseFormatter()

    def test_strips_bold(self):
        result = self.fmt.for_plain("This is **bold** text")
        self.assertEqual(result, "This is bold text")

    def test_strips_code_fences(self):
        result = self.fmt.for_plain("```python\nx = 1\n```")
        self.assertIn("x = 1", result)
        self.assertNotIn("```", result)

    def test_strips_inline_code(self):
        result = self.fmt.for_plain("Use `sorted()`")
        self.assertEqual(result, "Use sorted()")

    def test_strips_headers(self):
        result = self.fmt.for_plain("## Title\nContent")
        self.assertIn("Title", result)
        self.assertNotIn("##", result)


class TestForHtml(unittest.TestCase):

    def setUp(self):
        self.fmt = ResponseFormatter()

    def test_bold_to_strong(self):
        result = self.fmt.for_html("This is **bold**")
        self.assertIn("<strong>bold</strong>", result)

    def test_inline_code_to_tag(self):
        result = self.fmt.for_html("Use `sorted()`")
        self.assertIn("<code>sorted()</code>", result)

    def test_headers(self):
        result = self.fmt.for_html("# Title")
        self.assertIn("<h1>Title</h1>", result)

    def test_escapes_html(self):
        result = self.fmt.for_html("x < 5 & y > 3")
        self.assertIn("&lt;", result)
        self.assertIn("&amp;", result)


class TestForJson(unittest.TestCase):

    def setUp(self):
        self.fmt = ResponseFormatter()

    def test_text_only(self):
        result = self.fmt.for_json("Just some text")
        self.assertEqual(result["section_count"], 1)
        self.assertEqual(result["sections"][0]["type"], "text")

    def test_code_block_extracted(self):
        result = self.fmt.for_json("Before\n```python\nx = 1\n```\nAfter")
        code_sections = [s for s in result["sections"] if s["type"] == "code"]
        self.assertEqual(len(code_sections), 1)
        self.assertEqual(code_sections[0]["language"], "python")
        self.assertIn("x = 1", code_sections[0]["content"])

    def test_multiple_sections(self):
        text = "Intro\n```js\ncode1\n```\nMiddle\n```py\ncode2\n```\nEnd"
        result = self.fmt.for_json(text)
        self.assertGreater(result["section_count"], 3)


if __name__ == "__main__":
    unittest.main()
