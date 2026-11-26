"""Tests for HAL output validator."""
import os, sys, unittest
from lib.hal.output_validator import OutputValidator

class TestValidateJson(unittest.TestCase):
    def test_valid(self):
        r = OutputValidator().validate_json('Here: {"key": "value"}')
        self.assertTrue(r.valid)
        self.assertIn("key", r.extracted)

    def test_valid_array(self):
        r = OutputValidator().validate_json('Result: [1, 2, 3]')
        self.assertTrue(r.valid)

    def test_invalid(self):
        r = OutputValidator().validate_json("No JSON here")
        self.assertFalse(r.valid)

    def test_malformed(self):
        r = OutputValidator().validate_json('{"broken: json}')
        self.assertFalse(r.valid)

class TestValidateCode(unittest.TestCase):
    def test_valid_python(self):
        r = OutputValidator().validate_code("```python\ndef foo():\n    return 42\n```")
        self.assertTrue(r.valid)

    def test_invalid_python(self):
        r = OutputValidator().validate_code("```python\ndef foo(:\n```")
        self.assertFalse(r.valid)

    def test_no_code_block(self):
        r = OutputValidator().validate_code("Just text, no code")
        self.assertFalse(r.valid)

class TestValidateList(unittest.TestCase):
    def test_valid(self):
        r = OutputValidator().validate_list("- Item 1\n- Item 2\n- Item 3")
        self.assertTrue(r.valid)

    def test_numbered(self):
        r = OutputValidator().validate_list("1. First\n2. Second")
        self.assertTrue(r.valid)

    def test_no_list(self):
        r = OutputValidator().validate_list("Just a paragraph of text.")
        self.assertFalse(r.valid)

class TestValidateMarkdown(unittest.TestCase):
    def test_with_headers(self):
        self.assertTrue(OutputValidator().validate_markdown("# Title\nContent").valid)

    def test_with_code(self):
        self.assertTrue(OutputValidator().validate_markdown("Here: ```code```").valid)

    def test_plain_text(self):
        self.assertFalse(OutputValidator().validate_markdown("Just plain text").valid)

if __name__ == "__main__":
    unittest.main()
