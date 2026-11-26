"""Tests for HAL conversation export."""
import json
import os
import sys
import tempfile
import unittest

from lib.hal.export import (
    _escape_html,
    to_html_string,
    to_json,
    to_markdown,
    to_text,
)
from lib.hal.types import ChatMessage


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


class TestToMarkdown(unittest.TestCase):

    def test_basic_export(self):
        history = [_msg("user", "Hello"), _msg("assistant", "Hi there")]
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            to_markdown(history, path, title="Test Session")
            with open(path) as f:
                content = f.read()
            self.assertIn("# Test Session", content)
            self.assertIn("Hello", content)
            self.assertIn("Hi there", content)
        finally:
            os.unlink(path)

    def test_includes_model(self):
        history = [_msg("user", "Q")]
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            to_markdown(history, path, model="claude-opus-4-7")
            with open(path) as f:
                content = f.read()
            self.assertIn("claude-opus-4-7", content)
        finally:
            os.unlink(path)


class TestToJson(unittest.TestCase):

    def test_valid_json(self):
        history = [_msg("user", "Hello"), _msg("assistant", "Hi")]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            to_json(history, path, title="JSON Test")
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["title"], "JSON Test")
            self.assertEqual(len(data["messages"]), 2)
            self.assertEqual(data["turn_count"], 1)
        finally:
            os.unlink(path)


class TestToText(unittest.TestCase):

    def test_plain_text(self):
        history = [_msg("user", "Question"), _msg("assistant", "Answer")]
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            to_text(history, path)
            with open(path) as f:
                content = f.read()
            self.assertIn("[USER]", content)
            self.assertIn("[ASSISTANT]", content)
        finally:
            os.unlink(path)


class TestToHtmlString(unittest.TestCase):

    def test_contains_html_structure(self):
        history = [_msg("user", "Hello"), _msg("assistant", "Hi")]
        html = to_html_string(history, title="HTML Test")
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("HTML Test", html)
        self.assertIn("Hello", html)

    def test_escapes_html(self):
        history = [_msg("user", "<script>alert('xss')</script>")]
        html = to_html_string(history)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_empty_history(self):
        html = to_html_string([])
        self.assertIn("0 turns", html)


class TestEscapeHtml(unittest.TestCase):

    def test_escapes_angle_brackets(self):
        self.assertIn("&lt;", _escape_html("<div>"))

    def test_escapes_ampersand(self):
        self.assertIn("&amp;", _escape_html("A & B"))

    def test_newlines_to_br(self):
        self.assertIn("<br>", _escape_html("line1\nline2"))


if __name__ == "__main__":
    unittest.main()
