"""Tests for HAL conversation search."""
import json
import os
import sys
import tempfile
import unittest

from lib.hal.conversation_search import (
    ConversationSearch,
    _build_snippet,
    _search_replay,
    _tokenise,
)


class TestTokenise(unittest.TestCase):

    def test_basic(self):
        tokens = _tokenise("Hello world testing")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("testing", tokens)

    def test_short_words_excluded(self):
        tokens = _tokenise("I am ok at it")
        self.assertEqual(len(tokens), 0)

    def test_numbers_included(self):
        tokens = _tokenise("error 404 not found 200")
        self.assertIn("404", tokens)
        self.assertIn("found", tokens)


class TestBuildSnippet(unittest.TestCase):

    def test_short_content(self):
        snippet = _build_snippet("Hello world", {"hello"})
        self.assertEqual(snippet, "Hello world")

    def test_long_content_truncated(self):
        content = "prefix " * 20 + "TARGET_WORD " + "suffix " * 20
        snippet = _build_snippet(content, {"target_word"}, max_len=100)
        self.assertLessEqual(len(snippet), 120)  # max_len + prefix/suffix dots

    def test_ellipsis_added(self):
        content = "a " * 100 + "match here" + " b" * 100
        snippet = _build_snippet(content, {"match"}, max_len=50)
        self.assertTrue(snippet.startswith("...") or snippet.endswith("..."))


class TestSearchReplay(unittest.TestCase):

    def test_search_finds_matching_event(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({
                "session_id": "test-session",
                "events": [
                    {"event_type": "turn", "content": "Fix the authentication bug in the login module", "timestamp": 100.0},
                    {"event_type": "turn", "content": "Updated the CSS styles", "timestamp": 200.0},
                ],
            }, f)
            path = f.name
        try:
            results = _search_replay(path, {"authentication", "login"})
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].session_id, "test-session")
            self.assertGreater(results[0].relevance, 0)
        finally:
            os.unlink(path)

    def test_search_no_match(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({
                "session_id": "test",
                "events": [{"event_type": "turn", "content": "Hello world", "timestamp": 0}],
            }, f)
            path = f.name
        try:
            results = _search_replay(path, {"authentication"})
            self.assertEqual(len(results), 0)
        finally:
            os.unlink(path)

    def test_search_invalid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("not json")
            path = f.name
        try:
            results = _search_replay(path, {"test"})
            self.assertEqual(len(results), 0)
        finally:
            os.unlink(path)


class TestConversationSearch(unittest.TestCase):

    def test_empty_query(self):
        search = ConversationSearch()
        results = search.find("")
        self.assertEqual(len(results), 0)

    def test_recent_empty(self):
        search = ConversationSearch()
        recent = search.recent()
        self.assertIsInstance(recent, list)


if __name__ == "__main__":
    unittest.main()
