"""Tests for HAL knowledge absorber."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.knowledge_absorber import (
    AbsorptionReport,
    KnowledgeAbsorber,
    KnowledgeItem,
    _auto_wikilink,
    _detect_source_type,
    _extract_audio,
    _parse_items,
    _strip_html,
    _whisper_cli,
    _whisper_python,
)


class TestDetectSourceType(unittest.TestCase):

    def test_github(self):
        self.assertEqual(_detect_source_type("https://github.com/org/repo"), "github")

    def test_arxiv_url(self):
        self.assertEqual(_detect_source_type("https://arxiv.org/abs/2603.28990"), "arxiv")

    def test_arxiv_id(self):
        self.assertEqual(_detect_source_type("arxiv:2603.28990"), "arxiv")

    def test_pdf(self):
        self.assertEqual(_detect_source_type("/path/to/paper.pdf"), "paper")

    def test_url(self):
        self.assertEqual(_detect_source_type("https://example.com/article"), "article")

    def test_audio_mp3(self):
        self.assertEqual(_detect_source_type("/tmp/podcast.mp3"), "audio")

    def test_audio_wav(self):
        self.assertEqual(_detect_source_type("/tmp/recording.wav"), "audio")

    def test_audio_m4a(self):
        self.assertEqual(_detect_source_type("/tmp/voice.m4a"), "audio")

    def test_video_mp4(self):
        self.assertEqual(_detect_source_type("/tmp/lecture.mp4"), "audio")

    def test_youtube(self):
        self.assertEqual(_detect_source_type("https://youtube.com/watch?v=abc"), "audio")

    def test_youtu_be(self):
        self.assertEqual(_detect_source_type("https://youtu.be/abc"), "audio")


class TestStripHtml(unittest.TestCase):

    def test_strips_tags(self):
        self.assertEqual(_strip_html("<p>hello</p>"), "hello")

    def test_strips_scripts(self):
        result = _strip_html("<script>evil()</script>text")
        self.assertNotIn("evil", result)
        self.assertIn("text", result)

    def test_strips_styles(self):
        result = _strip_html("<style>.red{}</style>content")
        self.assertIn("content", result)


class TestParseItems(unittest.TestCase):

    def test_valid_json(self):
        text = '[{"concept": "test", "description": "desc", "relevance": 8}]'
        items = _parse_items(text, "source")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].concept, "test")

    def test_invalid(self):
        self.assertEqual(_parse_items("not json", "s"), [])

    def test_embedded(self):
        text = 'Here: [{"concept": "c1"}] done'
        self.assertEqual(len(_parse_items(text, "s")), 1)


class TestAbsorptionReport(unittest.TestCase):

    def test_to_dict(self):
        report = AbsorptionReport(
            source="test", source_type="article", title="Test",
            absorb_time=1.0, items=[], summary="test",
        )
        d = report.to_dict()
        self.assertEqual(d["source"], "test")


class TestExtractAudio(unittest.TestCase):

    def test_nonexistent_file_returns_placeholder(self):
        title, content = _extract_audio("/nonexistent/file.mp3")
        self.assertIn("Whisper not available", content)

    def test_whisper_cli_nonexistent(self):
        self.assertEqual(_whisper_cli("/nonexistent.mp3"), "")

    def test_whisper_python_nonexistent(self):
        self.assertEqual(_whisper_python("/nonexistent.mp3"), "")


class TestAutoWikilink(unittest.TestCase):

    def test_empty_items(self):
        self.assertEqual(_auto_wikilink([]), [])

    @patch("lib.hal.kono_bridge.recall_for_query", return_value="Known concept with extensive context here")
    def test_matching_concept_creates_wikilink(self, _mock):
        items = [KnowledgeItem(
            concept="RAILLM", description="test", category="technique",
            relevance=8, actionable=True, hal_module="", source="test",
        )]
        links = _auto_wikilink(items)
        self.assertIn("[[RAILLM]]", links)

    @patch("lib.hal.kono_bridge.recall_for_query", return_value="")
    def test_no_match_no_wikilink(self, _mock):
        items = [KnowledgeItem(
            concept="obscure", description="test", category="technique",
            relevance=3, actionable=False, hal_module="", source="test",
        )]
        links = _auto_wikilink(items)
        self.assertEqual(links, [])


class TestKnowledgeAbsorber(unittest.TestCase):

    def test_create(self):
        absorber = KnowledgeAbsorber()
        self.assertIsNone(absorber._registry)

    def test_create_with_registry(self):
        reg = MagicMock()
        absorber = KnowledgeAbsorber(reg)
        self.assertEqual(absorber._registry, reg)


if __name__ == "__main__":
    unittest.main()
