"""Tests for Vulavula / Lelapa AI provider adapter."""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.hal.providers.vulavula import (
    LANGUAGES,
    SentimentResult,
    TranscriptionResult,
    TranslationResult,
    VulavulaProvider,
    _API_CODE,
    _resolve_api_key,
    _to_api_code,
)


class TestTranslationResult(unittest.TestCase):

    def test_creation(self):
        result = TranslationResult(
            source_lang="zul", target_lang="eng",
            source_text="Sawubona", translated_text="Hello",
        )
        self.assertEqual(result.translated_text, "Hello")
        self.assertEqual(result.source_lang, "zul")

    def test_frozen(self):
        result = TranslationResult("zul", "eng", "test", "test")
        with self.assertRaises(AttributeError):
            result.translated_text = "changed"


class TestSentimentResult(unittest.TestCase):

    def test_creation(self):
        result = SentimentResult(
            text="Great!", sentiment="positive",
            confidence=0.95, language="eng",
        )
        self.assertEqual(result.sentiment, "positive")
        self.assertGreater(result.confidence, 0.9)


class TestTranscriptionResult(unittest.TestCase):

    def test_creation(self):
        result = TranscriptionResult(text="Hello world", language="eng", duration_s=2.5)
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.duration_s, 2.5)

    def test_defaults(self):
        result = TranscriptionResult(text="test", language="zul")
        self.assertEqual(result.duration_s, 0.0)


class TestLanguages(unittest.TestCase):

    def test_has_south_african_languages(self):
        self.assertIn("zul", LANGUAGES)
        self.assertIn("xho", LANGUAGES)
        self.assertIn("sot", LANGUAGES)
        self.assertIn("afr", LANGUAGES)
        self.assertIn("eng", LANGUAGES)

    def test_has_east_african(self):
        self.assertIn("swh", LANGUAGES)
        self.assertIn("kin", LANGUAGES)

    def test_count(self):
        self.assertGreaterEqual(len(LANGUAGES), 17)


class TestApiCodeMapping(unittest.TestCase):
    """Test TNMR-derived _Latn suffix mapping."""

    def test_zulu_gets_latn_suffix(self):
        self.assertEqual(_to_api_code("zul"), "zul_Latn")

    def test_english_gets_latn_suffix(self):
        self.assertEqual(_to_api_code("eng"), "eng_Latn")

    def test_all_languages_have_mapping(self):
        for code in LANGUAGES:
            self.assertIn(code, _API_CODE)
            self.assertTrue(_API_CODE[code].endswith("_Latn"))

    def test_unknown_code_still_gets_suffix(self):
        self.assertEqual(_to_api_code("xyz"), "xyz_Latn")


class TestResolveApiKey(unittest.TestCase):

    def test_env_source(self):
        with patch.dict(os.environ, {"TEST_KEY": "abc123"}):
            self.assertEqual(_resolve_api_key("env:TEST_KEY"), "abc123")

    def test_env_missing(self):
        self.assertEqual(_resolve_api_key("env:NONEXISTENT_KEY_XYZ"), "")

    def test_literal(self):
        self.assertEqual(_resolve_api_key("my-literal-key"), "my-literal-key")


class TestVulavulaProvider(unittest.TestCase):

    def setUp(self):
        self.provider = VulavulaProvider(api_key_source="test-key")

    @patch("lib.hal.providers.vulavula.VulavulaProvider._post")
    def test_translate(self, mock_post):
        mock_post.return_value = {
            "translation": [{"translated_text": "Hello world"}],
        }
        result = self.provider.translate("Sawubona umhlaba", "zul", "eng")
        self.assertEqual(result.translated_text, "Hello world")
        self.assertEqual(result.source_lang, "zul")
        self.assertEqual(result.target_lang, "eng")

    @patch("lib.hal.providers.vulavula.VulavulaProvider._post")
    def test_translate_sends_latn_codes(self, mock_post):
        """TNMR pattern: API expects _Latn suffix on language codes."""
        mock_post.return_value = {
            "translation": [{"translated_text": "Hello"}],
        }
        self.provider.translate("Sawubona", "zul", "eng")
        call_payload = mock_post.call_args[0][1]
        self.assertEqual(call_payload["source_lang"], "zul_Latn")
        self.assertEqual(call_payload["target_lang"], "eng_Latn")

    @patch("lib.hal.providers.vulavula.VulavulaProvider._post")
    def test_translate_caches_result(self, mock_post):
        """Repeated translations should hit cache, not API."""
        mock_post.return_value = {
            "translation": [{"translated_text": "Hello"}],
        }
        self.provider.translate("Sawubona", "zul", "eng")
        self.provider.translate("Sawubona", "zul", "eng")
        # Only one API call — second was cached
        self.assertEqual(mock_post.call_count, 1)

    @patch("lib.hal.providers.vulavula.time.sleep")
    @patch("lib.hal.providers.vulavula.VulavulaProvider._post")
    def test_translate_empty_response(self, mock_post, mock_sleep):
        mock_post.return_value = {}
        result = self.provider.translate("test", "zul", "eng")
        self.assertEqual(result.translated_text, "")
        self.assertGreaterEqual(mock_sleep.call_count, 4)  # Retried with backoff

    @patch("lib.hal.providers.vulavula.VulavulaProvider._post")
    def test_analyse_sentiment(self, mock_post):
        mock_post.return_value = {
            "sentiment": "positive",
            "confidence": 0.92,
        }
        result = self.provider.analyse_sentiment("This is wonderful", "eng")
        self.assertEqual(result.sentiment, "positive")
        self.assertAlmostEqual(result.confidence, 0.92)

    @patch("lib.hal.providers.vulavula.VulavulaProvider._post")
    def test_analyse_sentiment_empty(self, mock_post):
        mock_post.return_value = {}
        result = self.provider.analyse_sentiment("test", "eng")
        self.assertEqual(result.sentiment, "neutral")
        self.assertEqual(result.confidence, 0.0)

    def test_transcribe_file_not_found(self):
        result = self.provider.transcribe("/nonexistent/audio.wav", "zul")
        self.assertEqual(result.text, "")

    @patch("lib.hal.providers.vulavula.urllib.request.urlopen")
    def test_transcribe_uses_sync_endpoint_and_lang_code_query(self, mock_urlopen):
        """Regression anchor (proven live 2026-05-31): transcribe MUST POST to
        /transcribe/sync with lang_code as a URL QUERY param and a file-only
        multipart, then read `transcription_text`. Reverting to /transcribe/process,
        a `language` form field, or a `transcription` response key — the bugs that
        shipped a 404 + silent no-op — fails this test."""
        import tempfile
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = req.data
            resp = MagicMock()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps({
                "transcription_text": "ek is die koningin",
                "language_code": "afr",
                "audio_length_seconds": 12.5,
            }).encode()
            return resp

        mock_urlopen.side_effect = fake_urlopen
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFFfake-wav-bytes")
            path = f.name
        try:
            result = self.provider.transcribe(path, language="afr")
        finally:
            os.unlink(path)

        self.assertIn("/transcribe/sync", captured["url"])
        self.assertIn("lang_code=afr", captured["url"])
        self.assertNotIn("/transcribe/process", captured["url"])
        self.assertIn(b'name="file"', captured["body"])
        self.assertNotIn(b'name="language"', captured["body"])
        self.assertEqual(result.text, "ek is die koningin")
        self.assertEqual(result.language, "afr")
        self.assertEqual(result.duration_s, 12.5)

    def test_list_languages(self):
        langs = self.provider.list_languages()
        self.assertIn("zul", langs)
        self.assertEqual(langs["zul"], "isiZulu")

    @patch("lib.hal.providers.vulavula.VulavulaProvider._post")
    def test_bridge_from_english(self, mock_post):
        """Bridge with English input skips input translation."""
        llm_fn = MagicMock(return_value="The answer is 42")
        result = self.provider.bridge("What is the answer?", "eng", llm_fn)
        self.assertEqual(result, "The answer is 42")
        llm_fn.assert_called_once_with("What is the answer?")
        mock_post.assert_not_called()

    @patch("lib.hal.providers.vulavula.VulavulaProvider._post")
    def test_bridge_from_zulu(self, mock_post):
        """Bridge from isiZulu: translate in, LLM, translate back."""
        mock_post.side_effect = [
            {"translation": [{"translated_text": "What is the answer?"}]},
            {"translation": [{"translated_text": "Impendulo ngu-42"}]},
        ]
        llm_fn = MagicMock(return_value="The answer is 42")

        result = self.provider.bridge("Yini impendulo?", "zul", llm_fn)
        self.assertEqual(result, "Impendulo ngu-42")
        llm_fn.assert_called_once_with("What is the answer?")
        self.assertEqual(mock_post.call_count, 2)

    @patch("lib.hal.providers.vulavula.VulavulaProvider._post")
    def test_bridge_to_different_target(self, mock_post):
        """Bridge can output in a different language than input."""
        mock_post.side_effect = [
            {"translation": [{"translated_text": "Hello"}]},
            {"translation": [{"translated_text": "Molo"}]},
        ]
        llm_fn = MagicMock(return_value="Hello back!")

        result = self.provider.bridge("Sawubona", "zul", llm_fn, target_lang="xho")
        self.assertEqual(result, "Molo")


if __name__ == "__main__":
    unittest.main()
