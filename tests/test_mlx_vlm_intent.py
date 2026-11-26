"""Wave 3: vision intent detection tests.

Verifies that visual/multimodal queries route to the 'vision' category
with confidence >= 0.85, while text-only queries remain unaffected.
"""
import os
import sys
import unittest

from lib.hal.intent_detector import IntentDetector


class TestVisionIntentPositive(unittest.TestCase):
    """Vision queries should classify as 'vision' with confidence >= 0.85."""

    def setUp(self):
        self.det = IntentDetector()

    def test_describe_image(self):
        intent = self.det.classify("describe this image")
        self.assertEqual(intent.category, "vision")
        self.assertGreaterEqual(intent.confidence, 0.85)

    def test_ocr_request(self):
        intent = self.det.classify("OCR the receipt and extract the total")
        self.assertEqual(intent.category, "vision")
        self.assertGreaterEqual(intent.confidence, 0.85)

    def test_screenshot_mention(self):
        intent = self.det.classify("What does this screenshot show?")
        self.assertEqual(intent.category, "vision")
        self.assertGreaterEqual(intent.confidence, 0.85)

    def test_data_uri(self):
        intent = self.det.classify("data:image/jpeg;base64,/9j/4AAQ describe it")
        self.assertEqual(intent.category, "vision")
        self.assertGreaterEqual(intent.confidence, 0.85)

    def test_object_detection(self):
        # "detect objects" — scoped to CV context, should be vision
        intent = self.det.classify("detect all objects in this photo")
        self.assertEqual(intent.category, "vision")
        self.assertGreaterEqual(intent.confidence, 0.85)

    def test_segment_query(self):
        intent = self.det.classify("segment the foreground from this picture")
        self.assertEqual(intent.category, "vision")
        self.assertGreaterEqual(intent.confidence, 0.85)

    def test_image_file_extension_boost(self):
        # PNG path mid-sentence → base 0.85 + 0.1 booster = 0.95
        intent = self.det.classify("analyse diagram.png and list the components")
        self.assertEqual(intent.category, "vision")
        self.assertGreaterEqual(intent.confidence, 0.95)

    def test_bmp_extension_boosts(self):
        # bmp is in _IMAGE_FILE_RE and _PATTERNS — booster fires
        intent = self.det.classify("open photo.bmp")
        self.assertEqual(intent.category, "vision")
        self.assertGreaterEqual(intent.confidence, 0.95)

    def test_tiff_extension_boosts(self):
        intent = self.det.classify("process scan.tiff")
        self.assertEqual(intent.category, "vision")
        self.assertGreaterEqual(intent.confidence, 0.95)

    def test_pie_chart_is_vision(self):
        intent = self.det.classify("describe the pie chart in this report")
        self.assertEqual(intent.category, "vision")
        self.assertGreaterEqual(intent.confidence, 0.85)


class TestVisionIntentNegative(unittest.TestCase):
    """Edge cases that must NOT route to vision."""

    def setUp(self):
        self.det = IntentDetector()

    def test_detect_bugs_is_not_vision(self):
        # "detect" without a visual object must not route to vision;
        # "bugs" does not match the debug pattern's \bbug\b token so category is chat — that is fine
        intent = self.det.classify("detect all bugs in code")
        self.assertNotEqual(intent.category, "vision")

    def test_compound_extension_not_boosted(self):
        # report.png.zip — the .png is not a standalone image file
        intent = self.det.classify("extract report.png.zip")
        self.assertNotEqual(intent.category, "vision")

    def test_bare_chart_is_not_vision(self):
        # "chart" alone (no visual qualifier) must not trigger vision
        intent = self.det.classify("chart the performance metrics over time")
        self.assertNotEqual(intent.category, "vision")


class TestVisionIntentRegression(unittest.TestCase):
    """Existing intent categories must be unaffected."""

    def setUp(self):
        self.det = IntentDetector()

    def test_debug_query_unchanged(self):
        intent = self.det.classify("fix the null pointer in auth.py line 42")
        self.assertNotEqual(intent.category, "vision")
        self.assertEqual(intent.category, "debug")

    def test_code_query_unchanged(self):
        intent = self.det.classify("write a Python function to sort a list")
        self.assertNotEqual(intent.category, "vision")
        self.assertEqual(intent.category, "code")

    def test_vision_mode_is_vision(self):
        intent = self.det.classify("show me the screenshot of this error")
        self.assertEqual(intent.category, "vision")
        self.assertEqual(intent.suggested_mode, "vision")


if __name__ == "__main__":
    unittest.main()
