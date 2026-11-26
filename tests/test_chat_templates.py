"""Tests for HAL chat templates."""
import os, sys, unittest
from lib.hal.chat_templates import ChatTemplate, TemplateLibrary

class TestTemplateLibrary(unittest.TestCase):
    def test_get_code_review(self):
        t = TemplateLibrary().get("code_review")
        self.assertIsNotNone(t)
        self.assertEqual(t.name, "code_review")

    def test_get_unknown(self):
        self.assertIsNone(TemplateLibrary().get("nonexistent"))

    def test_list_all(self):
        templates = TemplateLibrary().list_templates()
        self.assertGreater(len(templates), 4)

    def test_list_by_category(self):
        dev = TemplateLibrary().list_templates(category="development")
        self.assertTrue(all(t.category == "development" for t in dev))

    def test_categories(self):
        cats = TemplateLibrary().categories()
        self.assertIn("development", cats)

    def test_template_has_follow_ups(self):
        t = TemplateLibrary().get("debug")
        self.assertGreater(len(t.follow_ups), 0)

    def test_frozen(self):
        t = TemplateLibrary().get("debug")
        with self.assertRaises(AttributeError):
            t.name = "changed"

if __name__ == "__main__":
    unittest.main()
