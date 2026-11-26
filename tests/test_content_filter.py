"""Tests for HAL content filter."""
import os, sys, unittest
from lib.hal.content_filter import ContentFilter

class TestContentFilter(unittest.TestCase):
    def setUp(self):
        self.filt = ContentFilter()

    def test_redacts_api_key(self):
        r = self.filt.redact("My key is key_test_aaabbbcccdddeeefffggg012")
        self.assertIn("[REDACTED_KEY]", r.text)
        self.assertGreater(r.redactions, 0)

    def test_redacts_email(self):
        r = self.filt.redact("Contact user@example.com for details")
        self.assertIn("[REDACTED_EMAIL]", r.text)

    def test_redacts_password(self):
        r = self.filt.redact("password: mysecret123")
        self.assertIn("[REDACTED]", r.text)

    def test_redacts_bearer(self):
        r = self.filt.redact("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5")
        self.assertIn("[REDACTED]", r.text)

    def test_no_redaction_for_clean(self):
        r = self.filt.redact("How do I sort a list in Python?")
        self.assertEqual(r.redactions, 0)
        self.assertEqual(r.text, "How do I sort a list in Python?")

    def test_is_sensitive(self):
        self.assertTrue(self.filt.is_sensitive("key sk_abc123def456ghi789jkl012"))
        self.assertFalse(self.filt.is_sensitive("just normal text"))

    def test_categories_tracked(self):
        r = self.filt.redact("Email user@test.com and key sk_abc123def456ghi789jkl012")
        self.assertIn("email", r.categories)

    def test_custom_pattern(self):
        filt = ContentFilter(extra_patterns=[("custom", r"SECRET\d+", "[CUSTOM]")])
        r = filt.redact("The code is SECRET42")
        self.assertIn("[CUSTOM]", r.text)

if __name__ == "__main__":
    unittest.main()
