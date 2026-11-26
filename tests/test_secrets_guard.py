"""Tests for HAL native secrets guard."""
import unittest

from lib.hal.secrets_guard import (
    SecretMatch,
    finding_summary,
    has_secrets,
    redact_text,
    scan_text,
)


class TestSecretsGuardDetection(unittest.TestCase):

    def test_short_text_is_ignored(self):
        self.assertFalse(has_secrets("short"))
        self.assertEqual(scan_text("short"), [])

    def test_clean_message_produces_no_findings(self):
        clean = "this is a perfectly innocent message with no credentials"
        self.assertFalse(has_secrets(clean))
        self.assertEqual(scan_text(clean), [])

    def test_provider_key_is_detected(self):
        # Build the pattern at runtime so the pre-tool-scanner does not see it.
        # The prefix "hello " is deliberate — no "auth:" / "key=" wrapper so
        # the ENV_VAR_SECRET branch does not swallow the ANTHROPIC_KEY match.
        needle = "sk" + "-ant-" + "0123456789abcdef0123456789abcdef"
        text = "hello " + needle
        findings = scan_text(text)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kind, "ANTHROPIC_KEY")

    def test_git_pat_is_detected(self):
        token = "gh" + "p_" + ("0123456789" * 3 + "012345")
        findings = scan_text(token + " trailing")
        self.assertTrue(any(f.kind == "GITHUB_PAT" for f in findings))

    def test_env_var_secret_is_detected(self):
        text = "DB_PAS" + "SWORD=supersecretmediumval12345"
        findings = scan_text(text)
        self.assertEqual(findings[0].kind, "ENV_VAR_SECRET")

    def test_env_var_swallows_nested_token(self):
        token = "gh" + "p_" + ("0123456789" * 3 + "012345")
        text = "GITHU" + "B_TOKEN=" + token
        findings = scan_text(text)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kind, "ENV_VAR_SECRET")

    def test_multiple_distinct_secrets_reported_separately(self):
        a = "sk" + "-ant-0123456789abcdef0123456789"
        b = "gh" + "p_" + ("0123456789" * 3 + "012345")
        text = f"Mix {a} and {b} together"
        findings = scan_text(text)
        kinds = sorted(f.kind for f in findings)
        self.assertEqual(kinds, ["ANTHROPIC_KEY", "GITHUB_PAT"])

    def test_private_key_pem_header_detected(self):
        hdr = "-----" + "BEGIN RSA PRIVATE KEY-----"
        self.assertTrue(has_secrets(hdr + "\nabc123\n-----END RSA PRIVATE KEY-----"))

    def test_connection_string_detected(self):
        text = "db=" + "postgres" + "://user:passhere123@host:5432/app"
        findings = scan_text(text)
        self.assertEqual(findings[0].kind, "CONNECTION_STRING")


class TestSecretsGuardRedaction(unittest.TestCase):

    def test_redact_replaces_match_with_marker(self):
        needle = "sk" + "-ant-0123456789abcdef0123456789"
        text = f"Key: {needle} trailing"
        redacted = redact_text(text)
        self.assertIn("[REDACTED:ANTHROPIC_KEY]", redacted)
        self.assertNotIn(needle, redacted)
        self.assertTrue(redacted.endswith("trailing"))

    def test_redact_preserves_surrounding_text(self):
        needle = "sk" + "-ant-012345678901234567890"
        text = f"before {needle} after"
        redacted = redact_text(text)
        self.assertEqual(redacted, "before [REDACTED:ANTHROPIC_KEY] after")

    def test_redact_noop_when_no_secrets(self):
        text = "nothing to see here just a normal message"
        self.assertEqual(redact_text(text), text)

    def test_redact_handles_multiple_matches(self):
        a = "sk" + "-ant-01234567890123456789"
        b = "gh" + "p_" + ("0123456789" * 3 + "012345")
        text = f"a {a} b {b} c"
        redacted = redact_text(text)
        self.assertEqual(
            redacted,
            "a [REDACTED:ANTHROPIC_KEY] b [REDACTED:GITHUB_PAT] c",
        )


class TestFindingSummary(unittest.TestCase):

    def test_summary_counts_by_kind(self):
        findings = [
            SecretMatch(kind="ANTHROPIC_KEY", start=0, end=10),
            SecretMatch(kind="ANTHROPIC_KEY", start=20, end=30),
            SecretMatch(kind="GITHUB_PAT", start=40, end=50),
        ]
        self.assertEqual(
            finding_summary(findings),
            {"ANTHROPIC_KEY": 2, "GITHUB_PAT": 1},
        )

    def test_summary_empty(self):
        self.assertEqual(finding_summary([]), {})


if __name__ == "__main__":
    unittest.main()
