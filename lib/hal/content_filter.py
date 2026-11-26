"""HAL Content Filter — redact sensitive data before sending to LLMs.

Strips API keys, passwords, email addresses, and other PII from prompts
before they reach external providers. Essential for privacy-conscious
environments where local models handle sensitive data but cloud providers
must not see it.

Usage:
    from lib.hal.content_filter import ContentFilter

    filt = ContentFilter()
    safe_prompt = filt.redact(user_message)
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RedactionResult:
    """Result of content redaction."""
    text: str
    redactions: int
    categories: tuple[str, ...]


# Patterns to redact: (name, compiled regex, replacement)
_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("api_key", re.compile(r"\b(?:sk|gsk|pk|ak|key)[_-][A-Za-z0-9_-]{20,}\b"), "[REDACTED_KEY]"),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"), "Bearer [REDACTED]"),
    ("password", re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE), "password: [REDACTED]"),
    ("email", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[REDACTED_EMAIL]"),
    ("ip_address", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[REDACTED_IP]"),
    ("credit_card", re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"), "[REDACTED_CC]"),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
]


class ContentFilter:
    """Redacts sensitive content from text."""

    def __init__(self, extra_patterns: list[tuple[str, str, str]] | None = None):
        self._patterns = list(_PATTERNS)
        if extra_patterns:
            for name, pattern, replacement in extra_patterns:
                self._patterns.append((name, re.compile(pattern), replacement))

    def redact(self, text: str) -> RedactionResult:
        """Redact all sensitive patterns from text."""
        result = text
        total = 0
        categories: set[str] = set()

        for name, pattern, replacement in self._patterns:
            new_text, count = pattern.subn(replacement, result)
            if count > 0:
                total += count
                categories.add(name)
                result = new_text

        return RedactionResult(text=result, redactions=total, categories=tuple(sorted(categories)))

    def is_sensitive(self, text: str) -> bool:
        """Check if text contains sensitive content without redacting."""
        return any(pattern.search(text) for _, pattern, _ in self._patterns)
