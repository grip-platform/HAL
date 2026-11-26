"""HAL Output Quality Filter — detect dismissive patterns in LLM responses.

Catches language that deflects responsibility for found issues. An LLM that
labels a bug "pre-existing" and moves on is optimising for self-image, not
for the user. Every found issue deserves equal attention.

Usage:
    from lib.hal.output_quality import OutputQualityChecker

    checker = OutputQualityChecker()
    warnings = checker.check("This is a pre-existing issue, not related to our changes.")
    # [QualityWarning(pattern="pre-existing", suggestion="Use 'found issue' — then fix or track it")]
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityWarning:
    """A detected dismissive pattern in output text."""
    pattern: str
    matched_text: str
    suggestion: str
    severity: str = "advisory"

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "matched_text": self.matched_text,
            "suggestion": self.suggestion,
            "severity": self.severity,
        }


_DISMISSAL_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (
        re.compile(r"\bpre-existing\b", re.IGNORECASE),
        "pre-existing",
        "Use 'found issue' — then fix it or track it",
    ),
    (
        re.compile(r"\bnot (?:related|relevant) to (?:this |the )?(sprint|change|PR|commit|task)\b", re.IGNORECASE),
        "not-related-to",
        "All found issues matter equally — investigate or file",
    ),
    (
        re.compile(r"\bunrelated (?:failure|error|issue|bug|problem)\b", re.IGNORECASE),
        "unrelated-issue",
        "Use 'additional failure' — then investigate the root cause",
    ),
    (
        re.compile(r"\bout of scope\b", re.IGNORECASE),
        "out-of-scope",
        "Only valid if the user explicitly scoped the work",
    ),
    (
        re.compile(r"\bnot introduced by\b", re.IGNORECASE),
        "not-introduced-by",
        "Irrelevant who introduced it — if you see it, own it",
    ),
    (
        re.compile(r"\balready (?:broken|failing|existed)\b", re.IGNORECASE),
        "already-broken",
        "Broken is broken — fix it or file it with a plan",
    ),
    (
        re.compile(r"\bwas already\b.{0,20}\bbefore\b", re.IGNORECASE),
        "was-already-before",
        "Acknowledge and act, do not deflect responsibility",
    ),
]


class OutputQualityChecker:
    """Checks LLM output for dismissive language patterns."""

    def __init__(self, patterns: list = None):
        self._patterns = patterns or _DISMISSAL_PATTERNS

    def check(self, text: str) -> list[QualityWarning]:
        """Scan text for dismissive patterns. Returns list of warnings."""
        warnings = []
        for regex, pattern_name, suggestion in self._patterns:
            match = regex.search(text)
            if match:
                warnings.append(QualityWarning(
                    pattern=pattern_name,
                    matched_text=match.group(0),
                    suggestion=suggestion,
                ))
        return warnings

    def has_dismissals(self, text: str) -> bool:
        """Quick check: does this text contain any dismissive patterns?"""
        return any(regex.search(text) for regex, _, _ in self._patterns)

    def rewrite_suggestion(self, text: str) -> str:
        """Return text with inline suggestions for each dismissal found."""
        result = text
        for regex, _, suggestion in self._patterns:
            match = regex.search(result)
            if match:
                result = result[:match.end()] + f" [FIX: {suggestion}]" + result[match.end():]
        return result
