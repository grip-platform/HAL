"""HAL Code Reviewer — wires SafetyScanner + Verifier into the RSI pipeline.

Mechanical review (safety scan + syntax check) always runs. Semantic review
(LLM analysis) runs only when an llm_fn is supplied. Both gates fail-safe:
mechanical failures surface as errors; LLM failures degrade gracefully to [].

Usage:
    from lib.hal.code_reviewer import CodeReviewer, ReviewResult

    reviewer = CodeReviewer()                       # mechanical only
    reviewer = CodeReviewer(llm_fn=registry.chat)  # + semantic gate

    result = reviewer.review(generated)
    if result.severity == "block":
        print(result.summary)   # "BLOCKED: 2 error(s)"
"""
import json
import logging
import os
import tempfile
from dataclasses import dataclass

from lib.hal.safety_scanner import SafetyScanner
from lib.hal.verify import Verifier

_log = logging.getLogger(__name__)

# Maps SafetyScanner severity -> ReviewIssue severity
_SEVERITY_MAP: dict[str, str] = {
    "critical": "error",
    "high": "error",
    "medium": "warning",
    "low": "info",
}


@dataclass(frozen=True)
class ReviewIssue:
    """A single issue found during code review."""

    gate: str       # "safety" | "syntax" | "semantic"
    severity: str   # "info" | "warning" | "error"
    line: int
    message: str


@dataclass(frozen=True)
class ReviewResult:
    """Aggregate result of all review gates."""

    passed: bool
    issues: tuple[ReviewIssue, ...]
    severity: str   # "pass" | "warn" | "block"
    summary: str


def _severity_from_issues(issues: list[ReviewIssue]) -> str:
    """Compute overall severity: block > warn > pass."""
    if any(i.severity == "error" for i in issues):
        return "block"
    if any(i.severity == "warning" for i in issues):
        return "warn"
    return "pass"


class CodeReviewer:
    """Multi-gate reviewer for RSI-generated Python modules."""

    def __init__(self, llm_fn=None):
        self._llm_fn = llm_fn
        self._scanner = SafetyScanner()
        self._verifier = Verifier()

    def review(self, generated: "GeneratedCode") -> ReviewResult:
        """Run all gates against generated code. Returns ReviewResult."""
        issues: list[ReviewIssue] = []
        issues.extend(self._safety_gate(generated))
        issues.extend(self._syntax_gate(generated))
        if self._llm_fn is not None:
            issues.extend(self._semantic_gate(generated))

        sev = _severity_from_issues(issues)
        return ReviewResult(
            passed=(sev != "block"),
            issues=tuple(issues),
            severity=sev,
            summary=_build_summary(issues, sev),
        )

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------

    def _safety_gate(self, generated: "GeneratedCode") -> list[ReviewIssue]:
        """Run SafetyScanner on module_source. O(lines × patterns)."""
        source = generated.module_source or ""
        if not source:
            return []
        raw_issues = self._scanner.scan_code(source, language="python")
        return [
            ReviewIssue(
                gate="safety",
                severity=_SEVERITY_MAP.get(issue.severity, "warning"),
                line=issue.line,
                message=f"[{issue.category}] {issue.description}",
            )
            for issue in raw_issues
        ]

    def _syntax_gate(self, generated: "GeneratedCode") -> list[ReviewIssue]:
        """Write source to a temp file and run Verifier.check().

        Verifier requires the file to exist on disk — safe here because
        review() is called after _write_and_test() in the build pipeline.
        We still use a temp file to keep review idempotent and side-effect-free.
        """
        source = generated.module_source or ""
        if not source:
            return []
        fd, tmp_path = tempfile.mkstemp(suffix=".py")
        try:
            os.write(fd, source.encode())
            os.close(fd)
            result = self._verifier.check(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        if result.passed:
            return []
        return [ReviewIssue(gate="syntax", severity="error", line=0, message=result.errors)]

    def _semantic_gate(self, generated: "GeneratedCode") -> list[ReviewIssue]:
        """Optional LLM semantic review. Non-fatal: exceptions return []."""
        source = generated.module_source or ""
        spec_text = generated.spec.description if generated.spec else ""
        prompt = (
            f"Review this Python code for logical errors, spec compliance, and edge cases.\n"
            f"Spec: {spec_text}\n\nCode:\n{source}\n\n"
            f"Return a JSON array of issues: "
            f'[{{"line": N, "severity": "error"|"warning"|"info", "message": "..."}}]. '
            f"Return [] if no issues. JSON only, no markdown."
        )
        try:
            raw = self._llm_fn(prompt)
            issues_data = json.loads(raw.strip())
            if not isinstance(issues_data, list):
                return []
            return [
                ReviewIssue(
                    gate="semantic",
                    severity=item.get("severity", "info"),
                    line=item.get("line", 0),
                    message=item.get("message", ""),
                )
                for item in issues_data
                if isinstance(item, dict) and item.get("message")
            ]
        except Exception:
            _log.debug("Semantic gate LLM call failed — degrading gracefully")
            return []


def _build_summary(issues: list[ReviewIssue], severity: str) -> str:
    """Module-level pure function — keeps CodeReviewer at ≤7 public methods."""
    if not issues:
        return "All gates passed."
    counts: dict[str, int] = {}
    for i in issues:
        counts[i.severity] = counts.get(i.severity, 0) + 1
    parts = [f"{v} {k}(s)" for k, v in counts.items() if v]
    verdict = {"block": "BLOCKED", "warn": "WARNED", "pass": "PASSED"}.get(severity, severity)
    return f"{verdict}: {', '.join(parts)}"
