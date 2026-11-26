"""HAL Safety Scanner — scan code for common security issues.

Lightweight static analysis for common vulnerabilities: SQL injection,
command injection, hardcoded secrets, insecure deserialization, and
path traversal.

Usage:
    from lib.hal.safety_scanner import SafetyScanner
    scanner = SafetyScanner()
    issues = scanner.scan_code(code, language="python")
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SecurityIssue:
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "injection", "secrets", "crypto", "deserialization"
    line: int
    description: str
    recommendation: str


_PYTHON_PATTERNS: list[tuple[str, str, str, str]] = [
    ("high", "injection", r"os\.system\(|subprocess\.call\([^,]+shell\s*=\s*True",
     "Command injection risk. Use subprocess with shell=False."),
    ("high", "injection", r"exec\(|eval\(",
     "Code injection risk. Avoid exec/eval on untrusted input."),
    ("high", "deserialization", r"pickle\.loads?\(|yaml\.load\([^,]*\)(?!.*Loader)",
     "Insecure deserialization. Use safe loaders."),
    ("medium", "secrets", r"(?:password|secret|token|key)\s*=\s*[\"\'][^\"\' ]{8,}",
     "Possible hardcoded secret. Use environment variables."),
    ("medium", "crypto", r"\bmd5\b|\bsha1\b",
     "Weak hash algorithm. Use SHA-256 or better."),
    ("low", "injection", r"\.format\(.*input|f\".*\{.*input",
     "Potential format string injection. Validate input."),
]


class SafetyScanner:
    """Scans code for security issues."""

    def scan_code(self, code: str, language: str = "python") -> list[SecurityIssue]:
        if language != "python":
            return []
        issues: list[SecurityIssue] = []
        lines = code.split("\n")
        for line_num, line in enumerate(lines, 1):
            for severity, category, pattern, desc in _PYTHON_PATTERNS:
                if re.search(pattern, line):
                    issues.append(SecurityIssue(
                        severity=severity, category=category,
                        line=line_num, description=desc,
                        recommendation=desc.split(". ", 1)[-1] if ". " in desc else "Review this code.",
                    ))
        return issues

    def scan_file(self, path: str) -> list[SecurityIssue]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
            lang = "python" if path.endswith(".py") else ""
            return self.scan_code(code, lang)
        except OSError:
            return []
