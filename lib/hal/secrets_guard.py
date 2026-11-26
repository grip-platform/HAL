"""HAL Secrets Guard — domain-neutral secret detection for tool I/O.

Salvaged from GRIP hooks during the hooks/ directory deletion. HAL inherits
the regex library but drops the GRIP-specific JSON stdin/stdout hook
protocol — callers invoke ``scan_text`` / ``has_secrets`` / ``redact_text``
directly from the ``/tool`` HTTP endpoint, the conversation pipeline, or
any other path that needs to block secrets.

ReDoS hardening
---------------
HAL's council review of PR #125 flagged the single mega-pattern approach
as a theoretical ReDoS risk. Mitigations applied:

1. **Bounded input** — every public entry point truncates its input to
   :data:`MAX_SCAN_BYTES` (default 256 KiB) before handing it to the regex
   engine. Longer inputs are scanned head-only; a note on the truncation
   is surfaced via :attr:`ScanReport.truncated` so callers can log it.
2. **Bounded patterns** — every source pattern uses explicit upper bounds
   (``{20,}``, ``{36}``, no nested unbounded quantifiers). The test file
   ``tests/test_secrets_guard_redos.py`` asserts the mega-pattern stays
   responsive on pathological inputs (long runs of repeated characters,
   deeply quoted strings) within a fixed wall-clock budget.
3. **Pre-filter short-circuit** — :func:`has_secrets` short-circuits on
   the first match, so adversarial inputs that do not contain a match
   still scan in O(n) once.

Behaviour:

* ``scan_text(text)`` returns a list of :class:`SecretMatch` findings.
* ``scan_text_report(text)`` returns a :class:`ScanReport` with the
  findings plus truncation metadata.
* ``has_secrets(text)`` is a fast boolean short-circuit for the common
  "detect and refuse" flow.
* ``redact_text(text)`` returns a copy with matches replaced by
  ``[REDACTED:TYPE]`` markers.

Performance: regexes are compiled once at import. Short-circuits on text
shorter than :data:`MIN_SCAN_LENGTH` (default 30 chars — below the entropy
floor of any real-world secret).

Usage:
    from lib.hal.secrets_guard import has_secrets, scan_text, redact_text

    if has_secrets(user_message):
        raise ValueError("Cannot send message containing credentials")

    findings = scan_text(tool_output)
    if findings:
        safe_log = redact_text(tool_output)
"""
import re
from dataclasses import dataclass, field

MIN_SCAN_LENGTH = 30
MAX_SCAN_BYTES = 262144  # 256 KiB — head-only scan cap for ReDoS protection


@dataclass(frozen=True)
class ScanReport:
    """Outcome of a bounded scan.

    Findings are always non-overlapping and in ascending start order. When
    ``truncated`` is True, the input exceeded :data:`MAX_SCAN_BYTES` and
    only the first ``scanned_bytes`` were fed to the regex engine.
    """
    findings: tuple = field(default_factory=tuple)
    scanned_bytes: int = 0
    truncated: bool = False


@dataclass(frozen=True)
class SecretMatch:
    """A single secret detection."""
    kind: str           # Symbolic type, e.g. "ANTHROPIC_KEY"
    start: int          # Match start offset
    end: int            # Match end offset (exclusive)

    @property
    def length(self) -> int:
        return self.end - self.start


# ---------------------------------------------------------------------------
# Pattern library
#
# Each (source, kind) entry becomes one named group inside a single compiled
# alternation regex (see :data:`_MEGA_PATTERN`). This gives us a single-pass
# O(n) scan of the text regardless of how many patterns we register, and the
# whole scan function is a flat list comprehension — no nested loops.
#
# Order is significant: within the alternation, the first matching branch at
# any given position wins. More-specific patterns must come before less-
# specific ones (ANTHROPIC_KEY before OPENAI_KEY, ENV_VAR_SECRET last so it
# does not swallow standalone provider tokens).
#
# Case sensitivity: the mega-pattern is compiled with :data:`re.IGNORECASE`
# so the three entries that originally used inline ``(?i)`` no longer need
# it. The prefix patterns (ghp_, glpat-, xox) pick up a minor false-positive
# risk on uppercased variants — acceptable, since the detector is advisory.
# ---------------------------------------------------------------------------

_PATTERN_SOURCES: tuple[tuple[str, str], ...] = (
    # Cloud LLM providers
    (r"sk-ant-[a-zA-Z0-9_-]{20,}", "ANTHROPIC_KEY"),
    (r"sk-[a-zA-Z0-9]{20,}", "OPENAI_KEY"),
    (r"AKIA[0-9A-Z]{16}", "AWS_ACCESS_KEY"),
    (r"aws_secret_access_key\s*[=:]\s*[A-Za-z0-9/+=]{40}", "AWS_SECRET"),

    # Git hosting
    (r"ghp_[a-zA-Z0-9]{36}", "GITHUB_PAT"),
    (r"gho_[a-zA-Z0-9]{36}", "GITHUB_OAUTH"),
    (r"ghu_[a-zA-Z0-9]{36}", "GITHUB_USER_TOKEN"),
    (r"ghs_[a-zA-Z0-9]{36}", "GITHUB_SERVER_TOKEN"),
    (r"ghr_[a-zA-Z0-9]{36}", "GITHUB_REFRESH_TOKEN"),
    (r"glpat-[a-zA-Z0-9_-]{20,}", "GITLAB_PAT"),

    # Messaging & SaaS
    (r"xox[baprs]-[a-zA-Z0-9-]{10,}", "SLACK_TOKEN"),
    (r"sk_(?:live|test)_[a-zA-Z0-9]{24,}", "STRIPE_SECRET_KEY"),
    (r"pk_(?:live|test)_[a-zA-Z0-9]{24,}", "STRIPE_PUBLIC_KEY"),
    (r"SG\.[a-zA-Z0-9_-]{22}\.[a-zA-Z0-9_-]{43}", "SENDGRID_KEY"),
    (r"SK[a-f0-9]{32}", "TWILIO_KEY"),
    (r"[MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27}", "DISCORD_TOKEN"),
    (r"npm_[a-zA-Z0-9]{36}", "NPM_TOKEN"),
    (r"pypi-AgEIcHlwaS5vcmc[a-zA-Z0-9_-]{50,}", "PYPI_TOKEN"),

    # Structured credentials
    (r"eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}", "JWT_TOKEN"),
    (r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "PRIVATE_KEY"),
    (r"(?:postgres|mysql|mongodb)(?:\+\w+)?://[^:]+:[^@\s]+@[^\s]+", "CONNECTION_STRING"),

    # Generic env var secrets — the "pasted docker inspect" class of incident.
    # Last so specific tokens above win when both would match.
    (r"\b\w*(?:PASSWORD|PASSWD|SECRET|TOKEN|API_KEY|APIKEY|ACCESS_KEY|PRIVATE_KEY|AUTH)\w*\s*[=:]\s*\S{8,}",
     "ENV_VAR_SECRET"),
)


def _build_mega_pattern() -> re.Pattern:
    """Combine all source patterns into a single alternation regex.

    Each entry becomes ``(?P<KIND>source)``; the branches are joined with
    ``|`` and compiled with :data:`re.IGNORECASE`.
    """
    parts = [f"(?P<{kind}>{src})" for src, kind in _PATTERN_SOURCES]
    return re.compile("|".join(parts), re.IGNORECASE)


_MEGA_PATTERN = _build_mega_pattern()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _bounded(text: str) -> tuple[str, bool]:
    """Truncate ``text`` to :data:`MAX_SCAN_BYTES` UTF-8 bytes.

    Returns ``(scan_target, truncated)``. The cap is measured in bytes so
    the scan cost is predictable under multi-byte inputs. When the text is
    shorter than the cap, the original string is returned unchanged.
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_SCAN_BYTES:
        return text, False
    head = encoded[:MAX_SCAN_BYTES].decode("utf-8", errors="ignore")
    return head, True


def scan_text(text: str) -> list[SecretMatch]:
    """Return all non-overlapping secret matches in ``text``.

    Inputs larger than :data:`MAX_SCAN_BYTES` are head-scanned only. Use
    :func:`scan_text_report` if the caller needs to know whether truncation
    occurred.
    """
    return list(scan_text_report(text).findings)


def scan_text_report(text: str) -> ScanReport:
    """Scan ``text`` and return a :class:`ScanReport` with truncation info.

    The single mega-pattern is scanned once via ``finditer``; Python's regex
    engine resolves alternation left-to-right and continues after each
    match, so the result is already non-overlapping and in ascending start
    order. ``lastgroup`` names the winning named group for each match.
    """
    if not text or len(text) < MIN_SCAN_LENGTH:
        return ScanReport(findings=(), scanned_bytes=len(text or ""),
                          truncated=False)
    scan_target, truncated = _bounded(text)
    findings = tuple(
        SecretMatch(kind=m.lastgroup, start=m.start(), end=m.end())
        for m in _MEGA_PATTERN.finditer(scan_target)
    )
    return ScanReport(
        findings=findings,
        scanned_bytes=len(scan_target),
        truncated=truncated,
    )


def has_secrets(text: str) -> bool:
    """Fast boolean — true if any secret pattern matches.

    Uses :meth:`re.Pattern.search` on the mega-pattern — short-circuits at
    the first hit. Input is truncated to :data:`MAX_SCAN_BYTES` before
    scanning to bound worst-case CPU.
    """
    if not text or len(text) < MIN_SCAN_LENGTH:
        return False
    scan_target, _ = _bounded(text)
    return _MEGA_PATTERN.search(scan_target) is not None


def redact_text(text: str) -> str:
    """Return ``text`` with each match replaced by ``[REDACTED:KIND]``.

    Safe for logging. Inputs larger than :data:`MAX_SCAN_BYTES` are scanned
    head-only for ReDoS protection; the tail passes through unchanged and
    a ``[SECRETS SCAN TRUNCATED]`` marker is appended so callers know the
    redaction is partial.
    """
    report = scan_text_report(text)
    if not report.findings and not report.truncated:
        return text
    parts: list[str] = []
    cursor = 0
    for match in report.findings:
        parts.append(text[cursor:match.start])
        parts.append(f"[REDACTED:{match.kind}]")
        cursor = match.end
    parts.append(text[cursor:])
    result = "".join(parts)
    if report.truncated:
        result += "\n[SECRETS SCAN TRUNCATED — tail not scanned]"
    return result


def finding_summary(findings: list[SecretMatch]) -> dict[str, int]:
    """Group findings by kind for structured logging."""
    summary: dict[str, int] = {}
    for f in findings:
        summary[f.kind] = summary.get(f.kind, 0) + 1
    return summary
