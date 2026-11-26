"""Circuit breaker for the HAL REPL tool loop.

Issue #200 surfaced a loop where a broken tool schema produced a 400 on
every turn and the REPL kept re-trying the same schema, burning tokens
and user patience. The root fix is in the schema wrapper
(:func:`~lib.hal.tool_adapter.to_openai_tools`); this module is the
defence-in-depth guard: if the same failure class fires ``threshold``
times in a row, tools are auto-disabled and the user is told — the REPL
falls back to plain-text mode so subsequent turns are productive even
under a provider-side regression.

Design shape
------------

The breaker is **stateful but intentionally minimal**. A single instance
tracks consecutive failures of a named category (``"tool_schema_400"``,
``"tool_dispatch_error"``, …). Any successful turn resets the counter
for that category — the breaker is consecutive-failure-sensitive, not
total-failure-sensitive. A handful of non-consecutive 400s across a
session does NOT trip the breaker; a real, persistent, systemic fault
(exactly the Issue #200 scenario) does.

The threshold defaults to 2 because the reproducing user saw the error
on "every subsequent user turn" — one 400 could still be a transient
network failure or an upstream deploy; two in a row is a pattern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CircuitBreaker:
    """Consecutive-failure counter with per-category threshold tripping.

    Usage from the tool-loop entry point::

        breaker = CircuitBreaker(threshold=2)
        try:
            send_with_tools_to_cli(...)
            breaker.record_success("tool_schema_400")
        except Exception as exc:
            if _is_tool_schema_400(exc):
                if breaker.record_failure("tool_schema_400"):
                    _disable_tools_with_reason(exc)
            else:
                raise

    The breaker is category-agnostic — callers choose category names
    that map 1:1 to the upstream error classes they care about.
    ``record_failure()`` returns ``True`` when the threshold is crossed
    so callers can use it as an if-branch directly, rather than a
    two-step ``increment + query`` pattern that invites TOCTOU bugs.
    """

    threshold: int = 2
    _counters: dict[str, int] = field(default_factory=dict)
    _tripped: dict[str, bool] = field(default_factory=dict)

    def record_failure(self, category: str) -> bool:
        """Record a failure. Return ``True`` when the threshold is newly crossed.

        Returns ``False`` on second-and-subsequent failures after the trip
        so callers don't re-fire the auto-disable message once tools are
        already disabled. Reset via :meth:`reset` (e.g. when the user
        explicitly re-enables tools with ``/tools on``).
        """
        current = self._counters.get(category, 0) + 1
        self._counters[category] = current
        already_tripped = self._tripped.get(category, False)
        crossed_now = current >= self.threshold and not already_tripped
        if crossed_now:
            self._tripped[category] = True
        return crossed_now

    def record_success(self, category: str) -> None:
        """Clear the consecutive-failure counter for ``category``.

        A success is evidence the system is healthy — past failures were
        transient. Does NOT clear the ``_tripped`` flag; once the breaker
        has surfaced the auto-disable decision, only :meth:`reset` (called
        when the user re-enables tools) acknowledges that surface.
        """
        self._counters[category] = 0

    def is_tripped(self, category: str) -> bool:
        """Return True when the category breaker has already tripped."""
        return self._tripped.get(category, False)

    def reset(self, category: Optional[str] = None) -> None:
        """Clear state for ``category`` (or all categories when ``None``).

        Called when the user explicitly re-enables tools — the breaker
        gives them a fresh chance to succeed. Also used by tests to
        isolate test cases.
        """
        if category is None:
            self._counters.clear()
            self._tripped.clear()
            return
        self._counters.pop(category, None)
        self._tripped.pop(category, None)


# ──────────────────────────────────────────────────────────────────────
# Classification helpers — decide which failure category an exception
# belongs to. Kept outside the breaker class so the breaker stays
# category-agnostic and tests can exercise classification separately.
# ──────────────────────────────────────────────────────────────────────

# Substrings that, when present in the body of a 400, indicate the tool
# schema is malformed. Conservative: we don't trip on generic 400s (a
# bad user message can also produce 400). Only obvious schema-rejection
# error bodies are treated as schema failures. Every substring listed
# here MUST appear in a real provider error — no speculative strings.
_SCHEMA_400_MARKERS: tuple[str, ...] = (
    "tools.0.type",         # Anthropic / Groq: "property 'type' is missing"
    "tools[0].type",        # OpenAI variant
    "tool_choice",
    "invalid_request_error",  # Anthropic error family
    "invalid tools",
    "tool schema",
)


def is_tool_schema_400(exc: BaseException) -> bool:
    """Return True when ``exc`` looks like a tool-schema rejection.

    Duck-typed on ``status`` + ``body`` attributes so we don't import
    :class:`AnthropicChatError` at module load time — tests that stub the
    chat_fn can raise plain exceptions with matching attributes and this
    classifier will still fire. Mirrors the ``ValueError`` / ``HTTPError``
    convention.

    A 400 without any of the schema markers is NOT classified as a
    schema failure — it's surfaced to the user as-is. Only unambiguous
    schema rejections trip the breaker.
    """
    status = getattr(exc, "status", None)
    body = getattr(exc, "body", "") or ""
    if status != 400:
        return False
    if not isinstance(body, str):
        return False
    body_lower = body.lower()
    return any(marker.lower() in body_lower for marker in _SCHEMA_400_MARKERS)


__all__ = ["CircuitBreaker", "is_tool_schema_400"]
