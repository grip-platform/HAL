"""auth_signal -- what a 401 or 403 from api.anthropic.com says about its account.

The HAL wire carries Claude Code's turns across a pool of the operator's own
Anthropic accounts (``lib/hal/account_switch.py``). Anthropic says it may enforce
its terms "without prior notice", and its Usage Policy names continuing activity
across accounts after an enforcement action. So an answer that may BE an
enforcement action has to stop the switching, never set it off. This module reads
one 401 or 403 and says which of three things it is:

  enforcement  its message names a revocation, a suspension, a ban or a
               violation (:data:`ENFORCEMENT_WORDS`). The wire stops ALL account
               switching and tells the operator.
  expiry       its message says the credential expired (:data:`EXPIRY_WORDS`) and
               names nothing above. A 401 like this keeps the dead-token switch.
  unknown      anything else, a body that is not JSON included.

What is documented, and what is a guess. Anthropic's errors page
(platform.claude.com/docs/en/api/errors, read 2026-10-08) documents the error
TYPES: "401 - authentication_error: There's an issue with your API key (for
example, it's malformed, revoked, or expired ...)" and "403 - permission_error:
Your API key does not have permission to use the specified resource". It does not
document the MESSAGE an account is sent when it is revoked, disabled or
suspended, and HAL's tests hold no real OAuth 401 or 403 body (only "invalid
x-api-key" and "invalid bearer"). So both word lists are a guess, to be corrected
from real bodies as they are seen. The wire relays such an answer to Claude Code
unchanged, so its message reaches the operator's screen.

Two consequences of that guess, stated rather than left to be found. A message
that names both ("expired or revoked") reads as enforcement: a stop that was not
needed costs one command to undo, and a missed one is the failure this module
exists to prevent. And "revoked" also covers a token the operator signed out of
himself, which is no enforcement action; that stops switching too, until he
turns it back on.

Custody. Only the JSON ``error.type`` and ``error.message`` are read, and the
message is never kept. A caller gets the error type when it is one the errors
page documents (else ``"other"``, or ``"none"`` when there is none) and the word
that matched, both from the closed lists below, so neither can carry text from
the body. Pure: no I/O, no logging.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from lib.hal.wire_classify import decoded_body

ENFORCEMENT = "enforcement"
EXPIRY = "expiry"
UNKNOWN = "unknown"

#: The statuses this module judges.
STATUSES = frozenset({401, 403})

#: Words that, in a 401's or a 403's message, name an enforcement action. A guess
#: (see the module docstring): correct it from real bodies.
ENFORCEMENT_WORDS = ("revoked", "revocation", "disabled", "suspended", "suspension",
                     "banned", "terminated", "termination", "violation", "violations")

#: Words that name a credential that has expired, which a re-login fixes.
EXPIRY_WORDS = ("expired",)

#: The error types Anthropic's errors page documents. Any other reads as "other".
ERROR_TYPES = frozenset({
    "invalid_request_error", "authentication_error", "billing_error",
    "permission_error", "not_found_error", "request_too_large", "rate_limit_error",
    "api_error", "timeout_error", "overloaded_error"})

_WORDS = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class AuthSignal:
    """One 401 or 403, judged. Every field comes from a closed list."""

    status: int
    kind: str
    error_type: str
    word: Optional[str] = None

    @property
    def enforcement(self) -> bool:
        return self.kind == ENFORCEMENT

    def fields(self) -> dict:
        """The judgement as an audit row's fields; never any text from the body."""
        return {"status": self.status, "signal": self.kind,
                "error_type": self.error_type, "word": self.word}


def classify(status: int, body: Optional[bytes], headers: Any = None) -> AuthSignal:
    """Judge one 401 or 403 from its body; ``headers`` give its Content-Encoding.

    Total: never raises. A body that cannot be decoded or parsed, or whose
    ``error.message`` is not a string, judges as ``unknown``.
    """
    error = _error_object(decoded_body(headers, body))
    message = error.get("message")
    words = set(_WORDS.findall(message.lower())) if isinstance(message, str) else set()
    error_type = _error_type(error.get("type"))
    word = _first(ENFORCEMENT_WORDS, words)
    if word is not None:
        return AuthSignal(status, ENFORCEMENT, error_type, word)
    word = _first(EXPIRY_WORDS, words)
    return AuthSignal(status, EXPIRY if word is not None else UNKNOWN, error_type, word)


def _error_object(raw: Optional[bytes]) -> dict:
    try:
        payload = json.loads(raw) if raw else None
    except (ValueError, RecursionError):
        return {}
    error = payload.get("error") if isinstance(payload, dict) else None
    return error if isinstance(error, dict) else {}


def _error_type(value: Any) -> str:
    if value is None:
        return "none"
    return value if isinstance(value, str) and value in ERROR_TYPES else "other"


def _first(vocabulary: tuple, words: set) -> Optional[str]:
    return next((word for word in vocabulary if word in words), None)


__all__ = ["AuthSignal", "ENFORCEMENT", "ENFORCEMENT_WORDS", "ERROR_TYPES", "EXPIRY",
           "EXPIRY_WORDS", "STATUSES", "UNKNOWN", "classify"]
