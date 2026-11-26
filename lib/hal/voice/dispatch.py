"""HAL Voice — ladder dispatch classifier.

Per plans/voice-tier3-cloud-fallover.md §4 W4.

When ``voice.speak`` invokes a ladder, each provider in the ladder may
succeed, raise a retryable exception (ladder falls through), or raise a
non-retryable one (ladder halts and re-raises). ``is_retryable`` is the
single source of truth for that classification so provider-specific
knowledge doesn't leak into ``voice.py``.

Policy:
- The four named ``VoiceError`` subclasses (``AuthFailure``,
  ``RateLimited``, ``QuotaExceeded``, ``NetworkError``) are ALWAYS
  retryable. Next provider might have different auth, capacity, quota,
  or network path.
- Bare ``VoiceError`` whose message matches a caller-error hint
  (``text too long`` / ``unsupported voice`` / ``not in allowed set`` /
  ``profile tier``) is NOT retryable. Every provider will reject the
  same input, burning budget on retries is waste.
- Other bare ``VoiceError`` (malformed-JSON, empty-output, schema-drift)
  IS retryable — those are provider-side malfunctions that a sibling
  provider may handle correctly.
- Non-``VoiceError`` exceptions (``TypeError``, ``KeyError``, arbitrary
  bugs) are NEVER retryable — they indicate a programming error that
  will recur across every provider in the ladder.

This is the plain-English contract V>>'s ladder expects:
    "Fall over on everything the server/network could fix by handing
    to a different provider. Don't fall over on caller mistakes."
"""
from __future__ import annotations

from .providers._errors import (
    AuthFailure,
    NetworkError,
    QuotaExceeded,
    RateLimited,
)
from .types import VoiceError


_RETRYABLE_SUBCLASSES: tuple[type[BaseException], ...] = (
    AuthFailure,
    NetworkError,
    QuotaExceeded,
    RateLimited,
)

NON_RETRYABLE_HINTS: tuple[str, ...] = (
    "text too long",
    "unsupported voice",
    "not in allowed set",
    "profile tier",
)


def is_retryable(exc: BaseException) -> bool:
    """Return True if the ladder should try the next provider.

    Safe on any exception type; never raises. Used inside
    ``voice._speak_with_ladder`` as the sole fall-through decision
    point.
    """
    if isinstance(exc, _RETRYABLE_SUBCLASSES):
        return True
    if isinstance(exc, VoiceError):
        message = str(exc).lower()
        for hint in NON_RETRYABLE_HINTS:
            if hint in message:
                return False
        # Bare VoiceError without a caller-error hint is treated as
        # provider-side malfunction — a sibling provider may succeed.
        return True
    return False


__all__ = ["NON_RETRYABLE_HINTS", "is_retryable"]
