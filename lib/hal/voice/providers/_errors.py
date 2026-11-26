"""HAL Voice — shared provider exception subclasses.

Three providers now raise these classes (W1 Piper, W2 Google Cloud TTS,
W3 Gemini 2.5 audio) and the W4 ladder dispatcher's ``is_retryable``
helper classifies off them. Defining them once prevents the type-
confusion trap where each provider carries its own ``AuthFailure`` and
``isinstance(exc, SomeProvider.AuthFailure)`` fails for a different
provider's identically-named class.

All four inherit from ``VoiceError`` so the outer ``voice.speak`` catch
still routes every provider failure through the same audit + fallover
path. The subclasses only matter inside the ladder — they let
``is_retryable`` skip provider-level failures (AuthFailure, Quota) while
retrying transient failures (RateLimited, NetworkError) at the next rung.
"""
from __future__ import annotations

from ..types import VoiceError


class AuthFailure(VoiceError):
    """Missing API key or HTTP 401/403.

    W4 ladder's ``is_retryable`` classifies this as NON-retryable at the
    same provider — the operator needs to fix the key, not retry.
    """


class RateLimited(VoiceError):
    """HTTP 429 from the server OR local RPM pre-check caught the limit.

    Distinguishable from ``QuotaExceeded`` in that rate limits are
    per-minute / per-second constraints that relax with time; quota
    exhaustion is a hard monthly ceiling.
    """


class QuotaExceeded(VoiceError):
    """Local char / call counter caught the free-tier limit BEFORE the
    API call. Never originates from the server.
    """


class NetworkError(VoiceError):
    """URLError, TimeoutError, or other transport-layer failure."""


__all__ = [
    "AuthFailure",
    "NetworkError",
    "QuotaExceeded",
    "RateLimited",
]
