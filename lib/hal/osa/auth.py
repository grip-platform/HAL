"""Capability-token authorisation for tier-2+ osa scripts.

The council RETHINK verdict on α-3 flagged the original ``confirmed: bool``
gate as unauthenticated — any caller could pass ``confirmed=True`` to bypass
tier-2 restrictions. α-3b replaces that with HMAC-signed, single-use,
time-bounded capability tokens.

Design choices (deliberate KISS):

*   **HMAC-SHA256 over a compact payload**. No asymmetric crypto, no
    certificates. The secret lives in ``HAL_OSA_CAPABILITY_SECRET``
    (environment variable, 32 bytes minimum after hex-decode).
*   **60-second default TTL**. A token is a single-use permission slip,
    not a session. Callers mint one per dispatch.
*   **Single-use enforced in-process**. Nonces go into a set; a second
    verify of the same nonce raises :class:`TokenReused`. A future β-2
    upgrade can promote this to a cross-process store if multi-worker
    harnesses need it, without changing the public API.
*   **No silent-insecure fallback**. If ``HAL_OSA_CAPABILITY_SECRET`` is
    unset when a tier-2+ script is dispatched, we raise — never auto-mint
    a placeholder token.

Public API::

    from lib.hal.osa.auth import (
        CapabilityToken, mint_token, verify_token, default_secret,
        TokenInvalid, TokenExpired, TokenReused,
    )

    token = mint_token("notification", secret=default_secret())
    osa("notification", {...}, capability_token=token)
"""
from __future__ import annotations

import hmac
import os
import secrets
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Set

__all__ = [
    "CapabilityToken",
    "mint_token",
    "verify_token",
    "default_secret",
    "TokenInvalid",
    "TokenExpired",
    "TokenReused",
    "MissingSecret",
]


_SECRET_ENV = "HAL_OSA_CAPABILITY_SECRET"
_DEFAULT_TTL_S = 60
_USED_NONCES: Set[str] = set()


class TokenInvalid(ValueError):
    """HMAC verification failed or token fields do not match the request."""


class TokenExpired(ValueError):
    """The token's ``expires_at`` is in the past."""


class TokenReused(ValueError):
    """The token's nonce has already been redeemed in this process."""


class MissingSecret(RuntimeError):
    """``HAL_OSA_CAPABILITY_SECRET`` is unset and cannot be derived."""


@dataclass(frozen=True)
class CapabilityToken:
    """HMAC-signed permission slip for one tier-2+ dispatch.

    Fields are JSON-serialisable so a token can cross a subprocess or MCP
    boundary without pickling. ``mac`` is the hex-encoded HMAC-SHA256 over
    the canonical string ``f"{script_name}|{issued_at}|{expires_at}|{nonce}"``.
    """
    script_name: str
    issued_at: float
    expires_at: float
    nonce: str
    mac: str


def default_secret() -> bytes:
    """Return the capability secret from the environment, or raise.

    The env var holds a hex-encoded secret. We enforce a 32-byte (64-hex-char)
    minimum so accidental short values raise loudly instead of producing
    a weak MAC silently.
    """
    raw = os.environ.get(_SECRET_ENV)
    if not raw:
        raise MissingSecret(
            f"{_SECRET_ENV} is not set; tier-2+ dispatch requires a secret"
        )
    try:
        decoded = bytes.fromhex(raw.strip())
    except ValueError as exc:
        raise MissingSecret(
            f"{_SECRET_ENV} must be hex-encoded"
        ) from exc
    if len(decoded) < 32:
        raise MissingSecret(
            f"{_SECRET_ENV} must decode to at least 32 bytes; got {len(decoded)}"
        )
    return decoded


def _canonical(script_name: str, issued: float, expires: float, nonce: str) -> bytes:
    """Canonical byte string for the HMAC input."""
    return f"{script_name}|{issued}|{expires}|{nonce}".encode("utf-8")


def _sign(secret: bytes, script_name: str, issued: float, expires: float, nonce: str) -> str:
    """Return the hex HMAC-SHA256 over the canonical payload."""
    mac = hmac.new(secret, _canonical(script_name, issued, expires, nonce), sha256)
    return mac.hexdigest()


def mint_token(
    script_name: str,
    *,
    secret: bytes,
    ttl_s: int = _DEFAULT_TTL_S,
    now: float = None,
) -> CapabilityToken:
    """Mint a fresh, single-use capability token for ``script_name``.

    Args:
        script_name: Name of the script this token authorises. The verifier
            checks that the dispatch target matches exactly — a token for
            ``notification`` cannot be redeemed for ``shutdown``.
        secret: HMAC key (at least 32 bytes). Normally :func:`default_secret`.
        ttl_s: Seconds until ``expires_at``. Default 60 — do not inflate this
            without a reason; a capability token is not a session.
        now: Injection point for tests. Defaults to :func:`time.time`.

    Returns:
        A :class:`CapabilityToken` ready for
        :func:`lib.hal.osa.osa` via the ``capability_token`` kwarg.
    """
    if ttl_s <= 0:
        raise ValueError("ttl_s must be positive")
    issued = time.time() if now is None else now
    expires = issued + ttl_s
    nonce = secrets.token_hex(16)
    return CapabilityToken(
        script_name=script_name,
        issued_at=issued,
        expires_at=expires,
        nonce=nonce,
        mac=_sign(secret, script_name, issued, expires, nonce),
    )


def verify_token(
    token: CapabilityToken,
    *,
    script_name: str,
    secret: bytes,
    now: float = None,
) -> None:
    """Raise if ``token`` cannot authorise a dispatch to ``script_name``.

    Enforces in order:

    1. Script-name match (token was minted for this exact script).
    2. HMAC validity under ``secret``.
    3. Expiry (``expires_at`` must be in the future).
    4. Single-use (nonce must not have been redeemed already).

    On success, the nonce is recorded as used and a second call raises
    :class:`TokenReused`. This matters most when a token is sent over the
    wire: an eavesdropper who captures it still cannot replay, because the
    legitimate verify has already consumed the nonce.
    """
    if token.script_name != script_name:
        raise TokenInvalid(
            f"token minted for {token.script_name!r}, "
            f"dispatch is for {script_name!r}"
        )

    expected = _sign(
        secret, token.script_name, token.issued_at, token.expires_at, token.nonce
    )
    if not hmac.compare_digest(expected, token.mac):
        raise TokenInvalid("HMAC mismatch")

    clock = time.time() if now is None else now
    if clock >= token.expires_at:
        raise TokenExpired(
            f"token expired {clock - token.expires_at:.1f}s ago"
        )

    if token.nonce in _USED_NONCES:
        raise TokenReused(f"nonce {token.nonce[:8]}… already redeemed")
    _USED_NONCES.add(token.nonce)


def _reset_for_tests() -> None:
    """Test-only hook to clear the used-nonce set between cases."""
    _USED_NONCES.clear()
