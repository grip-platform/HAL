"""osa dispatch — named-script public entry point.

Phase α-3a of the osascript control layer plan. Wraps the α-2 kernel
(:func:`lib.hal.osa.osa_raw`) with registry resolution, parameter
validation, audit logging, capability-token authorisation (α-3b), and
provenance tagging (α-3b)::

    from lib.hal.osa import osa, ProvenanceType, mint_token, default_secret

    # tier 1 — no token needed
    osa("battery_get", provenance=ProvenanceType.HUMAN_TYPED)

    # tier 2+ — token required
    token = mint_token("notification", secret=default_secret())
    osa(
        "notification",
        {"title": "GRIP", "body": "Build done"},
        capability_token=token,
        provenance=ProvenanceType.TOOL_CHAIN,
    )

α-3b changes relative to α-3a:

*   Accepts an optional :class:`CapabilityToken`. Tier-2+ dispatches
    **require** a valid token unless ``GRIP_OSA_UNSAFE=1`` is set in the
    caller's environment (the escape hatch is loud: a one-shot warning
    fires on import if any rc file would set the variable).
*   Accepts a :class:`ProvenanceType` (default :data:`ProvenanceType.LOCAL`).
    Recorded on the audit entry so downstream policy can branch on origin.
*   The legacy ``caller`` string is still accepted for back-compat with
    α-3a callers but is treated as a free-form annotation on the audit
    entry; new callers should pass ``provenance`` instead.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

from lib.hal.osa.audit import make_entry, write_audit
from lib.hal.osa.auth import (
    CapabilityToken,
    MissingSecret,
    default_secret,
    verify_token,
)
from lib.hal.osa.consent import (
    ConsentDeclined,
    get_consent_callback,
    redact_params_for_tier3,
)
from lib.hal.osa.rate_limit import (
    RateLimitExceeded,
    check_and_record,
)
from lib.hal.osa.escape import applescript_escape, js_escape
from lib.hal.osa.osa import OsaError, OsaResult
from lib.hal.osa.provenance import ProvenanceType
from lib.hal.osa.registry import (
    Script,
    get_script,
    validate_params,
)
from lib.hal.osa.shell_rc import UNSAFE_VAR_NAME, detect_unsafe_env_in_shell_rc
from lib.hal.osa.transport import (
    REMOTE_DISPATCH_ENV,
    LocalTransport,
    Transport,
)

__all__ = ["osa"]


_OSA_DIR = Path(__file__).parent
_UNSAFE_WARNING_EMITTED = False


def _warn_unsafe_rc_once() -> None:
    """Emit a :class:`UserWarning` the first time this module is imported
    if any shell rc file sets ``GRIP_OSA_UNSAFE``."""
    global _UNSAFE_WARNING_EMITTED
    if _UNSAFE_WARNING_EMITTED:
        return
    _UNSAFE_WARNING_EMITTED = True
    hits = detect_unsafe_env_in_shell_rc()
    if not hits:
        return
    paths = ", ".join(str(p) for p in hits)
    warnings.warn(
        f"{UNSAFE_VAR_NAME} is set in {paths}; tier-2+ osa dispatches will "
        f"bypass capability-token auth for every future shell. Remove it "
        f"unless you understand the risk.",
        UserWarning,
        stacklevel=2,
    )


_warn_unsafe_rc_once()


def _escape_for(language: str, value: str) -> str:
    """Return ``value`` quoted for the given osa language."""
    if language == "JavaScript":
        return js_escape(value)
    return applescript_escape(value)


def _resolve_body(script: Script) -> str:
    """Return the script source (inline or file-backed)."""
    if script.body is not None:
        return script.body
    if script.path is not None:
        return (_OSA_DIR / script.path).read_text(encoding="utf-8")
    raise ValueError(
        f"script {script.name!r} has neither 'body' nor 'path'"
    )


def _apply_params(body: str, params: Dict[str, object], language: str = "AppleScript") -> str:
    """Substitute ``{name}`` placeholders with language-appropriate escapes."""
    result = body
    for name, val in params.items():
        placeholder = "{" + name + "}"
        result = result.replace(placeholder, _escape_for(language, str(val)))
    return result


def _authorise(
    script: Script,
    token: Optional[CapabilityToken],
) -> None:
    """Enforce tier-2+ capability-token requirement.

    Tier-1 scripts are allowed without a token. Tier-2+ requires a valid,
    unexpired, unused token for this exact script name. The
    ``GRIP_OSA_UNSAFE=1`` environment variable bypasses the check — but
    the shell-rc scanner (see :func:`_warn_unsafe_rc_once`) already told
    the operator about that; we do not re-warn per dispatch.
    """
    if script.tier < 2:
        return
    if os.environ.get(UNSAFE_VAR_NAME) == "1":
        return
    if token is None:
        raise MissingSecret(
            f"tier-{script.tier} script {script.name!r} requires a "
            f"capability_token (mint one with mint_token())"
        )
    verify_token(token, script_name=script.name, secret=default_secret())


def _authorise_tier3_consent(
    script: Script,
    supplied: Dict[str, object],
    caller: str,
    effective_provenance: ProvenanceType,
    transport: Transport,
) -> None:
    """Fire the consent callback for tier-3 scripts. Yolo does NOT bypass.

    Invoked after tier-2 token / remote-consent has already cleared, so
    the callback runs only for dispatches that are otherwise permitted.
    Rejection writes an audit entry with ``provenance=user_rejected`` and
    redacted params, then raises :class:`ConsentDeclined`. Callback
    exceptions are coerced to rejection (never fail-open).
    """
    if script.tier < 3:
        return
    callback = get_consent_callback()
    try:
        approved = callback(script.name, dict(supplied))
    except Exception:  # noqa: BLE001 — fail-closed by design
        approved = False
    if approved:
        return
    write_audit(make_entry(
        script=script.name,
        tier=script.tier,
        ok=False,
        duration_s=0.0,
        params=redact_params_for_tier3(supplied),
        caller=caller,
        provenance=ProvenanceType.USER_REJECTED.value,
        host=transport.host,
        error="consent_declined",
    ))
    raise ConsentDeclined(
        f"tier-3 consent declined for {script.name!r}"
    )


def _authorise_with_transport(
    script: Script,
    token: Optional[CapabilityToken],
    transport: Transport,
    provenance: ProvenanceType,
) -> ProvenanceType:
    """Pick the right auth path for local vs remote and return the
    effective provenance.

    Local transports get the full α-3b capability-token enforcement.
    Remote transports require operator consent (``HAL_OSA_REMOTE_DISPATCH=1``)
    and auto-promote provenance to :data:`ProvenanceType.MCP_REMOTE` because
    the Tailscale SSH key, not a local HMAC token, is the cross-host auth
    anchor. Returning the (possibly promoted) provenance keeps the audit
    trail honest without requiring every caller to remember the rule.
    """
    if not transport.is_remote:
        _authorise(script, token)
        return provenance
    if os.environ.get(REMOTE_DISPATCH_ENV) != "1":
        raise MissingSecret(
            f"remote dispatch of {script.name!r} to {transport.host!r} requires "
            f"operator consent: set {REMOTE_DISPATCH_ENV}=1 after confirming "
            f"the remote host is a trusted tailnet peer."
        )
    return ProvenanceType.MCP_REMOTE


def _enforce_rate_limit(
    script: Script,
    supplied: Dict[str, object],
    caller: str,
    provenance: ProvenanceType,
    transport: Transport,
    audit: bool,
) -> None:
    """Apply the declarative rate limit (γ-2b) before any auth / consent work.

    Scripts without a ``rate_limit`` field pass through unconditionally.
    Scripts with a limit get a per-``(script_name)`` rolling-window check
    against the SQLite counter DB; at/over cap raises
    :class:`RateLimitExceeded` after writing one audit entry tagged
    ``error="ratelimit_exceeded"``. No transport.execute is ever called
    when the limit fires. Redacted params are used for tier-3 audit
    entries even in the rejection path.
    """
    if script.rate_limit is None:
        return
    try:
        check_and_record(script.name, script.rate_limit)
    except RateLimitExceeded as exc:
        if audit:
            audit_params = (
                redact_params_for_tier3(supplied)
                if script.tier >= 3 else supplied
            )
            write_audit(make_entry(
                script=script.name, tier=script.tier, ok=False,
                duration_s=0.0,
                params=audit_params, caller=caller,
                provenance=provenance.value,
                host=transport.host,
                error="ratelimit_exceeded",
            ))
        raise


def osa(
    name: str,
    params: Optional[Dict[str, object]] = None,
    *,
    timeout_s: Optional[int] = None,
    audit: bool = True,
    caller: str = "local",
    provenance: ProvenanceType = ProvenanceType.LOCAL,
    capability_token: Optional[CapabilityToken] = None,
    transport: Optional[Transport] = None,
) -> OsaResult:
    """Dispatch a named registry script with validated params.

    Args:
        name: Registry script name (see ``lib/hal/osa/registry.json``).
        params: Mapping of parameter name → value. Validated against
            :class:`Script.params` before interpolation.
        timeout_s: Override the registry's declared timeout. None uses the
            script's default.
        audit: When True (default), append one :class:`AuditEntry` to
            ``$CLAUDE_DIR/logs/osa.log`` for both success and failure.
        caller: Free-form annotation on the audit entry. Deprecated in
            favour of ``provenance``; kept for α-3a back-compat.
        provenance: Canonical :class:`ProvenanceType` recorded on the audit
            entry. Defaults to :data:`ProvenanceType.LOCAL`. Auto-promoted
            to :data:`ProvenanceType.MCP_REMOTE` when ``transport`` is
            remote.
        capability_token: HMAC-signed token required for tier-2+ **local**
            scripts. Remote dispatches use SSH key auth instead; see
            :mod:`lib.hal.osa.transport`.
        transport: Optional :class:`Transport` strategy. ``None`` means
            local (the α-2 behaviour). Pass :class:`SSHTransport` for
            remote dispatch to a Tailscale peer.

    Returns:
        The :class:`OsaResult` from the underlying kernel.

    Raises:
        ScriptNotRegistered: unknown ``name``.
        ParamValidationError: ``params`` fails the declared schema.
        MissingSecret / TokenInvalid / TokenExpired / TokenReused: auth
            rejected the token (local) or remote consent env missing.
        ConsentDeclined: tier-3 consent callback returned False or
            raised; dispatch is aborted and the rejection is audit-logged
            with ``provenance=user_rejected`` and redacted params.
        RateLimitExceeded: the script declared a ``rate_limit`` and the
            rolling-window cap was hit; the rejection is audit-logged as
            ``error="ratelimit_exceeded"`` and no transport call runs.
        OsaError / OsaTimeout / PlatformUnsupported: from the transport.
    """
    supplied = params or {}
    script = get_script(name)
    validate_params(script, supplied)

    effective_transport = transport or LocalTransport()
    _enforce_rate_limit(
        script, supplied, caller, provenance, effective_transport, audit,
    )
    effective_provenance = _authorise_with_transport(
        script, capability_token, effective_transport, provenance
    )
    _authorise_tier3_consent(
        script, supplied, caller, effective_provenance, effective_transport,
    )

    audit_params = (
        redact_params_for_tier3(supplied) if script.tier >= 3 else supplied
    )

    body = _apply_params(_resolve_body(script), supplied, script.language)
    effective_timeout = timeout_s if timeout_s is not None else script.timeout_s
    started = time.monotonic()

    try:
        result = effective_transport.execute(
            body, language=script.language, timeout_s=effective_timeout
        )
    except OsaError as exc:
        if audit:
            write_audit(make_entry(
                script=name, tier=script.tier, ok=False,
                duration_s=time.monotonic() - started,
                params=audit_params, caller=caller,
                provenance=effective_provenance.value,
                host=effective_transport.host,
                error=str(exc),
            ))
        raise

    if audit:
        write_audit(make_entry(
            script=name, tier=script.tier, ok=result.ok,
            duration_s=result.duration_s,
            params=audit_params, caller=caller,
            provenance=effective_provenance.value,
            host=effective_transport.host,
            returncode=result.returncode,
            error=None if result.ok else (result.stderr.strip() or "non-zero exit"),
        ))

    if not result.ok:
        raise OsaError(
            f"osa {name!r} on {effective_transport.host!r} exited "
            f"{result.returncode}: {result.stderr.strip() or '(no stderr)'}"
        )
    return result
