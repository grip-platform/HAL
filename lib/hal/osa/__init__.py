"""osa — AppleScript/JavaScript OSA execution primitive + policy layer.

Phase α-2 shipped the kernel (escape + osa_raw). Phase α-3a added the
public dispatch (registry resolution, parameter validation, audit log,
:func:`osa` entry point). Phase α-3b hardens the dispatch surface with:

*   Capability tokens (HMAC-SHA256, single-use, 60-second TTL) gating
    every tier-2+ script.
*   Provenance taxonomy (enum) recorded on every audit entry.
*   ``js_escape`` for JavaScript-for-Automation scripts.
*   Append-only ``chflags uappnd`` on the audit log (macOS).
*   Shell-rc scan that warns if ``GRIP_OSA_UNSAFE`` is about to silently
    bypass tier-2+ checks in future sessions.

Call sites will migrate onto :func:`osa` in β-4 (per the audit in
``plans/osa-consolidation-audit.md``).

Remaining for γ phases: registry git-signature verification, MCP
per-script allowlist, SSH ControlPersist cap, iPhone Shortcut SHA
pinning. Tracked in the parent plan ``plans/osascript-control-layer.md``.
"""
from lib.hal.osa.audit import AUDIT_LOG_PATH, AuditEntry, make_entry, write_audit
from lib.hal.osa.auth import (
    CapabilityToken,
    MissingSecret,
    TokenExpired,
    TokenInvalid,
    TokenReused,
    default_secret,
    mint_token,
    verify_token,
)
from lib.hal.osa.consent import (
    CONSENT_TIMEOUT_S,
    ConsentCallback,
    ConsentDeclined,
    default_consent_callback,
    get_consent_callback,
    redact_params_for_tier3,
    set_consent_callback,
)
from lib.hal.osa.dispatch import osa
from lib.hal.osa.escape import applescript_escape, js_escape
from lib.hal.osa.osa import (
    OsaError,
    OsaResult,
    OsaTimeout,
    PlatformUnsupported,
    osa_raw,
)
from lib.hal.osa.allowlist import (
    AllowlistNotFound,
    AllowlistRejection,
    allowlist_path,
    apply_immutable,
    check_allowed,
    load_allowlist,
)
from lib.hal.osa.provenance import ProvenanceType, UnknownProvenance, parse_provenance
from lib.hal.osa.rate_limit import (
    RateLimit,
    RateLimitExceeded,
    check_and_record,
    default_db_path,
)
from lib.hal.osa.registry import (
    ParamSpec,
    ParamValidationError,
    Script,
    ScriptNotRegistered,
    get_script,
    load_registry,
    validate_params,
)
from lib.hal.osa.shell_rc import UNSAFE_VAR_NAME, detect_unsafe_env_in_shell_rc
from lib.hal.osa.shortcuts import (
    ShortcutError,
    ShortcutResult,
    ShortcutUnavailable,
    run_shortcut,
)
from lib.hal.osa.transport import (
    REMOTE_DISPATCH_ENV,
    LocalTransport,
    SSHTransport,
    Transport,
)
from lib.hal.osa.fleet import (
    FleetPlan,
    FleetResult,
    default_transport_factory,
    osa_fleet,
)

__all__ = [
    # Kernel (α-2)
    "applescript_escape",
    "js_escape",
    "osa_raw",
    "OsaResult",
    "OsaError",
    "OsaTimeout",
    "PlatformUnsupported",
    # Registry (α-3a)
    "Script",
    "ParamSpec",
    "ScriptNotRegistered",
    "ParamValidationError",
    "load_registry",
    "get_script",
    "validate_params",
    # Audit (α-3a + α-3b immutability)
    "AuditEntry",
    "make_entry",
    "write_audit",
    "AUDIT_LOG_PATH",
    # Auth (α-3b)
    "CapabilityToken",
    "mint_token",
    "verify_token",
    "default_secret",
    "TokenInvalid",
    "TokenExpired",
    "TokenReused",
    "MissingSecret",
    # Provenance (α-3b)
    "ProvenanceType",
    "parse_provenance",
    "UnknownProvenance",
    # Tier-3 consent (γ-2a)
    "ConsentCallback",
    "ConsentDeclined",
    "CONSENT_TIMEOUT_S",
    "set_consent_callback",
    "get_consent_callback",
    "default_consent_callback",
    "redact_params_for_tier3",
    # Shell-rc scan (α-3b)
    "UNSAFE_VAR_NAME",
    "detect_unsafe_env_in_shell_rc",
    # Transport (γ-1)
    "Transport",
    "LocalTransport",
    "SSHTransport",
    "REMOTE_DISPATCH_ENV",
    # Rate limit (γ-2b)
    "RateLimit",
    "RateLimitExceeded",
    "check_and_record",
    "default_db_path",
    # Allowlist (γ-2c)
    "AllowlistNotFound",
    "AllowlistRejection",
    "allowlist_path",
    "apply_immutable",
    "check_allowed",
    "load_allowlist",
    # Shortcut backend (γ-2d)
    "ShortcutError",
    "ShortcutResult",
    "ShortcutUnavailable",
    "run_shortcut",
    # Fleet dispatch (γ-3)
    "osa_fleet",
    "default_transport_factory",
    "FleetPlan",
    "FleetResult",
    # Public dispatch (α-3a, upgraded α-3b + γ-1)
    "osa",
]
