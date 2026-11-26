"""osa tier-3 consent — interactive approval for irreversible external effects.

Tier 3 sits above tier 2 (capability token): tier 3 scripts cannot be
authorised purely by a pre-minted token. They require a fresh consent
prompt rendered at dispatch time with the resolved params visible to the
operator. Rejection is logged with ``provenance=user_rejected``, audit
params are replaced by sha256 prefixes, and yolo mode does NOT bypass
this check.

Gaps this closes (broly MESH RETHINK on γ-2 naive seed, 2026-04-19):

* Gap 4 — "tier-2 is insufficient for irreversible external effects"
* Gap 5 — "audit log leaks PII"

Public API::

    from lib.hal.osa.consent import (
        ConsentCallback, ConsentDeclined, CONSENT_TIMEOUT_S,
        set_consent_callback, get_consent_callback,
        redact_params_for_tier3,
        default_consent_callback,
    )

Integrators (GRIP, HAL CLI, test harnesses) swap the default macOS
``display dialog`` prompt with their own transport via
:func:`set_consent_callback`.
"""
from __future__ import annotations

import hashlib
import subprocess
from typing import Callable, Dict

__all__ = [
    "ConsentCallback",
    "ConsentDeclined",
    "CONSENT_TIMEOUT_S",
    "set_consent_callback",
    "get_consent_callback",
    "redact_params_for_tier3",
    "default_consent_callback",
    "_reset_for_tests",
]


CONSENT_TIMEOUT_S = 30


ConsentCallback = Callable[[str, Dict[str, object]], bool]
"""``(script_name, params) -> bool``. Return True to allow, False to reject.

Implementations must complete within :data:`CONSENT_TIMEOUT_S` — the
caller does not wrap with its own timeout. Exceptions propagate and are
treated as rejection by the dispatcher."""


class ConsentDeclined(Exception):
    """Raised when a tier-3 dispatch's consent callback returns False."""


def default_consent_callback(
    script_name: str,
    params: Dict[str, object],
) -> bool:
    """Prompt via ``osascript display dialog`` with the resolved params visible.

    Returns False on timeout, cancel, or any subprocess failure. The
    dialog times out at :data:`CONSENT_TIMEOUT_S` via AppleScript's
    ``giving up after``. Non-Darwin hosts and environments without
    ``osascript`` return False — no silent fall-through.
    """
    param_summary = _format_params_for_prompt(params)
    prompt = (
        f"Tier-3 osa script requires consent.\n\n"
        f"Script: {script_name}\n{param_summary}"
    )
    applescript = (
        f"display dialog {_applescript_literal(prompt)} "
        f'buttons {{"Cancel", "Approve"}} '
        f'default button "Cancel" '
        f"giving up after {CONSENT_TIMEOUT_S} "
        f'with title "GRIP/HAL tier-3 consent"'
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", applescript],
            capture_output=True,
            text=True,
            timeout=CONSENT_TIMEOUT_S + 5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    return "Approve" in result.stdout


def _format_params_for_prompt(params: Dict[str, object]) -> str:
    """Render params for the approval dialog — values shown, not hashed.

    The operator needs to see what they are approving. The hash goes in
    the audit log after dispatch, not in the prompt. Long values are
    truncated so the dialog stays on one screen.
    """
    if not params:
        return "(no params)"
    lines = []
    for key, val in sorted(params.items()):
        rendered = str(val)
        if len(rendered) > 120:
            rendered = rendered[:117] + "..."
        lines.append(f"  {key}: {rendered}")
    return "\n".join(lines)


def _applescript_literal(text: str) -> str:
    """Escape ``text`` for embedding in an AppleScript string literal."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def redact_params_for_tier3(
    params: Dict[str, object],
) -> Dict[str, str]:
    """Replace every value with ``sha256(str(value))[:12]``.

    Keeps correlation (same input → same hash) while dropping plaintext.
    Always returns 12-char lowercase hex. Empty-string values still hash
    so the audit line shape is identical across calls.
    """
    redacted: Dict[str, str] = {}
    for key, val in params.items():
        digest = hashlib.sha256(str(val).encode("utf-8")).hexdigest()
        redacted[key] = digest[:12]
    return redacted


_callback: ConsentCallback = default_consent_callback


def set_consent_callback(fn: ConsentCallback) -> None:
    """Swap the module-level consent callback.

    Pass :func:`default_consent_callback` to restore the macOS dialog
    prompt. Called once per process by integrators; tests stub per
    test-case via :func:`_reset_for_tests`.
    """
    global _callback
    _callback = fn


def get_consent_callback() -> ConsentCallback:
    """Return the currently installed consent callback."""
    return _callback


def _reset_for_tests() -> None:
    """Restore the default callback (test isolation)."""
    global _callback
    _callback = default_consent_callback
