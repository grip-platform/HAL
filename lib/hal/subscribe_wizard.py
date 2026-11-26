"""HAL `hal --subscribe` wizard — one-question OAuth token activation.

Asks the operator to paste an OAuth token, writes it to the macOS Keychain
under the canonical service name (``grip-claude-oauth``), and verifies the
write by re-reading via :class:`~lib.hal.subscription.CCHSubscription`.

On non-macOS platforms (no ``security`` binary), falls back to printing
the canonical env var export line so the user can persist it themselves.

Design:
- Pure functions where possible — :func:`run_wizard` is the only IO sink,
  and it accepts injectable readers/writers for testability.
- The token never lands in a log file or stdout; failures show provider-
  specific hints rather than echoing the value.
- `-U` upserts so re-running the wizard refreshes a stale token instead
  of erroring on duplicate.
"""
import getpass
import platform
import subprocess
from typing import Callable, Optional

from lib.hal.subscription import CCHSubscription, KimiCodingSubscription, QwenSubscription

_KEYCHAIN_ACCOUNT = "hal"
_MIN_TOKEN_LEN = 20  # Anthropic OAuth tokens are well over this; cheap sanity check.


# Provider routing — keychain service names MUST match
# :mod:`lib.hal.subscription` so ``auto_detect_subscriptions()`` finds them.
_PROVIDER_CONFIG = {
    "anthropic": {
        "label": "Anthropic subscription (Claude Opus / Sonnet / Haiku)",
        "service": "grip-claude-oauth",
        "loader": CCHSubscription.from_env,
    },
    # Kimi Code -- Moonshot's Kimi MEMBERSHIP coding service (GRIP Pool D,
    # sovereign-council seal e49c00c214c500be, 2026-09-07). Not the
    # pay-per-token Moonshot platform key: that is keychain grip-moonshot,
    # Pool B, metered, and never a subscription. The service name is the one
    # the GRIP side resolves too (lib/provider_secret.resolve_provider_key
    # ("kimi-coding"), GRIP#5336), so both repos arm off one keychain entry.
    "kimi-coding": {
        "label": ("Kimi Code membership (Moonshot, GRIP Pool D -- "
                  "k3 / k3-256k / kimi-for-coding)"),
        "service": "grip-kimi-coding",
        "loader": KimiCodingSubscription.from_env,
    },
    "qwen": {
        "label": "Alibaba Qwen subscription (DashScope)",
        "service": "hal-dashscope-subscription",
        "loader": QwenSubscription.from_env,
    },
}

DEFAULT_PROVIDER = "anthropic"


# Public exit codes — caller (cli.py) maps these to process exit status.
EXIT_OK = 0
EXIT_USER_CANCELLED = 1
EXIT_TOKEN_INVALID = 2
EXIT_KEYCHAIN_FAILED = 3
EXIT_VERIFY_FAILED = 4
EXIT_UNSUPPORTED_PLATFORM = 5
EXIT_UNKNOWN_PROVIDER = 6


def _is_macos() -> bool:
    """Return True when the wizard can use the macOS `security` keychain."""
    return platform.system() == "Darwin"


def _validate_token(raw: str) -> Optional[str]:
    """Return a stripped token if it looks plausible, else None.

    The check is intentionally permissive — we do not gatekeep on a regex
    that may break when upstream rotates the token format. We only catch
    obvious paste mistakes (empty, too short, or contains a clear header
    like 'Bearer ' prefix that should have been trimmed).
    """
    token = raw.strip()
    if not token:
        return None
    if len(token) < _MIN_TOKEN_LEN:
        return None
    # A common copy-paste mistake — stripping the header so the keychain
    # holds the raw token only.
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token or None


def _write_keychain(token: str, *, service: str,
                    account: str = _KEYCHAIN_ACCOUNT,
                    runner: Optional[Callable] = None) -> tuple[bool, str]:
    """Upsert the token into the macOS Keychain.

    Returns ``(success, message)``. ``runner`` is injectable for tests.
    """
    runner = runner or subprocess.run
    cmd = [
        "security", "add-generic-password",
        "-U",                       # upsert (update if exists)
        "-s", service,
        "-a", account,
        "-w", token,
    ]
    try:
        result = runner(cmd, capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return False, "macOS `security` binary not found"
    except subprocess.TimeoutExpired:
        return False, "Keychain write timed out (>10s)"
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "unknown failure").strip()
    return True, "ok"


def _verify_subscription(loader: Callable) -> bool:
    """Re-read the keychain through the given loader and health-check.

    The loader is a ``*Subscription.from_env`` bound method — returns the
    subscription instance (or None) already resolved from env/keychain.
    """
    sub = loader()
    return bool(sub) and sub.health_check()


def _print_unsupported_hint() -> None:
    """Tell the user how to set the env var on platforms without keychain."""
    print("Subscription activation on this platform is manual.")
    print("Set the canonical env var in your shell rc:")
    print()
    print("  export CLAUDE_CODE_OAUTH_TOKEN='<your-token>'")
    print()
    print("Then restart your shell and re-run `hal --status`.")


def _token_already_active(loader: Callable) -> bool:
    """Return True if the provider already has a live token in the Keychain/env."""
    try:
        sub = loader()
        return bool(sub)
    except Exception:
        return False


def _print_intro(printer: Callable[..., None], config: dict) -> None:
    """Render the wizard banner for a specific provider."""
    label = config["label"]
    printer(f"HAL subscription activation — {label}")
    printer("-" * min(60, len(label) + 34))
    printer("Paste your OAuth token below. The token will be written to your")
    printer(f"macOS Keychain under service `{config['service']}` (account "
            f"`{_KEYCHAIN_ACCOUNT}`).")
    printer("")
    printer("To get a token: visit your provider's account dashboard, find the")
    printer("OAuth/API section, and copy the long-lived token. The wizard does")
    printer("not echo what you type.")
    printer("")


def _capture_token(reader: Callable[[str], str], printer: Callable[..., None]) -> tuple[Optional[str], int]:
    """Read + validate a token. Returns (token, exit_code) where token is set on EXIT_OK."""
    try:
        raw = reader("Paste OAuth token (input hidden): ")
    except (EOFError, KeyboardInterrupt):
        printer("")
        printer("Cancelled. No changes made.")
        return None, EXIT_USER_CANCELLED

    token = _validate_token(raw)
    if token is None:
        printer("")
        printer("That doesn't look like a valid OAuth token (empty or too short).")
        printer("Re-run `hal --subscribe` to try again.")
        return None, EXIT_TOKEN_INVALID
    return token, EXIT_OK


def _persist_and_verify(token: str, write_fn: Callable, verify_fn: Callable,
                        printer: Callable[..., None], config: dict) -> int:
    """Write the token to Keychain and verify the round-trip. Returns exit code."""
    ok, msg = write_fn(token, service=config["service"],
                       account=_KEYCHAIN_ACCOUNT, runner=None)
    if not ok:
        printer("")
        printer(f"Keychain write failed: {msg}")
        printer("Set the provider's documented env var instead (see README).")
        return EXIT_KEYCHAIN_FAILED

    if not verify_fn():
        printer("")
        printer("Token written to Keychain, but subscription verification failed.")
        printer("Run `hal --status` to investigate.")
        return EXIT_VERIFY_FAILED

    printer("")
    printer(f"{config['label']} active. Run `hal --models subscription` to see")
    printer("what's available, or `hal --model <provider/model>` to start chatting.")
    return EXIT_OK


def _prompt_rotate_or_skip(
    config: dict,
    reader: Optional[Callable[[str], str]],
    printer: Callable[..., None],
) -> Optional[int]:
    """If a token is already active, ask to rotate or skip.

    Returns an exit code when the wizard should stop early, or ``None`` to
    proceed to the full token-capture flow.
    """
    if not _token_already_active(config["loader"]):
        return None
    printer(f"{config['label']} is already active.")
    printer("Options: [r] rotate token  [s] skip (keep existing)")
    printer("")
    try:
        choice = (reader or getpass.getpass)("Choice [s]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        printer("")
        printer("Cancelled. Existing token unchanged.")
        return EXIT_USER_CANCELLED
    if choice not in ("r", "rotate"):
        printer("Keeping existing token. Run `hal --status` to verify.")
        return EXIT_OK
    return None


def run_wizard(provider: str = DEFAULT_PROVIDER, *,
               reader: Optional[Callable[[str], str]] = None,
               writer: Optional[Callable[..., tuple[bool, str]]] = None,
               verifier: Optional[Callable[..., bool]] = None,
               printer: Callable[..., None] = print,
               platform_check: Callable[[], bool] = _is_macos,
               _active_loader: Optional[Callable] = None,
               ) -> int:
    """Orchestrate banner → token capture → keychain write → verify.

    ``provider`` picks which subscription transport to configure
    (anthropic / kimi-coding / qwen). All IO collaborators are injectable
    so tests can simulate every branch without touching a real terminal,
    the keychain, or the network. ``_active_loader`` overrides the
    provider's default loader in tests.
    """
    config = _PROVIDER_CONFIG.get(provider)
    if config is None:
        printer(f"Unknown subscription provider: {provider!r}. "
                f"Supported: {', '.join(_PROVIDER_CONFIG)}.")
        return EXIT_UNKNOWN_PROVIDER

    if not platform_check():
        _print_unsupported_hint()
        return EXIT_UNSUPPORTED_PLATFORM

    effective_config = {**config, "loader": _active_loader} if _active_loader else config
    early = _prompt_rotate_or_skip(effective_config, reader, printer)
    if early is not None:
        return early

    _print_intro(printer, config)
    token, code = _capture_token(reader or getpass.getpass, printer)
    if token is None:
        return code

    verify_fn = verifier if verifier is not None else (
        lambda: _verify_subscription(config["loader"])
    )
    return _persist_and_verify(
        token,
        writer or _write_keychain,
        verify_fn,
        printer,
        config,
    )
