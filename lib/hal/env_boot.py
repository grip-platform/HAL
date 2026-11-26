"""HAL .env bootstrap — load ``$HAL_HOME/.env`` into ``os.environ``.

HAL has historically resolved secrets via two rungs only: the process
environment, and the macOS Keychain (see :mod:`lib.hal.subscription`
``_fetch_token``). The Keychain rung is unreadable from pm2 / launchd /
cron / non-login-shell contexts, so a Claude Code subscription token
stored *only* in the Keychain never reaches
:meth:`CCHSubscription.from_env` inside the ``hal-server`` daemon — and
HAL silently falls back to a metered per-token Anthropic key.

Loading a git-ignored ``$HAL_HOME/.env`` here gives the env-var rung a
value in exactly those contexts, making subscription (CCH) routing the
canonical-first path everywhere, not just in an interactive login shell.

Contract:
  * Process environment wins by default (``override=False``) — an
    explicitly exported value is never clobbered by the file.
  * A missing or unreadable file is a no-op (returns 0), never raised:
    ``.env`` is an optional convenience, not a hard dependency, so a
    deploy without one keeps working exactly as before.
"""
import os
from pathlib import Path

__all__ = ["load_env", "default_env_path"]


def default_env_path() -> Path:
    """Resolve ``$HAL_HOME/.env`` at call time (HAL_HOME may be set late)."""
    home = os.environ.get("HAL_HOME") or str(Path.home() / ".hal")
    return Path(home) / ".env"


def _parse_line(line: str) -> tuple[str, str] | None:
    """Parse one ``KEY=VALUE`` line. Returns None for blanks/comments.

    Tolerates a leading ``export `` and surrounding matched quotes so an
    operator can ``source`` the same file from a shell if they want to.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, _, val = stripped.partition("=")
    key = key.strip()
    if key.startswith("export "):
        key = key[len("export "):].strip()
    if not key:
        return None
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        val = val[1:-1]
    return key, val


def load_env(path=None, *, override: bool = False) -> int:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into ``os.environ``.

    Returns the number of variables actually set. The default path is
    :func:`default_env_path`. Process env wins unless ``override=True``.
    Any read failure (missing file, not a file, no permission) yields 0
    — the loader degrades to the pre-existing keychain/env behaviour.
    """
    target = Path(path) if path is not None else default_env_path()
    try:
        text = target.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError,
            PermissionError, OSError, UnicodeDecodeError):
        return 0
    count = 0
    for line in text.splitlines():
        parsed = _parse_line(line)
        if parsed is None:
            continue
        key, val = parsed
        if not override and key in os.environ:
            continue
        os.environ[key] = val
        count += 1
    return count
