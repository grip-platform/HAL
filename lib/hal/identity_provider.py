"""HAL Identity Provider — user/org identity without any external harness.

Reads identity from facts/identity.md (GRIP native) with zero external
dependency. No environment variables from any external tool required.

Also provides ExternalIdentityResolver for the Fleet Gateway — maps
platform-native user IDs (Telegram user_id, Slack user, WhatsApp MSISDN,
email address) to a single canonical HAL user so conversations from any
transport land on the same thread.

Usage:
    from lib.hal.identity_provider import get_identity, ExternalIdentityResolver

    identity = get_identity()
    print(identity.name, identity.signature, identity.role)

    resolver = ExternalIdentityResolver()
    resolver.bind("telegram", "123456", "M>>")
    resolver.bind("slack", "U0EXAMPLE01", "M>>")
    assert resolver.resolve("telegram", "123456") == "M>>"
    assert resolver.resolve("slack", "U0EXAMPLE01") == "M>>"
"""
import json
import os
import re
import threading
from typing import Optional

from lib.hal.home import HAL_HOME as _HAL_HOME
_IDENTITY_PATH = os.path.join(_HAL_HOME, "facts", "identity.md")
_TEAM_PATH = os.path.join(_HAL_HOME, "facts", "team.md")
_BINDINGS_PATH = os.path.join(_HAL_HOME, "facts", "external_bindings.json")
_YAML_KV_RE = re.compile(r'^(\w+):\s*"?([^"]*)"?\s*$')


class Identity:
    """User identity resolved from GRIP's own sources."""

    def __init__(
        self,
        name: str = "",
        signature: str = "",
        role: str = "",
        mode: str = "solo",
    ):
        self.name = name
        self.signature = signature
        self.role = role
        self.mode = mode

    def to_dict(self) -> dict:
        return {
            "name": self.name, "signature": self.signature,
            "role": self.role, "mode": self.mode,
        }


def get_identity(identity_path: Optional[str] = None) -> Identity:
    """Resolve identity from CC env vars (fast path) or facts/identity.md (fallback).

    Fast path uses CC 2.1.51 env vars — zero subprocess, no race condition.
    """
    # Fast path: CC env vars (CC 2.1.51 — zero subprocess, no race condition)
    user_email = os.environ.get("CLAUDE_CODE_USER_EMAIL", "")
    if user_email:
        return Identity(
            name=user_email,
            signature=os.environ.get("CLAUDE_CODE_ACCOUNT_UUID", ""),
            role=os.environ.get("CLAUDE_CODE_ORGANIZATION_UUID", ""),
            mode="cc_env",
        )

    path = identity_path or _IDENTITY_PATH
    raw = _read_file(path)
    if not raw:
        return Identity()
    return _parse_identity(raw)


def _parse_identity(raw: str) -> Identity:
    """Parse identity from YAML block inside markdown."""
    in_yaml = False
    fields: dict[str, str] = {}

    for line in raw.split("\n"):
        stripped = line.strip()
        if stripped == "```yaml":
            in_yaml = True
            continue
        if stripped == "```" and in_yaml:
            break
        if in_yaml:
            match = _YAML_KV_RE.match(stripped)
            if match:
                fields[match.group(1)] = match.group(2)

    return Identity(
        name=fields.get("name", ""),
        signature=fields.get("signature", ""),
        role=fields.get("role", ""),
        mode=fields.get("mode", "solo"),
    )


def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return ""


class ExternalIdentityResolver:
    """Maps platform-native user IDs to a single canonical HAL user.

    The resolver is the Fleet Gateway's identity bridge — when the same
    human reaches HAL from Telegram, Slack, WhatsApp, and email, every
    inbound envelope should resolve to the same canonical id so thread
    state and preferences remain coherent across transports.

    Storage: JSON file at `facts/external_bindings.json`. Structure is
    `{"<platform>:<external_id>": "<canonical_id>"}`. Bindings are
    load-on-construct, write-on-mutation. Thread-safe via a lock.

    Canonical IDs are strings — typically the HAL signature marker
    (e.g. "M>>") but the resolver is agnostic. Callers pick the scheme.
    """

    def __init__(self, bindings_path: Optional[str] = None):
        self._path = bindings_path or _BINDINGS_PATH
        self._lock = threading.Lock()
        self._bindings: dict[str, str] = self._load()

    @staticmethod
    def _key(platform: str, external_id: str) -> str:
        if not platform or not external_id:
            raise ValueError("platform and external_id must both be non-empty")
        return f"{platform}:{external_id}"

    def _load(self) -> dict[str, str]:
        raw = _read_file(self._path)
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def _flush_locked(self) -> None:
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{self._path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._bindings, f, indent=2, sort_keys=True)
        os.replace(tmp, self._path)

    def bind(self, platform: str, external_id: str, canonical_id: str) -> None:
        """Associate (platform, external_id) with a canonical user id.

        Re-binding an existing pair overwrites the canonical id (useful
        when a user rotates accounts).
        """
        if not canonical_id:
            raise ValueError("canonical_id must be non-empty")
        key = self._key(platform, external_id)
        with self._lock:
            self._bindings[key] = canonical_id
            self._flush_locked()

    def unbind(self, platform: str, external_id: str) -> None:
        """Remove a binding if present. Idempotent."""
        key = self._key(platform, external_id)
        with self._lock:
            if self._bindings.pop(key, None) is not None:
                self._flush_locked()

    def resolve(self, platform: str, external_id: str) -> Optional[str]:
        """Return the canonical user id for (platform, external_id) or None."""
        key = self._key(platform, external_id)
        with self._lock:
            return self._bindings.get(key)

    def bindings_for(self, canonical_id: str) -> tuple:
        """Return a tuple of (platform, external_id) pairs for a canonical id."""
        if not canonical_id:
            return ()
        with self._lock:
            matches = [
                tuple(k.split(":", 1))
                for k, v in self._bindings.items()
                if v == canonical_id and ":" in k
            ]
        return tuple(sorted(matches))

    def all_platforms(self) -> tuple:
        """Return the distinct platforms with at least one binding."""
        with self._lock:
            platforms = {k.split(":", 1)[0] for k in self._bindings if ":" in k}
        return tuple(sorted(platforms))
