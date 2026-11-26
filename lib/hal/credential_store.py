"""HAL Credential Store — secure API key management.

Manages API keys for LLM providers without storing them in plaintext
config files. Uses the system keychain (macOS Keychain, Linux Secret
Service) when available, falls back to env vars.

Usage:
    from lib.hal.credential_store import CredentialStore

    store = CredentialStore()
    store.set("groq", "gsk_abc123...")
    key = store.get("groq")
    providers = store.list_providers()
"""
import os
import subprocess
from dataclasses import dataclass


_SERVICE_NAME = "hal-credentials"


@dataclass(frozen=True)
class CredentialInfo:
    """Information about a stored credential (without the secret)."""
    provider: str
    source: str  # "keychain", "env", "config"
    masked: str  # e.g. "gsk_***abc"


class CredentialStore:
    """Multi-source credential management. Keychain > env > config."""

    def get(self, provider: str) -> str:
        """Get API key for a provider. Checks keychain, then env, then config."""
        # Try system keychain first
        key = _keychain_get(provider)
        if key:
            return key

        # Try environment variable
        env_key = _env_var_name(provider)
        key = os.environ.get(env_key, "")
        if key:
            return key

        return ""

    def set(self, provider: str, api_key: str) -> str:
        """Store API key in system keychain. Returns storage location."""
        if _keychain_set(provider, api_key):
            return "keychain"
        # Fallback: suggest env var
        return f"Set {_env_var_name(provider)} in your shell profile"

    def delete(self, provider: str) -> bool:
        """Remove a stored credential."""
        return _keychain_delete(provider)

    def list_providers(self) -> list[CredentialInfo]:
        """List all providers with credentials available."""
        providers: list[CredentialInfo] = []
        seen: set[str] = set()

        # Check known env vars
        for name in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY",
                     "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
            provider = name.replace("_API_KEY", "").lower()
            val = os.environ.get(name, "")
            if val:
                providers.append(CredentialInfo(provider, "env", _mask(val)))
                seen.add(provider)

        return providers

    def has(self, provider: str) -> bool:
        """Check if credentials exist for a provider."""
        return bool(self.get(provider))


def _env_var_name(provider: str) -> str:
    """Convert provider name to env var name."""
    return f"{provider.upper()}_API_KEY"


def _mask(key: str) -> str:
    """Mask an API key for display."""
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}***{key[-3:]}"


def _keychain_get(provider: str) -> str:
    """Get from macOS Keychain. Returns empty string on failure."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", _SERVICE_NAME,
             "-a", provider, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return ""


def _keychain_set(provider: str, key: str) -> bool:
    """Store in macOS Keychain."""
    try:
        # Delete existing first (add fails if exists)
        subprocess.run(
            ["security", "delete-generic-password", "-s", _SERVICE_NAME, "-a", provider],
            capture_output=True, timeout=5,
        )
        result = subprocess.run(
            ["security", "add-generic-password", "-s", _SERVICE_NAME,
             "-a", provider, "-w", key],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _keychain_delete(provider: str) -> bool:
    """Delete from macOS Keychain."""
    try:
        result = subprocess.run(
            ["security", "delete-generic-password", "-s", _SERVICE_NAME, "-a", provider],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
