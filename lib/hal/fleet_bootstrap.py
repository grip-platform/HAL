"""Fleet Gateway Bootstrap — wire transport adapters with credential loading.

Loads bot tokens from macOS keychain (security find-generic-password) and
registers transport adapters with the ChannelRouter. Transports without
credentials are logged but not registered — partial fleet is better than
no fleet.

Usage:
    from lib.hal.fleet_bootstrap import bootstrap_fleet

    router = ChannelRouter()
    report = bootstrap_fleet(router)
    print(f"Active: {report.active}, Skipped: {report.skipped}")

Credential storage pattern (macOS keychain):
    security add-generic-password -s "hal-telegram" -a "bot" -w "123:ABC"
    security add-generic-password -s "hal-slack" -a "bot" -w "xoxb-..."
    security add-generic-password -s "hal-discord" -a "bot" -w "MTIz..."
"""
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BootstrapReport:
    """Result of fleet bootstrap."""
    active: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


# Transport -> (keychain service name, adapter import path, adapter class name)
_TRANSPORT_CONFIG = {
    "telegram": ("hal-telegram", "apps.gateway.telegram.adapter", "TelegramAdapter"),
    "slack": ("hal-slack", "apps.gateway.slack.adapter", "SlackAdapter"),
    "discord": ("hal-discord", "apps.gateway.discord.adapter", "DiscordAdapter"),
}


def _keychain_get(service: str) -> Optional[str]:
    """Read a password from macOS keychain. Returns None if not found."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _import_adapter(module_path: str, class_name: str):
    """Dynamic import of adapter class."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def bootstrap_fleet(router: "ChannelRouter") -> BootstrapReport:
    """Load credentials and register available transport adapters.

    Each transport is attempted independently — one failure doesn't
    prevent others from registering.
    """
    report = BootstrapReport()

    for transport, (service, mod_path, cls_name) in _TRANSPORT_CONFIG.items():
        token = _keychain_get(service)
        if not token:
            report.skipped.append(transport)
            continue
        try:
            adapter_cls = _import_adapter(mod_path, cls_name)
            adapter = adapter_cls(router, token=token)
            router.register(transport, adapter)
            report.active.append(transport)
        except Exception as e:
            report.errors[transport] = str(e)[:200]

    return report


def fleet_status_text(report: BootstrapReport) -> str:
    """Format a human-readable fleet status line."""
    parts = [f"Fleet: {len(report.active)} active"]
    if report.active:
        parts.append(f"({', '.join(report.active)})")
    if report.skipped:
        parts.append(f"| {len(report.skipped)} need credentials")
    if report.errors:
        parts.append(f"| {len(report.errors)} errors")
    return " ".join(parts)
