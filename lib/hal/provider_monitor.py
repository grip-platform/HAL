"""HAL Provider Monitor — continuous background health checking.

Periodically pings all registered providers and tracks their availability
over time. Integrates with the notification system to alert on outages.

Usage:
    from lib.hal.provider_monitor import ProviderMonitor

    monitor = ProviderMonitor(registry)
    monitor.check_all()  # One-shot health check
    status = monitor.get_status()
"""
import time
from dataclasses import dataclass, field


@dataclass
class ProviderHealth:
    """Health state for a single provider.

    Includes a self-healing integrity model: each provider has a target
    integrity (desired health, 0.0-1.0) and an actual integrity (current
    health). Failures decay actual integrity; heal_tick() regenerates it
    toward the target. This creates graceful degradation and recovery
    rather than binary on/off health transitions.
    """
    provider: str
    healthy: bool = True
    last_check: float = 0.0
    last_success: float = 0.0
    consecutive_failures: int = 0
    total_checks: int = 0
    total_failures: int = 0
    # Self-healing integrity model
    target_integrity: float = 1.0
    actual_integrity: float = 1.0
    regen_rate: float = 0.05
    decay_rate: float = 0.20

    @property
    def uptime_pct(self) -> float:
        if self.total_checks == 0:
            return 100.0
        return ((self.total_checks - self.total_failures) / self.total_checks) * 100

    def to_dict(self) -> dict:
        return {
            "provider": self.provider, "healthy": self.healthy,
            "uptime_pct": round(self.uptime_pct, 1),
            "consecutive_failures": self.consecutive_failures,
            "total_checks": self.total_checks,
            "integrity": round(self.actual_integrity, 3),
            "target_integrity": self.target_integrity,
        }


@dataclass
class MonitorReport:
    """Health check report across all providers."""
    providers: list[ProviderHealth] = field(default_factory=list)
    check_time: float = field(default_factory=time.time)

    @property
    def all_healthy(self) -> bool:
        return all(p.healthy for p in self.providers)

    @property
    def summary(self) -> str:
        healthy = sum(1 for p in self.providers if p.healthy)
        return f"Monitor: {healthy}/{len(self.providers)} healthy"

    def to_dict(self) -> dict:
        return {
            "all_healthy": self.all_healthy,
            "providers": [p.to_dict() for p in self.providers],
        }


class ProviderMonitor:
    """Monitors provider health with state tracking."""

    def __init__(self, registry=None):
        self._registry = registry
        self._health: dict[str, ProviderHealth] = {}

    def check_all(self) -> MonitorReport:
        """Check health of all registered providers."""
        if not self._registry:
            return MonitorReport()

        for provider in self._registry.list_providers():
            name = provider.name
            if name not in self._health:
                self._health[name] = ProviderHealth(provider=name)

            health = self._health[name]
            health.total_checks += 1
            health.last_check = time.time()

            is_healthy = bool(provider.models) or provider.healthy
            if is_healthy:
                health.healthy = True
                health.last_success = time.time()
                health.consecutive_failures = 0
            else:
                health.consecutive_failures += 1
                health.total_failures += 1
                if health.consecutive_failures >= 3:
                    health.healthy = False

        return MonitorReport(providers=list(self._health.values()))

    def get_status(self, provider: str = "") -> dict:
        """Get health status for one or all providers."""
        if provider:
            h = self._health.get(provider)
            return h.to_dict() if h else {"provider": provider, "healthy": False, "error": "not monitored"}
        return {name: h.to_dict() for name, h in self._health.items()}

    def get_healthy_providers(self) -> list[str]:
        """List names of currently healthy providers."""
        return [name for name, h in self._health.items() if h.healthy]

    def heal_tick(self) -> None:
        """One healing cycle: move actual integrity toward target.

        Call periodically (e.g. every health check cycle). Each tick
        moves actual_integrity closer to target_integrity by regen_rate,
        never overshooting. Providers that haven't failed stay at 1.0.
        """
        for health in self._health.values():
            if health.actual_integrity < health.target_integrity:
                health.actual_integrity = min(
                    health.target_integrity,
                    health.actual_integrity + health.regen_rate,
                )

    def record_failure(self, provider: str) -> None:
        """Record a provider failure: decay integrity by decay_rate.

        Called by Sentinel or any component that detects a provider error.
        Integrity never drops below 0.0.
        """
        health = self._health.get(provider)
        if health is None:
            health = ProviderHealth(provider=provider)
            self._health[provider] = health
        health.actual_integrity = max(0.0, health.actual_integrity - health.decay_rate)
        health.consecutive_failures += 1
        health.total_failures += 1
        health.total_checks += 1
        if health.consecutive_failures >= 3:
            health.healthy = False

    def get_integrity(self, provider: str) -> float:
        """Get current integrity for a provider (0.0-1.0)."""
        health = self._health.get(provider)
        return health.actual_integrity if health else 1.0
