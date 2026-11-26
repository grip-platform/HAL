"""HAL Health Dashboard — unified system health at a glance.

Aggregates health signals from all HAL subsystems into a single dashboard.
Designed for both CLI display and API response. Checks providers, memory,
disk, and module integrity.

Usage:
    from lib.hal.health import HealthDashboard

    dashboard = HealthDashboard(registry)
    report = dashboard.check()
    print(report.summary)
    for check in report.checks:
        print(f"  [{check.status}] {check.name}: {check.detail}")
"""
import os
import time
from dataclasses import dataclass, field

from lib.hal.home import HAL_HOME as _HAL_HOME


@dataclass(frozen=True)
class HealthCheck:
    """A single health check result."""
    name: str
    status: str  # "ok", "warn", "error"
    detail: str
    duration_ms: int = 0


@dataclass
class HealthReport:
    """Aggregated health report from all subsystems."""
    checks: list[HealthCheck] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def overall(self) -> str:
        """Overall status: worst check wins."""
        statuses = {c.status for c in self.checks}
        if "error" in statuses:
            return "error"
        if "warn" in statuses:
            return "warn"
        return "ok"

    @property
    def summary(self) -> str:
        ok = sum(1 for c in self.checks if c.status == "ok")
        warn = sum(1 for c in self.checks if c.status == "warn")
        err = sum(1 for c in self.checks if c.status == "error")
        return f"Health: {self.overall} ({ok} ok, {warn} warn, {err} error)"

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail, "duration_ms": c.duration_ms}
                for c in self.checks
            ],
            "timestamp": self.timestamp,
        }


class HealthDashboard:
    """Runs all health checks and aggregates results."""

    def __init__(self, registry=None):
        self._registry = registry

    def check(self, reveal_path: bool = True) -> HealthReport:
        """Run all health checks.

        ``reveal_path=False`` redacts the ``hal_home`` check's absolute-path
        detail (see :meth:`_check_hal_home`) for callers outside the
        loopback boundary — e.g. the public, unauthenticated /api/health
        route. Every other check, and ``hal_home``'s ``name``/``status``,
        are unaffected either way. Defaults to ``True`` so direct/CLI use
        (see this module's docstring) and the existing dataclass-level tests
        are unchanged.
        """
        checks = [
            self._check_hal_home(reveal_path=reveal_path),
            self._check_modules(),
            self._check_memory(),
            self._check_state_dir(),
        ]
        if self._registry:
            checks.append(self._check_providers())
        return HealthReport(checks=checks)

    def _check_hal_home(self, reveal_path: bool = True) -> HealthCheck:
        """Verify HAL_HOME exists and is writable.

        ``reveal_path=False`` replaces the literal HAL_HOME path in
        ``detail`` with a generic word, so an absolute filesystem path (and
        the host account name it embeds) is never served to a remote,
        unauthenticated caller. ``name`` and ``status`` are unchanged
        either way, so a monitor alerting on this check's status keeps
        working against either body shape.
        """
        start = time.time()
        if not os.path.isdir(_HAL_HOME):
            detail = f"{_HAL_HOME} does not exist" if reveal_path else "missing"
            return HealthCheck("hal_home", "error", detail)
        if not os.access(_HAL_HOME, os.W_OK):
            detail = f"{_HAL_HOME} is not writable" if reveal_path else "not writable"
            return HealthCheck("hal_home", "error", detail)
        ms = int((time.time() - start) * 1000)
        detail = _HAL_HOME if reveal_path else "configured"
        return HealthCheck("hal_home", "ok", detail, duration_ms=ms)

    def _check_modules(self) -> HealthCheck:
        """Count importable HAL modules."""
        start = time.time()
        module_dir = os.path.join(_HAL_HOME, "lib", "hal")
        if not os.path.isdir(module_dir):
            return HealthCheck("modules", "error", "lib/hal/ not found")
        count = sum(1 for f in os.listdir(module_dir) if f.endswith(".py") and f != "__init__.py")
        ms = int((time.time() - start) * 1000)
        status = "ok" if count >= 10 else "warn"
        return HealthCheck("modules", status, f"{count} modules in lib/hal/", duration_ms=ms)

    def _check_memory(self) -> HealthCheck:
        """Check memory directory health."""
        start = time.time()
        memory_dir = os.path.join(_HAL_HOME, "memory")
        if not os.path.isdir(memory_dir):
            return HealthCheck("memory", "warn", "No memory directory found")
        entries = [f for f in os.listdir(memory_dir) if f.endswith(".md")]
        ms = int((time.time() - start) * 1000)
        return HealthCheck("memory", "ok", f"{len(entries)} memory entries", duration_ms=ms)

    def _check_state_dir(self) -> HealthCheck:
        """Check state directory for disk space."""
        start = time.time()
        state_dir = os.path.join(_HAL_HOME, "state")
        if not os.path.isdir(state_dir):
            return HealthCheck("state", "warn", "No state directory — first run?")
        total_bytes = _dir_size(state_dir)
        mb = total_bytes / (1024 * 1024)
        ms = int((time.time() - start) * 1000)
        status = "warn" if mb > 100 else "ok"
        return HealthCheck("state", status, f"{mb:.1f} MB in state/", duration_ms=ms)

    def _check_providers(self) -> HealthCheck:
        """Check provider availability."""
        start = time.time()
        providers = self._registry.list_providers()
        healthy = sum(1 for p in providers if p.models)
        total = len(providers)
        ms = int((time.time() - start) * 1000)
        if healthy == 0:
            return HealthCheck("providers", "error", "No healthy providers", duration_ms=ms)
        status = "ok" if healthy == total else "warn"
        return HealthCheck("providers", status, f"{healthy}/{total} providers healthy", duration_ms=ms)


def _dir_size(path: str) -> int:
    """Calculate total size of directory in bytes. O(n) single walk."""
    total = 0
    for root, _dirs, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total
