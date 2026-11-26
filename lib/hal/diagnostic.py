"""HAL Diagnostic Mode — automated issue detection and resolution.

When something goes wrong (provider errors, import failures, config issues),
HAL's diagnostic engine runs targeted checks and suggests fixes.

Usage:
    from lib.hal.diagnostic import DiagnosticEngine

    engine = DiagnosticEngine()
    report = engine.run_all()
    for issue in report.issues:
        print(f"[{issue.severity}] {issue.description}")
        print(f"  Fix: {issue.suggested_fix}")
"""
import importlib
import os
import subprocess
from dataclasses import dataclass, field

from lib.hal.home import HAL_HOME as _HAL_HOME


@dataclass(frozen=True)
class DiagnosticIssue:
    """A detected issue with a suggested fix."""
    category: str  # "config", "dependency", "provider", "filesystem", "module"
    severity: str  # "info", "warning", "error"
    description: str
    suggested_fix: str


@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""
    issues: list[DiagnosticIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def summary(self) -> str:
        errors = sum(1 for i in self.issues if i.severity == "error")
        warnings = sum(1 for i in self.issues if i.severity == "warning")
        if not self.issues:
            return "Diagnostics: all clear"
        return f"Diagnostics: {errors} errors, {warnings} warnings, {len(self.issues)} total"


class DiagnosticEngine:
    """Runs automated diagnostic checks."""

    def run_all(self) -> DiagnosticReport:
        """Run all diagnostic checks."""
        issues: list[DiagnosticIssue] = []
        issues.extend(_check_hal_home())
        issues.extend(_check_core_modules())
        issues.extend(_check_config_files())
        issues.extend(_check_git())
        issues.extend(_check_python_version())
        return DiagnosticReport(issues=issues)


def _issue(cat: str, sev: str, desc: str, fix: str) -> DiagnosticIssue:
    """Build a DiagnosticIssue — single construction point (DRY)."""
    return DiagnosticIssue(category=cat, severity=sev, description=desc, suggested_fix=fix)


def _check_hal_home() -> list[DiagnosticIssue]:
    """Check HAL_HOME directory structure."""
    issues = [
        _issue("filesystem", "warning", f"Directory missing: {d}", f"mkdir -p {os.path.join(_HAL_HOME, d)}")
        for d in ("lib/hal", "tests", "config", "state")
        if not os.path.isdir(os.path.join(_HAL_HOME, d))
    ]
    if not os.access(_HAL_HOME, os.W_OK):
        issues.append(_issue("filesystem", "error", f"HAL_HOME ({_HAL_HOME}) is not writable", f"chmod u+w {_HAL_HOME}"))
    return issues


def _check_core_modules() -> list[DiagnosticIssue]:
    """Check that core HAL modules can be imported."""
    issues: list[DiagnosticIssue] = []
    for mod in ("lib.hal.home", "lib.hal.types", "lib.hal.registry", "lib.hal.conversation", "lib.hal.cost_tracker"):
        try:
            importlib.import_module(mod)
        except ImportError as e:
            issues.append(_issue("module", "error", f"Cannot import {mod}: {e}", f"Check {mod.replace('.', '/')}.py exists"))
    return issues


def _check_config_files() -> list[DiagnosticIssue]:
    """Check for expected config files."""
    checks = [
        (os.path.join(_HAL_HOME, "HAL.md"), "HAL.md not found — using defaults", "Create HAL.md with HAL configuration"),
        (os.path.join(_HAL_HOME, "config", "providers.json"), "No providers.json — auto-discovery only", "Configure providers in config/providers.json"),
    ]
    return [_issue("config", "info", desc, fix) for path, desc, fix in checks if not os.path.exists(path)]


def _check_git() -> list[DiagnosticIssue]:
    """Check git repository status."""
    if not os.path.isdir(os.path.join(_HAL_HOME, ".git")):
        return [_issue("config", "info", "HAL_HOME is not a git repository", "git init to enable version tracking")]
    try:
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10, cwd=_HAL_HOME)
        if result.returncode == 0 and result.stdout.strip():
            count = len(result.stdout.strip().split("\n"))
            return [_issue("config", "info", f"{count} uncommitted changes in HAL_HOME", "git add -A && git commit")]
    except (subprocess.TimeoutExpired, OSError):
        pass
    return []


def _check_python_version() -> list[DiagnosticIssue]:
    """Check Python version compatibility."""
    import sys
    if sys.version_info < (3, 10):
        v = f"{sys.version_info.major}.{sys.version_info.minor}"
        return [_issue("dependency", "error", f"Python {v} detected — HAL requires 3.10+", "Install Python 3.10+")]
    return []
