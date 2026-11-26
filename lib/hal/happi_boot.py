"""HAPPI Boot — Python substrate for the HAPPI.md polyglot boot document.

HAPPI.md is a single file that is simultaneously valid Markdown, executable
bash, and a HAPPI/1.0 boot envelope. This module provides the testable Python
backend that the bash script delegates to for non-trivial logic.

"AI is a syscall." — any machine with bash + python3 + ~/.hal can boot HAL
by running `bash HAPPI.md`. This module makes that behaviour verifiable.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HAL_ROOT = Path(os.environ.get("HAL_ROOT", Path.home() / ".hal"))
CLAUDE_DIR = Path(os.environ.get("CLAUDE_DIR", Path.home() / ".claude"))

_MARKER_DIR = HAL_ROOT / "state" / "boot-markers"
_CAPSULE_DIR = HAL_ROOT / "state" / "witness-capsules"
_WATERMARK = "V>>--<<V"

# Advisory warmup threshold (ms). Not enforced — measurement forms a baseline.
WARMUP_ADVISORY_MS = 5_000


# ---------------------------------------------------------------------------
# Boot marker — idempotency token written once per calendar day
# ---------------------------------------------------------------------------

@dataclass
class BootMarker:
    """Idempotency marker written by each HAPPI.md run.

    One marker per calendar day. Re-running HAPPI.md the same day is a no-op
    unless --force is given; this class owns that contract.
    """
    date: str
    capsule: str
    pass_count: int
    fail_count: int
    warmup_ms: int
    polyglot_sha256: str = ""
    watermark: str = _WATERMARK

    @classmethod
    def read(cls, path: Path) -> Optional["BootMarker"]:
        """Return a BootMarker loaded from *path*, or None if absent/invalid."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls(
                date=data["date"],
                capsule=data["capsule"],
                pass_count=int(data["pass"]),
                fail_count=int(data["fail"]),
                warmup_ms=int(data["warmup_ms"]),
                polyglot_sha256=data.get("polyglot_sha256", ""),
                watermark=data.get("watermark", _WATERMARK),
            )
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return None

    def write(self, path: Path) -> None:
        """Persist this marker to *path* (parent dirs created if absent)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "date": self.date,
            "capsule": self.capsule,
            "pass": self.pass_count,
            "fail": self.fail_count,
            "warmup_ms": self.warmup_ms,
            "polyglot_sha256": self.polyglot_sha256,
            "watermark": self.watermark,
        }, indent=2))

    @classmethod
    def for_today(
        cls,
        *,
        capsule: str,
        pass_count: int,
        fail_count: int,
        warmup_ms: int,
        polyglot_sha256: str = "",
    ) -> "BootMarker":
        """Factory: produce a marker stamped with today's date."""
        from datetime import date
        return cls(
            date=date.today().isoformat(),
            capsule=capsule,
            pass_count=pass_count,
            fail_count=fail_count,
            warmup_ms=warmup_ms,
            polyglot_sha256=polyglot_sha256,
        )

    @property
    def all_passed(self) -> bool:
        return self.fail_count == 0

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "capsule": self.capsule,
            "pass": self.pass_count,
            "fail": self.fail_count,
            "warmup_ms": self.warmup_ms,
            "polyglot_sha256": self.polyglot_sha256,
            "watermark": self.watermark,
        }


def marker_path_for_today(marker_dir: Path = _MARKER_DIR) -> Path:
    """Return the canonical marker path for today's date."""
    from datetime import date
    return marker_dir / f"{date.today().isoformat()}.json"


def is_already_run_today(marker_dir: Path = _MARKER_DIR) -> bool:
    """True when a boot marker for today already exists (idempotency gate)."""
    return marker_path_for_today(marker_dir).exists()


# ---------------------------------------------------------------------------
# Capsule discovery
# ---------------------------------------------------------------------------

def find_latest_capsule(capsule_dir: Path = _CAPSULE_DIR) -> Optional[Path]:
    """Return the most-recently-modified .md capsule, or None if dir is empty."""
    if not capsule_dir.is_dir():
        return None
    capsules = sorted(
        capsule_dir.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return capsules[0] if capsules else None


def is_dry_run_capsule(capsule_path: Path) -> bool:
    """Return True when *capsule_path* was written in dry-run preview mode."""
    try:
        content = capsule_path.read_text(encoding="utf-8", errors="replace")
        return bool(re.search(r"\*\*Mode:\*\*.*DRY-RUN", content))
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Prediction results
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Outcome of verifying one prediction from a witness capsule."""
    label: str
    passed: bool
    got: str
    expected: str

    @classmethod
    def pass_(cls, label: str, got: str, expected: str) -> "PredictionResult":
        return cls(label=label, passed=True, got=got, expected=expected)

    @classmethod
    def fail(cls, label: str, got: str, expected: str) -> "PredictionResult":
        return cls(label=label, passed=False, got=got, expected=expected)


# ---------------------------------------------------------------------------
# Boot report
# ---------------------------------------------------------------------------

@dataclass
class BootReport:
    """Structured summary of a single HAPPI.md boot run."""
    date: str
    capsule: Optional[str]
    predictions: list[PredictionResult] = field(default_factory=list)
    warmup_ms: int = -1
    is_dry_run: bool = False
    idempotent_skip: bool = False
    force: bool = False
    dry_mode: bool = False

    @property
    def pass_count(self) -> int:
        return sum(1 for p in self.predictions if p.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for p in self.predictions if not p.passed)

    @property
    def exit_code(self) -> int:
        """Canonical exit code matching HAPPI.md bash contract.

        0 — all passed, or idempotent no-op
        1 — one or more predictions falsified
        2 — no witness capsule found
        """
        if self.idempotent_skip:
            return 0
        if self.capsule is None:
            return 2
        if self.fail_count > 0 and not self.is_dry_run:
            return 1
        return 0

    @property
    def warmup_under_advisory(self) -> bool:
        return 0 <= self.warmup_ms < WARMUP_ADVISORY_MS

    def summary_line(self) -> str:
        """Single-line summary for the glanceable report."""
        if self.idempotent_skip:
            return f"already ran today ({self.date}) — no-op"
        suffix = " (dry-run capsule — informational)" if self.is_dry_run else ""
        return f"{self.pass_count} pass · {self.fail_count} fail{suffix}"


# ---------------------------------------------------------------------------
# HAL warmup measurement
# ---------------------------------------------------------------------------

def measure_hal_warmup(hal_root: Path = HAL_ROOT) -> int:
    """Time `lib.hal` + `lib.hal.registry` module import from *hal_root*.

    Returns elapsed milliseconds, or -1 on import failure or missing path.
    Validates that hal_root contains lib/hal/__init__.py before importing —
    this prevents sys.modules cache from masking a genuinely invalid root.
    """
    import sys
    import importlib

    if not (hal_root / "lib" / "hal" / "__init__.py").exists():
        return -1

    root_str = str(hal_root)
    inserted = root_str not in sys.path
    if inserted:
        sys.path.insert(0, root_str)
    try:
        t0 = time.monotonic()
        importlib.import_module("lib.hal")
        importlib.import_module("lib.hal.registry")
        return int((time.monotonic() - t0) * 1000)
    except Exception:
        return -1
    finally:
        if inserted and root_str in sys.path:
            sys.path.remove(root_str)


# ---------------------------------------------------------------------------
# Polyglot document validation
# ---------------------------------------------------------------------------

def validate_polyglot(happi_path: Path) -> dict[str, bool]:
    """Verify that *happi_path* is simultaneously valid Markdown and valid bash.

    Returns a dict with keys:
      - "has_markdown_headers": True if the file contains at least one # heading
      - "has_bash_shebang": True if the first line is a bash shebang
      - "bash_syntax_ok": True if `bash -n <path>` reports no syntax errors
      - "has_happi_envelope": True if the HAPPI/1.0 capability block is present
      - "has_watermark": True if "V>>--<<V" appears in the file
    """
    import subprocess

    try:
        content = happi_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    lines = content.splitlines()

    has_markdown_headers = any(line.startswith("#") for line in lines)
    has_bash_shebang = lines[0].startswith("#!/") if lines else False
    has_happi_envelope = '"v": "happi/1.0"' in content
    has_watermark = _WATERMARK in content

    try:
        result = subprocess.run(
            ["bash", "-n", str(happi_path)],
            capture_output=True,
            timeout=5,
        )
        bash_syntax_ok = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        bash_syntax_ok = False

    return {
        "has_markdown_headers": has_markdown_headers,
        "has_bash_shebang": has_bash_shebang,
        "bash_syntax_ok": bash_syntax_ok,
        "has_happi_envelope": has_happi_envelope,
        "has_watermark": has_watermark,
    }


# ---------------------------------------------------------------------------
# Capsule prediction extraction
# ---------------------------------------------------------------------------

def extract_memory_prediction(capsule_path: Path) -> Optional[int]:
    """Extract the predicted active-memory cap (MB) from a capsule.

    Capsules embed a line like: "active memory at boot + 60 s < 8000 MB"
    Returns the integer cap, or None if the pattern is not found.
    """
    try:
        content = capsule_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"boot \+ 60 s < (\d{2,})", content)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Bootstrap — fresh-machine HAL installation and prerequisite checks
# ---------------------------------------------------------------------------

_MIN_PYTHON = (3, 10)
_HAL_REPO_URL = "https://github.com/grip-platform/HAL.git"
_REQUIRED_PACKAGES = ["anthropic", "rich", "httpx"]
_API_KEY_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
]


@dataclass
class PrerequisiteCheck:
    """Result of verifying one bootstrap prerequisite."""
    name: str
    required: str
    found: str
    ok: bool


@dataclass
class BootstrapResult:
    """Outcome of a bootstrap_hal() call.

    ok is True only when all prerequisite checks pass and no errors occurred.
    """
    hal_root: Path
    already_installed: bool
    prereqs: list[PrerequisiteCheck] = field(default_factory=list)
    api_keys_found: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(p.ok for p in self.prereqs) and not self.errors

    @property
    def missing_prereqs(self) -> list[PrerequisiteCheck]:
        return [p for p in self.prereqs if not p.ok]

    def summary(self) -> str:
        parts: list[str] = []
        status = "installed" if self.already_installed else "fresh machine"
        parts.append(f"HAL root: {self.hal_root} ({status})")
        for p in self.prereqs:
            mark = "✓" if p.ok else "✗"
            parts.append(f"  {mark} {p.name}: {p.found}")
        if self.api_keys_found:
            parts.append(f"  keys: {', '.join(self.api_keys_found)}")
        else:
            parts.append("  keys: none configured — set at least one API key")
        if self.errors:
            for e in self.errors:
                parts.append(f"  ! {e}")
        return "\n".join(parts)


def check_prerequisites(hal_root: Path = HAL_ROOT) -> list[PrerequisiteCheck]:
    """Return prerequisite checks for Python version, pip, and git."""
    import sys
    import shutil

    checks: list[PrerequisiteCheck] = []

    # Python version
    major, minor = sys.version_info[:2]
    min_str = f"{_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}"
    checks.append(PrerequisiteCheck(
        name="python",
        required=f">= {min_str}",
        found=f"{major}.{minor}",
        ok=(major, minor) >= _MIN_PYTHON,
    ))

    # pip
    import importlib.util
    pip_ok = importlib.util.find_spec("pip") is not None
    checks.append(PrerequisiteCheck(
        name="pip",
        required="available",
        found="present" if pip_ok else "missing",
        ok=pip_ok,
    ))

    # git
    git_path = shutil.which("git")
    checks.append(PrerequisiteCheck(
        name="git",
        required="available",
        found=git_path or "not found",
        ok=git_path is not None,
    ))

    return checks


def check_api_keys() -> list[str]:
    """Return list of API key env var names that are set to non-empty values."""
    return [k for k in _API_KEY_ENV_VARS if os.environ.get(k, "").strip()]


def bootstrap_hal(
    hal_root: Path = HAL_ROOT,
    *,
    dry_run: bool = False,
    repo_url: str = _HAL_REPO_URL,
) -> BootstrapResult:
    """Bootstrap HAL on a fresh machine or validate an existing installation.

    On a fresh machine (hal_root absent):
      1. Verify prerequisites (Python >= 3.10, pip, git).
      2. Clone repo_url to hal_root (unless dry_run or git missing).
      3. Install _REQUIRED_PACKAGES via pip (unless dry_run or pip missing).
      4. Detect configured API keys.

    On an existing installation (hal_root present):
      1. Verify prerequisites are still met.
      2. Detect configured API keys.
      3. Return immediately — morning validation handles the rest.

    Returns a BootstrapResult whose .ok reflects whether all checks passed.
    """
    import subprocess

    already_installed = hal_root.is_dir()
    prereqs = check_prerequisites(hal_root)
    api_keys = check_api_keys()
    errors: list[str] = []

    def _result(errs: Optional[list[str]] = None) -> BootstrapResult:
        return BootstrapResult(
            hal_root=hal_root,
            already_installed=already_installed,
            prereqs=prereqs,
            api_keys_found=api_keys,
            errors=errs if errs is not None else errors,
        )

    if already_installed or dry_run:
        return _result([])

    missing = [p for p in prereqs if not p.ok]
    if missing:
        errors.append("Prerequisites not met: " + ", ".join(p.name for p in missing))
        return _result()

    # Clone
    git_check = next((p for p in prereqs if p.name == "git"), None)
    if git_check and git_check.ok:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(hal_root)],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            errors.append(f"git clone failed: {e}")

    # Install pip deps
    if not errors:
        pip_check = next((p for p in prereqs if p.name == "pip"), None)
        if pip_check and pip_check.ok:
            try:
                import sys as _sys
                subprocess.run(
                    [_sys.executable, "-m", "pip", "install", "--quiet",
                     *_REQUIRED_PACKAGES],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
                errors.append(f"pip install failed: {e}")

    return _result()
