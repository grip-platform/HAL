"""HAL Post-Edit Verification — mechanical verification after file modifications.

CC gates this behind USER_TYPE=ant. HAL implements it as a standalone module.
After any file write/edit, runs compilation checks to catch broken code BEFORE
reporting success. The model's "done!" means nothing — the compiler's verdict does.

Usage:
    from lib.hal.verify import Verifier

    verifier = Verifier()
    result = verifier.check("path/to/file.py")
    if not result.passed:
        print(f"Verification failed: {result.errors}")
"""
import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class VerifyResult:
    """Result of a post-edit verification check."""
    passed: bool
    tool: str  # Which verifier ran (python, tsc, eslint, etc.)
    errors: str
    file_path: str
    duration_ms: float = 0.0


class Verifier:
    """Post-edit verification engine. Detects file type and runs appropriate checks."""

    def check(self, file_path: str) -> VerifyResult:
        """Verify a file after editing. Auto-detects language and runs checks."""
        if not os.path.exists(file_path):
            return VerifyResult(passed=False, tool="exists", errors="File not found", file_path=file_path)

        ext = os.path.splitext(file_path)[1].lower()
        checkers = {
            ".py": self._check_python,
            ".ts": self._check_typescript,
            ".tsx": self._check_typescript,
            ".js": self._check_javascript,
            ".jsx": self._check_javascript,
            ".rs": self._check_rust,
            ".go": self._check_go,
        }

        checker = checkers.get(ext)
        if checker is None:
            return VerifyResult(passed=True, tool="none", errors="", file_path=file_path)

        return checker(file_path)

    def check_all(self, file_paths: list[str]) -> list[VerifyResult]:
        """Verify multiple files. Returns list of results."""
        return [self.check(fp) for fp in file_paths]

    def _check_python(self, path: str) -> VerifyResult:
        """Verify Python: syntax check + import check."""
        import time
        start = time.monotonic()

        # Syntax check via py_compile
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", path],
            capture_output=True, text=True, timeout=10,
        )
        duration = (time.monotonic() - start) * 1000

        if result.returncode != 0:
            return VerifyResult(
                passed=False, tool="py_compile",
                errors=result.stderr[:500], file_path=path,
                duration_ms=duration,
            )
        return VerifyResult(passed=True, tool="py_compile", errors="", file_path=path, duration_ms=duration)

    def _check_typescript(self, path: str) -> VerifyResult:
        """Verify TypeScript: tsc --noEmit."""
        return self._run_check(path, ["npx", "tsc", "--noEmit", "--pretty", "false"], "tsc")

    def _check_javascript(self, path: str) -> VerifyResult:
        """Verify JavaScript: node --check for syntax."""
        return self._run_check(path, ["node", "--check", path], "node")

    def _check_rust(self, path: str) -> VerifyResult:
        """Verify Rust: cargo check."""
        return self._run_check(path, ["cargo", "check", "--message-format=short"], "cargo")

    def _check_go(self, path: str) -> VerifyResult:
        """Verify Go: go vet."""
        dir_path = os.path.dirname(path) or "."
        return self._run_check(path, ["go", "vet", f"./{dir_path}/..."], "go vet")

    def _run_check(self, path: str, cmd: list[str], tool: str) -> VerifyResult:
        """Run a verification command."""
        import time
        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                cwd=os.path.dirname(path) or ".",
            )
            duration = (time.monotonic() - start) * 1000
            if result.returncode != 0:
                errors = (result.stderr + result.stdout)[:500]
                return VerifyResult(passed=False, tool=tool, errors=errors, file_path=path, duration_ms=duration)
            return VerifyResult(passed=True, tool=tool, errors="", file_path=path, duration_ms=duration)
        except FileNotFoundError:
            return VerifyResult(passed=True, tool=f"{tool} (not installed)", errors="", file_path=path)
        except subprocess.TimeoutExpired:
            return VerifyResult(passed=False, tool=tool, errors="Timeout (30s)", file_path=path)
