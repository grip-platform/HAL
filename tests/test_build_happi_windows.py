"""Cross-platform regression tests for scripts/build-happi.py (task #2190).

The reported bug: `--build` tracebacks on Windows because validate_layers ran
`subprocess.run(["bash", -n, ...])` with no guard — bash is not on PATH on a
default Windows host, so it raised an unhandled FileNotFoundError.

The Goodhart property is test_check_bash_present_still_validates: the fix must
skip ONLY on bash-unavailability (OSError — FileNotFoundError when bash is
absent, PermissionError when bash exists but is not executable), never neuter
the real syntax check. Removing the OSError guard makes the absence/permission
tests fail; over-broadening it to swallow real failures makes the present-bash
test fail.
"""
import importlib.util
from pathlib import Path

import pytest

from tests._public_tree import reason, withheld

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build-happi.py"
# Public HAL leaves build-happi.py out of scripts/ (which otherwise ships only
# install-starter.sh), so on a public checkout every test here is skipped with
# that reason (HAL#859); in the full repository they all run. withheld(), not
# the coarser folder_absent(), because scripts/ itself now exists on a public
# checkout (it ships install-starter.sh) -- folder_absent("scripts") would be
# False there and this skip would never fire.
pytestmark = pytest.mark.skipif(
    withheld("scripts/build-happi.py"), reason=reason("scripts/build-happi.py")
)

# A genuinely valid layer set, which means it must declare HAPPI_VERSION in both
# languages and accept its own version -- validate_layers enforces that (#516).
# This fixture previously carried "x = 1" for the runtime layer, an invented
# minimal shape rather than one derived from the real sources, so it did not
# represent anything the build could actually produce.
_VALID = {
    "01-shebang.bash": "#!/usr/bin/env bash",
    "02-markdown-body.md": "# HAPPI\nbody",
    "03-bash-prelude.sh": 'set -euo pipefail\nHAPPI_VERSION="happi/1.3"',
    "04-python-runtime.py": (
        'HAPPI_VERSION = "happi/1.3"\n'
        '_ACCEPTED_VERSIONS = ("happi/1.0", "happi/1.1", "happi/1.2", "happi/1.3")\n'
        "x = 1\n"
    ),
    "05-bash-dispatcher.sh": 'echo "hi"',
}


def _load():
    spec = importlib.util.spec_from_file_location("build_happi", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_bash_skips_when_bash_absent(monkeypatch):
    """THE Windows fix: bash-not-on-PATH is skipped, not tracebacked."""
    mod = _load()

    def _raise(*a, **k):
        raise FileNotFoundError("[WinError 2] bash not found")

    monkeypatch.setattr(mod.subprocess, "run", _raise)
    assert mod._check_bash(_VALID) == []  # skipped, no exception


def test_validate_layers_survives_missing_bash(monkeypatch, capsys):
    """The whole validate path must not traceback when bash is absent, AND the
    bash-skip warning must fire — proving the FileNotFoundError guard branch
    actually executed, not merely that validate_layers happens not to sys.exit."""
    mod = _load()

    def _raise(*a, **k):
        raise FileNotFoundError("bash")

    monkeypatch.setattr(mod.subprocess, "run", _raise)
    mod.validate_layers(_VALID)  # must not raise (FileNotFoundError caught in _check_bash)
    assert "bash unavailable" in capsys.readouterr().err  # the guard branch ran


def test_check_bash_present_still_validates(monkeypatch):
    """Goodhart guard: when bash IS present and reports a syntax error, it still fails."""
    mod = _load()

    class _R:
        returncode = 2
        stderr = "line 3: syntax error"

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _R())
    failures = mod._check_bash(_VALID)
    assert failures and "bash -n" in failures[0]


def test_check_bash_skips_when_bash_not_executable(monkeypatch, capsys):
    """AGORA #487 falsifier: bash PRESENT but not executable raises PermissionError
    (not FileNotFoundError) on Windows. The OSError guard must skip it too, not
    traceback — this is exactly the crash path a FileNotFoundError-only guard missed."""
    mod = _load()

    def _raise(*a, **k):
        raise PermissionError("[WinError 5] Access is denied: 'bash.exe'")

    monkeypatch.setattr(mod.subprocess, "run", _raise)
    assert mod._check_bash(_VALID) == []  # skipped, no exception
    mod.validate_layers(_VALID)  # the whole validate path must not traceback either
    assert "bash unavailable" in capsys.readouterr().err  # the guard branch ran


def test_check_bash_returncode_zero_yields_no_failures(monkeypatch):
    """Negative Goodhart guard: bash present, clean syntax (returncode 0) => [].
    The OSError broadening must not turn a genuine pass into a spurious failure."""
    mod = _load()

    class _R:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _R())
    assert mod._check_bash(_VALID) == []
