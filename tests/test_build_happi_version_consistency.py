"""Regression tests for the HAPPI_VERSION consistency check (issue #516).

happi.md is a polyglot. The bash prelude and the embedded Python runtime each
declare HAPPI_VERSION separately, in their own syntax, and nothing in either
language forces them to agree. They HAVE diverged: before PR #515 the prelude
said happi/1.2 while the Python runtime said happi/1.3 and stamped 1.3 onto
every event, so `bash happi.md version` and the events it produced disagreed
about what the runtime was.

The round-trip identity gate structurally cannot catch that. It compares the
assembled happi.md against the sources it was assembled from, and in that
incident both sides were equally wrong -- the drift was internal to the sources.
This check asserts the property round-trip cannot: that the constant means ONE
thing.

The Goodhart property is test_agreeing_versions_pass. The check must fire ONLY
on genuine disagreement, never on the many REFERENCES to HAPPI_VERSION scattered
through the Python layer. Weakening the declaration regex to match references
makes that test fail; removing the check makes the divergence tests fail.
"""
import importlib.util
from pathlib import Path

import pytest

from tests._public_tree import reason, withheld

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build-happi.py"
# Public HAL ships only install-starter.sh from scripts/, so on a public checkout
# scripts/build-happi.py is absent and every test here is skipped with that reason (HAL#859);
# in the full repository they all run. withheld(), not folder_absent(): scripts/
# exists on a public checkout now, so a folder-level check would never skip.
pytestmark = pytest.mark.skipif(withheld("scripts/build-happi.py"), reason=reason("scripts/build-happi.py"))


def _load():
    spec = importlib.util.spec_from_file_location("build_happi", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _regions(bash_version="happi/1.3", py_version="happi/1.3",
             accepted='("happi/1.0", "happi/1.1", "happi/1.2", "happi/1.3")'):
    """Minimal layer set carrying only what the version check reads."""
    return {
        "01-shebang.bash": "#!/usr/bin/env bash",
        "02-markdown-body.md": "# HAPPI\nbody",
        "03-bash-prelude.sh": f'HAPPI_VERSION="{bash_version}"',
        "04-python-runtime.py": (
            f'HAPPI_VERSION = "{py_version}"\n'
            f"_ACCEPTED_VERSIONS = {accepted}\n"
        ),
        "05-bash-dispatcher.sh": 'echo "hi"',
    }


def test_agreeing_versions_pass():
    """The Goodhart anchor: no complaint when the declarations agree.

    Guards against a check that fires on every build by mistaking the Python
    layer's dozen REFERENCES to HAPPI_VERSION for extra declarations.
    """
    mod = _load()
    assert mod._check_version_consistency(_regions()) == []


def test_references_are_not_declarations():
    """Uses of the constant must not be read as competing sources of truth."""
    mod = _load()
    regions = _regions()
    regions["04-python-runtime.py"] += (
        '\n_emit({"v": HAPPI_VERSION, "type": "started"})\n'
        "    emit_delta(env, HAPPI_VERSION)\n"
    )
    assert mod._check_version_consistency(regions) == []


def test_catches_the_real_pre_pr515_divergence():
    """Replay of the actual incident: prelude on 1.2, Python runtime on 1.3."""
    mod = _load()
    failures = mod._check_version_consistency(
        _regions(bash_version="happi/1.2", py_version="happi/1.3")
    )
    assert len(failures) == 1
    assert "disagrees across layers" in failures[0]
    # Both sides named, so the message tells you where to look.
    assert "03-bash-prelude.sh" in failures[0]
    assert "04-python-runtime.py" in failures[0]
    assert "happi/1.2" in failures[0] and "happi/1.3" in failures[0]


def test_catches_bump_that_forgets_accepted_versions():
    """A runtime must accept its own version, or it rejects what it stamps."""
    mod = _load()
    failures = mod._check_version_consistency(
        _regions(bash_version="happi/1.4", py_version="happi/1.4")
    )
    assert len(failures) == 1
    assert "_ACCEPTED_VERSIONS" in failures[0]
    assert "happi/1.4" in failures[0]


def test_missing_declaration_is_a_failure_not_a_silent_pass():
    """No declaration anywhere means the check could not verify anything.

    Reporting that as success would make the gate unfalsifiable -- it would pass
    hardest precisely when it had the least to go on.
    """
    mod = _load()
    regions = _regions()
    regions["03-bash-prelude.sh"] = "set -euo pipefail"
    regions["04-python-runtime.py"] = "x = 1\n"
    failures = mod._check_version_consistency(regions)
    assert len(failures) == 1
    assert "no HAPPI_VERSION declaration" in failures[0]


def test_real_sources_are_consistent():
    """The checked-in happi-sources/ must satisfy the invariant it now enforces."""
    mod = _load()
    assert mod._check_version_consistency(mod.read_sources()) == []
