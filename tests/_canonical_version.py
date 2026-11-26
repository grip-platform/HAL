"""The version the CANONICAL polyglot runtime stamps, read from its source.

Shared by every test that asserts the lib's emission version. Deliberately an
INDEPENDENT source: reading the expected value out of the module under test
would make the assertion unable to fail (tests-must-be-able-to-fail.md shape 3),
since `_resolve_protocol_version()` returns `PROTOCOL_VERSION_DEFAULT` by
construction when the env var is unset.

This encodes H428's actual invariant — "the lib runtime's emission version
matches the canonical runtime's emission version" (plans/happi-lib-v11-alignment-
ultraplan.md:376). The tests that used to pin the literal "happi/1.1" pinned the
PARENTHETICAL EXAMPLE in that hypothesis rather than its claim, so when the
canonical runtime moved 1.1 -> 1.4 they demanded the stale value and made the
correct fix look like a regression.
"""
from __future__ import annotations

from pathlib import Path

from tests._public_tree import require

_RUNTIME_SOURCE = (
    Path(__file__).resolve().parents[1] / "happi-sources" / "04-python-runtime.py"
)


def canonical_runtime_version() -> str:
    """Return e.g. ``"happi/1.4"`` — the polyglot runtime's own declaration.

    Reads the single ``HAPPI_VERSION = "..."`` DECLARATION, which
    ``build-happi``'s version-consistency gate already guarantees is unique and
    agrees with the bash prelude.
    """
    require("happi-sources/04-python-runtime.py")
    for line in _RUNTIME_SOURCE.read_text(encoding="utf-8").splitlines():
        if line.startswith("HAPPI_VERSION = "):
            return line.split('"')[1]
    raise AssertionError(f"no HAPPI_VERSION declaration in {_RUNTIME_SOURCE}")
