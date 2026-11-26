"""Regression tests for ``hal --tools``.

The flag is a thin capability-reflection surface. These tests pin:

- The argv parser recognises ``--tools`` (regresses if we drop the arg key)
- ``_show_tools`` output actually lists every tool in ``_BUILTIN_SPECS``
  (Goodhart anchor — a stub that printed only a header would "pass" weaker
  assertions)
- Parallel-vs-serial semantics are communicated in the output so operators can
  reason about concurrency without reading source
- Required input-schema fields are rendered per tool (so operators don't have
  to guess what ``Bash`` needs)
"""
import io
import os
import sys
from contextlib import redirect_stdout

import pytest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.cli import _parse_args, _show_tools
from lib.hal.tool_adapter import _BUILTIN_SPECS


@pytest.fixture(autouse=True)
def _restore_argv():
    saved = sys.argv[:]
    yield
    sys.argv = saved


def _capture_tools_output() -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _show_tools()
    return buf.getvalue()


class TestToolsArgParsing:
    def test_tools_flag_recognised(self):
        sys.argv = ["hal", "--tools"]
        args = _parse_args()
        assert args["tools"] is True

    def test_tools_flag_absent_by_default(self):
        sys.argv = ["hal"]
        args = _parse_args()
        assert args["tools"] is False

    def test_tools_flag_does_not_consume_extra(self):
        sys.argv = ["hal", "--tools", "--model", "groq/llama"]
        args = _parse_args()
        assert args["tools"] is True
        assert args["model"] == "groq/llama"


class TestShowToolsOutput:
    def test_lists_every_builtin_tool(self):
        output = _capture_tools_output()
        for spec in _BUILTIN_SPECS:
            assert spec.name in output, (
                f"Tool {spec.name!r} missing from --tools output. "
                "Operators rely on this for capability reflection."
            )

    def test_shows_concurrency_tag_per_tool(self):
        """Operators must see parallel-vs-serial — affects batching assumptions."""
        output = _capture_tools_output()
        assert "parallel" in output
        assert "serial" in output

    def test_shows_total_count(self):
        """Header must reflect actual tool count, not a hardcoded number.

        Goodhart anchor — if someone hardcodes '8 built-in tools' and the spec
        table shrinks to 6, the output would lie. This test binds the header
        to ``_BUILTIN_SPECS`` length.
        """
        output = _capture_tools_output()
        assert f"{len(_BUILTIN_SPECS)} built-in tools" in output

    def test_shows_required_input_fields(self):
        """Each tool's required inputs must appear — sampled to avoid brittleness."""
        output = _capture_tools_output()
        # Sample three that have distinct required sets
        assert "command" in output  # Bash
        assert "file_path" in output  # Read / Write / Edit
        assert "pattern" in output  # Glob / Grep

    def test_parallel_tools_listed_before_serial(self):
        """Sort order matters — operators scanning top-down should see fast
        tools first. If this flips, concurrency intuition inverts."""
        output = _capture_tools_output()
        read_pos = output.index("Read")
        bash_pos = output.index("Bash")
        assert read_pos < bash_pos, (
            "Parallel-safe tools should render before serial tools so the "
            "operator can reason about batching from the top of the list."
        )
