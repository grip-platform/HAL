"""Regression tests for lib/hal/tool_adapter.py.

Closes the gap where ToolExecutor had no concrete ToolProtocol implementations,
so the concurrent-execution layer had nothing to execute. These tests fail if:

- The adapter stops wrapping every tool registered in the native dispatcher
- Concurrency flags drift (Read accidentally marked non-safe, Bash accidentally
  marked safe, etc) — regressions would silently break parallel dispatch OR
  serialise-safe tools
- ``execute`` stops delegating to the dispatcher (Goodhart anchor — a stub that
  always returned an empty ToolOutput would pass weaker checks)
- Dispatcher blocked/error results aren't translated into ``ToolOutput.is_error``
"""
import os
import sys
from unittest import mock

import pytest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.tool_adapter import (
    _BUILTIN_SPECS,
    DispatchAdaptedTool,
    ToolSpec,
    create_builtin_tools,
)
from lib.hal.tool_dispatch import ToolDispatcher, ToolResult
from lib.hal.tool_executor import ToolExecutor
from lib.hal.tool_protocol import ToolContext, ToolOutput, ToolProtocol


class TestBuiltinSpecs:
    """The spec table is the contract between the dispatcher and the executor."""

    def test_spec_count_matches_native_dispatcher(self):
        dispatcher_tools = set(ToolDispatcher().available_tools())
        adapter_tools = {spec.name for spec in _BUILTIN_SPECS}
        assert adapter_tools == dispatcher_tools, (
            "Adapter spec table has drifted from ToolDispatcher._TOOL_REGISTRY. "
            f"Adapter has {adapter_tools}, dispatcher has {dispatcher_tools}."
        )

    @pytest.mark.parametrize(
        "name,expected_safe",
        [
            ("Bash", False),
            ("Write", False),
            ("Edit", False),
            ("Read", True),
            ("Glob", True),
            ("Grep", True),
            ("CodeLookup", True),
            ("CodeSearch", True),
        ],
    )
    def test_concurrency_flag_matches_tool_semantics(self, name, expected_safe):
        spec = next(s for s in _BUILTIN_SPECS if s.name == name)
        assert spec.is_concurrent_safe is expected_safe

    def test_every_spec_has_required_input_schema(self):
        for spec in _BUILTIN_SPECS:
            assert spec.input_schema.get("type") == "object"
            assert "required" in spec.input_schema, spec.name
            assert spec.input_schema["required"], spec.name


class TestDispatchAdaptedTool:
    """Adapter wraps dispatcher — no behaviour of its own."""

    def _make_tool(self, spec: ToolSpec, dispatcher) -> DispatchAdaptedTool:
        return DispatchAdaptedTool(spec, dispatcher)

    def test_name_description_schema_from_spec(self):
        spec = _BUILTIN_SPECS[0]
        dispatcher = mock.Mock(spec=ToolDispatcher)
        tool = self._make_tool(spec, dispatcher)
        assert tool.name == spec.name
        assert tool.description == spec.description
        assert tool.input_schema is spec.input_schema
        assert tool.is_concurrent_safe is spec.is_concurrent_safe

    def test_execute_delegates_to_dispatcher(self):
        """Goodhart anchor — the adapter must actually call the dispatcher,
        not return a canned ToolOutput."""
        spec = next(s for s in _BUILTIN_SPECS if s.name == "Read")
        dispatcher = mock.Mock(spec=ToolDispatcher)
        dispatcher.execute.return_value = ToolResult(output="file contents")
        tool = self._make_tool(spec, dispatcher)

        output = tool.execute({"file_path": "/tmp/x"}, ToolContext(session_id="s"))

        dispatcher.execute.assert_called_once_with("Read", {"file_path": "/tmp/x"})
        assert output.content == "file contents"
        assert output.is_error is False
        assert output.metadata["tool"] == "Read"

    def test_blocked_result_becomes_error_output(self):
        spec = next(s for s in _BUILTIN_SPECS if s.name == "Bash")
        dispatcher = mock.Mock(spec=ToolDispatcher)
        dispatcher.execute.return_value = ToolResult(
            blocked=True, reason="guard denied: rm -rf /",
        )
        tool = self._make_tool(spec, dispatcher)

        output = tool.execute({"command": "rm -rf /"}, ToolContext())

        assert output.is_error is True
        assert "guard denied" in output.content
        assert output.metadata["blocked"] is True

    def test_error_result_becomes_error_output(self):
        spec = next(s for s in _BUILTIN_SPECS if s.name == "Bash")
        dispatcher = mock.Mock(spec=ToolDispatcher)
        dispatcher.execute.return_value = ToolResult(
            output="partial stdout", error="exit code 1",
        )
        tool = self._make_tool(spec, dispatcher)

        output = tool.execute({"command": "false"}, ToolContext())

        assert output.is_error is True
        assert output.content == "exit code 1"
        assert output.metadata["output"] == "partial stdout"

    def test_blocked_without_reason_has_fallback_message(self):
        spec = next(s for s in _BUILTIN_SPECS if s.name == "Write")
        dispatcher = mock.Mock(spec=ToolDispatcher)
        dispatcher.execute.return_value = ToolResult(blocked=True, reason="")
        tool = self._make_tool(spec, dispatcher)

        output = tool.execute({"file_path": "x", "content": "y"}, ToolContext())

        assert output.is_error is True
        assert output.content  # non-empty fallback message


class TestCreateBuiltinTools:
    """Factory wiring: ToolExecutor must see every native tool."""

    def test_factory_returns_protocol_instances(self):
        tools = create_builtin_tools()
        for tool in tools:
            assert isinstance(tool, ToolProtocol)

    def test_factory_covers_all_builtin_specs(self):
        tools = create_builtin_tools()
        returned_names = {t.name for t in tools}
        spec_names = {s.name for s in _BUILTIN_SPECS}
        assert returned_names == spec_names

    def test_factory_shares_dispatcher_when_provided(self):
        """All adapters from one factory call share the same dispatcher, so
        hook state and guard rules apply once across the whole tool set."""
        dispatcher = ToolDispatcher()
        tools = create_builtin_tools(dispatcher=dispatcher)
        for tool in tools:
            assert isinstance(tool, DispatchAdaptedTool)
            assert tool._dispatcher is dispatcher

    def test_factory_result_is_executor_ready(self):
        """End-to-end wiring: plug the factory output into ToolExecutor and the
        executor knows about every tool by name."""
        executor = ToolExecutor(tools=create_builtin_tools())
        for spec in _BUILTIN_SPECS:
            assert executor.get_tool(spec.name) is not None
        assert set(executor.list_tools()) == {s.name for s in _BUILTIN_SPECS}
