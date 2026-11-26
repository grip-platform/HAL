"""Regression tests for the ``/api/tools`` payload builder.

``build_tools_payload()`` is the source of truth for the Web tool-listing
response, and it MUST mirror what ``hal --tools`` prints so Web/TUI parity
holds. These tests fail if:

- The payload drifts away from ``_BUILTIN_SPECS`` (drift guard)
- Concurrency flags disappear from the payload (front-end batch planning
  would silently misbehave)
- Required input fields vanish (UI form generators would break)
- The ``registered`` hint stops tracking native-dispatcher presence
- The count stops matching the tool list length (Goodhart anchor)
"""
import os
import sys

import pytest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.server import build_tools_payload
from lib.hal.tool_adapter import _BUILTIN_SPECS


@pytest.fixture()
def payload() -> dict:
    return build_tools_payload()


class TestPayloadStructure:
    def test_has_tools_and_count(self, payload):
        assert "tools" in payload
        assert "count" in payload
        assert isinstance(payload["tools"], list)
        assert isinstance(payload["count"], int)

    def test_count_matches_tools_length(self, payload):
        """Goodhart anchor — count must be bound to the list, not hardcoded."""
        assert payload["count"] == len(payload["tools"])

    def test_count_matches_builtin_spec_count(self, payload):
        """Drift guard — Web payload and adapter spec table agree on cardinality."""
        assert payload["count"] == len(_BUILTIN_SPECS)


class TestPayloadContent:
    def test_every_spec_appears(self, payload):
        names_in_payload = {t["name"] for t in payload["tools"]}
        spec_names = {s.name for s in _BUILTIN_SPECS}
        assert names_in_payload == spec_names

    def test_every_entry_has_rich_fields(self, payload):
        """Each entry must expose the capability surface front-ends need."""
        required_keys = {
            "name", "type", "description", "is_concurrent_safe",
            "required", "input_schema", "registered",
        }
        for tool in payload["tools"]:
            missing = required_keys - set(tool.keys())
            assert not missing, f"Entry {tool.get('name')!r} missing {missing}"

    def test_concurrency_flag_matches_spec(self, payload):
        spec_by_name = {s.name: s for s in _BUILTIN_SPECS}
        for tool in payload["tools"]:
            spec = spec_by_name[tool["name"]]
            assert tool["is_concurrent_safe"] is spec.is_concurrent_safe

    def test_required_fields_mirror_input_schema(self, payload):
        for tool in payload["tools"]:
            required_from_schema = list(tool["input_schema"].get("required", []))
            assert tool["required"] == required_from_schema

    def test_tools_sorted_by_name(self, payload):
        """Stable alphabetical sort — UI tables don't need to re-sort."""
        names = [t["name"] for t in payload["tools"]]
        assert names == sorted(names)


class TestRegisteredFlag:
    def test_registered_true_for_stock_hal(self, payload):
        """Every spec-listed tool exists in the stock native dispatcher."""
        for tool in payload["tools"]:
            assert tool["registered"] is True, (
                f"Tool {tool['name']} missing from ToolDispatcher._TOOL_REGISTRY. "
                "Either the spec table drifted or the dispatcher lost a handler."
            )


class TestParityWithCli:
    """Web payload must match what the CLI --tools flag surfaces."""

    def test_required_fields_for_bash(self, payload):
        bash = next(t for t in payload["tools"] if t["name"] == "Bash")
        assert "command" in bash["required"]
        assert bash["is_concurrent_safe"] is False

    def test_required_fields_for_read(self, payload):
        read = next(t for t in payload["tools"] if t["name"] == "Read")
        assert "file_path" in read["required"]
        assert read["is_concurrent_safe"] is True

    def test_edit_marked_serial(self, payload):
        edit = next(t for t in payload["tools"] if t["name"] == "Edit")
        assert edit["is_concurrent_safe"] is False
        assert set(edit["required"]) == {"file_path", "old_string", "new_string"}
