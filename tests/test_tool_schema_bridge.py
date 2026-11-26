"""Tests for the HAL → Anthropic tool-schema bridge.

W1 deliverable of the tool-loop wiring sprint. Verifies:

- ``to_anthropic_tools()`` projects the 8 built-in specs into Anthropic's
  expected ``{name, description, input_schema}`` shape with zero drift
- ``parse_tool_use_blocks()`` recovers tool calls from a realistic
  Anthropic response payload, ignores text/thinking blocks, and degrades
  safely on missing or malformed input

Goodhart anchor: ``test_names_match_dispatcher_registry`` ties the offered
tool surface back to what ``ToolDispatcher`` can actually execute, so a
silent drift between the two surfaces can't sneak through test-count
gaming (e.g. renaming a spec without renaming the handler).
"""
import os
import sys
import unittest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.tool_adapter import (
    _BUILTIN_SPECS,
    parse_tool_use_blocks,
    to_anthropic_tools,
)


class TestToAnthropicTools(unittest.TestCase):
    def test_returns_all_builtin_specs(self):
        # Count tracks _BUILTIN_SPECS so new tools (e.g. osa_run added in
        # PR #226) don't require a parallel edit here.
        self.assertEqual(len(to_anthropic_tools()), len(_BUILTIN_SPECS))

    def test_every_tool_has_anthropic_required_fields(self):
        for tool in to_anthropic_tools():
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("input_schema", tool)

    def test_no_extra_fields_beyond_anthropic_schema(self):
        """Anthropic rejects unknown fields at the top level of a tool entry."""
        allowed = {"name", "description", "input_schema"}
        for tool in to_anthropic_tools():
            self.assertEqual(set(tool.keys()), allowed)

    def test_input_schema_has_required_subfields(self):
        for tool in to_anthropic_tools():
            schema = tool["input_schema"]
            self.assertEqual(schema.get("type"), "object")
            self.assertIn("properties", schema)
            self.assertIn("required", schema)

    def test_names_match_dispatcher_registry(self):
        """Spec names must exactly match ToolDispatcher's registered handlers.

        Otherwise the round-trip ``LLM → tool_use(name) → dispatcher`` breaks
        silently — the model emits a name like ``Read`` and the dispatcher
        can't resolve it.
        """
        from lib.hal.tool_dispatch import ToolDispatcher
        registered = set(ToolDispatcher().available_tools())
        spec_names = {t["name"] for t in to_anthropic_tools()}
        missing = spec_names - registered
        self.assertFalse(
            missing,
            f"spec names not registered in dispatcher: {sorted(missing)}",
        )

    def test_accepts_custom_spec_tuple(self):
        """Lets callers send a filtered subset — e.g. only read-only tools."""
        read_only = tuple(s for s in _BUILTIN_SPECS if s.is_concurrent_safe)
        tools = to_anthropic_tools(read_only)
        self.assertEqual(len(tools), 5)
        names = {t["name"] for t in tools}
        self.assertEqual(names, {"Read", "Glob", "Grep", "CodeLookup", "CodeSearch"})

    def test_drops_is_concurrent_safe_field(self):
        """Concurrency is a HAL-side concern; must not leak into Anthropic payload."""
        for tool in to_anthropic_tools():
            self.assertNotIn("is_concurrent_safe", tool)


class TestParseToolUseBlocks(unittest.TestCase):
    def test_extracts_single_tool_use(self):
        raw = {
            "content": [
                {"type": "text", "text": "I'll read that."},
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "Read",
                    "input": {"file_path": "/tmp/x"},
                },
            ],
            "stop_reason": "tool_use",
        }
        calls = parse_tool_use_blocks(raw)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "Read")
        self.assertEqual(calls[0]["id"], "toolu_abc")
        self.assertEqual(calls[0]["input"], {"file_path": "/tmp/x"})

    def test_extracts_multiple_tool_uses(self):
        raw = {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a"}},
                {"type": "tool_use", "id": "t2", "name": "Grep", "input": {"pattern": "foo"}},
            ],
        }
        calls = parse_tool_use_blocks(raw)
        self.assertEqual([c["name"] for c in calls], ["Read", "Grep"])

    def test_ignores_text_blocks(self):
        raw = {"content": [{"type": "text", "text": "hello"}]}
        self.assertEqual(parse_tool_use_blocks(raw), [])

    def test_ignores_thinking_blocks(self):
        raw = {
            "content": [
                {"type": "thinking", "thinking": "..."},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
            ],
        }
        calls = parse_tool_use_blocks(raw)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "Read")

    def test_empty_content_returns_empty_list(self):
        self.assertEqual(parse_tool_use_blocks({"content": []}), [])

    def test_missing_content_key_returns_empty_list(self):
        self.assertEqual(parse_tool_use_blocks({}), [])

    def test_none_input_returns_empty_list(self):
        self.assertEqual(parse_tool_use_blocks(None), [])

    def test_non_dict_root_returns_empty_list(self):
        self.assertEqual(parse_tool_use_blocks("not a dict"), [])
        self.assertEqual(parse_tool_use_blocks(42), [])
        self.assertEqual(parse_tool_use_blocks([{"type": "tool_use"}]), [])

    def test_non_list_content_returns_empty_list(self):
        self.assertEqual(parse_tool_use_blocks({"content": "nope"}), [])
        self.assertEqual(parse_tool_use_blocks({"content": None}), [])

    def test_malformed_block_missing_name_defaults_to_empty_string(self):
        raw = {"content": [{"type": "tool_use", "id": "t1", "input": {}}]}
        calls = parse_tool_use_blocks(raw)
        self.assertEqual(calls[0]["name"], "")

    def test_malformed_block_missing_input_defaults_to_empty_dict(self):
        raw = {"content": [{"type": "tool_use", "id": "t1", "name": "Read"}]}
        calls = parse_tool_use_blocks(raw)
        self.assertEqual(calls[0]["input"], {})

    def test_malformed_block_missing_id_defaults_to_empty_string(self):
        raw = {"content": [{"type": "tool_use", "name": "Read", "input": {}}]}
        calls = parse_tool_use_blocks(raw)
        self.assertEqual(calls[0]["id"], "")

    def test_non_dict_block_entries_are_skipped(self):
        raw = {"content": ["string", 42, None, {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]}
        calls = parse_tool_use_blocks(raw)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "Read")


if __name__ == "__main__":
    unittest.main()
