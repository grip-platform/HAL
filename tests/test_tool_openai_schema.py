"""Regression anchors for Issue #200 — OpenAI tool schema wrapping.

The failure reproduced by the user: ``hal --model groq/llama-3.3-70b`` +
``/tools on`` emitted ``'tools.0.type' : property 'type' is missing`` on
every request because :func:`to_anthropic_tools` produced
``{name, description, input_schema}`` (Anthropic shape) but the
OpenAI-compat endpoint (Groq, OpenAI, Gemini, Ollama-via-/v1) requires
``{type: "function", function: {name, description, parameters}}``.

Goodhart-protection: every assertion checks actual emitted shape. If
:func:`to_openai_tools` is neutered (identity, always-empty, wrong keys)
at least one assertion fails. Tests cover all three input paths
(:class:`ToolSpec`, Anthropic-shape dict, already-OpenAI-shape dict).
"""
from __future__ import annotations

import importlib.util
import pathlib
import unittest

_REPO = pathlib.Path(__file__).parent.parent


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tool_adapter = _load("lib/hal/tool_adapter.py", "_wt_tool_adapter_openai")
to_anthropic_tools = _tool_adapter.to_anthropic_tools
to_openai_tools = _tool_adapter.to_openai_tools
ToolSpec = _tool_adapter.ToolSpec


class ToOpenAiToolsShapeTests(unittest.TestCase):
    """Output shape assertions against the OpenAI /chat/completions API."""

    def test_output_is_list(self):
        self.assertIsInstance(to_openai_tools([]), list)

    def test_empty_input_produces_empty_output(self):
        self.assertEqual(to_openai_tools([]), [])

    def test_wraps_anthropic_shape_dict(self):
        anthropic = [{
            "name": "Read",
            "description": "Read a file from disk.",
            "input_schema": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        }]
        result = to_openai_tools(anthropic)
        self.assertEqual(len(result), 1)
        entry = result[0]
        self.assertEqual(entry["type"], "function")
        self.assertIn("function", entry)
        inner = entry["function"]
        self.assertEqual(inner["name"], "Read")
        self.assertEqual(inner["description"], "Read a file from disk.")
        # The JSON Schema moves from ``input_schema`` to ``parameters``.
        self.assertEqual(
            inner["parameters"]["properties"]["file_path"]["type"],
            "string",
        )

    def test_wraps_tool_spec_directly(self):
        spec = ToolSpec(
            name="Bash",
            description="Execute a shell command.",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
            is_concurrent_safe=False,
        )
        result = to_openai_tools([spec])
        self.assertEqual(result[0]["type"], "function")
        self.assertEqual(result[0]["function"]["name"], "Bash")
        self.assertEqual(
            result[0]["function"]["parameters"]["properties"]["command"]["type"],
            "string",
        )

    def test_passes_through_already_openai_shape(self):
        """Idempotency: a pre-wrapped entry is passed through unchanged."""
        preshaped = {
            "type": "function",
            "function": {
                "name": "Grep",
                "description": "Search with regex.",
                "parameters": {"type": "object"},
            },
        }
        result = to_openai_tools([preshaped])
        self.assertEqual(result, [preshaped])
        # Shape must not double-wrap.
        self.assertNotIn("function", result[0]["function"])

    def test_idempotent_on_round_trip(self):
        """``to_openai_tools(to_openai_tools(x)) == to_openai_tools(x)``."""
        anthropic = to_anthropic_tools()
        once = to_openai_tools(anthropic)
        twice = to_openai_tools(once)
        self.assertEqual(once, twice)

    def test_unknown_shape_is_dropped_not_raised(self):
        """A bad entry in a mixed batch must not poison the whole batch."""
        anthropic_entry = {"name": "OK", "description": "", "input_schema": {}}
        mixed = [anthropic_entry, "not-a-dict", 42, None, anthropic_entry]
        result = to_openai_tools(mixed)
        self.assertEqual(len(result), 2)
        for entry in result:
            self.assertEqual(entry["type"], "function")
            self.assertEqual(entry["function"]["name"], "OK")

    def test_all_builtins_convert_cleanly(self):
        """Every HAL built-in must survive the conversion with valid shape."""
        result = to_openai_tools(to_anthropic_tools())
        self.assertEqual(len(result), 9)  # 8 core + osa_run (PR #226)
        names = {entry["function"]["name"] for entry in result}
        self.assertSetEqual(
            names,
            {
                "Bash", "Read", "Write", "Edit", "Glob", "Grep",
                "CodeLookup", "CodeSearch", "osa_run",
            },
        )
        for entry in result:
            # The bug this test protects against: missing "type" on an entry.
            self.assertEqual(entry["type"], "function")
            self.assertIn("name", entry["function"])
            self.assertIn("description", entry["function"])
            self.assertIn("parameters", entry["function"])


class MutationGuardTests(unittest.TestCase):
    """Neuter-proof assertions — a broken :func:`to_openai_tools` fails these."""

    def test_identity_would_fail_this(self):
        """If to_openai_tools returned input unchanged, this would fail."""
        anthropic = [{"name": "x", "description": "", "input_schema": {}}]
        result = to_openai_tools(anthropic)
        # Identity would leave "input_schema" at top level and omit "type".
        self.assertNotIn("input_schema", result[0])
        self.assertIn("type", result[0])

    def test_always_empty_would_fail_this(self):
        """If to_openai_tools returned [] always, this would fail."""
        result = to_openai_tools(to_anthropic_tools())
        self.assertTrue(result)  # must be non-empty for 8 built-ins


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
