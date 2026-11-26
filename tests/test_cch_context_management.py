"""Tests for lib/hal/utils/cch.py::_compute_context_management and body-inject.

Hypothesis H-HAL-W4A-1: HAL's CCH build_body produces an Anthropic
Messages API body that includes a ``context_management`` field shaped
exactly like upstream's ``getAPIContextManagement`` when thinking is
active and the ``context-management-2025-06-27`` beta is announced.

Shape reference: ``plans/cch-context-management-shape.md`` (W3 Track B
research doc — verbatim port from paoloanzn/free-code).

Coverage:
- helper returns None when thinking is off (field must be omitted, NOT
  emitted as an empty edits list)
- helper returns None when redact-thinking is active
- helper returns ``keep: "all"`` by default
- helper returns the thinking_turns=1 form when clear_all_thinking is True
- beta header string includes ``context-management-2025-06-27`` so the
  upstream gate will pass the field through
- build_body payload JSON contains the ``context_management`` key with
  the expected shape for a thinking-active request
- the emitted field is a dict with a single edit of type
  ``clear_thinking_20251015``
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.utils import cch as cch_module
from lib.hal.utils.cch import (
    CCHSigner,
    _ANTHROPIC_BETA,
    _CONTEXT_MANAGEMENT_BETA_HEADER,
    _compute_context_management,
)


class TestComputeContextManagement(unittest.TestCase):

    def test_thinking_off_returns_none(self):
        self.assertIsNone(_compute_context_management(has_thinking=False))

    def test_redact_thinking_active_returns_none(self):
        self.assertIsNone(
            _compute_context_management(
                has_thinking=True, is_redact_thinking_active=True,
            ),
        )

    def test_default_returns_keep_all_edit(self):
        result = _compute_context_management(has_thinking=True)
        self.assertEqual(
            result,
            {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        )

    def test_clear_all_thinking_returns_thinking_turns_shape(self):
        result = _compute_context_management(
            has_thinking=True, clear_all_thinking=True,
        )
        self.assertEqual(
            result,
            {"edits": [{
                "type": "clear_thinking_20251015",
                "keep": {"type": "thinking_turns", "value": 1},
            }]},
        )

    def test_edits_is_list_not_tuple(self):
        """The Anthropic API rejects tuples in JSON serialisation paths.
        Must be a plain list so json.dumps emits a JSON array."""
        result = _compute_context_management(has_thinking=True)
        self.assertIsInstance(result["edits"], list)

    def test_single_edit_shape_no_tool_clearing(self):
        """Default call emits only the thinking edit — tool-clearing is
        opt-in and off by default."""
        result = _compute_context_management(has_thinking=True)
        self.assertEqual(len(result["edits"]), 1)
        self.assertEqual(result["edits"][0]["type"], "clear_thinking_20251015")

    def test_default_uses_thinking_true(self):
        """Convenience default: _compute_context_management() with no
        kwargs returns the thinking-active shape (HAL's default path)."""
        result = _compute_context_management()
        self.assertIsNotNone(result)
        self.assertEqual(result["edits"][0]["keep"], "all")


class TestBetaHeaderConstant(unittest.TestCase):

    def test_context_management_beta_in_anthropic_beta_string(self):
        self.assertIn(_CONTEXT_MANAGEMENT_BETA_HEADER, _ANTHROPIC_BETA)

    def test_beta_header_constant_matches_upstream(self):
        self.assertEqual(
            _CONTEXT_MANAGEMENT_BETA_HEADER, "context-management-2025-06-27",
        )


class TestBuildBodyInjectsContextManagement(unittest.TestCase):

    def setUp(self):
        self.signer = CCHSigner()

    def _build_and_parse(self, prompt: str) -> dict:
        body_str, _suffix = self.signer.build_body(
            prompt, model="claude-opus-4-7", session_id="test-session",
        )
        return json.loads(body_str)

    def test_payload_includes_context_management_key(self):
        payload = self._build_and_parse("hello world")
        self.assertIn("context_management", payload)

    def test_context_management_value_is_edits_dict(self):
        payload = self._build_and_parse("hello world")
        cm = payload["context_management"]
        self.assertIsInstance(cm, dict)
        self.assertIn("edits", cm)
        self.assertIsInstance(cm["edits"], list)

    def test_context_management_contains_thinking_edit(self):
        payload = self._build_and_parse("hello world")
        edits = payload["context_management"]["edits"]
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["type"], "clear_thinking_20251015")
        self.assertEqual(edits[0]["keep"], "all")

    def test_thinking_field_still_present(self):
        """Regression guard: context_management must not displace the
        existing ``thinking`` field — both are required."""
        payload = self._build_and_parse("hello world")
        self.assertIn("thinking", payload)
        self.assertEqual(payload["thinking"], {"type": "adaptive"})

    def test_body_is_still_signable(self):
        """sign() replaces the cch=00000 placeholder. Adding
        context_management must not corrupt that string — the placeholder
        must still exist in the rendered body."""
        body_str, _suffix = self.signer.build_body(
            "hello", model="claude-opus-4-7",
        )
        self.assertIn("cch=00000", body_str)


class TestBuildBodyPayloadShape(unittest.TestCase):

    def test_no_extra_top_level_keys_beyond_expected_set(self):
        """Guard against accidental leakage of internal fields into the
        outgoing body. The expected top-level keys are frozen here so
        any future addition forces an intentional test update."""
        signer = CCHSigner()
        body_str, _suffix = signer.build_body(
            "hello", model="claude-opus-4-7",
        )
        payload = json.loads(body_str)
        expected = {
            "model", "max_tokens", "stream", "thinking",
            "metadata", "system", "messages",
            "context_management",
        }
        self.assertEqual(set(payload.keys()), expected)


class TestClearToolUseEdits(unittest.TestCase):
    """Coverage for the ``clear_tool_uses_20250919`` strategy introduced
    in HAL's port of TF>>'s ``apiMicrocompact.ts``."""

    def test_clear_tool_results_emits_clear_tool_inputs_edit(self):
        result = _compute_context_management(
            has_thinking=False, clear_tool_results=True,
        )
        self.assertIsNotNone(result)
        edit = result["edits"][0]
        self.assertEqual(edit["type"], "clear_tool_uses_20250919")
        self.assertIn("clear_tool_inputs", edit)
        self.assertIn("bash", edit["clear_tool_inputs"])
        self.assertIn("read", edit["clear_tool_inputs"])
        self.assertNotIn("str_replace_editor", edit["clear_tool_inputs"])

    def test_clear_tool_uses_emits_exclude_tools_edit(self):
        result = _compute_context_management(
            has_thinking=False, clear_tool_uses=True,
        )
        self.assertIsNotNone(result)
        edit = result["edits"][0]
        self.assertEqual(edit["type"], "clear_tool_uses_20250919")
        self.assertIn("exclude_tools", edit)
        self.assertIn("str_replace_editor", edit["exclude_tools"])
        self.assertNotIn("bash", edit["exclude_tools"])

    def test_clear_tool_results_default_trigger_threshold(self):
        result = _compute_context_management(
            has_thinking=False, clear_tool_results=True,
        )
        edit = result["edits"][0]
        self.assertEqual(edit["trigger"], {"type": "input_tokens", "value": 180_000})
        self.assertEqual(
            edit["clear_at_least"], {"type": "input_tokens", "value": 40_000},
        )

    def test_both_modes_combined_with_thinking(self):
        result = _compute_context_management(
            has_thinking=True,
            clear_tool_results=True,
            clear_tool_uses=True,
        )
        self.assertEqual(len(result["edits"]), 3)
        types = [e["type"] for e in result["edits"]]
        self.assertEqual(types.count("clear_thinking_20251015"), 1)
        self.assertEqual(types.count("clear_tool_uses_20250919"), 2)

    def test_no_tool_clearing_and_no_thinking_returns_none(self):
        self.assertIsNone(
            _compute_context_management(has_thinking=False),
        )

    def test_tool_results_mode_empty_clearable_list_returns_none(self):
        result = _compute_context_management(
            has_thinking=False, clear_tool_results=True, clearable_results=(),
        )
        self.assertIsNone(result)

    def test_build_body_accepts_clear_tool_results_kwarg(self):
        signer = CCHSigner()
        body_str, _ = signer.build_body(
            "hello", model="claude-opus-4-7", clear_tool_results=True,
        )
        payload = json.loads(body_str)
        edits = payload["context_management"]["edits"]
        types = [e["type"] for e in edits]
        self.assertIn("clear_tool_uses_20250919", types)


if __name__ == "__main__":
    unittest.main()
