"""Tests for ``lib.hal.osa.provenance`` — provenance enum + parser."""
from __future__ import annotations

import json
import unittest

from lib.hal.osa.provenance import (
    ProvenanceType,
    UnknownProvenance,
    parse_provenance,
)


class EnumShape(unittest.TestCase):
    """``ProvenanceType`` carries the five taxonomy values + LOCAL."""

    def test_values_match_council_taxonomy(self) -> None:
        self.assertEqual(ProvenanceType.HUMAN_TYPED.value, "human_typed")
        self.assertEqual(ProvenanceType.TOOL_CHAIN.value, "tool_chain")
        self.assertEqual(ProvenanceType.MCP_REMOTE.value, "mcp_remote")
        self.assertEqual(ProvenanceType.AUTONOMOUS_LOOP.value, "autonomous_loop")
        self.assertEqual(ProvenanceType.YOLO.value, "yolo")
        self.assertEqual(ProvenanceType.LOCAL.value, "local")

    def test_is_str_subclass_for_json(self) -> None:
        """Inheritance from str → JSON encoder produces the plain value."""
        encoded = json.dumps({"p": ProvenanceType.MCP_REMOTE})
        self.assertIn('"mcp_remote"', encoded)


class Parser(unittest.TestCase):
    """``parse_provenance`` accepts both enum name and value, case-insensitive."""

    def test_parse_lowercase_value(self) -> None:
        self.assertIs(parse_provenance("tool_chain"), ProvenanceType.TOOL_CHAIN)

    def test_parse_uppercase_name(self) -> None:
        self.assertIs(parse_provenance("HUMAN_TYPED"), ProvenanceType.HUMAN_TYPED)

    def test_parse_trims_whitespace(self) -> None:
        self.assertIs(parse_provenance("  yolo  "), ProvenanceType.YOLO)

    def test_parse_mixed_case(self) -> None:
        self.assertIs(parse_provenance("McP_Remote"), ProvenanceType.MCP_REMOTE)

    def test_unknown_raises(self) -> None:
        with self.assertRaises(UnknownProvenance):
            parse_provenance("super_user_bypass")

    def test_empty_string_raises(self) -> None:
        with self.assertRaises(UnknownProvenance):
            parse_provenance("")

    def test_exception_lists_valid_options(self) -> None:
        """Goodhart guard — the error message must help the caller fix it."""
        try:
            parse_provenance("bogus")
        except UnknownProvenance as exc:
            self.assertIn("bogus", str(exc))
            self.assertIn("human_typed", str(exc))
        else:
            self.fail("parse_provenance accepted an unknown value")


if __name__ == "__main__":
    unittest.main()
