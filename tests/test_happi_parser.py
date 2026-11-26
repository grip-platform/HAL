"""Tests for lib/hal/happi/parser.py — HAPPI fenced-block extraction.

Goodhart guard: an always-empty parser would pass the "missing block"
tests but fail content-extraction. An identity parser (everything to
prose) would pass prose tests but fail block-extraction. Every
assertion here forces real behaviour to exist.
"""
from __future__ import annotations

import textwrap
import unittest

from lib.hal.happi.parser import (
    CANONICAL_LABELS,
    HappiDocument,
    parse_happi,
)


_MINIMAL_DOC = textwrap.dedent(
    """\
    # HAPPI example

    Prose before any fences.

    ```yaml-openapi
    openapi: 3.1.0
    info:
      title: Example
    ```

    Middle prose between fences.

    ```bash-bootstrap
    #!/usr/bin/env bash
    echo "hello from bootstrap"
    ```

    ```llm-system-prompt
    You are HAL. Emit JSON matching the schema above.
    ```

    Trailing prose.
    """
)

_MULTI_SAME_LABEL = textwrap.dedent(
    """\
    # multi-block

    ```bash-bootstrap
    echo first
    ```

    ```bash-bootstrap
    echo second
    ```
    """
)

_UNLABELLED_FENCE = textwrap.dedent(
    """\
    An unlabelled fence below.

    ```
    no-label content
    ```
    """
)

_NO_FENCES = "Just prose. No fences anywhere here."


class BasicStructureTests(unittest.TestCase):

    def test_empty_input_returns_empty_document(self):
        doc = parse_happi("")
        self.assertEqual(doc.prose, "")
        self.assertEqual(doc.blocks, {})

    def test_no_fences_returns_prose_only(self):
        doc = parse_happi(_NO_FENCES)
        self.assertIn("Just prose", doc.prose)
        self.assertEqual(doc.blocks, {})

    def test_all_canonical_labels_are_extracted(self):
        doc = parse_happi(_MINIMAL_DOC)
        for label in ("yaml-openapi", "bash-bootstrap", "llm-system-prompt"):
            self.assertTrue(
                doc.has(label),
                msg=f"missing canonical label: {label}",
            )

    def test_openapi_block_contents_preserved_verbatim(self):
        doc = parse_happi(_MINIMAL_DOC)
        openapi = doc.get_first("yaml-openapi")
        self.assertIsNotNone(openapi)
        self.assertIn("openapi: 3.1.0", openapi)
        self.assertIn("title: Example", openapi)

    def test_bash_block_contents_preserved_verbatim(self):
        doc = parse_happi(_MINIMAL_DOC)
        bash = doc.get_first("bash-bootstrap")
        self.assertIsNotNone(bash)
        self.assertIn("#!/usr/bin/env bash", bash)
        self.assertIn('echo "hello from bootstrap"', bash)

    def test_llm_prompt_block_contents_preserved_verbatim(self):
        doc = parse_happi(_MINIMAL_DOC)
        prompt = doc.get_first("llm-system-prompt")
        self.assertIsNotNone(prompt)
        self.assertIn("You are HAL", prompt)

    def test_prose_between_fences_is_retained(self):
        doc = parse_happi(_MINIMAL_DOC)
        self.assertIn("Prose before any fences", doc.prose)
        self.assertIn("Middle prose between fences", doc.prose)
        self.assertIn("Trailing prose", doc.prose)

    def test_prose_does_not_include_fenced_content(self):
        doc = parse_happi(_MINIMAL_DOC)
        self.assertNotIn("openapi: 3.1.0", doc.prose)
        self.assertNotIn('echo "hello from bootstrap"', doc.prose)


class MultipleBlocksTests(unittest.TestCase):

    def test_multiple_blocks_same_label_all_preserved(self):
        doc = parse_happi(_MULTI_SAME_LABEL)
        blocks = doc.get_all("bash-bootstrap")
        self.assertEqual(len(blocks), 2)
        self.assertIn("first", blocks[0])
        self.assertIn("second", blocks[1])

    def test_document_order_preserved_for_same_label(self):
        doc = parse_happi(_MULTI_SAME_LABEL)
        blocks = doc.get_all("bash-bootstrap")
        self.assertTrue(blocks[0].strip().endswith("first"))
        self.assertTrue(blocks[1].strip().endswith("second"))


class UnlabelledFenceTests(unittest.TestCase):

    def test_unlabelled_fence_stored_under_empty_key(self):
        doc = parse_happi(_UNLABELLED_FENCE)
        self.assertIn("", doc.blocks)
        self.assertIn("no-label content", doc.get_first("") or "")

    def test_unlabelled_fence_body_not_in_prose(self):
        doc = parse_happi(_UNLABELLED_FENCE)
        self.assertNotIn("no-label content", doc.prose)


class AbsentLabelTests(unittest.TestCase):

    def test_get_first_missing_label_returns_none(self):
        doc = parse_happi(_MINIMAL_DOC)
        self.assertIsNone(doc.get_first("ssml"))

    def test_get_all_missing_label_returns_empty_list(self):
        doc = parse_happi(_MINIMAL_DOC)
        self.assertEqual(doc.get_all("nope"), [])

    def test_has_returns_false_for_missing_label(self):
        doc = parse_happi(_MINIMAL_DOC)
        self.assertFalse(doc.has("ssml"))


class EdgeCaseTests(unittest.TestCase):

    def test_label_with_dot_and_plus_chars(self):
        raw = "```app.json+happi\npayload\n```\n"
        doc = parse_happi(raw)
        self.assertTrue(doc.has("app.json+happi"))

    def test_unclosed_fence_at_eof_still_commits_partial(self):
        raw = "prose\n```bash-bootstrap\noops no close\n"
        doc = parse_happi(raw)
        self.assertTrue(doc.has("bash-bootstrap"))
        self.assertIn("oops no close", doc.get_first("bash-bootstrap") or "")

    def test_fence_with_trailing_metadata_keeps_label_only(self):
        raw = '```yaml-openapi title="x"\nopenapi: 3.1.0\n```\n'
        doc = parse_happi(raw)
        self.assertTrue(doc.has("yaml-openapi"))
        self.assertFalse(doc.has('yaml-openapi title="x"'))


class CanonicalLabelsTests(unittest.TestCase):

    def test_canonical_labels_is_a_tuple(self):
        self.assertIsInstance(CANONICAL_LABELS, tuple)

    def test_expected_canonical_labels_are_present(self):
        for label in (
            "yaml-openapi",
            "bash-bootstrap",
            "llm-system-prompt",
            "json-provenance",
        ):
            self.assertIn(label, CANONICAL_LABELS)


class GoodhartGuardTests(unittest.TestCase):
    """Tests constructed so a broken parser has no chance of passing by accident."""

    def test_non_trivial_block_extraction(self):
        doc = parse_happi(_MINIMAL_DOC)
        openapi = doc.get_first("yaml-openapi")
        self.assertIsNotNone(openapi)
        self.assertTrue(len(openapi.strip()) > 0)

    def test_non_trivial_prose_separation(self):
        doc = parse_happi(_MINIMAL_DOC)
        self.assertGreater(len(doc.blocks), 0)
        self.assertLess(len(doc.prose), len(_MINIMAL_DOC))


if __name__ == "__main__":
    unittest.main()
