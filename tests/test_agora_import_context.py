"""Goodhart-proof tests for AGORA import-context extraction (fix for N=3
false-block false-positives: GRIP #2723, HAL #450, HAL #451).

Core contract: when a changed file has imports at the top and the diff hunk
starts further down (so imports are NOT in the hunk), the assembled review
context for that file MUST contain the import line.

T1 — _extract_import_block: import at line 2, hunk at line 80 -> import visible.
     MUTATION GUARD: removing the header-lines logic makes this fail.

T2 — _changed_files_from_diff: parses `diff --git a/foo b/foo` form correctly.

T3 — extract_pr_file_import_context: with a mocked file fetch returning a file
     whose `import json` is at line 2 and the diff hunk is at line 80, the
     returned section CONTAINS `import json`.
     MUTATION GUARD: reverting to diff-only (_fetch_file_at_head returning '')
     makes the context section empty -> assertion fails.

T4 — fail-open: extract_pr_file_import_context with a failing gh api call
     (mock returns '') returns '' (no crash, no exception).

T5 — _fetch_pr_metadata includes import context in the returned diff_excerpt
     when extract_pr_file_import_context produces a non-empty block.
     MUTATION GUARD: commenting out the import_ctx prepend -> assertion fails.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("HAL_HOME", str(_REPO_ROOT))

from lib.pr_diff import (  # noqa: E402
    _extract_import_block,
    _changed_files_from_diff,
    extract_pr_file_import_context,
    _IMPORT_CONTEXT_MAX_LINES,
    _IMPORT_CONTEXT_MAX_BYTES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A realistic Python file: imports at lines 1-5, actual code starts at line 10.
_SAMPLE_FILE_CONTENT = """\
from __future__ import annotations
import json
import time
from typing import Optional
from lib.hal_council_dispatch import _dispatch_one

_CONSTANT = "foo"


def resolve_model(provider: str) -> str:
    \"\"\"Resolve the canonical model name for a provider.\"\"\"
    data = json.loads('{}')
    return data.get(provider, provider)
""" + "\n".join(f"# line {i}" for i in range(14, 90))

# A diff where the changed hunk is at line 80 — imports are NOT in the hunk.
_SAMPLE_DIFF = """\
diff --git a/lib/hal/resolver.py b/lib/hal/resolver.py
index abc123..def456 100644
--- a/lib/hal/resolver.py
+++ b/lib/hal/resolver.py
@@ -78,6 +78,8 @@ class Resolver:
     def resolve(self, name: str) -> str:
-        return name
+        resolved = resolve_model(name)
+        return resolved
"""


# ---------------------------------------------------------------------------
# T1 — _extract_import_block: imports at file top survive into the block
# ---------------------------------------------------------------------------

def test_extract_import_block_includes_top_imports():
    block = _extract_import_block(_SAMPLE_FILE_CONTENT)
    assert "import json" in block, (
        "import json is at line 2 of the file — must be in the extracted block"
    )
    assert "import time" in block
    assert "from lib.hal_council_dispatch import _dispatch_one" in block


def test_extract_import_block_bounded_by_max_bytes():
    long_content = "import os\n" + ("x = 1\n" * 10_000)
    block = _extract_import_block(long_content)
    assert len(block) <= _IMPORT_CONTEXT_MAX_BYTES


def test_extract_import_block_empty_content_returns_empty():
    assert _extract_import_block("") == ""


# ---------------------------------------------------------------------------
# T2 — _changed_files_from_diff: correct path extraction
# ---------------------------------------------------------------------------

def test_changed_files_from_diff_git_format():
    paths = _changed_files_from_diff(_SAMPLE_DIFF)
    assert paths == ["lib/hal/resolver.py"]


def test_changed_files_from_diff_empty_diff():
    assert _changed_files_from_diff("") == []


def test_changed_files_from_diff_patch_fallback():
    patch_diff = "+++ b/lib/foo.py\n--- a/lib/foo.py\n@@ -1 +1 @@\n"
    paths = _changed_files_from_diff(patch_diff)
    assert "lib/foo.py" in paths


# ---------------------------------------------------------------------------
# T3 — extract_pr_file_import_context: import visible when hunk is at line 80
# ---------------------------------------------------------------------------

def test_import_context_contains_import_when_not_in_hunk():
    """The core regression test for GRIP #2723 / HAL #450 / HAL #451.

    The diff hunk starts at line 78 — `import json` at line 2 is NOT in it.
    The context section MUST contain `import json` so the reviewer sees it.
    """
    with (
        patch("lib.pr_diff._run") as mock_run,
    ):
        import base64
        import json

        # First _run call: gh pr view --json headRefOid
        head_response = json.dumps({"headRefOid": "deadbeef123"})
        # Second _run call: gh api repos/.../contents/...
        encoded = base64.b64encode(_SAMPLE_FILE_CONTENT.encode()).decode()
        contents_response = json.dumps({"content": encoded + "\n", "encoding": "base64"})

        mock_run.side_effect = [head_response, contents_response]

        result = extract_pr_file_import_context(42, "CodeTonight-SA/HAL", _SAMPLE_DIFF)

    assert result, "Expected non-empty import context section"
    assert "import json" in result, (
        "import json is at line 2 of the file but not in the diff hunk — "
        "the context section must include it so the reviewer sees it"
    )
    assert "## File context (imports / module top" in result
    assert "NOT part of the diff" in result


# ---------------------------------------------------------------------------
# T4 — fail-open: gh api failure -> empty string, no exception
# ---------------------------------------------------------------------------

def test_import_context_fail_open_on_fetch_error():
    with patch("lib.pr_diff._run", return_value=""):
        result = extract_pr_file_import_context(42, "CodeTonight-SA/HAL", _SAMPLE_DIFF)
    assert result == "", "Fetch failure must return '' (fail-open, no crash)"


def test_import_context_empty_diff_returns_empty():
    result = extract_pr_file_import_context(42, "CodeTonight-SA/HAL", "")
    assert result == ""


# ---------------------------------------------------------------------------
# T5 — _fetch_pr_metadata prepends import context into diff_excerpt
# ---------------------------------------------------------------------------

def test_fetch_pr_metadata_prepends_import_context():
    """Mutation guard: removing the import_ctx prepend makes this fail."""
    import base64
    import json as json_mod

    head_response = json_mod.dumps({"headRefOid": "cafebabe"})
    title_response = json_mod.dumps({"title": "test: add resolver fix"})
    encoded = base64.b64encode(_SAMPLE_FILE_CONTENT.encode()).decode()
    contents_response = json_mod.dumps({"content": encoded + "\n", "encoding": "base64"})

    from lib.agora_pr_review import _fetch_pr_metadata

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = title_response

    with (
        patch("lib.agora_pr_review.subprocess.run", return_value=mock_proc),
        patch("lib.pr_diff.robust_pr_diff", return_value=_SAMPLE_DIFF),
        patch("lib.pr_diff._run", side_effect=[head_response, contents_response]),
    ):
        _, diff_excerpt = _fetch_pr_metadata(42, "CodeTonight-SA/HAL")

    assert "## File context (imports / module top" in diff_excerpt, (
        "Import context section must be prepended to diff_excerpt"
    )
    assert "import json" in diff_excerpt
    # The original diff must still be present.
    assert "resolve_model" in diff_excerpt
