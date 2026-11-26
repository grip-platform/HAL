"""Robust PR diff fetcher with mergeability-poll + multi-rung fallback ladder.

Public API
----------
robust_pr_diff(pr_number, repo, *, repo_root=None, timeout=60, max_chars=PR_REVIEW_DIFF_BUDGET) -> str

Fallback ladder (returns first non-empty stripped result):
  1. Poll gh pr view until mergeable != UNKNOWN (Fibonacci back-off, cap ~30 s).
  2. gh pr diff <N> --repo <R>
  3. gh pr diff <N> --repo <R> --patch
  4. git diff <baseRefName>...<headRefOid>   (local fetch + diff)
  5. "" — caller keeps its existing empty-result guard.

Budget + source-priority
------------------------
When the assembled diff exceeds ``max_chars`` (default PR_REVIEW_DIFF_BUDGET):
  1. Source files are ordered before lock/min/asset/binary/generated hunks.
  2. The result is truncated to ``max_chars``.
  3. An explicit banner is appended so models never hallucinate completeness.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from typing import Callable, Optional

_FIB_DELAYS = (2, 3, 5, 8, 13)  # total ≤ 31 s

# ~50k tokens — covers virtually every real PR; frontier models handle it.
# Import this constant wherever a diff-budget cap is needed (DRY).
PR_REVIEW_DIFF_BUDGET = 200_000

# File path fragments that sort LAST in the budget window (non-source).
# Source hunks matching none of these come FIRST.
_LOW_PRIORITY_RE = re.compile(
    r"(?:package-lock\.json|yarn\.lock|pnpm-lock\.yaml|Pipfile\.lock|"
    r"Cargo\.lock|poetry\.lock|composer\.lock|Gemfile\.lock|"
    r"\.min\.js|\.min\.css|\.map\b|\.svg\b|\.png\b|\.jpg\b|\.jpeg\b|\.gif\b|"
    r"\.ico\b|\.woff2?\b|\.ttf\b|\.eot\b|\.pdf\b|\.zip\b|\.tar\b|"
    r"dist/|build/|\.next/|__pycache__/|node_modules/|vendor/)"
)

_HUNK_HEADER_RE = re.compile(r"^diff --git ", re.MULTILINE)

# Searched only in a file's header, before its first hunk.
_DELETED_MODE_RE = re.compile(r"^deleted file mode ", re.MULTILINE)


def _source_priority_sort(diff: str) -> str:
    """Re-order diff hunks: source files first, lock/asset/generated last."""
    if not diff:
        return diff
    # Split on 'diff --git ' boundaries; first element may be empty header.
    parts = _HUNK_HEADER_RE.split(diff)
    prefix = parts[0]  # anything before the first hunk (may be "")
    hunks = ["diff --git " + p for p in parts[1:]]

    source_hunks = [h for h in hunks if not _LOW_PRIORITY_RE.search(h)]
    low_hunks = [h for h in hunks if _LOW_PRIORITY_RE.search(h)]
    return prefix + "".join(source_hunks + low_hunks)


def _omit_deleted_body(chunk: str) -> str:
    """One deleted file's diff reduced to its header and removed-line count."""
    head, sep, body = chunk.partition("\n@@")
    if not sep or not _DELETED_MODE_RE.search(head):
        return chunk
    removed = sum(1 for line in body.splitlines() if line.startswith("-"))
    return f"{head}\n[deleted file: {removed} lines removed; contents not sent for review]\n"


def _omit_deleted_bodies(diff: str) -> str:
    """Omit the contents of every deleted file while retaining its header."""
    parts = _HUNK_HEADER_RE.split(diff)
    return parts[0] + "".join(_omit_deleted_body("diff --git " + p) for p in parts[1:])


def _apply_budget(diff: str, max_chars: int) -> str:
    """Source-prioritise then truncate with an explicit banner if over budget."""
    if len(diff) <= max_chars:
        return diff
    total = len(diff)
    prioritised = _source_priority_sort(diff)
    truncated = prioritised[:max_chars]
    shown = len(truncated)
    omitted = total - shown
    banner = (
        f"\n\n[DIFF TRUNCATED: showing {shown} of {total} chars; "
        f"{omitted} omitted — source files prioritised. "
        f"Treat unseen code as UNKNOWN, do not infer completeness.]"
    )
    return truncated + banner


def _for_review(
    diff: str,
    max_chars: int,
    scrub: Optional[Callable[[str], str]] = None,
) -> str:
    """Omit deleted bodies, scrub the whole diff, then apply the budget."""
    whole = _omit_deleted_bodies(diff)
    return _apply_budget(scrub(whole) if scrub else whole, max_chars)


def _run(cmd: list[str], *, cwd: Optional[str] = None, timeout: int = 60) -> str:
    """Run a subprocess, return stdout; return '' on any failure."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd,
        )
        return proc.stdout if proc.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def _poll_mergeability(pr_number: int, repo: str, timeout: int) -> None:
    """Best-effort poll until GitHub resolves mergeability. Never raises."""
    for delay in _FIB_DELAYS:
        try:
            proc = subprocess.run(
                [
                    "gh", "pr", "view", str(pr_number), "--repo", repo,
                    "--json", "mergeable,mergeStateStatus",
                ],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
            if proc.returncode != 0:
                break
            data = json.loads(proc.stdout)
            if data.get("mergeable", "UNKNOWN") != "UNKNOWN":
                return
        except Exception:  # noqa: BLE001
            break
        time.sleep(delay)


def robust_pr_diff(
    pr_number: int,
    repo: str,
    *,
    repo_root: Optional[str] = None,
    timeout: int = 60,
    max_chars: int = PR_REVIEW_DIFF_BUDGET,
    scrub: Optional[Callable[[str], str]] = None,
) -> str:
    """Return the PR diff text via a multi-rung fallback ladder.

    Rung 1: poll mergeability (best-effort, then fall through).
    Rung 2: gh pr diff --repo.
    Rung 3: gh pr diff --repo --patch.
    Rung 4: local git diff via base..head.
    Rung 5: "" (empty; caller handles last-resort guard).

    The result is source-prioritised and capped at ``max_chars`` chars.
    If truncated, an explicit banner is appended (never silent).
    """
    _poll_mergeability(pr_number, repo, timeout)

    # Rung 2 — plain diff
    out = _run(["gh", "pr", "diff", str(pr_number), "--repo", repo], timeout=timeout)
    if out.strip():
        return _for_review(out, max_chars, scrub)

    # Rung 3 — patch endpoint (works when merge-base diff cannot be computed)
    out = _run(
        ["gh", "pr", "diff", str(pr_number), "--repo", repo, "--patch"],
        timeout=timeout,
    )
    if out.strip():
        return _for_review(out, max_chars, scrub)

    # Rung 4 — local git diff
    view_out = _run(
        ["gh", "pr", "view", str(pr_number), "--repo", repo,
         "--json", "baseRefName,headRefOid"],
        timeout=timeout,
    )
    if view_out.strip():
        try:
            data = json.loads(view_out)
            base = data.get("baseRefName", "")
            head = data.get("headRefOid", "")
            if base and head:
                cwd = str(repo_root) if repo_root else None
                _run(["git", "fetch", "--quiet", "origin"], cwd=cwd, timeout=timeout)
                out = _run(
                    ["git", "diff", f"{base}...{head}"],
                    cwd=cwd, timeout=timeout,
                )
                if out.strip():
                    return _for_review(out, max_chars, scrub)
        except Exception:  # noqa: BLE001
            pass

    return ""


# ---------------------------------------------------------------------------
# Import-context extractor — fixes AGORA false-blocks on truncated diffs
#
# Reviewers (speakers/judge) see only the changed hunks, not the file top.
# When an import lives at line 2 and the hunk starts at line 80, the reviewer
# falsely flags "symbol X not imported / NameError". N=3 confirmed instances:
# GRIP #2723 (_LIB undefined), HAL #450 (resolve_model), HAL #451 (json/time).
# Fix: for each changed file, fetch its import block at the PR head and
# prepend it to the review context under a clearly-labelled section.
# ---------------------------------------------------------------------------

# Cap per file so we never blow the prompt budget.
_IMPORT_CONTEXT_MAX_LINES = 60
_IMPORT_CONTEXT_MAX_BYTES = 2048
# Regex: lines that are import statements anywhere in the file.
_IMPORT_LINE_RE = re.compile(r"^(?:import |from \S+ import )", re.MULTILINE)


def _extract_import_block(content: str) -> str:
    """Return the import block for a file's content string.

    Strategy (language-agnostic, Python-biased):
    1. Take the first _IMPORT_CONTEXT_MAX_LINES lines as the header block.
    2. Union with any import/from lines in the remainder (catches deferred
       imports that still matter for symbol-existence checks).
    3. Deduplicate, cap at _IMPORT_CONTEXT_MAX_BYTES, return.
    Fail-open: empty content -> empty string.
    """
    if not content:
        return ""
    lines = content.splitlines()
    header = lines[:_IMPORT_CONTEXT_MAX_LINES]
    # Collect import lines from the tail that aren't already in header.
    header_set = set(header)
    tail_imports = [
        ln for ln in lines[_IMPORT_CONTEXT_MAX_LINES:]
        if _IMPORT_LINE_RE.match(ln) and ln not in header_set
    ]
    block = "\n".join(header + tail_imports)
    return block[:_IMPORT_CONTEXT_MAX_BYTES]


def _changed_files_from_diff(diff: str) -> list[str]:
    """Extract changed file paths from a unified diff string.

    Handles both `diff --git a/foo b/foo` and `+++ b/foo` forms.
    Returns de-duplicated list preserving first-seen order.
    """
    seen: dict[str, None] = {}
    for m in re.finditer(r"^diff --git a/.+ b/(.+)$", diff, re.MULTILINE):
        path = m.group(1).strip()
        seen[path] = None
    # Fallback: +++ b/path lines for patch-format diffs.
    if not seen:
        for m in re.finditer(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE):
            path = m.group(1).strip()
            if path != "/dev/null":
                seen[path] = None
    return list(seen)


def _fetch_file_at_head(repo: str, pr_number: int, path: str, timeout: int = 20) -> str:
    """Fetch file content at the PR head ref via gh api. Fail-open -> ''."""
    view = _run(
        ["gh", "pr", "view", str(pr_number), "--repo", repo,
         "--json", "headRefOid"],
        timeout=timeout,
    )
    if not view:
        return ""
    try:
        head_sha = json.loads(view).get("headRefOid", "")
    except Exception:  # noqa: BLE001
        return ""
    if not head_sha:
        return ""
    owner, repo_name = (repo.split("/", 1) + [""])[:2]
    if not repo_name:
        return ""
    import base64
    api_path = f"repos/{owner}/{repo_name}/contents/{path}?ref={head_sha}"
    raw = _run(["gh", "api", api_path], timeout=timeout)
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        encoded = data.get("content", "")
        if not encoded:
            return ""
        return base64.b64decode(encoded.replace("\n", "")).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def extract_pr_file_import_context(
    pr_number: int,
    repo: str,
    diff_excerpt: str,
    *,
    timeout: int = 20,
) -> str:
    """Return a formatted import-context section for all changed files.

    For each changed file in ``diff_excerpt``:
    - Fetches the file at the PR head ref via gh api.
    - Extracts its import/header block (<= 60 lines, <= 2KB).
    - Collects non-empty blocks into a clearly-labelled section.

    Fail-open: any fetch failure silently skips that file. If no blocks are
    collected, returns '' so the caller falls back to diff-only behaviour.

    The returned string is suitable for prepending to the diff in the review
    prompt, under a section the model is told is NOT part of the diff.
    """
    if not diff_excerpt:
        return ""
    paths = _changed_files_from_diff(diff_excerpt)
    if not paths:
        return ""

    blocks: list[str] = []
    for path in paths:
        content = _fetch_file_at_head(repo, pr_number, path, timeout=timeout)
        block = _extract_import_block(content)
        if block.strip():
            blocks.append(f"### {path}\n```\n{block}\n```")

    if not blocks:
        return ""

    header = (
        "## File context (imports / module top -- NOT part of the diff)\n"
        "The sections below show each changed file's import block at the PR "
        "head. These symbols ARE defined even if not visible in the diff hunk. "
        "Use this to verify symbol existence before flagging a NameError or "
        "missing import.\n\n"
    )
    return header + "\n\n".join(blocks) + "\n\n"
