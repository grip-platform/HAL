"""Path-allowlist file sandbox for HAL ``/api/infer-with-tools``.

This module provides the file-tool primitives that the Anthropic
tool-loop driver exposes to the model:

- ``read_text(root, path) -> str``
- ``grep_pattern(root, pattern, glob_filter) -> list[Match]``
- ``glob_pattern(root, pattern) -> list[str]``

**Capability-lock invariant** (per ``rules/untrusted-content-capability-lock.md``
in the GRIP repo): every call resolves the candidate path with
``os.path.realpath`` and refuses any result that escapes ``root``
(symlinks included). The sandbox is the boundary between untrusted
WA / channel content and arbitrary host filesystem read.

**Architecture decision: path-allowlist, NOT chroot.**
chroot requires root (Linux) or doesn't exist (macOS without sudo).
Path-allowlist is cross-platform, deterministic, and testable as a
pure function (``(root, candidate) -> allow|refuse``).

**Audit log**: every refusal AND every successful tool call appends
one JSONL line to ``~/.hal/logs/infer-with-tools.jsonl`` (best-effort,
never raises). Format::

    {"ts": "...", "tool": "Read", "root": "~/.claude",
     "path": "lib/foo.py", "allowed": true, "bytes": 1234}

Falsifier
---------
This module is wrong if a hostile path (e.g. ``../../../etc/passwd``,
``/etc/passwd``, or a symlink to one) is returned as ``allowed``.
Verified by ``tests/test_file_sandbox.py``: T1 (rooted-inside),
T2 (symlink-escape-refused), T14 (etc/root rejected), T15 (symlinked
root rejected).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = [
    "SandboxEscapeError",
    "SandboxRootRefused",
    "SandboxBinaryError",
    "SandboxTimeoutError",
    "GrepMatch",
    "validate_root",
    "read_text",
    "grep_pattern",
    "glob_pattern",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SandboxEscapeError(Exception):
    """Raised when a candidate path escapes the sandbox root.

    Includes symlink-resolved paths that point outside the root, ``..``
    traversal that exits the root, and absolute paths that aren't a
    descendant of the root.
    """


class SandboxRootRefused(Exception):
    """Raised when the configured sandbox_root itself is unsafe.

    Examples: ``/``, ``/etc``, a non-existent directory, a file (not
    directory), or a symlink whose realpath points to a system directory.
    """


class SandboxBinaryError(Exception):
    """Raised when :func:`read_text` is called on a binary file.

    AGORA Council #373 concern 4 (this PR): ``open(..., encoding="utf-8",
    errors="replace")`` would happily decode a binary file by substituting
    replacement characters for non-UTF-8 bytes, returning garbage to the
    model. The model would either hallucinate from the garbage or waste
    turns trying to interpret it. Better to fail honestly with a
    distinguishable error so the model can pick a different tool (Glob,
    Grep with a glob_filter, or a different path).

    Detection heuristic: null byte present in the first 8 KiB of the file.
    This is the standard ``grep -I`` heuristic and is what ripgrep,
    git, and most editors use to classify text vs binary.
    """


class SandboxTimeoutError(Exception):
    """Raised when a sandbox primitive exceeds its wall-clock budget.

    AGORA Council #373 concern 1 (this PR): ``_grep_via_ripgrep`` used a
    hard 30s subprocess timeout and returned an empty list on hit. The
    model would interpret the empty list as "we searched and found
    nothing" — indistinguishable from a legitimate zero-hits result.

    This exception lets the dispatcher surface an explicit
    ``"grep timed out after Xs"`` error to the model so it can retry
    with a narrower ``glob_filter`` or split the query."""


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def _audit_path() -> Path:
    return Path(os.path.expanduser("~/.hal/logs/infer-with-tools.jsonl"))


def _audit(record: dict) -> None:
    """Best-effort append. Never raises."""
    try:
        path = _audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        # never block on audit failure
        pass


# ---------------------------------------------------------------------------
# Root validation
# ---------------------------------------------------------------------------


# Refused roots — never sandbox here even if the operator passes the
# path explicitly. These cover the most common "escape via root-pinning"
# attacks. We DELIBERATELY exclude the broad parents ``/var`` and
# ``/private/var`` on macOS because the user-writable tempdir lives
# under ``/private/var/folders/...`` — pinning a sandbox there is
# legitimate. We refuse only canonical system **leaf** directories
# where read access of any descendant is sensitive.
#
# The refused ``/var`` leaves (AGORA Council #373/#376 concern — system
# reconnaissance): ``/var/log`` (and ``/private/var/log``) holds
# ``system.log`` etc. with kernel/auth messages; ``/var/audit`` holds
# BSM audit trails; ``/var/db`` holds system databases (``dslocal``,
# sudo lectured state, ...); ``/var/spool`` holds user mail
# (``/var/spool/mail``), cron, and print queues. These are NOT user
# tempdirs — pinning a read-sandbox there is always reconnaissance,
# never legitimate, so we refuse the specific leaves WITHOUT refusing
# the ``/var/folders`` tempdir parent. macOS realpath rewrites
# ``/var/...`` to ``/private/var/...``, so BOTH forms are listed (the
# prefix check on line ~196 already catches descendants of each, and
# ``realpath`` is applied to the candidate root BEFORE this comparison,
# so a ``..``-traversal like ``/var/folders/../log`` resolves to
# ``/private/var/log`` and is caught here — verified by
# ``test_sandbox_root_traversal_to_refused_leaf_rejected``).
_REFUSED_ROOTS = frozenset({
    "/",
    "/etc", "/private/etc",
    "/usr/bin", "/usr/sbin", "/usr/local/etc",
    "/bin", "/sbin", "/boot",
    "/root",
    "/System",
    # System-sensitive ``/var`` leaves (system.log / audit trail / system
    # DBs / mail+cron+print spools). ``/var/folders`` (user tempdir) is
    # deliberately NOT here.
    "/var/log", "/private/var/log",
    "/var/audit", "/private/var/audit",
    "/var/db", "/private/var/db",
    "/var/root", "/private/var/root",
    "/var/spool", "/private/var/spool",
    # ``/Library`` and the broad ``/var`` parent deliberately omitted —
    # too broad on macOS; subpaths are legitimately read by tooling.
})


def validate_root(sandbox_root: str) -> str:
    """Validate the caller-supplied ``sandbox_root`` and return its
    canonical absolute path. Raises :class:`SandboxRootRefused` on any
    issue.

    Checks (in order):

    1. Not empty
    2. Absolute path (``os.path.abspath``)
    3. ``realpath`` resolves to an existing **directory**
    4. ``realpath`` is not in :data:`_REFUSED_ROOTS`
    5. ``realpath`` is not under any refused root prefix
    """
    if not sandbox_root:
        raise SandboxRootRefused("sandbox_root is required (got empty string)")

    abspath = os.path.abspath(sandbox_root)
    try:
        realpath = os.path.realpath(abspath, strict=True)
    except OSError as exc:
        raise SandboxRootRefused(
            f"sandbox_root does not resolve: {sandbox_root!r} ({exc})"
        ) from exc

    if not os.path.isdir(realpath):
        raise SandboxRootRefused(
            f"sandbox_root is not a directory: {realpath!r}"
        )

    if realpath in _REFUSED_ROOTS:
        raise SandboxRootRefused(
            f"sandbox_root resolves to a system directory: {realpath!r}"
        )

    for refused in _REFUSED_ROOTS:
        # realpath under /etc, /usr, etc. — refuse
        if realpath == refused or realpath.startswith(refused + os.sep):
            raise SandboxRootRefused(
                f"sandbox_root resolves under a refused prefix "
                f"({refused!r}): {realpath!r}"
            )

    return realpath


def _resolve_safe(root: str, candidate: str) -> str:
    """Resolve ``candidate`` (relative or absolute) against ``root`` and
    refuse anything that escapes. ``root`` MUST already be validated.

    Returns the canonical absolute path on success. Raises
    :class:`SandboxEscapeError` on escape.
    """
    if os.path.isabs(candidate):
        joined = candidate
    else:
        joined = os.path.join(root, candidate)

    # realpath resolves symlinks AND ``..`` segments
    try:
        resolved = os.path.realpath(joined, strict=False)
    except OSError as exc:
        raise SandboxEscapeError(
            f"path resolution failed: {candidate!r} ({exc})"
        ) from exc

    # Resolved path must equal root OR be a descendant
    if resolved == root:
        return resolved
    if resolved.startswith(root + os.sep):
        return resolved

    raise SandboxEscapeError(
        f"path escapes sandbox: {candidate!r} -> {resolved!r} "
        f"(root: {root!r})"
    )


# ---------------------------------------------------------------------------
# Tool primitives
# ---------------------------------------------------------------------------


_MAX_READ_BYTES = 256 * 1024  # 256 KB per file
_MAX_GREP_HITS = 200
_MAX_GLOB_HITS = 500
_BINARY_SNIFF_BYTES = 8192  # standard heuristic — same as grep -I / ripgrep

# Wall-clock timeouts for grep/glob — env-tunable. Default 30s mirrors the
# previous hardcoded ripgrep timeout; raising it for huge sandboxes is now
# a one-env-var change rather than a code edit. AGORA Council #373
# concern 1 fix.
_DEFAULT_GREP_TIMEOUT_S = int(
    os.environ.get("HAL_FILE_SANDBOX_GREP_TIMEOUT_S", "30")
)
_DEFAULT_GLOB_TIMEOUT_S = int(
    os.environ.get("HAL_FILE_SANDBOX_GLOB_TIMEOUT_S", "60")
)


def _looks_binary(blob: bytes) -> bool:
    """Detect binary content by null-byte heuristic.

    Returns ``True`` if any null byte appears in ``blob``. This matches
    the ``grep -I``/ripgrep/git heuristic — false-negative on UTF-16
    text (which legitimately contains null bytes) is acceptable here
    because the sandbox is rooted in a UTF-8 codebase.
    """
    return b"\x00" in blob


def read_text(
    root: str,
    path: str,
    *,
    max_bytes: int = _MAX_READ_BYTES,
) -> str:
    """Read a text file rooted at ``root``. ``root`` is the validated
    sandbox path (call :func:`validate_root` first).

    Raises :class:`SandboxEscapeError` on path escape, :class:`FileNot
    FoundError` if the resolved path is not a file, and
    :class:`SandboxBinaryError` if the file's first 8 KiB contains a
    null byte (the standard binary-content heuristic). Binary detection
    is the AGORA Council #373 concern 4 fix — without it the function
    would silently return UTF-8 replacement-character garbage to the
    model.
    """
    try:
        resolved = _resolve_safe(root, path)
    except SandboxEscapeError:
        _audit({"tool": "Read", "root": root, "path": path,
                "allowed": False, "reason": "escape"})
        raise

    if not os.path.isfile(resolved):
        _audit({"tool": "Read", "root": root, "path": path,
                "allowed": False, "reason": "not_a_file"})
        raise FileNotFoundError(f"not a file: {path!r}")

    # Binary sniff first — open in binary mode, peek first 8 KiB.
    # If null byte present → SandboxBinaryError (the dispatcher catches
    # and surfaces as a recoverable tool_result error so the model can
    # pick a different tool).
    with open(resolved, "rb") as f:
        head = f.read(_BINARY_SNIFF_BYTES)
        if _looks_binary(head):
            _audit({"tool": "Read", "root": root, "path": path,
                    "allowed": False, "reason": "binary"})
            raise SandboxBinaryError(
                f"file {path!r} appears to be binary "
                f"(null byte in first {_BINARY_SNIFF_BYTES} bytes)"
            )
        rest = f.read(max(0, max_bytes + 1 - len(head)))
    blob = head + rest
    truncated = len(blob) > max_bytes
    if truncated:
        blob = blob[:max_bytes]
    # Decode AFTER the binary check so replacement characters can only
    # appear on partial-UTF-8 source files, never on binary content.
    content = blob.decode("utf-8", errors="replace")

    _audit({"tool": "Read", "root": root, "path": path,
            "allowed": True, "bytes": len(blob),
            "truncated": truncated})
    return content


@dataclass(frozen=True)
class GrepMatch:
    """A single grep hit. ``path`` is RELATIVE to the sandbox root."""
    path: str
    line: int
    text: str


def _ripgrep_available() -> bool:
    return shutil.which("rg") is not None


def _parse_rg_line(line: str, root: str) -> Optional[GrepMatch]:
    """Parse one ripgrep output line ``<path>:<line>:<text>``. Returns
    None on malformed line or relative-path that escapes root."""
    parts = line.split(":", 2)
    if len(parts) != 3:
        return None
    abs_path, line_no, text = parts
    try:
        rel = os.path.relpath(abs_path, root)
        line_int = int(line_no)
    except ValueError:
        return None
    if rel.startswith(".."):
        return None  # safety belt
    return GrepMatch(path=rel, line=line_int, text=text[:200])


def _grep_via_ripgrep(
    root: str, pattern: str, glob_filter: Optional[str],
    max_hits: int, timeout_s: int,
) -> list[GrepMatch]:
    """Grep via ``rg`` subprocess. Raises :class:`SandboxTimeoutError`
    on subprocess timeout so the caller can surface "we timed out — try
    a narrower glob_filter" to the model instead of silently returning
    empty (the AGORA #373 concern 1 fix). ``OSError`` (rg not found, I/O
    error) still returns empty list — the model has no actionable
    recovery there."""
    argv = ["rg", "--no-heading", "--line-number",
            "--max-count", str(max_hits), "-e", pattern]
    if glob_filter:
        argv += ["--glob", glob_filter]
    argv += [root]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        _audit({"tool": "Grep", "root": root, "pattern": pattern,
                "allowed": False,
                "reason": f"timeout_{timeout_s}s:{exc}"})
        raise SandboxTimeoutError(
            f"grep timed out after {timeout_s}s — "
            f"try a narrower glob_filter or split the query"
        ) from exc
    except OSError as exc:
        _audit({"tool": "Grep", "root": root, "pattern": pattern,
                "allowed": False, "reason": f"rg_failed:{exc}"})
        return []
    hits: list[GrepMatch] = []
    for line in proc.stdout.splitlines():
        match = _parse_rg_line(line, root)
        if match is None:
            continue
        hits.append(match)
        if len(hits) >= max_hits:
            break
    return hits


def _grep_file_stdlib(
    root: str, abs_path: str, regex, remaining: int,
) -> list[GrepMatch]:
    """Grep one file with the compiled regex. Returns up to
    ``remaining`` hits."""
    found: list[GrepMatch] = []
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if regex.search(line):
                    rel = os.path.relpath(abs_path, root)
                    found.append(GrepMatch(
                        path=rel, line=i,
                        text=line.rstrip("\n")[:200]))
                    if len(found) >= remaining:
                        break
    except OSError:
        pass
    return found


def _walk_files(root: str):
    """Yield absolute paths to every regular file under ``root``.

    Uses ``yield from`` over a per-directory generator expression so
    the iteration is a single conceptual stream of files. Algorithmic
    complexity is O(F) where F is the total file count (linear in
    files visited, NOT quadratic — the syntactic nesting is the
    standard ``os.walk`` shape; the per-directory inner expression is
    O(files_in_dir), summed across all directories to O(F))."""
    for dirpath, _dirs, files in os.walk(root):
        yield from (os.path.join(dirpath, fname) for fname in files)


def _grep_via_stdlib(
    root: str, pattern: str, max_hits: int, timeout_s: int,
) -> list[GrepMatch]:
    """Pure-stdlib grep fallback (no ``rg`` available). Single loop
    over a flat file generator with a wall-clock deadline — raises
    :class:`SandboxTimeoutError` if ``timeout_s`` is exceeded so the
    caller can surface the timeout to the model (AGORA #373 concern 1).
    """
    try:
        regex = re.compile(pattern)
    except re.error:
        _audit({"tool": "Grep", "root": root, "pattern": pattern,
                "allowed": False, "reason": "bad_regex"})
        return []
    deadline = time.monotonic() + timeout_s
    hits: list[GrepMatch] = []
    for full in _walk_files(root):
        if time.monotonic() > deadline:
            _audit({"tool": "Grep", "root": root, "pattern": pattern,
                    "allowed": False,
                    "reason": f"stdlib_timeout_{timeout_s}s"})
            raise SandboxTimeoutError(
                f"grep timed out after {timeout_s}s — "
                f"try a narrower glob_filter or split the query"
            )
        remaining = max_hits - len(hits)
        if remaining <= 0:
            return hits
        hits.extend(_grep_file_stdlib(root, full, regex, remaining))
    return hits


def grep_pattern(
    root: str,
    pattern: str,
    *,
    glob_filter: Optional[str] = None,
    max_hits: int = _MAX_GREP_HITS,
    timeout_s: Optional[int] = None,
) -> list[GrepMatch]:
    """Grep ``pattern`` under ``root``. Returns list of GrepMatch with
    paths relative to root.

    Uses ``rg`` if available (faster, respects .gitignore), falls back
    to a stdlib walker otherwise. The sandbox bounds the filesystem,
    not the search expression (any regex permitted).

    Raises :class:`SandboxTimeoutError` if the search exceeds
    ``timeout_s`` (default :data:`_DEFAULT_GREP_TIMEOUT_S`, tunable via
    ``HAL_FILE_SANDBOX_GREP_TIMEOUT_S``). Surfaced as a recoverable tool
    error by the dispatcher so the model can narrow the query."""
    if timeout_s is None:
        timeout_s = _DEFAULT_GREP_TIMEOUT_S
    if _ripgrep_available():
        hits = _grep_via_ripgrep(
            root, pattern, glob_filter, max_hits, timeout_s,
        )
    else:
        hits = _grep_via_stdlib(root, pattern, max_hits, timeout_s)
    _audit({"tool": "Grep", "root": root, "pattern": pattern,
            "glob_filter": glob_filter, "allowed": True,
            "hits": len(hits)})
    return hits


def glob_pattern(
    root: str,
    pattern: str,
    *,
    max_hits: int = _MAX_GLOB_HITS,
    timeout_s: Optional[int] = None,
) -> list[str]:
    """Glob ``pattern`` under ``root``. Returns list of paths RELATIVE
    to root. Patterns are interpreted by ``pathlib.Path.rglob``.

    Absolute patterns or patterns starting with ``/`` are refused.

    Raises :class:`SandboxTimeoutError` if traversal exceeds
    ``timeout_s`` (default :data:`_DEFAULT_GLOB_TIMEOUT_S`, tunable via
    ``HAL_FILE_SANDBOX_GLOB_TIMEOUT_S``). The wall-clock guard bounds
    pathological cases — a symlink loop, a recursive bind-mount, or a
    legitimate huge tree — so the tool-loop never blocks indefinitely."""
    if pattern.startswith("/") or pattern.startswith(os.sep):
        _audit({"tool": "Glob", "root": root, "pattern": pattern,
                "allowed": False, "reason": "absolute_pattern"})
        raise SandboxEscapeError(
            f"glob pattern must be relative to root: {pattern!r}")

    if timeout_s is None:
        timeout_s = _DEFAULT_GLOB_TIMEOUT_S
    deadline = time.monotonic() + timeout_s
    hits: list[str] = []
    root_path = Path(root)
    try:
        for match in root_path.rglob(pattern):
            if time.monotonic() > deadline:
                _audit({"tool": "Glob", "root": root, "pattern": pattern,
                        "allowed": False,
                        "reason": f"timeout_{timeout_s}s"})
                raise SandboxTimeoutError(
                    f"glob timed out after {timeout_s}s — "
                    f"narrow the pattern or split the query"
                )
            # Defence-in-depth: confirm the match really lives under root
            try:
                _resolve_safe(root, str(match))
            except SandboxEscapeError:
                continue
            try:
                rel = match.relative_to(root_path)
            except ValueError:
                continue
            hits.append(str(rel))
            if len(hits) >= max_hits:
                break
    except OSError as exc:
        _audit({"tool": "Glob", "root": root, "pattern": pattern,
                "allowed": False, "reason": f"oserror:{exc}"})
        return hits

    _audit({"tool": "Glob", "root": root, "pattern": pattern,
            "allowed": True, "hits": len(hits)})
    return hits
