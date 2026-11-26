"""HAPPI/1.1 v1.1-only cmd handlers (P3 of v1.1 alignment).

Mirrors the canonical Python runtime in ``~/.hal/happi.md`` (also embedded
at ``happi-sources/04-python-runtime.py``). The four cmds added here:

    idr.emit             — sha256 of (envelope_json + ndjson_events)
    pr.reference         — informational PR reference (fractal-seed metadata)
    hypothesis.register  — append a falsifiable hypothesis to NDJSON log
    quine.spawn          — DRY-RUN by default; LIVE creates child issue via gh

Each handler yields NDJSON event lines via the shared encoders in
``lib/hal/happi/events.py`` so the protocol-version flag (P2) propagates
automatically. Argument validation matches the canonical runtime's
``parse_error`` / ``runtime_error`` codes byte-for-byte (modulo the
non-deterministic fields ``ts`` and generated issue numbers).

Side-effects gated:
    - ``hypothesis.register`` writes to ``$HAL_HYPOTHESES_PATH`` (default
      ``~/.hal/data/hypotheses.jsonl``). Tests must set the env var to a
      tmp path to avoid polluting the production registry.
    - ``quine.spawn`` LIVE branch invokes the ``gh`` CLI. DRY-RUN (the
      default unless ``flags.live=true``) emits no side effects.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Iterator

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.events import emit_completed, emit_delta, emit_error, emit_event


V11_RUNTIME_COMMANDS = frozenset({
    "idr.emit",
    "pr.reference",
    "hypothesis.register",
    "quine.spawn",
})


# ---------- idr.emit -------------------------------------------------------

def dispatch_idr_emit(envelope: HappiEnvelope) -> Iterator[str]:
    """Compute IDR for a recorded (envelope, event-stream) pair.

    args[0] = envelope JSON string; args[1] = NDJSON event stream string.
    Optional flags: ``model_versions`` (list[str]), ``block_anchor`` (int|null).
    """
    args = envelope.args
    if len(args) < 2 or not all(isinstance(a, str) for a in args[:2]):
        yield emit_error(
            envelope.id,
            "idr.emit: requires args=[envelope_json, ndjson_events]",
            code="parse_error",
        )
        return
    envelope_str, events_str = args[0], args[1]
    h = hashlib.sha256()
    h.update(envelope_str.encode("utf-8"))
    h.update(events_str.encode("utf-8"))
    flags = envelope.flags or {}
    idr_ref = {
        "sha256": h.hexdigest(),
        "cid": None,
        "model_versions": list(flags.get("model_versions") or []),
        "block_anchor": flags.get("block_anchor"),
    }
    yield emit_event(envelope.id, "idr", idr_ref=idr_ref)
    yield emit_completed(envelope.id)


# ---------- pr.reference ---------------------------------------------------

def _pr_reference_parts(pr: int, repo: str, description: str,
                        flags: dict[str, Any]) -> list[str]:
    """Build the joined-by-' | ' parts list for the pr.reference delta."""
    parts: list[str] = ["PR #" + str(pr) + " (" + repo + ")"]
    if description:
        parts.append(description)
    if flags.get("status"):
        parts.append("status=" + str(flags["status"]))
    if flags.get("contributes_axiom"):
        parts.append("axiom=" + str(flags["contributes_axiom"]))
    return parts


def dispatch_pr_reference(envelope: HappiEnvelope) -> Iterator[str]:
    """Record a pull request as part of a HAPPI seed (informational only).

    Required: flags.pr (int), flags.repo ("org/repo" str)
    Optional: args[0] (description), flags.status, flags.contributes_axiom
    """
    flags = envelope.flags or {}
    pr = flags.get("pr")
    repo = flags.get("repo")
    if not isinstance(pr, int) or isinstance(pr, bool):
        yield emit_error(
            envelope.id,
            "pr.reference: requires flags.pr (int)",
            code="parse_error",
        )
        return
    if not isinstance(repo, str) or not repo:
        yield emit_error(
            envelope.id,
            "pr.reference: requires flags.repo (str)",
            code="parse_error",
        )
        return
    description = envelope.args[0] if envelope.args else ""
    parts = _pr_reference_parts(pr, repo, description, flags)
    yield emit_delta(envelope.id, " | ".join(parts))
    yield emit_completed(envelope.id, usage={"pr": pr, "repo": repo})


# ---------- hypothesis.register --------------------------------------------

_HYP_REQUIRED_FLAGS = ("claim", "metric", "prediction", "deadline")


def _hyp_validate(envelope: HappiEnvelope) -> tuple[str, dict[str, Any]] | str:
    """Validate envelope; return (hyp_id, flags) on success, error-message str on failure."""
    args = envelope.args
    if not args or not isinstance(args[0], str) or not args[0]:
        return "hypothesis.register: requires args[0] = hypothesis ID"
    flags = envelope.flags or {}
    missing = [k for k in _HYP_REQUIRED_FLAGS
               if not isinstance(flags.get(k), str) or not flags[k]]
    if missing:
        return "hypothesis.register: missing flags: " + ",".join(missing)
    return args[0], flags


def _hyp_write(record: dict[str, Any]) -> str:
    """Append one NDJSON record to the hypotheses log; return resolved out_path."""
    default = os.path.expanduser("~/.hal/data/hypotheses.jsonl")
    out_path = os.environ.get("HAL_HYPOTHESES_PATH", default)
    parent_dir = os.path.dirname(out_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
    return out_path


def dispatch_hypothesis_register(envelope: HappiEnvelope) -> Iterator[str]:
    """Append a falsifiable hypothesis to NDJSON log.

    Required: args[0] (hyp_id), flags.{claim, metric, prediction, deadline}.
    Override path via ``HAL_HYPOTHESES_PATH`` env var.
    """
    validated = _hyp_validate(envelope)
    if isinstance(validated, str):
        yield emit_error(envelope.id, validated, code="parse_error")
        return
    hyp_id, flags = validated
    record = {
        "v": "happi/1.1", "id": hyp_id,
        "claim": flags["claim"], "metric": flags["metric"],
        "prediction": flags["prediction"], "deadline": flags["deadline"],
        "registered_at_unix": int(time.time()), "status": "pending",
    }
    try:
        out_path = _hyp_write(record)
    except OSError as e:
        yield emit_error(
            envelope.id,
            "hypothesis.register: cannot write — " + str(e),
            code="runtime_error",
        )
        return
    yield emit_delta(envelope.id,
                     "registered " + hyp_id + " | claim: " + flags["claim"][:80])
    yield emit_delta(envelope.id,
                     "deadline " + flags["deadline"] + " | logged to " + out_path)
    yield emit_completed(envelope.id,
                         usage={"hypothesis_id": hyp_id, "path": out_path})


# ---------- quine.spawn ----------------------------------------------------


class _QuineError(Exception):
    """Internal signal for quine.spawn helpers to surface a runtime_error."""


def _fetch_parent_issue(repo: str, issue_num: int) -> tuple[str, str]:
    """Fetch parent title + body via gh CLI. Raises _QuineError on failure."""
    cmd = ["gh", "issue", "view", str(issue_num), "--repo", repo,
           "--json", "title,body"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise _QuineError("gh CLI unavailable: " + type(e).__name__)
    if result.returncode != 0:
        raise _QuineError("gh issue view failed: " + result.stderr.strip()[:200])
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise _QuineError("gh returned non-JSON: " + str(e))
    return data["title"], data["body"]


def _check_and_bump(title: str, body: str, depth_limit: int,
                    parent_issue: int) -> tuple[str, int]:
    """Validate parent is a HAPPI seed and bump generation counter in title."""
    if "quine.spawn" not in body:
        raise _QuineError(
            "parent #" + str(parent_issue)
            + " is not a HAPPI seed (no quine.spawn envelope)"
        )
    m = re.search(r"generation\s+(\d+)", title, re.IGNORECASE)
    if not m:
        raise _QuineError("parent title lacks 'generation N' counter")
    current = int(m.group(1))
    if current >= depth_limit:
        raise _QuineError(
            "generation " + str(current) + " >= depth_limit " + str(depth_limit)
        )
    new_title = re.sub(
        r"(generation\s+)(\d+)",
        lambda x: x.group(1) + str(current + 1),
        title, count=1, flags=re.IGNORECASE,
    )
    return new_title, current + 1


def _spawn_dry_run(envelope: HappiEnvelope, parent: int, new_title: str,
                   body: str, next_gen: int, depth_limit: int) -> Iterator[str]:
    """Emit DRY-RUN events without creating any GitHub issue."""
    yield emit_delta(envelope.id, "DRY-RUN: would spawn from #" + str(parent))
    yield emit_delta(envelope.id, "proposed_title: " + new_title)
    yield emit_delta(envelope.id,
                     "body_preview: " + body[:200].replace("\n", " "))
    yield emit_delta(envelope.id, "generation: " + str(next_gen))
    yield emit_delta(envelope.id,
                     "depth_limit: " + str(depth_limit)
                     + " (set flags.live=true to actually create)")
    yield emit_completed(envelope.id,
                         usage={"dry_run": True, "next_generation": next_gen})


def _create_child_issue(repo: str, title: str, body: str) -> int:
    """Create child issue via gh CLI. Returns child issue number; raises on failure."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md",
                                     delete=False, encoding="utf-8") as f:
        f.write(body)
        body_file = f.name
    try:
        cmd = ["gh", "issue", "create", "--repo", repo,
               "--title", title, "--body-file", body_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise _QuineError(
                "gh issue create failed: " + result.stderr.strip()[:200]
            )
        m = re.search(r"/issues/(\d+)", result.stdout)
        if not m:
            raise _QuineError(
                "could not parse issue # from: " + result.stdout[:200]
            )
        return int(m.group(1))
    finally:
        os.unlink(body_file)


def _append_quine_audit(parent: int, child: int, generation: int,
                        repo: str) -> str:
    """Append shared audit record to data/quine-spawn-audit.jsonl. Returns path."""
    audit_path = os.path.expanduser("~/.hal/data/quine-spawn-audit.jsonl")
    parent_dir = os.path.dirname(audit_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    record = {
        "v": "happi/1.1",
        "parent_issue": parent, "child_issue": child,
        "generation": generation, "repo": repo,
        "ts_unix": int(time.time()),
    }
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
    return audit_path


def _spawn_live(envelope: HappiEnvelope, parent: int, repo: str,
                new_title: str, body: str, next_gen: int) -> Iterator[str]:
    """LIVE branch: create child issue + append audit. Emits live events."""
    try:
        child = _create_child_issue(repo, new_title, body)
    except _QuineError as e:
        yield emit_error(envelope.id, "quine.spawn LIVE: " + str(e),
                         code="runtime_error")
        return
    audit_path = _append_quine_audit(parent, child, next_gen, repo)
    yield emit_delta(envelope.id,
                     "LIVE: spawned child #" + str(child)
                     + " from #" + str(parent))
    yield emit_delta(envelope.id, "title: " + new_title)
    yield emit_delta(envelope.id, "generation: " + str(next_gen))
    yield emit_delta(envelope.id, "audit: " + audit_path)
    yield emit_completed(envelope.id, usage={
        "parent": parent, "child": child,
        "generation": next_gen, "audit": audit_path,
    })


def _quine_validate_flags(envelope: HappiEnvelope) -> tuple[int, str] | str:
    """Validate parent_issue + repo flags. Return (parent, repo) on success, error str on failure."""
    flags = envelope.flags or {}
    parent_issue = flags.get("parent_issue")
    repo = flags.get("repo")
    if not isinstance(parent_issue, int) or isinstance(parent_issue, bool):
        return "quine.spawn: requires flags.parent_issue (int)"
    if not isinstance(repo, str) or not repo:
        return "quine.spawn: requires flags.repo (str)"
    return parent_issue, repo


def dispatch_quine_spawn(envelope: HappiEnvelope) -> Iterator[str]:
    """Spawn a child HAPPI seed issue from a parent (DRY-RUN by default).

    Required: flags.parent_issue (int), flags.repo ("org/repo")
    Optional: flags.live (bool, default False — DRY-RUN unless explicit),
              flags.depth_limit (int, default 16; HAPPI_QUINE_DEPTH_MAX env override)
    """
    validated = _quine_validate_flags(envelope)
    if isinstance(validated, str):
        yield emit_error(envelope.id, validated, code="parse_error")
        return
    parent_issue, repo = validated
    flags = envelope.flags or {}
    default_depth = int(os.environ.get("HAPPI_QUINE_DEPTH_MAX", "16"))
    depth_limit = flags.get("depth_limit", default_depth)
    try:
        title, body = _fetch_parent_issue(repo, parent_issue)
        new_title, next_gen = _check_and_bump(title, body, depth_limit,
                                              parent_issue)
    except _QuineError as e:
        yield emit_error(envelope.id, "quine.spawn: " + str(e),
                         code="runtime_error")
        return
    if not bool(flags.get("live", False)):
        yield from _spawn_dry_run(envelope, parent_issue, new_title, body,
                                  next_gen, depth_limit)
        return
    yield from _spawn_live(envelope, parent_issue, repo, new_title, body,
                           next_gen)


# ---------- Dispatch table -------------------------------------------------

V11_HANDLERS = {
    "idr.emit": dispatch_idr_emit,
    "pr.reference": dispatch_pr_reference,
    "hypothesis.register": dispatch_hypothesis_register,
    "quine.spawn": dispatch_quine_spawn,
}
