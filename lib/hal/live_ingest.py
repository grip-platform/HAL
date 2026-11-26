"""HAL Live Ingestion — opt-in file watcher primitives (Phase β W5).

Decision primitives + provenance population + audit log contract for
the ``hal --watch`` daemon. The watchdog / pyperclip integration lives
downstream; this module ships the safety scaffolding the council
RETHINK requires: opt-in activation, source allow-list, provenance-
populated MemoryEntry output, and a one-line audit record per decision.

See ``plans/live-ingestion-verdict.md`` for the full council transcript
and the concerns-to-invariants mapping.

Public API:
    LiveIngestConfig — opt-in flag + allow-list + audit path
    IngestionCandidate — path + source + detected_at + metadata
    IngestionDecision — (should_ingest, reason)
    should_ingest(candidate, config) -> IngestionDecision
    build_memory_entry(candidate, config, content) -> MemoryEntry  [1.1]
    audit_record(decision, candidate) -> str  (one line)
    IngestionDisabledError — raised when daemon tries to ingest without opt-in
"""
from __future__ import annotations

import fnmatch
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from lib.hal.memory_engine import MemoryEntry


class IngestionDisabledError(RuntimeError):
    """Raised when ingestion is attempted without explicit opt-in."""


@dataclass(frozen=True)
class IngestionCandidate:
    path: str
    source: str  # "downloads" | "clipboard" | "watch-dir" | arbitrary string
    detected_at: float = field(default_factory=time.time)
    summary: str = ""  # short human description for the entry name


@dataclass(frozen=True)
class IngestionDecision:
    should_ingest: bool
    reason: str


@dataclass
class LiveIngestConfig:
    enabled: bool = False
    allow_sources: tuple[str, ...] = ()
    allow_extensions: tuple[str, ...] = (".md", ".pdf", ".txt", ".wav", ".mp3")
    allow_path_globs: tuple[str, ...] = ()  # e.g. ("~/Downloads/*.pdf",)
    session_id: str = ""
    audit_log_path: str = ""

    @classmethod
    def from_env(cls) -> "LiveIngestConfig":
        """Build a config from environment variables — opt-in by default.

        * ``HAL_LIVE_INGEST=1`` enables the daemon.
        * ``HAL_LIVE_INGEST_SOURCES=comma,separated`` gates sources.
        * ``HAL_LIVE_INGEST_GLOBS=pattern1:pattern2`` path allow-list.
        * ``HAL_LIVE_INGEST_EXTS=.md,.pdf`` override extension allow.
        * ``HAL_INSTANCE_ID`` populates ``source_session_id`` in entries.
        """
        def _split(val: str, sep: str) -> tuple[str, ...]:
            return tuple(p.strip() for p in val.split(sep) if p.strip())

        return cls(
            enabled=os.environ.get("HAL_LIVE_INGEST", "").lower() in ("1", "true", "yes"),
            allow_sources=_split(os.environ.get("HAL_LIVE_INGEST_SOURCES", ""), ","),
            allow_extensions=_split(os.environ.get("HAL_LIVE_INGEST_EXTS", ""), ",") or
                             (".md", ".pdf", ".txt", ".wav", ".mp3"),
            allow_path_globs=_split(os.environ.get("HAL_LIVE_INGEST_GLOBS", ""), ":"),
            session_id=os.environ.get("HAL_INSTANCE_ID", ""),
            audit_log_path=os.environ.get("HAL_LIVE_INGEST_AUDIT_LOG", ""),
        )


def should_ingest(
    candidate: IngestionCandidate,
    config: LiveIngestConfig,
) -> IngestionDecision:
    """Decide whether ``candidate`` may be ingested under ``config``.

    Fail-closed: every reject path returns a specific reason string so
    the audit log can distinguish "disabled" from "wrong source" from
    "wrong extension" from "glob mismatch". The positive path requires
    the candidate to pass ALL four checks (enabled, source, extension,
    glob) — every barrier is in AND.
    """
    if not config.enabled:
        return IngestionDecision(False, "daemon disabled (HAL_LIVE_INGEST not set)")
    if config.allow_sources and candidate.source not in config.allow_sources:
        return IngestionDecision(
            False, f"source '{candidate.source}' not in allow-list",
        )
    if config.allow_extensions:
        ext = Path(candidate.path).suffix.lower()
        if ext not in config.allow_extensions:
            return IngestionDecision(False, f"extension '{ext}' not in allow-list")
    if config.allow_path_globs:
        expanded = os.path.expanduser(candidate.path)
        matched = any(
            fnmatch.fnmatch(expanded, os.path.expanduser(g))
            for g in config.allow_path_globs
        )
        if not matched:
            return IngestionDecision(False, "path does not match any allow-list glob")
    return IngestionDecision(True, "allowed")


def build_memory_entry(
    candidate: IngestionCandidate,
    config: LiveIngestConfig,
    content: str,
) -> MemoryEntry:
    """Produce a 1.1 MemoryEntry with live-ingestion provenance populated.

    Fails loudly via ``IngestionDisabledError`` when the config is not
    enabled — a defence-in-depth check so a downstream bug cannot
    silently ingest without opt-in.
    """
    if not config.enabled:
        raise IngestionDisabledError(
            "build_memory_entry called without HAL_LIVE_INGEST=1",
        )
    stem = Path(candidate.path).stem
    filename = f"ingest-{stem}-{int(candidate.detected_at)}.md"
    name = candidate.summary or f"Ingested: {Path(candidate.path).name}"
    description = f"auto-ingested from {candidate.source}"
    return MemoryEntry(
        filename=filename,
        name=name,
        description=description,
        memory_type="reference",
        content=content,
        source_session_id=config.session_id or "live-ingest-daemon",
        tool_calls_used=["LiveIngest"],
        council_vote_hashes=[],
    )


def audit_record(
    decision: IngestionDecision,
    candidate: IngestionCandidate,
) -> str:
    """Render a single ``time\tsource\tverdict\treason\tpath`` audit line."""
    verdict = "ACCEPT" if decision.should_ingest else "REJECT"
    iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(candidate.detected_at))
    return f"{iso}\t{candidate.source}\t{verdict}\t{decision.reason}\t{candidate.path}"


def append_audit(log_path: str, line: str) -> None:
    """Append one audit line, creating the parent dir if needed."""
    if not log_path:
        return
    path = os.path.expanduser(log_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


__all__ = [
    "IngestionCandidate",
    "IngestionDecision",
    "IngestionDisabledError",
    "LiveIngestConfig",
    "append_audit",
    "audit_record",
    "build_memory_entry",
    "should_ingest",
]
