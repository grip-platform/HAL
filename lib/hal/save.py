"""HAL session save — lightweight persistence for session boundaries.

Phase B deliverable (H155): a HAL-native equivalent of GRIP's
``lib/grip_save.py`` that does NOT inject ``$CLAUDE_DIR`` on the
PYTHONPATH. Intentionally narrower than GRIP's orchestrator:

* Write one JSONL line to ``$HAL_HOME/state/session-log.jsonl``
* Update ``$HAL_HOME/state/session-context.md`` with the latest entry
* Touch the instance-next-up pointer at
  ``$HAL_HOME/state/instance-next-up.txt`` if an achievement was given

Out of scope (by design): genome evolution, KONO embedding sync,
DELIGHT snapshot, hypothesis verify. Those are GRIP-internal and
should not be reimplemented in HAL — they belong behind an integration
seam on the GRIP side.

CLI: ``hal save [achievement]`` — dispatched from ``bin/hal``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

__all__ = [
    "SaveEntry",
    "SaveResult",
    "save_session",
    "session_log_path",
    "session_context_path",
    "instance_next_up_path",
    "main",
]


def _hal_home() -> Path:
    return Path(os.environ.get("HAL_HOME", str(Path.home() / ".hal")))


def session_log_path() -> Path:
    return _hal_home() / "state" / "session-log.jsonl"


def session_context_path() -> Path:
    return _hal_home() / "state" / "session-context.md"


def instance_next_up_path() -> Path:
    return _hal_home() / "state" / "instance-next-up.txt"


@dataclass(frozen=True)
class SaveEntry:
    """One session-save record."""
    ts: str
    achievement: str
    cwd: str
    pid: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SaveResult:
    """Outcome of one save invocation."""
    log_path: Path
    context_path: Path
    next_up_written: bool
    entry: SaveEntry


def _iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _render_context(
    entries: List[SaveEntry], max_entries: int = 10,
) -> str:
    """Format the last N entries as the session-context.md surface.

    Newest-first. Truncates long achievement strings to keep the file
    scannable by a human."""
    header = "# HAL Session Context (auto-generated)\n\n"
    lines = ["## Last saves\n"]
    for entry in reversed(entries[-max_entries:]):
        achievement = entry.achievement or "(no achievement message)"
        if len(achievement) > 400:
            achievement = achievement[:397] + "..."
        lines.append(f"- `{entry.ts}` — {achievement}")
    return header + "\n".join(lines) + "\n"


def save_session(
    achievement: str = "",
    *,
    log_path: Optional[Path] = None,
    context_path: Optional[Path] = None,
    next_up_path: Optional[Path] = None,
) -> SaveResult:
    """Append one save entry, refresh context, optionally touch next-up."""
    entry = SaveEntry(
        ts=_iso_utc(),
        achievement=achievement,
        cwd=os.getcwd(),
        pid=os.getpid(),
    )
    log = log_path or session_log_path()
    ctx = context_path or session_context_path()
    next_up = next_up_path or instance_next_up_path()

    _append_jsonl(log, entry.to_dict())

    all_entries = _read_all_entries(log)
    ctx.parent.mkdir(parents=True, exist_ok=True)
    ctx.write_text(_render_context(all_entries), encoding="utf-8")

    wrote_next_up = False
    if achievement.strip():
        next_up.parent.mkdir(parents=True, exist_ok=True)
        next_up.write_text(f"{entry.ts} {achievement}\n", encoding="utf-8")
        wrote_next_up = True

    return SaveResult(
        log_path=log,
        context_path=ctx,
        next_up_written=wrote_next_up,
        entry=entry,
    )


def _read_all_entries(log_path: Path) -> List[SaveEntry]:
    if not log_path.exists():
        return []
    out: List[SaveEntry] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
            out.append(SaveEntry(**{
                k: payload[k] for k in ("ts", "achievement", "cwd", "pid")
                if k in payload
            }))
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point invoked by ``bin/hal save``.

    Accepts either positional message or stdin-piped input. Zero
    external dependencies (stdlib only).
    """
    args = argv if argv is not None else sys.argv[1:]
    message = " ".join(a for a in args if a).strip()
    if not message and not sys.stdin.isatty():
        try:
            message = sys.stdin.read().strip()
        except (OSError, ValueError):
            message = ""
    result = save_session(message)
    print(f"hal save: {result.entry.ts}")
    print(f"  log:     {result.log_path}")
    print(f"  context: {result.context_path}")
    if result.next_up_written:
        print(f"  next-up: {instance_next_up_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
