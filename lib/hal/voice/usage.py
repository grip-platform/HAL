"""HAL Voice — usage telemetry aggregator (Tier 3 W5).

Per plans/voice-tier3-cloud-fallover.md §4 W5.

JSONL-append counter for per-provider synthesis volume. W2/W3 each
carry their own provider-local quota/RPM accounting; this module is
the GLOBAL reporting surface — `hal voice usage` consumes it, and in
a later wave the provider-local helpers will migrate here.

State file: ``~/.hal/state/voice/usage.jsonl``

Record shape: ``{"ts": ISO8601, "ym": "YYYY-MM", "provider": str,
"chars": int, "profile": str}``

Concurrency: ``record()`` acquires an exclusive ``fcntl.flock`` on the
append fd so two concurrent ``voice.speak`` calls from separate
processes cannot interleave their writes.

Operator report model: monthly buckets (UTC) because every free-tier
provider we dispatch to (Google Cloud TTS, Gemini) tracks quotas at
the calendar-month grain. Cross-month totals are NOT reported at the
CLI boundary — operators care about "will I blow the cap this
month", not historical totals.
"""
from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_USAGE_PATH = Path.home() / ".hal" / "state" / "voice" / "usage.jsonl"
WARN_THRESHOLD_PCT = 80.0

# Known free-tier monthly character limits for CLI reporting. Providers
# whose budget is not char-based (Gemini uses RPM; Piper + macos_say are
# local/unlimited) appear here as None so the CLI can render "n/a".
FREE_TIER_LIMITS: dict[str, int | None] = {
    "google_cloud_tts_free": 1_000_000,
    "gemini_tts_free": None,       # RPM, not chars — tracked provider-side
    "piper": None,                  # local, unlimited
    "macos_say": None,              # local, unlimited
}


def _now_ym() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class UsageRow:
    """One row of the per-provider monthly usage summary."""

    provider: str
    chars_this_month: int
    free_tier_limit: int | None
    percent_used: float | None

    @property
    def over_warn_threshold(self) -> bool:
        return (
            self.percent_used is not None
            and self.percent_used >= WARN_THRESHOLD_PCT
        )


class UsageCounter:
    """Append-only per-provider usage tracker.

    Thread-safe and process-safe via ``fcntl.flock``. Malformed lines
    (partial writes, manual edits) are skipped silently rather than
    propagated as exceptions — operator-diagnosable debris must not
    break the synthesis path.
    """

    def __init__(self, *, path: Path | None = None) -> None:
        self._path = path or DEFAULT_USAGE_PATH

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        provider: str,
        chars: int,
        *,
        profile: str = "default",
    ) -> None:
        """Append one synthesis record.

        Called from ``voice.speak`` after the provider returns a
        successful ``TtsSynthResult``. Counter records the LOGICAL
        synthesis size (``len(text)``) rather than the audio byte
        count — quotas are char-priced, not byte-priced.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": _now_ts(),
            "ym": _now_ym(),
            "provider": str(provider),
            "chars": int(chars),
            "profile": str(profile),
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with self._path.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def chars_this_month(
        self,
        provider: str,
        *,
        ym: str | None = None,
    ) -> int:
        """Sum chars for ``provider`` in the named YYYY-MM bucket.

        ``ym`` defaults to the current UTC month; tests override it
        to simulate cross-month behaviour deterministically.
        """
        target_ym = ym or _now_ym()
        if not self._path.exists():
            return 0
        total = 0
        with self._path.open("r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    record = _safe_json_loads(line)
                    if record is None:
                        continue
                    if (
                        record.get("ym") == target_ym
                        and record.get("provider") == provider
                    ):
                        total += int(record.get("chars", 0))
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return total

    def summary(
        self,
        *,
        ym: str | None = None,
        limits: dict[str, int | None] | None = None,
    ) -> list[UsageRow]:
        """Per-provider monthly summary, sorted by name.

        Providers present in ``limits`` but absent from the file are
        reported with ``chars_this_month=0`` so operators see the
        complete fallover ladder at a glance. Providers in the file
        but not in ``limits`` are reported with ``free_tier_limit=None``
        (unknown cap) so unexpected providers don't disappear.
        """
        limits_map = dict(FREE_TIER_LIMITS if limits is None else limits)
        target_ym = ym or _now_ym()
        seen: set[str] = set()
        rows: list[UsageRow] = []
        for provider in sorted(limits_map.keys()):
            chars = self.chars_this_month(provider, ym=target_ym)
            limit = limits_map.get(provider)
            pct = (chars / limit * 100.0) if limit else None
            rows.append(
                UsageRow(
                    provider=provider,
                    chars_this_month=chars,
                    free_tier_limit=limit,
                    percent_used=pct,
                )
            )
            seen.add(provider)
        # Sweep file for any extra providers not in limits_map.
        extras: set[str] = set()
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    for line in f:
                        record = _safe_json_loads(line)
                        if record is None:
                            continue
                        if record.get("ym") != target_ym:
                            continue
                        provider = record.get("provider")
                        if isinstance(provider, str) and provider not in seen:
                            extras.add(provider)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        for provider in sorted(extras):
            chars = self.chars_this_month(provider, ym=target_ym)
            rows.append(
                UsageRow(
                    provider=provider,
                    chars_this_month=chars,
                    free_tier_limit=None,
                    percent_used=None,
                )
            )
        return rows


def _safe_json_loads(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


__all__ = [
    "DEFAULT_USAGE_PATH",
    "FREE_TIER_LIMITS",
    "UsageCounter",
    "UsageRow",
    "WARN_THRESHOLD_PCT",
]
