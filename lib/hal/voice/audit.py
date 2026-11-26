"""HAL Voice — HMAC-chained, append-only JSONL audit log.

Per plans/voice-first-harness-triphase.md Rev 2.3 section 4.1:

- Each record is a single-line JSON object whose `hmac` field is a
  sha256-HMAC over the canonicalised body (audit_id, ts, event,
  payload, prev_hmac). `prev_hmac` chains the record to the previous
  one, so any mutation anywhere in the chain breaks verification from
  that point onwards.
- `audit_id` is generated internally here via `secrets.token_hex(16)`
  and returned to the caller. The public `voice.listen` / `voice.speak`
  primitives attach it to their result dataclasses; callers cannot
  supply their own audit_id through this module's API.
- `chflags uappnd` (macOS) is applied to the log file so it cannot be
  truncated or overwritten — only appended. Rotation at 10 MB briefly
  unsets the flag to rename the file, re-applies the flag to the
  rotated file, and seeds the fresh file with a
  `voice.audit.rotation` record whose `prev_hmac` is the last HMAC of
  the rotated file so the chain survives across files.
- `verify_chain()` returns (True, None) on an unmodified chain and
  (False, line_number) at the first record that fails. Two-sided
  Goodhart-proof tests in tests/test_voice_audit.py exercise both
  branches (Rule 14).

Shipped at PR alpha-2 W2. The primitive bodies (voice.listen,
voice.speak, voice.chat_loop) consume this module via the public
`AuditLogger.append(event=..., payload=...)` call.
"""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .types import VoiceError


# ---------- constants ----------

ROTATION_BYTES_DEFAULT = 10 * 1024 * 1024  # 10 MB per plan §4.1
GENESIS_PREV_HMAC = "GENESIS"
_HMAC_FIELDS = ("audit_id", "ts", "event", "payload", "prev_hmac")
_AUDIT_KEY_FILENAME = "audit.key"
_KEY_MIN_BYTES = 32


# ---------- helpers ----------


def _now_iso() -> str:
    """ISO-8601 UTC timestamp with microseconds and trailing Z."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _canonical_bytes(body: Mapping[str, Any]) -> bytes:
    """Deterministic JSON over the HMAC body.

    Sorted keys + no whitespace + UTF-8 means the bytes used for
    verification match the bytes used to sign, regardless of dict
    ordering or Python version.
    """
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _compute_hmac(key: bytes, canonical: bytes) -> str:
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()


def _load_or_generate_key(key_path: Path) -> bytes:
    """Load a >=32-byte HMAC key, generating one on first use."""
    if key_path.exists():
        raw = key_path.read_bytes()
        if len(raw) < _KEY_MIN_BYTES:
            raise VoiceError(
                f"audit key at {key_path} is {len(raw)} bytes, need >={_KEY_MIN_BYTES}"
            )
        return raw
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(_KEY_MIN_BYTES)
    key_path.write_bytes(key)
    os.chmod(key_path, 0o600)
    return key


def _set_append_only(path: Path) -> None:
    """Set the OS-level append-only flag on ``path`` (macOS / Linux).

    - macOS: ``chflags uappnd`` — blocks truncate/overwrite by the
      file's owner; only open-append succeeds. Requires the calling
      user to own the file.
    - Linux: ``chattr +a`` — analogous; typically requires CAP_LINUX_IMMUTABLE.
    - Other platforms: silent no-op. Voice is macOS-first; the log
      still functions, it's just not hardware-backed append-only.
    """
    if sys.platform == "darwin":
        subprocess.run(["chflags", "uappnd", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["chattr", "+a", str(path)], check=False)


def _unset_append_only(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["chflags", "nouappnd", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["chattr", "-a", str(path)], check=False)


def _read_last_line(path: Path, tail_bytes: int = 4096) -> bytes | None:
    """Return the last non-empty line of ``path`` without slurping the whole file.

    Seeks from the end, reads up to ``tail_bytes``, and picks the last
    newline-delimited fragment. Returns None if the file is empty.
    """
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return None
        read_from = max(0, size - tail_bytes)
        f.seek(read_from)
        tail = f.read()
    # Strip trailing newline so rsplit's last fragment is the last record.
    if tail.endswith(b"\n"):
        tail = tail[:-1]
    if b"\n" not in tail:
        return tail or None
    return tail.rsplit(b"\n", 1)[-1] or None


# ---------- API ----------


@dataclass(frozen=True)
class AuditRecord:
    """Public representation of one audit entry (parsing side)."""

    audit_id: str
    ts: str
    event: str
    payload: Mapping[str, Any]
    prev_hmac: str
    hmac: str


class AuditLogger:
    """HMAC-chained append-only JSONL audit log.

    Threading/multi-process: append() acquires an exclusive
    ``fcntl.flock`` on the file before reading the last hmac AND
    writing the new line, so two concurrent processes cannot produce a
    fork in the chain.

    The ``set_append_only`` flag exists so tests can disable the
    privileged ``chflags`` calls and run in a sandbox; production
    voice.listen / voice.speak use the default True.
    """

    def __init__(
        self,
        *,
        path: Path | str,
        key: bytes | None = None,
        key_path: Path | str | None = None,
        rotation_bytes: int = ROTATION_BYTES_DEFAULT,
        set_append_only: bool = True,
    ) -> None:
        self._path = Path(path)
        self._rotation_bytes = int(rotation_bytes)
        self._set_append_only = bool(set_append_only)
        if key is not None:
            if len(key) < _KEY_MIN_BYTES:
                raise VoiceError(
                    f"audit key must be >={_KEY_MIN_BYTES} bytes, got {len(key)}"
                )
            self._key = key
        else:
            resolved_key_path = (
                Path(key_path) if key_path else self._path.parent / _AUDIT_KEY_FILENAME
            )
            self._key = _load_or_generate_key(resolved_key_path)

    @property
    def path(self) -> Path:
        return self._path

    def append(
        self,
        *,
        event: str,
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        """Append a new record; return the generated audit_id."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode; fcntl locks this specific fd so readers
        # (including our own _last_hmac inside the lock) are consistent.
        with self._path.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                first_write = f.tell() == 0
                prev_hmac = (
                    GENESIS_PREV_HMAC if first_write else self._last_hmac_unlocked()
                )
                record = self._build_record(
                    event=event,
                    payload=dict(payload or {}),
                    prev_hmac=prev_hmac,
                )
                line = (
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        if first_write and self._set_append_only:
            _set_append_only(self._path)

        if self._path.stat().st_size >= self._rotation_bytes:
            self._rotate()

        return record["audit_id"]

    def _build_record(
        self,
        *,
        event: str,
        payload: Mapping[str, Any],
        prev_hmac: str,
    ) -> dict[str, Any]:
        body = {
            "audit_id": secrets.token_hex(16),
            "ts": _now_iso(),
            "event": event,
            "payload": payload,
            "prev_hmac": prev_hmac,
        }
        body["hmac"] = _compute_hmac(self._key, _canonical_bytes(body))
        return body

    def _last_hmac_unlocked(self) -> str:
        """Peek the last record's hmac. Caller must hold the flock."""
        last = _read_last_line(self._path)
        if last is None:
            return GENESIS_PREV_HMAC
        try:
            return json.loads(last)["hmac"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise VoiceError(
                f"audit log {self._path} tail is malformed: {exc}"
            ) from exc

    def _rotate(self) -> None:
        """Rotate at the 10 MB threshold preserving chain continuity."""
        if not self._path.exists() or self._path.stat().st_size == 0:
            return
        last_hmac = self._last_hmac_unlocked()
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        rotated = self._path.with_name(f"{self._path.name}.{stamp}")
        collision = 0
        while rotated.exists():
            collision += 1
            rotated = self._path.with_name(f"{self._path.name}.{stamp}.{collision}")

        if self._set_append_only:
            _unset_append_only(self._path)
        self._path.rename(rotated)
        if self._set_append_only:
            _set_append_only(rotated)

        seed = self._build_record(
            event="voice.audit.rotation",
            payload={"rotated_to": rotated.name},
            prev_hmac=last_hmac,
        )
        line = (
            json.dumps(seed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n"
        )
        self._path.write_text(line, encoding="utf-8")
        if self._set_append_only:
            _set_append_only(self._path)


def verify_chain(
    path: Path | str,
    key: bytes,
    *,
    initial_prev_hmac: str = GENESIS_PREV_HMAC,
) -> tuple[bool, int | None]:
    """Verify the HMAC chain of a single audit.jsonl file.

    Returns (True, None) when every record's HMAC recomputes correctly
    AND each record's ``prev_hmac`` equals the previous record's
    ``hmac`` (or ``initial_prev_hmac`` for the first record).

    Returns (False, line_number) at the first non-verifying record.
    ``line_number`` is 1-indexed and counts non-blank lines only.

    Cross-file verification (for rotated histories): call this on each
    file in order, passing the previous file's last hmac as
    ``initial_prev_hmac`` of the next call.
    """
    prev_hmac = initial_prev_hmac
    line_number = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            line_number += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                return False, line_number
            if rec.get("prev_hmac") != prev_hmac:
                return False, line_number
            missing = [k for k in _HMAC_FIELDS if k not in rec]
            if missing:
                return False, line_number
            body = {k: rec[k] for k in _HMAC_FIELDS}
            expected = _compute_hmac(key, _canonical_bytes(body))
            actual = rec.get("hmac", "")
            if not isinstance(actual, str) or not hmac.compare_digest(expected, actual):
                return False, line_number
            prev_hmac = actual
    return True, None


__all__ = [
    "AuditLogger",
    "AuditRecord",
    "GENESIS_PREV_HMAC",
    "ROTATION_BYTES_DEFAULT",
    "verify_chain",
]
