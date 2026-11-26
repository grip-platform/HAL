"""infinite_thread -- the provable thread: public-key signed rows, an append-only
store, and checkpoints held outside the writer's control.

InfiniteMachine v2, Step 4 (``docs/infinite-machine-contract.md``; GRIP plan
``ULTRAPLAN-2026-10-04-infinite-machine-v2.md``, "Step 4 -- The provable
thread"; the operator's Decision 5: "Records public-key signed, with checkpoints
held outside the writer's control").

**What it replaces.** The gateway's signed record was an HMAC chain whose key
sits beside the log (``gateway_witness.sign_record``), so anyone who can write
the file can re-sign it. Here every row is signed with this host's PRIVATE key
by GRIP's own Ed25519 + ML-DSA-65 signer (``<grip>/lib/pq_dual_sign.py``), and
anyone holding only the PUBLIC key can check it (``verify_thread``). A
signature cannot show that rows were cut from the end, so each thread also
keeps a head file, and ``anchor_heads`` commits every head to an RFC 3161
timestamp authority (``<grip>/lib/rfc3161_anchor.py``), a party this host does
not control.

**Fail closed.** When GRIP's signer cannot run here -- no GRIP checkout, or an
interpreter whose ``cryptography`` has no ML-DSA (it needs 48 or later) --
``append`` writes nothing and says why, once, in the log. There is no weaker
fallback, and never an HMAC row. The gateway keeps serving; only the signing
is refused.

Layout, under ``$HAL_HOME/state/infinite/`` (resolved at call time)::

    threads/<id>.jsonl        rows, one canonical-JSON line each, append-only
    threads/<id>.head         {"seq", "sha256", "signed_at"} of the last row
    threads/<id>.torn         torn tails moved aside by recovery, never dropped
    threads/index/<k>.thread  the thread a session writes to (k: sha256 of it)
    heads/<stamp>.json        a heads snapshot and its sha256
    heads/<stamp>.tsa.json    the timestamp authority's answer, or why none came
    keys/<host>/              this host's key pair, in GRIP's file layout
    host                      this host's random label (never its hostname)

**A row** is canonical JSON (sorted keys, no whitespace, UTF-8; floats are
refused, so another language rebuilds the same bytes) carrying ``v``, ``class``
(served, aux, hold, failover, pause, bypass or unreachable), ``seq`` (1 for the
first row), ``prev`` (sha256 of the previous row's stored line, or the thread
id for the first row), ``ts``, ``host``, ``session_id``, ``parent_session_id``,
the caller's allowlisted fields (route class, model, provider, status, timing,
token counts, the prompt id Claude Code sends) and ``sig``: GRIP's audit dict over the row's canonical bytes
without ``sig``. A header, a bearer or a request body has no field to ride in;
any other field name is refused.

**A thread id** is the first 16 hex of the sha256 of the first row's canonical
bytes without ``prev`` and ``sig``, the two fields that cannot exist before the
id does. A thread is keyed by its session id (HAL sees no conversation key
today). A subagent writes under its own session id and names its parent in
``parent_session_id``, so its rows chain in their own thread and never
interleave with the parent's.

**Concurrency.** One whole line per ``O_APPEND`` write, under a portable advisory
lock on the thread file. Before each append the last line is
checked; a torn one (no newline, not JSON, or a ``prev`` that does not match)
is moved to ``<id>.torn`` and logged. The head file advances by
compare-and-set (a temporary file, then ``os.replace``) and never moves
backwards, and a thread whose head names a row the file no longer holds is not
extended. POSIX and Windows use the same lock protocol.

**What verification can and cannot show.** ``verify_thread`` captures the head
before reading rows, walks every complete row with pinned public keys only,
including rows the head does not yet cover, and names the first failing row: an
edited, deleted, reordered or foreign-signed row fails where it sits. A cut tail fails
against the head file and, if the writer also rewrote the head, against the
newest heads snapshot that covers the thread. Rows written after the newest
snapshot are protected by the head file alone, which the writer controls, until
the next anchor; ``anchor_heads`` and ``publish_heads`` (the Step 6 seam to the
VPS aggregator) are what move a checkpoint out of the writer's reach. The
timestamp token's message imprint is checked (with ``asn1crypto``); the
authority's own signature on the token is not.

Key custody today: the private keys are 0600 files under ``keys/<host>/``.
Step 6 moves them into each OS's secret store.
"""
from __future__ import annotations

import base64
import contextlib
import dataclasses
import enum
import hashlib
import importlib.util
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from lib.hal.file_lock import lock, unlock

log = logging.getLogger(__name__)

ROW_VERSION = "hal-thread/1"
SNAPSHOT_VERSION = "hal-heads/1"
ROW_CLASSES = frozenset(
    {"served", "aux", "hold", "failover", "pause", "bypass", "unreachable"})
#: The fields a caller may put on a row. Any other name is refused (allowlist).
#: Token counts ride inside ``usage``; the three cost figures (observed usage,
#: list price, actual spend) are written by ``lib/hal/infinite_cost.py``.
#: ``prompt_id`` is Claude Code's ``x-claude-code-prompt-id`` (2.1.283), the key
#: that groups the requests serving one user prompt; the wire puts it on a row
#: only when the header carries a value in the session-id safe set.
PAYLOAD_FIELDS = frozenset({
    "route", "model", "provider", "status", "latency_ms",
    "usage", "list_price_usd", "price_version", "billing", "actual_spend_usd",
    "prompt_id",
})
#: What a row's ``usage`` may count. ``variant`` (a short note) may sit beside them.
USAGE_COUNTS = ("input_tokens", "output_tokens", "cache_read_tokens",
                "cache_write_tokens", "cache_write_1h_tokens", "reasoning_tokens")
THREAD_ID_LEN = 16
#: Every stored row carries these, with exactly these types (bool is not int).
_FIELD_TYPES = {"v": str, "class": str, "seq": int, "prev": str, "ts": str,
                "host": str, "session_id": str, "sig": dict}
_KEY_FILES = ("signing.key", "verifying.pub",
              "signing-mldsa65.key", "verifying-mldsa65.pub")
_MAX_TEXT = 200
_SESSION_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_HOST_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_THREAD_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_TAIL_CHUNK = 16384
# Canonical rows and tail offsets require untranslated bytes on Windows.
_BINARY = getattr(os, "O_BINARY", 0)

Signer = Callable[[bytes], dict]
Failure = Tuple[Optional[int], str]


class StoreRefused(RuntimeError):
    """The store wrote nothing; the message says why (never a secret)."""


class SignerUnavailable(StoreRefused):
    """GRIP's signer cannot run here, so no row may be written."""


# --- canonical bytes ----------------------------------------------------------

def canonical(obj: Any) -> bytes:
    """RFC 8785-style canonical JSON: sorted keys, no whitespace, UTF-8.

    Floats are refused: languages print them differently, and a verifier in
    another language must rebuild these exact bytes. The keys used here are
    ASCII, so Python's key order is RFC 8785's.
    """
    _refuse_floats(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _refuse_floats(obj: Any) -> None:
    if isinstance(obj, float):
        raise TypeError("a float has no single canonical form; use an integer or a string")
    children = obj.values() if isinstance(obj, dict) else obj
    if isinstance(obj, (dict, list, tuple)):
        for child in children:
            _refuse_floats(child)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def line_hash(line: bytes) -> str:
    """sha256 of a stored line without its newline: what ``prev`` and heads name."""
    return sha256_hex(line[:-1] if line.endswith(b"\n") else line)


def signed_bytes(row: Mapping[str, Any]) -> bytes:
    """The bytes a row's signature covers: its canonical form without ``sig``."""
    return canonical({key: value for key, value in row.items() if key != "sig"})


def thread_id_of(first_row: Mapping[str, Any]) -> str:
    """A thread's id: the first row hashed without ``prev`` and ``sig``."""
    genesis = {key: value for key, value in first_row.items()
               if key not in ("prev", "sig")}
    return sha256_hex(canonical(genesis))[:THREAD_ID_LEN]


# --- building and sealing a row -----------------------------------------------

def build_row(row_class: str, fields: Mapping[str, Any], *, host: str, ts: str,
              session_id: str, parent_session_id: Optional[str] = None) -> dict:
    """A row before sealing: the caller's allowlisted fields plus the store's own."""
    if row_class not in ROW_CLASSES:
        raise ValueError(f"unknown row class {row_class!r}")
    row = {name: _checked_value(name, value) for name, value in _allowed(fields)}
    row.update({"v": ROW_VERSION, "class": row_class, "host": host, "ts": ts,
                "session_id": checked_session(session_id),
                "parent_session_id": _checked_parent(parent_session_id, session_id)})
    return row


def _allowed(fields: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    for name, value in fields.items():
        if name not in PAYLOAD_FIELDS:
            raise ValueError(f"field {name!r} is not allowed on a thread row")
        yield name, value


def _checked_value(name: str, value: Any) -> Any:
    """A short string, an integer, a boolean or null; never a dict of headers.

    ``usage`` is the one structured field, and it holds token counts only.
    """
    if name == "usage":
        return checked_usage(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str) and len(value) <= _MAX_TEXT:
        return value
    raise ValueError(
        f"field {name!r} must be a short string, an integer, a boolean or null")


def checked_usage(value: Any) -> Optional[dict]:
    """``None`` (nothing was measured) or token counts, with an optional ``variant``."""
    if value is None:
        return None
    if not isinstance(value, dict) or not set(value) <= set(USAGE_COUNTS) | {"variant"}:
        raise ValueError("usage must be null or a dict of token counts")
    counts = [value[name] for name in value if name != "variant"]
    if not all(_is_count(count) and count >= 0 for count in counts):
        raise ValueError("usage counts must be non-negative integers")
    variant = value.get("variant")
    if variant is not None and not (isinstance(variant, str) and len(variant) <= 60):
        raise ValueError("a usage variant must be a short string")
    return dict(value)


def is_session_id(value: Any) -> bool:
    """Whether this store accepts ``value`` as a session id.

    The bound is HAL's own acceptance rule (128 characters of a safe set), not
    a format any vendor documents; anything else is not used as a thread key.
    """
    return isinstance(value, str) and bool(_SESSION_RE.match(value))


def checked_session(value: Any) -> str:
    if is_session_id(value):
        return value
    raise ValueError(f"a session id must match {_SESSION_RE.pattern}")


def _checked_parent(parent: Optional[str], session_id: str) -> Optional[str]:
    if parent is None:
        return None
    if checked_session(parent) == session_id:
        raise ValueError("a session cannot be its own parent")
    return parent


def _seal(row: dict, *, seq: int, prev: str, sign: Signer) -> bytes:
    """The stored line: ``seq``, ``prev`` and ``sig`` added, canonical, one newline."""
    body = dict(row, seq=seq, prev=prev)
    try:
        body["sig"] = sign(signed_bytes(body))
    except StoreRefused:
        raise
    except Exception as exc:  # a broken key file must refuse, never write unsigned
        raise StoreRefused(f"the signer failed: {type(exc).__name__}") from exc
    return canonical(body) + b"\n"


# --- files ------------------------------------------------------------------------

def state_root() -> Path:
    """``$HAL_HOME/state/infinite``, read at call time."""
    home = os.environ.get("HAL_HOME") or str(Path.home() / ".hal")
    return Path(home) / "state" / "infinite"


def _root(root: Optional[Path]) -> Path:
    return Path(root) if root is not None else state_root()


def _iso(moment: float) -> str:
    stamp = datetime.fromtimestamp(moment, timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _stamp(moment: float) -> str:
    """A file name that starts with the UTC date and sorts by time."""
    return datetime.fromtimestamp(moment, timezone.utc).strftime("%Y-%m-%dT%H%M%S.%fZ")


def _stamp_moment(name: str) -> Optional[float]:
    """The moment a snapshot name ``<_stamp>.json`` stands for, or None."""
    try:
        when = datetime.strptime(name[:-len(".json")], "%Y-%m-%dT%H%M%S.%fZ")
    except ValueError:
        return None
    return when.replace(tzinfo=timezone.utc).timestamp()


@contextlib.contextmanager
def _flocked(path: Path, *, create: bool = True) -> Iterator[int]:
    """Hold an exclusive advisory lock on ``path`` for the block; yield its fd."""
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_APPEND | (os.O_CREAT if create else 0)
    try:
        fd = os.open(path, flags | _BINARY, 0o644)
    except FileNotFoundError as exc:
        raise StoreRefused(f"{path.name} is missing") from exc
    held = False
    try:
        lock(fd)
        held = True
        yield fd
    finally:
        if held:
            unlock(fd)
        os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:  # never spin on a descriptor that accepts nothing
            raise OSError("the file accepted no bytes")
        view = view[written:]


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` whole or not at all: temporary file, fsync, ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _BINARY, 0o644)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _publish_new(path: Path, data: bytes) -> bool:
    """Publish ``data`` whole at ``path``, unless the name is taken: never replaces.

    The bytes go to a temporary file first and a hard link then names them, so
    a reader sees the whole file or none, and ``os.link`` refuses a name in use.
    Where hard links are unsupported (FAT, some network shares) it falls back
    to ``os.replace`` after a fresh check, which narrows the race to that gap.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _BINARY, 0o644)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(tmp, path)
    except FileExistsError:
        published = False
    except OSError:
        published = not path.exists()
        if published:
            os.replace(tmp, path)
    else:
        published = True
    if tmp.exists():
        os.unlink(tmp)
    return published


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _parse_row(line: bytes) -> Optional[dict]:
    try:
        row = json.loads(line)
    except (ValueError, RecursionError):
        return None
    return row if isinstance(row, dict) else None


def _is_count(value: Any) -> bool:
    return type(value) is int


# --- the tail of a thread file --------------------------------------------------

def _pread(fd: int, size: int, offset: int) -> bytes:
    """A positioned read that never disturbs the descriptor's seek position."""
    position = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, offset, os.SEEK_SET)
        return os.read(fd, size)
    finally:
        os.lseek(fd, position, os.SEEK_SET)


def _tail_lines(fd: int, size: int, want: int = 2) -> List[Tuple[int, bytes]]:
    """The last ``want`` lines as ``(offset, bytes)``, oldest first.

    A final line without its newline comes back as it is: that is a torn write.
    """
    pos, buf = size, b""
    while pos > 0 and _newlines_before_end(buf) < want:
        step = min(_TAIL_CHUNK, pos)
        pos -= step
        buf = _pread(fd, step, pos) + buf
    lines, offset = [], pos
    for piece in _with_newlines(buf):
        lines.append((offset, piece))
        offset += len(piece)
    return (lines[1:] if pos > 0 else lines)[-want:]


def _newlines_before_end(buf: bytes) -> int:
    return (buf[:-1] if buf.endswith(b"\n") else buf).count(b"\n")


def _with_newlines(buf: bytes) -> List[bytes]:
    """``buf`` as lines that keep their newline; a final partial line stays as is."""
    pieces = buf.split(b"\n")
    lines = [piece + b"\n" for piece in pieces[:-1]]
    return lines + [pieces[-1]] if pieces[-1] else lines


def _torn_reason(last: bytes, before: Optional[bytes], thread_id: str) -> Optional[str]:
    """Why the final line cannot be chained onto, or None when it can."""
    if not last.endswith(b"\n"):
        return "the line has no newline"
    row = _parse_row(last[:-1])
    if row is None:
        return "the line is not JSON"
    expected = thread_id if before is None else line_hash(before)
    if row.get("prev") != expected:
        return "prev does not match the row before it"
    if not _is_count(row.get("seq")):
        return "the row has no seq"
    return None


def _recover_tail(fd: int, path: Path, thread_id: str) -> Tuple[bytes, int]:
    """Move torn final lines aside until the last one chains; return it and the count."""
    moved = 0
    while True:
        size = os.fstat(fd).st_size
        if size == 0:
            raise StoreRefused(f"thread {thread_id} has no intact row left")
        lines = _tail_lines(fd, size)
        offset, last = lines[-1]
        before = lines[-2][1] if len(lines) > 1 else None
        reason = _torn_reason(last, before, thread_id)
        if reason is None:
            return last, moved
        _quarantine(fd, path, offset, last, reason)
        moved += 1


def _quarantine(fd: int, path: Path, offset: int, data: bytes, reason: str) -> None:
    """Keep a torn tail in ``<id>.torn`` (where, why, and its bytes), then cut it off."""
    record = {"at": _iso(time.time()), "offset": offset, "reason": reason,
              "bytes_b64": base64.b64encode(data).decode("ascii")}
    side = path.with_suffix(".torn")
    with open(side, "ab") as out:
        out.write(canonical(record) + b"\n")
        out.flush()
        os.fsync(out.fileno())
    os.ftruncate(fd, offset)
    log.warning("infinite_thread: moved a torn tail of %s (byte %d: %s) to %s",
                path.name, offset, reason, side.name)


# --- heads ------------------------------------------------------------------------

def _head_record(seq: int, line: bytes, signed_at: str) -> dict:
    return {"seq": seq, "sha256": line_hash(line), "signed_at": signed_at}


def cas_head(path: Path, head: dict) -> bool:
    """Advance a head file by compare-and-set; it never moves backwards.

    Called with the thread's lock held, so no other appender is between the
    compare and the set. Returns whether the head was written.
    """
    current = _read_json(path)
    if current is not None and _is_count(current.get("seq")) \
            and current["seq"] >= head["seq"]:
        return False
    _atomic_write(path, canonical(head))
    return True


def _refuse_if_cut(head_path: Path, last_seq: int, last_hash: str, thread_id: str) -> None:
    """Refuse to extend a thread whose head names a row the file no longer holds.

    Extending it would carry the head past the cut and bury the evidence. A head
    BEHIND the last row is a crash between the write and the head update, and
    the next head update repairs it.
    """
    head = _read_json(head_path)
    seq = head.get("seq") if head else None
    if not _is_count(seq) or seq < last_seq:
        return
    if seq > last_seq or head.get("sha256") != last_hash:
        raise StoreRefused(
            f"thread {thread_id}: its head names row {seq}, which its file no "
            "longer holds as written; a cut or rewritten thread is not extended")


def heads_snapshot(root: Optional[Path] = None) -> Dict[str, dict]:
    """Every thread's head, keyed by thread id, read from the head files."""
    heads = {}
    for path in sorted((_root(root) / "threads").glob("*.head")):
        head = _read_json(path)
        if head is not None and _THREAD_ID_RE.match(path.stem):
            heads[path.stem] = head
    return heads


_ANCHOR_LOCK = threading.Lock()


def anchor_heads(root: Optional[Path] = None, *,
                 anchor: Optional[Callable[[bytes], dict]] = None,
                 clock: Callable[[], float] = time.time) -> dict:
    """Snapshot every head, keep it with its sha256, and timestamp that digest.

    The snapshot goes to ``heads/<stamp>.json`` and the timestamp authority's
    answer, or the reason none came, beside it in ``heads/<stamp>.tsa.json``.
    Anchoring is fail-open, as GRIP's ``rfc3161_anchor`` is (it never raises and
    says why it failed): the snapshot is written either way, because a
    checkpoint without a timestamp still catches a cut tail on this host. No
    network call is made beyond the authority, and none at all unless
    ``GRIP_TSA_URL`` is set.
    """
    root = _root(root)
    heads_dir = root / "heads"
    # Name order is the verifier's chronology (_snapshots, _snapshot_check), so
    # the clock read, the heads capture and the name claim happen under one
    # lock, and a name always sorts after every snapshot already present. A
    # coarse clock (time.time on Python 3.12 for Windows moves in ~15.6 ms
    # steps), a clock stepped back, or an overlapping call would otherwise give
    # a stale capture a name at or after a newer one. _publish_new claims the
    # name atomically and never replaces, as a second line of defence.
    with _ANCHOR_LOCK, _flocked(heads_dir / ".anchor.lock"):
        now = clock()
        existing = _snapshots(heads_dir)
        floor = _stamp_moment(existing[0].name) if existing else None
        if floor is not None and now <= floor:
            now = floor + 1e-6
        host, heads = host_label(root), heads_snapshot(root)
        while True:
            snapshot = {"v": SNAPSHOT_VERSION, "host": host, "taken_at": _iso(now),
                        "heads": heads}
            digest = hashlib.sha256(canonical(snapshot))
            path = heads_dir / f"{_stamp(now)}.json"
            if _publish_new(path, canonical({"sha256": digest.hexdigest(), "snapshot": snapshot})):
                break
            now += 1e-6
    result = (anchor or grip_anchor)(digest.digest())
    _atomic_write(_tsa_path(path),
                  json.dumps(result, sort_keys=True, default=str).encode("utf-8"))
    return {"path": str(path), "sha256": digest.hexdigest(), "anchor": result}


def publish_heads(destination: Path, root: Optional[Path] = None) -> List[Path]:
    """Copy the newest heads snapshot and its timestamp record to ``destination``.

    The seam for Step 6, which ships them to the VPS aggregator so the
    checkpoint sits outside this host's control. Today it writes to a local
    directory and makes no network call. A file already published under the
    same name is never replaced with other bytes.
    """
    snapshots = _snapshots(_root(root) / "heads")
    if not snapshots:
        return []
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    sources = [src for src in (snapshots[0], _tsa_path(snapshots[0])) if src.is_file()]
    return [_publish_one(src, target / src.name) for src in sources]


def _tsa_path(snapshot: Path) -> Path:
    return snapshot.with_name(snapshot.name[:-len(".json")] + ".tsa.json")


def _snapshots(heads_dir: Path) -> List[Path]:
    """Heads snapshots, newest first (their names sort by time)."""
    found = [path for path in heads_dir.glob("*.json")
             if not path.name.endswith(".tsa.json")]
    return sorted(found, key=lambda path: path.name, reverse=True)


def _publish_one(src: Path, dst: Path) -> Path:
    data = src.read_bytes()
    if dst.exists():
        if dst.read_bytes() != data:
            raise ValueError(
                f"{dst.name} is already published with other bytes; it is never replaced")
        return dst
    _atomic_write(dst, data)
    return dst


# --- the store ----------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Appended:
    """What one ``append`` did. ``reason`` says why nothing was written."""

    written: bool
    thread_id: Optional[str] = None
    seq: Optional[int] = None
    reason: str = ""
    quarantined: int = 0  # torn tails moved to ``<id>.torn`` on the way


class ThreadStore:
    """Append-only, signed threads under one root (``$HAL_HOME/state/infinite``).

    It writes rows and nothing else; checkpoints are ``heads_snapshot``,
    ``anchor_heads`` and ``publish_heads``, and checking is ``verify_thread``.
    """

    def __init__(self, root: Optional[Path] = None, *, signer: Optional[Signer] = None,
                 host: Optional[str] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.root = _root(root)
        self._host = host
        self._signer = signer
        self._clock = clock

    @property
    def host(self) -> str:
        if self._host is None:
            self._host = host_label(self.root)
        return self._host

    def thread_path(self, thread_id: str) -> Path:
        return self.root / "threads" / f"{thread_id}.jsonl"

    def append(self, row_class: str, fields: Mapping[str, Any], *, session_id: str,
               parent_session_id: Optional[str] = None) -> Appended:
        """Sign one row and append it to the session's thread; never a weaker row.

        Raises ``ValueError`` for a bad row (an unknown class or field, a bad
        id). Returns ``Appended(written=False, reason=...)``, and logs the
        reason once, when nothing may be written: no signer, a damaged thread,
        or a platform without file locks.
        """
        try:
            row = build_row(row_class, fields, host=self.host,
                            ts=_iso(self._clock()), session_id=session_id,
                            parent_session_id=parent_session_id)
            return self._append(row, self._sign())
        except (StoreRefused, OSError) as exc:
            reason = str(exc) if isinstance(exc, StoreRefused) \
                else f"the store could not write: {type(exc).__name__}"
            warn_once(f"no signed row written: {reason}")
            return Appended(written=False, reason=reason)

    def _sign(self) -> Signer:
        if self._signer is None:
            self._signer = default_signer(self.root, self.host)
        return self._signer

    def _index_path(self, session_id: str) -> Path:
        digest = sha256_hex(session_id.encode("utf-8"))[:32]
        return self.root / "threads" / "index" / f"{digest}.thread"

    def _append(self, row: dict, sign: Signer) -> Appended:
        index = self._index_path(row["session_id"])
        with _flocked(index.with_suffix(".lock")):
            thread_id = _read_thread_id(index)
            if thread_id is None:
                return self._create(row, sign, index)
        return self._extend(thread_id, row, sign)

    def _create(self, row: dict, sign: Signer, index: Path) -> Appended:
        """Start a thread; its whole first row appears at once (temporary file, rename)."""
        thread_id = thread_id_of(dict(row, seq=1))
        path = self.thread_path(thread_id)
        if path.exists():
            raise StoreRefused(f"thread {thread_id} already exists")
        line = _seal(row, seq=1, prev=thread_id, sign=sign)
        _atomic_write(path, line)
        cas_head(path.with_suffix(".head"), _head_record(1, line, row["ts"]))
        _atomic_write(index, thread_id.encode("ascii"))
        return Appended(written=True, thread_id=thread_id, seq=1)

    def _extend(self, thread_id: str, row: dict, sign: Signer) -> Appended:
        """Chain one row onto an existing thread, holding its file lock."""
        path = self.thread_path(thread_id)
        with _flocked(path, create=False) as fd:
            last, moved = _recover_tail(fd, path, thread_id)
            seq, last_hash = json.loads(last)["seq"] + 1, line_hash(last)
            _refuse_if_cut(path.with_suffix(".head"), seq - 1, last_hash, thread_id)
            line = _seal(row, seq=seq, prev=last_hash, sign=sign)
            _write_all(fd, line)
            os.fsync(fd)
            cas_head(path.with_suffix(".head"), _head_record(seq, line, row["ts"]))
        return Appended(written=True, thread_id=thread_id, seq=seq, quarantined=moved)


def default_store() -> ThreadStore:
    """The store under the current ``$HAL_HOME``, signing with GRIP's signer."""
    return ThreadStore()


def _read_thread_id(index: Path) -> Optional[str]:
    try:
        value = index.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return value if _THREAD_ID_RE.match(value) else None


_WARNED: set = set()
_WARN_LOCK = threading.Lock()


def warn_once(message: str, *, logger: Optional[logging.Logger] = None) -> None:
    """Log ``message`` once per process (a refusal, a missing price): never twice."""
    with _WARN_LOCK:
        if message in _WARNED:
            return
        _WARNED.add(message)
    (logger or log).warning("%s", message)


# --- the host label ----------------------------------------------------------------

def host_label(root: Optional[Path] = None) -> str:
    """This host's label: ``$HAL_HOST_LABEL``, else a random one kept in ``<root>/host``.

    Random on purpose: a hash of the hostname could be reversed by guessing the
    name, and the label rides on every row.
    """
    configured = os.environ.get("HAL_HOST_LABEL", "").strip()
    if configured:
        return _checked_host(configured)
    path = _root(root) / "host"
    try:
        return _checked_host(path.read_text(encoding="ascii").strip())
    except FileNotFoundError:
        return _mint_host(path)


def _checked_host(label: str) -> str:
    if _HOST_RE.match(label):
        return label
    raise StoreRefused(f"the host label must match {_HOST_RE.pattern}")


def _mint_host(path: Path) -> str:
    """Create the label file exactly once, even with several writers racing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    label = "h-" + secrets.token_hex(6)
    tmp = path.with_name(f".host.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(label + "\n", encoding="ascii")
    try:
        os.link(tmp, path)  # atomic create-if-absent: readers never see it half-written
    except FileExistsError:
        label = _checked_host(path.read_text(encoding="ascii").strip())
    finally:
        tmp.unlink()
    return label


# --- GRIP's signer, loaded without touching HAL's own ``lib`` package ----------

_GRIP_LOCK = threading.Lock()
_PROVISION_LOCK = threading.Lock()
_GRIP_MODULES: Dict[Path, types.ModuleType] = {}


class _Verdict(str, enum.Enum):
    """GRIP's verdict vocabulary (``lib/audit_chain.py::Verdict``), same three values.

    ``pq_dual_sign`` imports ``Verdict`` from ``lib.audit_chain``, and GRIP's
    ``audit_chain`` itself imports GRIP-only modules, so HAL supplies just this
    vocabulary while ``pq_dual_sign`` loads. If ``pq_dual_sign`` ever imports
    anything else from ``lib``, loading fails and signing is refused.
    """

    VERIFIED = "verified"
    DEGRADED = "degraded"
    BROKEN = "broken"


def grip_dir() -> Optional[Path]:
    """GRIP's checkout: ``$CLAUDE_DIR``, then ``~/.grip``, then ``~/.claude``.

    The first that holds ``lib/pq_dual_sign.py`` wins.
    """
    for raw in (os.environ.get("CLAUDE_DIR", ""), "~/.grip", "~/.claude"):
        if raw.strip():
            path = Path(raw.strip()).expanduser()
            if (path / "lib" / "pq_dual_sign.py").is_file():
                return path
    return None


def _grip_module(relpath: str,
                 provides: Optional[Mapping[str, types.ModuleType]] = None) -> types.ModuleType:
    """One GRIP module, loaded once per checkout under a private name."""
    root = grip_dir()
    if root is None:
        raise SignerUnavailable(
            "GRIP is not installed here (no lib/pq_dual_sign.py under "
            "$CLAUDE_DIR, ~/.grip or ~/.claude)")
    path = (root / "lib" / relpath).resolve()
    with _GRIP_LOCK:
        if path not in _GRIP_MODULES:
            _GRIP_MODULES[path] = _load_isolated(path, provides or {})
        return _GRIP_MODULES[path]


def _load_isolated(path: Path,
                   provides: Mapping[str, types.ModuleType]) -> types.ModuleType:
    """Run one GRIP file under a private module name, leaving HAL's ``lib`` alone.

    Both repositories ship a top-level ``lib`` package and HAL's wins in this
    process, so a GRIP file importing ``lib.<x>`` would search HAL's. ``provides``
    stands in for such modules while the file runs; ``sys.path`` and
    ``sys.modules`` are put back afterwards.
    """
    name = f"hal_grip_{path.stem}_{sha256_hex(str(path).encode('utf-8'))[:8]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SignerUnavailable(f"GRIP's {path.name} cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    saved_path = list(sys.path)
    saved = {key: sys.modules.get(key) for key in provides}
    sys.modules.update(provides)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise SignerUnavailable(
            f"GRIP's {path.name} failed to load: {type(exc).__name__}") from exc
    finally:
        sys.path[:] = saved_path
        _restore_modules(saved)
    return module


def _restore_modules(saved: Mapping[str, Optional[types.ModuleType]]) -> None:
    for key, module in saved.items():
        if module is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = module


def _pq_module() -> types.ModuleType:
    """GRIP's ``pq_dual_sign``, with the one ``lib`` name it imports supplied."""
    vocabulary = types.ModuleType("lib.audit_chain")
    vocabulary.Verdict = _Verdict  # type: ignore[attr-defined]
    return _grip_module("pq_dual_sign.py", {"lib.audit_chain": vocabulary})


def mldsa_ready() -> bool:
    """Whether this interpreter's ``cryptography`` has ML-DSA (48 or later)."""
    try:
        from cryptography.hazmat.primitives.asymmetric import mldsa  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "cannot sign"
        return False
    return True


def default_signer(root: Path, host: str, *, provision: bool = True) -> Signer:
    """GRIP's dual signer; read-only workers use ``provision=False``.

    Without provisioning, missing keys fail when signing; they are never created
    or replaced. Thread writers retain the existing first-use provisioning.
    """
    if not mldsa_ready():
        raise SignerUnavailable(
            f"this interpreter ({sys.executable}) has no cryptography with ML-DSA "
            "(cryptography 48 or later is required)")
    pq = _pq_module()
    base = Path(root) / "keys"
    if provision:
        _ensure_keys(base, host)

    def sign(message: bytes) -> dict:
        return pq.dual_sign(message, host, base=base)

    return sign


def _ensure_keys(base: Path, host: str) -> None:
    """Make this host's key pair once, with GRIP's own provisioning code."""
    key_dir = base / host
    if all((key_dir / name).is_file() for name in _KEY_FILES):
        return
    provisioning = _grip_module("aide/key_provisioning.py")
    base.mkdir(parents=True, exist_ok=True)
    os.chmod(base, 0o700)
    with _PROVISION_LOCK, _flocked(base / ".provision.lock"):
        try:
            # GRIP's documented base-path seam (read at call time), set on
            # HAL's private copy of the module, never on GRIP's own process.
            provisioning._TENANT_KEY_BASE = base
            provisioning.provision_tenant_pq_keys(host)
        except Exception as exc:
            raise SignerUnavailable(
                f"provisioning this host's keys failed: {type(exc).__name__}") from exc
    os.chmod(key_dir, 0o700)


def grip_anchor(digest: bytes) -> dict:
    """GRIP's RFC 3161 anchor: it sends a digest only, and only when ``GRIP_TSA_URL`` is set."""
    try:
        module = _grip_module("rfc3161_anchor.py")
    except SignerUnavailable as exc:
        return {"status": "unavailable", "anchored": False, "reason": str(exc)}
    return module.anchor_digest(digest)


# --- verification, with public keys only ---------------------------------------

@dataclasses.dataclass(frozen=True)
class PinnedKey:
    """One host's public keys as raw bytes, held by whoever verifies."""

    ed25519: bytes
    mldsa65: Optional[bytes] = None


Verifier = Callable[[bytes, dict, PinnedKey], Tuple[str, str]]


def host_public_key(root: Optional[Path] = None, host: Optional[str] = None) -> PinnedKey:
    """``host``'s public keys, read from its key directory with GRIP's PEM reader.

    Read them once and keep them where this host cannot change them: a key read
    from the disk the thread sits on proves nothing against that disk's owner.
    """
    root = _root(root)
    key_dir = root / "keys" / (host or host_label(root))
    pq = _pq_module()
    return PinnedKey(
        ed25519=pq.public_key_raw_from_pem((key_dir / "verifying.pub").read_bytes()),
        mldsa65=pq.public_key_raw_from_pem(
            (key_dir / "verifying-mldsa65.pub").read_bytes()))


def grip_verifier(message: bytes, sig: dict, key: PinnedKey) -> Tuple[str, str]:
    """Check one signature with GRIP's verifier and public keys only.

    Both limbs when an ML-DSA-65 key is pinned and this interpreter can check
    it; otherwise the Ed25519 limb alone, and the second value says the
    post-quantum limb went unchecked.
    """
    pq = _pq_module()
    if key.mldsa65 is not None and mldsa_ready():
        both = pq.dual_verify_external(message, sig, key.ed25519, key.mldsa65)
        return _verdict_text(both), "ed25519+ml-dsa-65"
    one = pq.ed25519_limb_verify_external(message, sig, key.ed25519)
    return _verdict_text(one), "ed25519 only (the post-quantum limb was not checked)"


def _verdict_text(verdict: Any) -> str:
    return str(getattr(verdict, "value", verdict))


@dataclasses.dataclass(frozen=True)
class ThreadVerdict:
    """What ``verify_thread`` found; ``line`` names the first failing row (1-based)."""

    ok: bool
    thread_id: str
    rows: int  # rows that passed before the walk stopped
    line: Optional[int]
    reason: str
    limbs: str
    head: str
    snapshot: str
    anchor: str


@dataclasses.dataclass
class _Walk:
    thread_id: str
    lines: List[bytes] = dataclasses.field(default_factory=list)
    host: Optional[str] = None
    limbs: str = "none"

    def expected_prev(self) -> str:
        return sha256_hex(self.lines[-1]) if self.lines else self.thread_id


def verify_thread(path: Path, public_keys: Mapping[str, Any], *,
                  heads_dir: Optional[Path] = None,
                  verifier: Optional[Verifier] = None) -> ThreadVerdict:
    """Verify one thread with pinned PUBLIC keys only.

    ``public_keys`` maps a host label to a ``PinnedKey`` (or to raw Ed25519
    bytes). Capture the snapshot list, then the head, before reading rows, so
    concurrent appends cannot move the checkpoint ahead of the rows. Walk
    every complete row -- canonical form, ``seq``, ``prev``, host, the first
    row's hash, the signature -- including rows past the captured head, since
    readers such as ``infinite_cost.reconcile`` read every row. Compare only
    the head's prefix with the captured head and the newest captured snapshot
    covering the thread, and count only those rows in ``rows``. An unfinished
    last line past the head is an append in progress and is left out; one the
    head covers is a torn write, because the writer fsyncs a row before it
    moves the head. The walk stops at the first failure and the verdict names
    its row.
    """
    path = Path(path)
    name = path.name
    walk = _Walk(thread_id=name[:-len(".jsonl")] if name.endswith(".jsonl") else path.stem)
    heads = Path(heads_dir) if heads_dir is not None else path.parent.parent / "heads"
    snapshots = _snapshots(heads)
    mark = _read_json(path.with_suffix(".head"))
    seq = mark.get("seq") if isinstance(mark, dict) else None
    head_seq = seq if _is_count(seq) and seq > 0 else None
    failure = _walk_rows(walk, path, _pins(public_keys), verifier or grip_verifier,
                         head_seq=head_seq)
    if failure is None and head_seq is not None:
        del walk.lines[head_seq:]  # rows past the head: checked, but vouched for by nothing yet
    head = snapshot = anchor = "not checked"
    if failure is None:
        head, failure = _checkpoint(walk, mark, "head file")
    if failure is None:
        snapshot, anchor, failure = _snapshot_check(walk, snapshots)
    line, reason = failure if failure is not None else (None, "verified")
    return ThreadVerdict(ok=failure is None, thread_id=walk.thread_id,
                         rows=len(walk.lines), line=line, reason=reason,
                         limbs=walk.limbs, head=head, snapshot=snapshot, anchor=anchor)


def _pins(public_keys: Mapping[str, Any]) -> Dict[str, PinnedKey]:
    return {host: key if isinstance(key, PinnedKey) else PinnedKey(ed25519=bytes(key))
            for host, key in public_keys.items()}


def _walk_rows(walk: _Walk, path: Path, pins: Mapping[str, PinnedKey],
               verifier: Verifier, *, head_seq: Optional[int] = None) -> Optional[Failure]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return 1, f"the thread file cannot be read: {type(exc).__name__}"
    *complete, tail = data.split(b"\n")
    for number, line in enumerate(complete, start=1):
        problem = _row_problem(walk, number, line, pins, verifier)
        if problem is not None:
            return number, problem
        walk.lines.append(line)
    if tail and (head_seq is None or len(complete) < head_seq):
        return len(complete) + 1, "the last line has no newline: a torn write"
    if not complete:
        return 1, "the thread file is empty"
    return None


def _row_problem(walk: _Walk, number: int, line: bytes,
                 pins: Mapping[str, PinnedKey], verifier: Verifier) -> Optional[str]:
    row = _parse_row(line)
    if row is None:
        return "the row is not JSON"
    problem = (_form_problem(row, line) or _link_problem(walk, number, row)
               or _signature_problem(walk, row, pins, verifier))
    if problem is None and number == 1:
        walk.host = row["host"]
    return problem


def _form_problem(row: dict, line: bytes) -> Optional[str]:
    wrong = [name for name, kind in _FIELD_TYPES.items() if type(row.get(name)) is not kind]
    if wrong:
        return f"the row has no well-typed {', '.join(wrong)}"
    if row["v"] != ROW_VERSION or row["class"] not in ROW_CLASSES:
        return "the row's version or class is not one this store writes"
    try:
        same = canonical(row) == line
    except (TypeError, ValueError):
        same = False
    return None if same else "the row is not in canonical form"


def _link_problem(walk: _Walk, number: int, row: dict) -> Optional[str]:
    if row["seq"] != number:
        return (f"seq {row['seq']} sits where row {number} belongs: "
                "a row is missing or out of order")
    if row["prev"] != walk.expected_prev():
        return "prev does not match the row before it"
    if number == 1:
        if thread_id_of(row) != walk.thread_id:
            return "the first row does not hash to the thread id"
        return None
    return None if row["host"] == walk.host else "the host changed inside the thread"


def _signature_problem(walk: _Walk, row: dict, pins: Mapping[str, PinnedKey],
                       verifier: Verifier) -> Optional[str]:
    key = pins.get(row["host"])
    if key is None:
        return "no public key is pinned for this row's host"
    try:
        verdict, walk.limbs = verifier(signed_bytes(row), row["sig"], key)
    except Exception as exc:  # noqa: BLE001 - an unusable checker fails closed
        return f"the signature could not be checked: {type(exc).__name__}"
    return None if verdict == "verified" else f"the signature is {verdict} under the pinned key"


def _checkpoint(walk: _Walk, mark: Optional[dict],
                label: str) -> Tuple[str, Optional[Failure]]:
    """Compare one checkpoint (``{"seq", "sha256"}``) with the rows walked."""
    rows = len(walk.lines)
    seq = mark.get("seq") if isinstance(mark, dict) else None
    if not _is_count(seq) or seq < 1:
        return "missing", (rows + 1, f"the {label} is missing or unreadable, "
                                     "so a cut tail cannot be ruled out")
    if seq > rows:
        return "ahead", (rows + 1, f"the {label} names row {seq} but the thread "
                                   f"ends at row {rows}: rows were cut")
    if sha256_hex(walk.lines[seq - 1]) != mark.get("sha256"):
        return "differs", (seq, f"row {seq} does not match the {label}: it was rewritten")
    return f"matches row {seq} of {rows}", None


def _snapshot_check(walk: _Walk, snapshots: List[Path]) -> Tuple[str, str, Optional[Failure]]:
    """The newest snapshot covering this thread, after checking every newer one."""
    for path in snapshots:
        doc = _read_json(path) or {}
        anchor, problem = _snapshot_integrity(path, doc)
        if problem is not None:
            return path.name, anchor, (None, problem)
        mark = _snapshot_heads(doc).get(walk.thread_id)
        if mark is not None:
            status, failure = _checkpoint(walk, mark, f"heads snapshot {path.name}")
            return f"{path.name}: {status}", anchor, failure
    return "no snapshot covers this thread", "none", None


def _snapshot_heads(doc: Mapping[str, Any]) -> dict:
    snapshot = doc.get("snapshot")
    heads = snapshot.get("heads") if isinstance(snapshot, dict) else None
    return heads if isinstance(heads, dict) else {}


def _snapshot_integrity(path: Path, doc: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
    """A snapshot must match its own sha256, and an anchored one its token."""
    try:
        digest = sha256_hex(canonical(doc.get("snapshot")))
    except (TypeError, ValueError):
        digest = ""
    if not isinstance(doc.get("snapshot"), dict) or digest != doc.get("sha256"):
        return "not checked", f"heads snapshot {path.name} does not match its own sha256"
    return _anchor_check(path, digest)


def _anchor_check(path: Path, digest_hex: str) -> Tuple[str, Optional[str]]:
    result = _read_json(_tsa_path(path))
    status = str(result.get("status", "unknown")) if result else "no timestamp record"
    if status != "anchored":
        return status, None
    imprint, readable = token_imprint(str(result.get("token_b64", "")))
    if not readable:
        return "anchored (the token was not checked: asn1crypto is missing)", None
    if imprint != digest_hex:
        return "anchored", f"the timestamp token of {path.name} does not commit to it"
    return "anchored; the token commits to this snapshot", None


def token_imprint(token_b64: str) -> Tuple[Optional[str], bool]:
    """The sha256 an RFC 3161 token commits to, and whether it could be read here.

    ``(None, False)``: no ``asn1crypto``, so nothing was checked.
    ``(None, True)``: the token was examined and is not a readable token.
    """
    try:
        from asn1crypto import cms
    except ImportError:
        return None, False
    try:
        signed = cms.ContentInfo.load(base64.b64decode(token_b64, validate=True))["content"]
        info = signed["encap_content_info"]["content"].parsed
        return info["message_imprint"]["hashed_message"].native.hex(), True
    except Exception:  # noqa: BLE001 - any parse failure is an unreadable token
        return None, True


__all__ = [
    "Appended",
    "PAYLOAD_FIELDS",
    "PinnedKey",
    "ROW_CLASSES",
    "SignerUnavailable",
    "StoreRefused",
    "ThreadStore",
    "ThreadVerdict",
    "anchor_heads",
    "USAGE_COUNTS",
    "build_row",
    "canonical",
    "cas_head",
    "checked_usage",
    "default_signer",
    "default_store",
    "grip_anchor",
    "grip_dir",
    "grip_verifier",
    "heads_snapshot",
    "host_label",
    "host_public_key",
    "is_session_id",
    "line_hash",
    "mldsa_ready",
    "publish_heads",
    "signed_bytes",
    "state_root",
    "thread_id_of",
    "token_imprint",
    "verify_thread",
    "warn_once",
]
