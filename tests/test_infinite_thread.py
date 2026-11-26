"""InfiniteMachine v2, Step 4 -- the provable thread (``lib/hal/infinite_thread.py``).

What these tests pin, and how each can fail:

* Tamper, delete, cut, reorder and a foreign key each make ``verify_thread``
  fail AT THE RIGHT ROW (property-style: every row, many byte positions).
* A crash mid-write leaves a torn tail; the next append moves it aside and
  succeeds.
* Eight threads appending at once still make one chain that verifies.
* With no signer nothing is written -- nothing weaker either -- and the refusal
  is logged once, while the gateway keeps serving.
* A request carrying a bearer and an API key leaves neither string in any file.

Two signing backends. ``grip`` is the real thing, GRIP's Ed25519 + ML-DSA-65
signer and verifier; it needs a GRIP checkout and cryptography >= 48, which HAL
CI does not have, so it skips there. ``ed25519-double`` is a test double
(Ed25519 straight from ``cryptography``) so that the chain logic -- prev, seq,
heads, snapshots, torn tails, locking -- also runs in CI. Every chain test runs
under both. Nothing here reaches a provider or a timestamp authority.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lib.hal import gateway_witness, infinite_thread  # noqa: E402
from lib.hal.gateway import UpstreamResponse, handle_proxy_request  # noqa: E402
from lib.hal.infinite_thread import (  # noqa: E402
    PinnedKey,
    ROW_CLASSES,
    ThreadStore,
    anchor_heads,
    build_row,
    canonical,
    cas_head,
    line_hash,
    publish_heads,
    signed_bytes,
    thread_id_of,
    verify_thread,
)

GRIP_SKIP = ("needs GRIP's signer (lib/pq_dual_sign.py and cryptography>=48); "
             "HAL CI has no GRIP checkout")
HOST = "h-test"
CLASSES = sorted(ROW_CLASSES)
N = 7
DOUBLE_SCHEME = "test-ed25519"


# --- signing backends ------------------------------------------------------------

def _grip_ready() -> bool:
    if not infinite_thread.mldsa_ready():
        return False
    try:
        infinite_thread._pq_module()
    except infinite_thread.SignerUnavailable:
        return False
    return True


def _double_verify(message: bytes, sig: dict, key: PinnedKey):
    """The double's checker: strict about the whole sig dict, like GRIP's."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    limbs = "ed25519 (test double)"
    if set(sig) != {"scheme", "signature"} or sig["scheme"] != DOUBLE_SCHEME:
        return "broken", limbs
    try:
        Ed25519PublicKey.from_public_bytes(key.ed25519).verify(
            bytes.fromhex(sig["signature"]), message)
    except Exception:
        return "broken", limbs
    return "verified", limbs


class _Double:
    """Ed25519 from cryptography directly: a test double for GRIP's signer."""

    name = "ed25519-double"
    verifier = staticmethod(_double_verify)

    def __init__(self, root: Path):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        self.root = root
        self._key = Ed25519PrivateKey.generate()

    def signer(self):
        key = self._key
        return lambda message: {"scheme": DOUBLE_SCHEME,
                                "signature": key.sign(message).hex()}

    def other_signer(self):
        return _Double(self.root).signer()

    def pins(self):
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        raw = self._key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return {HOST: PinnedKey(ed25519=raw)}


class _Grip:
    """GRIP's own signer and verifier, with this host's keys under ``root``."""

    name = "grip"
    verifier = None  # verify_thread's default: GRIP's verifier

    def __init__(self, root: Path):
        self.root = root

    def signer(self):
        return infinite_thread.default_signer(self.root, HOST)

    def other_signer(self):
        return infinite_thread.default_signer(self.root, "h-other")

    def pins(self):
        return {HOST: infinite_thread.host_public_key(self.root, HOST)}


@pytest.fixture(params=["ed25519-double", "grip"])
def backend(request, monkeypatch):
    pytest.importorskip("cryptography")
    monkeypatch.setenv("HAL_HOST_LABEL", HOST)
    root = infinite_thread.state_root()  # under this test's own HAL_HOME
    if request.param == "grip":
        if not _grip_ready():
            pytest.skip(GRIP_SKIP)
        return _Grip(root)
    double = _Double(root)
    monkeypatch.setattr(infinite_thread, "default_signer",
                        lambda _root, _host: double.signer())
    return double


@pytest.fixture
def grip_only(monkeypatch):
    pytest.importorskip("cryptography")
    if not _grip_ready():
        pytest.skip(GRIP_SKIP)
    monkeypatch.setenv("HAL_HOST_LABEL", HOST)
    return _Grip(infinite_thread.state_root())


def _store(backend) -> ThreadStore:
    return ThreadStore(backend.root, signer=backend.signer(), host=HOST)


def _fields(i: int) -> dict:
    return {"model": "claude-opus-5-5", "provider": "anthropic", "status": 200,
            "latency_ms": i, "route": "inference",
            "usage": {"input_tokens": 10 + i, "output_tokens": 3}}


def _thread(backend, rows: int = N, session: str = "sess-main") -> Path:
    store, result = _store(backend), None
    for i in range(rows):
        result = store.append(CLASSES[i % len(CLASSES)], _fields(i), session_id=session)
        assert result.written, result.reason
    return store.thread_path(result.thread_id)


def _verify(backend, path: Path, **kw):
    return verify_thread(path, backend.pins(), verifier=backend.verifier, **kw)


def _lines(path: Path):
    return path.read_bytes().split(b"\n")[:-1]


def _rows_of(path: Path):
    return [json.loads(line) for line in _lines(path)]


def _write_lines(path: Path, lines) -> None:
    path.write_bytes(b"".join(line + b"\n" for line in lines))


def _spans(data: bytes):
    spans, start = [], 0
    for line in data.split(b"\n")[:-1]:
        spans.append((start, start + len(line) + 1))  # the newline included
        start += len(line) + 1
    return spans


def _resign(line: bytes, sign) -> bytes:
    row = json.loads(line)
    row["sig"] = sign(signed_bytes(row))
    return canonical(row)


def _disabled_anchor(_digest: bytes) -> dict:
    return {"status": "disabled", "anchored": False}


# --- a fresh thread ------------------------------------------------------------

def test_a_fresh_thread_verifies_with_public_keys_only(backend):
    path = _thread(backend)
    verdict = _verify(backend, path)
    assert verdict.ok, verdict
    assert (verdict.rows, verdict.line, verdict.reason) == (N, None, "verified")
    assert verdict.head == f"matches row {N} of {N}"
    assert verdict.limbs != "none"


def test_the_thread_id_is_the_first_row_hashed_without_prev_and_sig(backend):
    path = _thread(backend, rows=1)
    row = _rows_of(path)[0]
    genesis = {k: v for k, v in row.items() if k not in ("prev", "sig")}
    expected = hashlib.sha256(canonical(genesis)).hexdigest()[:16]
    assert path.stem == row["prev"] == thread_id_of(row) == expected
    assert row["seq"] == 1


def test_rows_carry_only_the_allowlisted_fields(backend):
    row = _rows_of(_thread(backend, rows=1))[0]
    store_fields = {"v", "class", "seq", "prev", "ts", "host", "session_id",
                    "parent_session_id", "sig"}
    assert set(row) <= store_fields | infinite_thread.PAYLOAD_FIELDS
    assert row["host"] == HOST


# --- tamper, delete, cut, reorder, key swap --------------------------------

def _assert_flips_fail_at(backend, path: Path, original: bytes, number: int, positions):
    for pos in sorted(positions):
        flipped = bytearray(original)
        flipped[pos] ^= 0x01
        path.write_bytes(bytes(flipped))
        verdict = _verify(backend, path)
        assert not verdict.ok and verdict.line == number, (number, pos, verdict)


def test_flipping_one_byte_of_any_row_fails_at_that_row(backend):
    path = _thread(backend)
    original = path.read_bytes()
    rng = random.Random(4004)
    for number, (start, end) in enumerate(_spans(original), start=1):
        positions = {start, end - 1} | set(rng.sample(range(start, end), 6))
        _assert_flips_fail_at(backend, path, original, number, positions)
    path.write_bytes(original)
    assert _verify(backend, path).ok


def test_deleting_any_row_fails_where_it_sat(backend):
    path = _thread(backend)
    lines = _lines(path)
    for number in range(1, N + 1):
        _write_lines(path, lines[:number - 1] + lines[number:])
        verdict = _verify(backend, path)
        assert not verdict.ok and verdict.line == number, (number, verdict)


def test_cutting_the_tail_fails_at_the_first_missing_row(backend):
    path = _thread(backend)
    lines = _lines(path)
    for keep in range(1, N):
        _write_lines(path, lines[:keep])
        verdict = _verify(backend, path)
        assert not verdict.ok and verdict.line == keep + 1, (keep, verdict)
        assert "cut" in verdict.reason and verdict.head == "ahead"


def test_swapping_two_neighbours_fails_at_the_first(backend):
    path = _thread(backend)
    lines = _lines(path)
    for number in range(1, N):
        swapped = list(lines)
        swapped[number - 1], swapped[number] = swapped[number], swapped[number - 1]
        _write_lines(path, swapped)
        verdict = _verify(backend, path)
        assert not verdict.ok and verdict.line == number, (number, verdict)


def test_a_row_signed_by_another_key_fails_at_that_row(backend):
    path = _thread(backend)
    lines, other = _lines(path), backend.other_signer()
    for number in range(1, N + 1):
        forged = list(lines)
        forged[number - 1] = _resign(lines[number - 1], other)
        _write_lines(path, forged)
        verdict = _verify(backend, path)
        assert not verdict.ok and verdict.line == number, (number, verdict)
        assert "signature" in verdict.reason


def test_a_thread_re_signed_end_to_end_with_another_key_fails_at_row_one(backend):
    path = _thread(backend)
    other = backend.other_signer()
    _write_lines(path, [_resign(line, other) for line in _lines(path)])
    verdict = _verify(backend, path)
    assert not verdict.ok and verdict.line == 1


def test_a_host_with_no_pinned_key_is_not_trusted(backend):
    path = _thread(backend, rows=2)
    verdict = verify_thread(path, {"h-someone-else": b"\x00" * 32},
                            verifier=backend.verifier)
    assert not verdict.ok and verdict.line == 1
    assert "no public key is pinned" in verdict.reason


# --- checkpoints: the head file and the heads snapshot ----------------------

def _cut_and_rewrite_head(path: Path, keep: int) -> None:
    """What a writer who holds the files can do: cut rows, then fix the head."""
    kept = _lines(path)[:keep]
    _write_lines(path, kept)
    head = {"seq": keep, "sha256": line_hash(kept[-1]), "signed_at": "x"}
    path.with_suffix(".head").write_bytes(canonical(head))


def test_a_snapshot_catches_a_cut_tail_even_with_the_head_rewritten(backend):
    path = _thread(backend)
    anchor_heads(backend.root, anchor=_disabled_anchor)
    _cut_and_rewrite_head(path, keep=4)
    verdict = _verify(backend, path)
    assert not verdict.ok and verdict.line == 5, verdict
    assert "heads snapshot" in verdict.reason and verdict.head == "matches row 4 of 4"


def test_limit_with_no_snapshot_a_rewritten_head_hides_a_cut_tail(backend):
    """The documented limit, pinned: past the newest snapshot only the head file
    guards the tail, and the writer controls it. The verdict says so."""
    path = _thread(backend)
    _cut_and_rewrite_head(path, keep=4)
    verdict = _verify(backend, path)
    assert verdict.ok and verdict.rows == 4
    assert verdict.snapshot == "no snapshot covers this thread"


def test_an_edited_snapshot_fails(backend):
    path = _thread(backend)
    snap_path = Path(anchor_heads(backend.root, anchor=_disabled_anchor)["path"])
    doc = json.loads(snap_path.read_bytes())
    doc["snapshot"]["heads"][path.stem]["seq"] = 3
    snap_path.write_bytes(canonical(doc))
    verdict = _verify(backend, path)
    assert not verdict.ok and "does not match its own sha256" in verdict.reason


def _token(digest: bytes) -> str:
    """An RFC 3161-shaped token committing to ``digest`` (unsigned: the
    authority's signature is out of scope; the imprint is what is checked)."""
    from asn1crypto import cms, tsp
    info = tsp.TSTInfo({
        "version": "v1", "policy": "1.2.3.4", "serial_number": 1,
        "message_imprint": {"hash_algorithm": {"algorithm": "sha256"},
                            "hashed_message": digest},
        "gen_time": datetime.now(timezone.utc),
    })
    signed = cms.SignedData({
        "version": "v3", "digest_algorithms": [], "signer_infos": [],
        "encap_content_info": {"content_type": "tst_info", "content": info},
    })
    token = cms.ContentInfo({"content_type": "signed_data", "content": signed})
    return base64.b64encode(token.dump()).decode("ascii")


def _token_anchor(digest: bytes) -> dict:
    return {"status": "anchored", "anchored": True, "token_b64": _token(digest)}


def test_a_timestamp_token_must_commit_to_its_snapshot(backend):
    pytest.importorskip("asn1crypto")
    path = _thread(backend)
    snap_path = Path(anchor_heads(backend.root, anchor=_token_anchor)["path"])
    verdict = _verify(backend, path)
    assert verdict.ok and verdict.anchor == "anchored; the token commits to this snapshot"
    doc = json.loads(snap_path.read_bytes())  # rewrite it consistently, token untouched
    doc["snapshot"]["heads"][path.stem]["seq"] = 3
    doc["sha256"] = hashlib.sha256(canonical(doc["snapshot"])).hexdigest()
    snap_path.write_bytes(canonical(doc))
    verdict = _verify(backend, path)
    assert not verdict.ok and "does not commit" in verdict.reason


def test_anchor_heads_timestamps_the_snapshot_digest(backend):
    _thread(backend, rows=2)
    seen = []
    result = anchor_heads(backend.root, anchor=lambda d: seen.append(d) or _disabled_anchor(d))
    doc = json.loads(Path(result["path"]).read_bytes())
    digest = hashlib.sha256(canonical(doc["snapshot"])).digest()
    assert seen == [digest] and doc["sha256"] == digest.hex() == result["sha256"]
    tsa = Path(result["path"][:-len(".json")] + ".tsa.json")
    assert json.loads(tsa.read_bytes()) == {"anchored": False, "status": "disabled"}
    assert Path(result["path"]).name.startswith(datetime.now(timezone.utc).strftime("%Y-%m-%d"))


def test_two_snapshots_at_one_clock_reading_keep_both(backend):
    # On a coarse clock two snapshots can be taken at the same reading; the
    # second must not overwrite the first checkpoint.
    _thread(backend, rows=2)
    first = anchor_heads(backend.root, anchor=_disabled_anchor, clock=lambda: 1_800_000_000.0)
    second = anchor_heads(backend.root, anchor=_disabled_anchor, clock=lambda: 1_800_000_000.0)
    assert first["path"] != second["path"]
    assert Path(first["path"]).is_file() and Path(second["path"]).is_file()
    assert Path(second["path"]).name > Path(first["path"]).name  # names still sort by time


def test_publish_new_never_replaces_a_taken_name(tmp_path):
    target = tmp_path / "heads" / "one.json"
    assert infinite_thread._publish_new(target, b"first")
    assert not infinite_thread._publish_new(target, b"second")
    assert target.read_bytes() == b"first"
    assert sorted(p.name for p in target.parent.iterdir()) == ["one.json"]  # no temp file left


def test_a_snapshot_name_taken_mid_write_moves_on(backend, monkeypatch):
    # Another writer claims the name between this call choosing it and
    # publishing it: this snapshot takes the next microsecond, never overwrites.
    _thread(backend, rows=2)
    real, rival = infinite_thread._publish_new, []

    def racing(path, data):
        if not rival:
            rival.append(path)
            real(path, b'{"rival": true}')
        return real(path, data)

    monkeypatch.setattr(infinite_thread, "_publish_new", racing)
    result = anchor_heads(backend.root, anchor=_disabled_anchor, clock=lambda: 1_800_000_000.0)
    assert Path(result["path"]) != rival[0]
    assert rival[0].read_bytes() == b'{"rival": true}'
    assert Path(result["path"]).name > rival[0].name


def test_a_later_snapshot_never_sorts_before_an_earlier_one(backend):
    # Names are the verifier's chronology: a clock read behind the newest
    # snapshot's name (a clock stepped back) still yields a later name.
    _thread(backend, rows=2)
    first = anchor_heads(backend.root, anchor=_disabled_anchor, clock=lambda: 1_800_000_000.0)
    second = anchor_heads(backend.root, anchor=_disabled_anchor, clock=lambda: 1_799_999_000.0)
    assert Path(second["path"]).name > Path(first["path"]).name


def test_capture_and_claim_happen_under_the_anchor_lock(backend, monkeypatch):
    # Overlapping calls must not capture heads in one order and claim names in
    # the other, so both steps run while the anchor lock is held.
    _thread(backend, rows=2)
    seen, capture, publish = [], infinite_thread.heads_snapshot, infinite_thread._publish_new
    monkeypatch.setattr(infinite_thread, "heads_snapshot",
                        lambda root: seen.append(infinite_thread._ANCHOR_LOCK.locked()) or capture(root))
    monkeypatch.setattr(infinite_thread, "_publish_new",
                        lambda path, data: seen.append(infinite_thread._ANCHOR_LOCK.locked()) or publish(path, data))
    anchor_heads(backend.root, anchor=_disabled_anchor)
    assert seen == [True, True]
    assert not infinite_thread._ANCHOR_LOCK.locked()


def test_publish_heads_copies_the_newest_checkpoint_and_never_replaces_it(backend, tmp_path):
    _thread(backend, rows=2)
    anchor_heads(backend.root, anchor=_disabled_anchor)
    out = tmp_path / "published"
    first = publish_heads(out, backend.root)
    assert sorted(p.name.endswith(".tsa.json") for p in first) == [False, True]
    assert publish_heads(out, backend.root) == first  # identical bytes: a no-op
    first[0].write_bytes(b"{}")
    with pytest.raises(ValueError, match="never replaced"):
        publish_heads(out, backend.root)


# --- crash recovery, the head, concurrency -----------------------------------

def test_a_torn_tail_is_moved_aside_and_the_next_append_succeeds(backend):
    path = _thread(backend, rows=3)
    whole = path.read_bytes()
    last = _lines(path)[-1]
    half = last[:len(last) // 2]
    with open(path, "ab") as fh:
        fh.write(half)  # what a crash in the middle of a write leaves
    committed = _verify(backend, path)
    assert committed.ok and committed.rows == 3  # the torn tail is beyond the head
    result = _store(backend).append("served", _fields(9), session_id="sess-main")
    assert result.written and result.seq == 4 and result.quarantined == 1
    assert _verify(backend, path).ok
    record = json.loads(path.with_suffix(".torn").read_bytes().splitlines()[0])
    assert base64.b64decode(record["bytes_b64"]) == half
    assert record["offset"] == len(whole)


def test_a_complete_line_with_a_wrong_prev_is_moved_aside_too(backend):
    path = _thread(backend, rows=2)
    forged = json.loads(_lines(path)[-1])
    forged["prev"] = "0" * 64
    with open(path, "ab") as fh:
        fh.write(canonical(forged) + b"\n")
    result = _store(backend).append("served", _fields(5), session_id="sess-main")
    assert result.written and result.seq == 3 and result.quarantined == 1
    assert _verify(backend, path).ok


def test_a_cut_thread_is_not_extended(backend):
    path = _thread(backend)
    _write_lines(path, _lines(path)[:N - 2])
    result = _store(backend).append("served", _fields(1), session_id="sess-main")
    assert not result.written and "not extended" in result.reason
    head = json.loads(path.with_suffix(".head").read_bytes())
    assert head["seq"] == N  # the evidence of the cut is still there
    assert _verify(backend, path).line == N - 1


def test_a_head_never_moves_backwards(tmp_path):
    head = tmp_path / "x.head"
    assert cas_head(head, {"seq": 5, "sha256": "a", "signed_at": "t"})
    assert not cas_head(head, {"seq": 4, "sha256": "b", "signed_at": "t"})
    assert json.loads(head.read_bytes())["seq"] == 5
    assert cas_head(head, {"seq": 6, "sha256": "c", "signed_at": "t"})


def _append_many(backend, count: int, errors: list) -> None:
    store = _store(backend)  # one store per request, as the gateway makes them
    for i in range(count):
        result = store.append("served", _fields(i), session_id="sess-busy")
        if not result.written:
            errors.append(result.reason)


def test_eight_threads_appending_at_once_make_one_chain(backend):
    errors, per_worker = [], 20
    workers = [threading.Thread(target=_append_many, args=(backend, per_worker, errors))
               for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert errors == []
    paths = list((backend.root / "threads").glob("*.jsonl"))
    assert len(paths) == 1
    verdict = _verify(backend, paths[0])
    assert verdict.ok and verdict.rows == 8 * per_worker, verdict


# --- subagents --------------------------------------------------------------------

def test_a_subagent_chains_in_its_own_thread_and_points_at_its_parent(backend):
    store = _store(backend)
    for i in range(3):
        store.append("served", _fields(i), session_id="parent-1")
        store.append("aux", _fields(i), session_id="child-1", parent_session_id="parent-1")
    by_session = {}
    for path in sorted((backend.root / "threads").glob("*.jsonl")):
        rows = _rows_of(path)
        sessions = {row["session_id"] for row in rows}
        assert len(sessions) == 1, sessions  # never interleaved
        by_session[sessions.pop()] = rows
        assert _verify(backend, path).ok
    assert sorted(by_session) == ["child-1", "parent-1"]
    assert {r["parent_session_id"] for r in by_session["parent-1"]} == {None}
    assert {r["parent_session_id"] for r in by_session["child-1"]} == {"parent-1"}


def test_a_session_cannot_be_its_own_parent():
    with pytest.raises(ValueError, match="own parent"):
        build_row("served", {}, host=HOST, ts="t", session_id="s",
                  parent_session_id="s")


# --- what may and may not be on a row --------------------------------------

def test_a_field_outside_the_allowlist_is_refused_without_echoing_its_value():
    with pytest.raises(ValueError) as caught:
        build_row("served", {"authorization": "Bearer sk-ant-TESTSECRET"},
                  host=HOST, ts="t", session_id="s")
    assert "authorization" in str(caught.value)
    assert "TESTSECRET" not in str(caught.value)


@pytest.mark.parametrize("value", [{"Authorization": "x"}, ["x"], "m" * 201, 1.5])
def test_an_allowed_field_takes_only_a_short_scalar(value):
    with pytest.raises(ValueError):
        build_row("served", {"model": value}, host=HOST, ts="t", session_id="s")


@pytest.mark.parametrize("row_class", sorted(ROW_CLASSES))
def test_every_row_class_is_accepted(row_class):
    assert build_row(row_class, {}, host=HOST, ts="t", session_id="s")["class"] == row_class


def test_an_unknown_row_class_is_refused():
    with pytest.raises(ValueError, match="unknown row class"):
        build_row("served-ish", {}, host=HOST, ts="t", session_id="s")


def test_canonical_bytes_are_sorted_compact_utf8_and_refuse_floats():
    assert canonical({"b": 1, "a": "é"}) == b'{"a":"\xc3\xa9","b":1}'
    with pytest.raises(TypeError):
        canonical({"a": [1, {"b": 0.5}]})


# --- fail closed --------------------------------------------------------------------

def _no_signer(_root, _host):
    raise infinite_thread.SignerUnavailable("test: no signer here")


def test_with_no_signer_nothing_is_written_and_it_is_logged_once(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(infinite_thread, "_WARNED", set())
    monkeypatch.setattr(infinite_thread, "default_signer", _no_signer)
    store = ThreadStore(tmp_path / "infinite", host=HOST)
    with caplog.at_level(logging.WARNING, logger="lib.hal.infinite_thread"):
        first = store.append("served", _fields(0), session_id="s1")
        second = store.append("served", _fields(1), session_id="s1")
    assert not first.written and not second.written
    assert "no signer here" in first.reason
    assert not (tmp_path / "infinite").exists()  # nothing, and nothing weaker
    logged = [r for r in caplog.records if "no signed row written" in r.getMessage()]
    assert len(logged) == 1


def test_without_grip_the_real_signer_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    assert infinite_thread.grip_dir() is None
    result = ThreadStore(tmp_path / "infinite", host=HOST).append(
        "served", _fields(0), session_id="s1")
    assert not result.written
    assert "GRIP is not installed" in result.reason or "cryptography" in result.reason
    assert not (tmp_path / "infinite" / "threads").exists()


def test_a_bad_configured_host_label_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("HAL_HOST_LABEL", "Alex's MacBook")
    result = ThreadStore(tmp_path / "infinite").append("served", {}, session_id="s1")
    assert not result.written and "host label" in result.reason


def test_the_host_label_is_random_kept_and_not_the_hostname(tmp_path, monkeypatch):
    monkeypatch.delenv("HAL_HOST_LABEL", raising=False)
    root = tmp_path / "infinite"
    label = infinite_thread.host_label(root)
    assert re.fullmatch(r"h-[0-9a-f]{12}", label)
    assert infinite_thread.host_label(root) == label
    assert socket.gethostname().split(".")[0].lower() not in label


# --- the gateway wiring -------------------------------------------------------------

class _Headers(dict):
    def get(self, key, default=None):
        return next((v for k, v in self.items() if k.lower() == key.lower()), default)

    def items(self):
        return list(super().items())


class _Handler:
    """A BaseHTTPRequestHandler stand-in."""

    def __init__(self, body: bytes, *, headers=None, path="/v1/messages", command="POST"):
        self._body, self.path, self.command = body, path, command
        self.headers = _Headers({"content-length": str(len(body)), **(headers or {})})
        self.wfile = io.BytesIO()
        self.status = None

    @property
    def rfile(self):
        return io.BytesIO(self._body)

    def send_response(self, code):
        self.status = code

    def send_header(self, *_a):
        pass

    def end_headers(self):
        pass


class _Proxy:
    def forward(self, path, body, headers, *, method="POST", max_retries=None):
        reply = (b'{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5",'
                 b'"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",'
                 b'"usage":{"input_tokens":3,"output_tokens":1}}')
        return UpstreamResponse(status=200, headers=[("content-type", "application/json")],
                                _reader=io.BytesIO(reply))


def _turn(**extra) -> bytes:
    body = {"model": "claude-opus-5-5", "max_tokens": 512,
            "messages": [{"role": "user", "content": "hello"}]}
    return json.dumps(dict(body, **extra)).encode("utf-8")


def _all_rows():
    rows = []
    for path in sorted((infinite_thread.state_root() / "threads").glob("*.jsonl")):
        rows.extend(_rows_of(path))
    return rows


def test_a_turn_through_the_gateway_is_a_signed_served_row(backend):
    handler = _Handler(_turn(), headers={"x-claude-code-session-id": "cc-sess-42"})
    handle_proxy_request(handler, _Proxy(), "/v1/messages?beta=true")
    assert handler.status == 200
    (row,) = _all_rows()
    assert (row["class"], row["route"], row["session_id"]) == ("served", "inference", "cc-sess-42")
    assert (row["model"], row["provider"], row["status"]) == ("claude-opus-5-5", "anthropic", 200)
    path = next((backend.root / "threads").glob("*.jsonl"))
    assert _verify(backend, path).ok


def test_a_quota_check_is_an_aux_row_and_no_session_means_the_launch_id(backend):
    handle_proxy_request(_Handler(_turn(max_tokens=1)), _Proxy(), "/v1/messages")
    (row,) = _all_rows()
    assert (row["class"], row["route"]) == ("aux", "aux")
    assert row["session_id"] == gateway_witness.LAUNCH_SESSION_ID


def test_the_chat_rail_writes_a_served_row(backend, monkeypatch):
    from lib.hal.gateway_openai import handle_chat_completions
    monkeypatch.setenv("HAL_GATEWAY_MULTIPROVIDER", "0")
    body = json.dumps({"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]})
    handle_chat_completions(_Handler(body.encode(), path="/v1/chat/completions"), _Proxy())
    (row,) = _all_rows()
    # The route classifier reads Anthropic's wire, so the chat rail does not
    # claim a class it never checked.
    assert (row["class"], row["route"], row["provider"]) == ("served", "unclassified", "anthropic")


def test_an_account_bound_call_earns_no_thread_row(backend):
    handle_proxy_request(_Handler(b"{}", path="/v1/code/sessions"), _Proxy(), "/v1/code/sessions")
    assert _all_rows() == []


def test_with_no_signer_the_gateway_still_serves(monkeypatch):
    monkeypatch.setattr(infinite_thread, "default_signer", _no_signer)
    handler = _Handler(_turn(), headers={"x-claude-code-session-id": "cc-sess-1"})
    handle_proxy_request(handler, _Proxy(), "/v1/messages")
    assert handler.status == 200 and b'"text":"ok"' in handler.wfile.getvalue()
    assert _all_rows() == []
    audit = Path(os.environ["HAL_HOME"]) / "logs" / "gateway-audit.jsonl"
    assert audit.is_file()  # the unsigned audit row still lands


def test_a_row_the_witness_cannot_build_is_logged_once_not_raised(backend, monkeypatch, caplog):
    monkeypatch.setattr(infinite_thread, "_WARNED", set())
    with caplog.at_level(logging.WARNING, logger="lib.hal.infinite_thread"):
        gateway_witness.witness_chat(provider="anthropic", model="m" * 201, status=200,
                                     served=True, started=0.0, path="/v1/chat/completions")
    assert _all_rows() == []
    assert [r.getMessage() for r in caplog.records
            if "could not build a thread row: ValueError" in r.getMessage()]


def _files_holding(home: Path, needles) -> list:
    files = [p for p in home.rglob("*") if p.is_file()]
    return [str(p) for p in files if any(n in p.read_bytes() for n in needles)]


def test_no_bearer_or_api_key_reaches_any_file(backend, monkeypatch):
    from lib.hal.gateway_openai import handle_chat_completions
    monkeypatch.setenv("HAL_GATEWAY_MULTIPROVIDER", "0")
    seeded = {"Authorization": "Bearer sk-ant-TESTSECRET", "x-api-key": "TESTKEY2",
              "x-claude-code-session-id": "cc-sess-secret"}
    handle_proxy_request(_Handler(_turn(), headers=seeded), _Proxy(), "/v1/messages")
    handle_proxy_request(_Handler(_turn(max_tokens=1), headers=seeded), _Proxy(), "/v1/messages")
    chat = json.dumps({"model": "gpt-5", "messages": [{"role": "user", "content": "x"}]})
    handle_chat_completions(_Handler(chat.encode(), headers=seeded,
                                     path="/v1/chat/completions"), _Proxy())
    home = Path(os.environ["HAL_HOME"])
    anchor_heads(anchor=_disabled_anchor)
    publish_heads(home / "published")
    assert len(_all_rows()) == 3  # rows were written, so the scan below can fail
    assert (home / "logs" / "gateway-audit.jsonl").is_file()
    assert list((infinite_thread.state_root() / "heads").glob("*.json"))
    assert _files_holding(home, (b"sk-ant-TESTSECRET", b"TESTKEY2")) == []


# --- GRIP only ----------------------------------------------------------------------

def test_grip_keys_are_readable_by_this_user_only(grip_only):
    grip_only.signer()
    key_dir = grip_only.root / "keys" / HOST
    assert oct(key_dir.stat().st_mode & 0o777) == "0o700"
    assert {oct((key_dir / name).stat().st_mode & 0o777)
            for name in ("signing.key", "signing-mldsa65.key")} == {"0o600"}


def test_grip_verifier_checks_both_limbs_or_says_it_did_not(grip_only):
    path = _thread(grip_only, rows=2)
    both = verify_thread(path, grip_only.pins())
    assert both.ok and both.limbs == "ed25519+ml-dsa-65"
    ed_only = verify_thread(path, {HOST: grip_only.pins()[HOST].ed25519})
    assert ed_only.ok and "not checked" in ed_only.limbs


def test_loading_grip_leaves_hals_lib_package_alone(grip_only, monkeypatch):
    before = list(sys.path)
    monkeypatch.setattr(infinite_thread, "_GRIP_MODULES", {})
    infinite_thread._pq_module()
    assert "lib.audit_chain" not in sys.modules
    assert sys.path == before


def test_grip_anchor_sends_nothing_without_a_tsa_url(grip_only, monkeypatch):
    monkeypatch.delenv("GRIP_TSA_URL", raising=False)
    assert infinite_thread.grip_anchor(b"\x00" * 32)["status"] == "disabled"
