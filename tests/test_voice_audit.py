"""Two-sided Goodhart-proof tests for lib/hal/voice/audit.py.

Per Rule 14 (scientific-method.md + rules/efficiency-rules.md): each
invariant gets BOTH a must-pass and must-fail assertion, so the test
actually exercises the contract instead of being a stub that always
passes.

The must-fail branches explicitly tamper with the on-disk JSONL, call
verify_chain, and assert the correct (False, line_number) return.
Without those tamper assertions a 'verify succeeds' test is vacuous:
a logger that always returned True would pass it silently.

Shipped at PR alpha-2 W2 alongside audit.py. Rotation + HMAC-chain
continuity are covered here. Concurrency via fcntl.flock is deferred
to the multi-process empirical test in alpha-2 W5 when voice.listen
and voice.speak are actually invoked from separate processes.
"""
from __future__ import annotations

import json
import re
import secrets
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.voice.audit import (  # noqa: E402
    GENESIS_PREV_HMAC,
    AuditLogger,
    verify_chain,
)
from lib.hal.voice.types import VoiceError  # noqa: E402


# ---------- fixtures ----------


@pytest.fixture
def audit_key() -> bytes:
    """Deterministic 32-byte key for tests — NOT for production."""
    return b"\x00" * 32


@pytest.fixture
def logger(tmp_path: Path, audit_key: bytes) -> AuditLogger:
    """Logger that writes to a tmp file with chflags disabled."""
    return AuditLogger(
        path=tmp_path / "audit.jsonl",
        key=audit_key,
        set_append_only=False,
    )


def _read_lines(path: Path) -> list[dict]:
    return [
        json.loads(raw)
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]


# ---------- audit_id shape ----------


def test_audit_id_is_32_hex_chars(logger: AuditLogger) -> None:
    """Must-pass: secrets.token_hex(16) → 32-char lowercase hex."""
    audit_id = logger.append(event="voice.listen", payload={"text": "hi"})
    assert re.fullmatch(r"[0-9a-f]{32}", audit_id), audit_id


def test_audit_id_is_unique_across_appends(logger: AuditLogger) -> None:
    """Must-pass: 50 appends → 50 distinct audit_ids (rejects shared-state bug)."""
    ids = {logger.append(event="voice.speak", payload={"i": i}) for i in range(50)}
    assert len(ids) == 50


def test_append_rejects_caller_supplied_audit_id(logger: AuditLogger) -> None:
    """Must-fail: API does not accept audit_id kwarg.

    Per plan §4.1: callers must not be able to control audit_id. A
    regression that accepted it would allow an attacker in a compromised
    call path to inject a forged id and smear provenance.
    """
    with pytest.raises(TypeError):
        logger.append(  # type: ignore[call-arg]
            event="voice.listen",
            payload={},
            audit_id="attacker-chosen",
        )


# ---------- first-record genesis ----------


def test_first_record_prev_hmac_is_genesis(logger: AuditLogger) -> None:
    """Must-pass: chain seed uses the literal GENESIS sentinel."""
    logger.append(event="voice.listen", payload={"text": "hi"})
    first = _read_lines(logger.path)[0]
    assert first["prev_hmac"] == GENESIS_PREV_HMAC


def test_non_genesis_seed_rejected_by_verify(tmp_path: Path, audit_key: bytes) -> None:
    """Must-fail: if the log's first prev_hmac isn't GENESIS, verify fails.

    Without this, an attacker could splice in a fresh file with a
    fabricated prev_hmac value and verify_chain would happily accept
    it.
    """
    logger = AuditLogger(
        path=tmp_path / "audit.jsonl", key=audit_key, set_append_only=False
    )
    logger.append(event="voice.listen", payload={"text": "hi"})
    ok, bad = verify_chain(
        logger.path, audit_key, initial_prev_hmac="WRONG-SEED"
    )
    assert (ok, bad) == (False, 1)


# ---------- chain verification ----------


def test_verify_succeeds_on_unmodified_chain(logger: AuditLogger, audit_key: bytes) -> None:
    """Must-pass: five sequential appends verify cleanly."""
    for i in range(5):
        logger.append(event="voice.listen", payload={"turn": i})
    ok, bad = verify_chain(logger.path, audit_key)
    assert (ok, bad) == (True, None)


def _tamper_line(path: Path, line_index: int, mutator) -> None:
    """Load line_index (0-based), apply mutator(dict)->dict, rewrite."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[line_index])
    rec = mutator(rec)
    lines[line_index] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_verify_rejects_payload_tamper(logger: AuditLogger, audit_key: bytes) -> None:
    """Must-fail: mutating any record's payload invalidates its HMAC."""
    for i in range(3):
        logger.append(event="voice.listen", payload={"turn": i, "text": "a"})

    def attacker(rec: dict) -> dict:
        rec["payload"] = {"turn": 1, "text": "EXFILTRATED"}
        return rec

    _tamper_line(logger.path, line_index=1, mutator=attacker)
    ok, bad = verify_chain(logger.path, audit_key)
    assert ok is False and bad == 2  # 1-indexed line numbers


def test_verify_rejects_hmac_tamper(logger: AuditLogger, audit_key: bytes) -> None:
    """Must-fail: mutating the hmac field itself fails recomputation."""
    logger.append(event="voice.listen", payload={"turn": 0})
    logger.append(event="voice.listen", payload={"turn": 1})

    def attacker(rec: dict) -> dict:
        rec["hmac"] = "0" * 64
        return rec

    _tamper_line(logger.path, line_index=0, mutator=attacker)
    ok, bad = verify_chain(logger.path, audit_key)
    assert ok is False and bad == 1


def test_verify_rejects_prev_hmac_tamper(
    logger: AuditLogger, audit_key: bytes
) -> None:
    """Must-fail: mutating prev_hmac breaks chain linkage."""
    logger.append(event="voice.listen", payload={"turn": 0})
    logger.append(event="voice.listen", payload={"turn": 1})
    logger.append(event="voice.listen", payload={"turn": 2})

    def attacker(rec: dict) -> dict:
        rec["prev_hmac"] = "f" * 64
        return rec

    _tamper_line(logger.path, line_index=1, mutator=attacker)
    ok, bad = verify_chain(logger.path, audit_key)
    assert ok is False and bad == 2


def test_verify_rejects_inserted_line(logger: AuditLogger, audit_key: bytes) -> None:
    """Must-fail: splicing in an attacker-crafted record between valid ones fails.

    The inserted record cannot satisfy both its own HMAC AND the next
    record's prev_hmac — so verify fails either on the insert or on
    the record after.
    """
    logger.append(event="voice.listen", payload={"turn": 0})
    logger.append(event="voice.listen", payload={"turn": 1})

    lines = logger.path.read_text(encoding="utf-8").splitlines()
    fake = {
        "audit_id": "deadbeef" * 4,
        "ts": "2026-04-18T00:00:00.000000Z",
        "event": "voice.listen",
        "payload": {"injected": True},
        "prev_hmac": GENESIS_PREV_HMAC,
        "hmac": "0" * 64,
    }
    lines.insert(1, json.dumps(fake, sort_keys=True, separators=(",", ":")))
    logger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, bad = verify_chain(logger.path, audit_key)
    assert ok is False and bad in (2, 3)


def test_verify_rejects_malformed_json(
    logger: AuditLogger, audit_key: bytes, tmp_path: Path
) -> None:
    """Must-fail: a corrupted non-JSON line is detected, not swallowed."""
    logger.append(event="voice.listen", payload={"turn": 0})
    logger.path.write_text(
        logger.path.read_text(encoding="utf-8") + "definitely-not-json\n",
        encoding="utf-8",
    )
    ok, bad = verify_chain(logger.path, audit_key)
    assert ok is False and bad == 2


# ---------- rotation ----------


def test_rotation_triggers_at_threshold(tmp_path: Path, audit_key: bytes) -> None:
    """Must-pass: when file grows past rotation_bytes, original is renamed
    and a new file is seeded with a voice.audit.rotation record."""
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(
        path=log_path,
        key=audit_key,
        rotation_bytes=600,  # ~2-3 small records triggers it
        set_append_only=False,
    )
    for i in range(8):
        logger.append(event="voice.listen", payload={"turn": i, "text": "ab" * 40})

    rotated = list(tmp_path.glob("audit.jsonl.*"))
    assert rotated, "rotation should have produced at least one rotated file"

    new_records = _read_lines(log_path)
    assert new_records, "new active log should not be empty"
    assert new_records[0]["event"] == "voice.audit.rotation"
    assert "rotated_to" in new_records[0]["payload"]


def test_rotation_preserves_chain_across_files(
    tmp_path: Path, audit_key: bytes
) -> None:
    """Must-pass: every rotated file + the active file verify when
    walked in order, each seeded with the previous file's last hmac.

    The FIRST rotated file is seeded from GENESIS; subsequent files
    are seeded from the last hmac of the prior file. This is the
    contract an auditor follows to reconstruct a multi-file history.
    """
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(
        path=log_path,
        key=audit_key,
        rotation_bytes=1200,
        set_append_only=False,
    )
    for i in range(8):
        logger.append(event="voice.listen", payload={"turn": i, "text": "ab" * 40})

    rotated_files = sorted(tmp_path.glob("audit.jsonl.*"))
    assert rotated_files, "rotation should produce at least one rotated file"

    chain = [*rotated_files, log_path]
    prev_hmac = GENESIS_PREV_HMAC
    for f in chain:
        ok, bad = verify_chain(f, audit_key, initial_prev_hmac=prev_hmac)
        assert (ok, bad) == (True, None), f"{f.name} failed at line {bad}"
        records = _read_lines(f)
        if records:
            prev_hmac = records[-1]["hmac"]


def test_rotation_without_seed_fails_new_file_verify(
    tmp_path: Path, audit_key: bytes
) -> None:
    """Must-fail-if-missing-seed: without the rotated file's last hmac,
    verifying the new file against GENESIS must fail on line 1.

    This is the contract: rotation is only safe BECAUSE the chain
    survives across files. Mis-verifying the new file standalone
    must be detectable.
    """
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(
        path=log_path,
        key=audit_key,
        rotation_bytes=600,
        set_append_only=False,
    )
    for i in range(8):
        logger.append(event="voice.listen", payload={"turn": i, "text": "ab" * 40})

    ok, bad = verify_chain(log_path, audit_key)
    assert ok is False and bad == 1


# ---------- key loading ----------


def test_auto_generated_key_has_restrictive_mode(tmp_path: Path) -> None:
    """Must-pass: first use creates audit.key at 0o600."""
    log_path = tmp_path / "audit.jsonl"
    AuditLogger(path=log_path, set_append_only=False).append(
        event="voice.listen", payload={}
    )
    key_path = log_path.parent / "audit.key"
    assert key_path.exists()
    mode = key_path.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


def test_short_supplied_key_raises_voice_error(tmp_path: Path) -> None:
    """Must-fail: supplying a <32-byte key raises before any write."""
    with pytest.raises(VoiceError):
        AuditLogger(
            path=tmp_path / "audit.jsonl",
            key=b"\x00" * 16,
            set_append_only=False,
        )


def test_persisted_short_key_raises_voice_error(tmp_path: Path) -> None:
    """Must-fail: a pre-existing too-short key file is rejected."""
    key_path = tmp_path / "audit.key"
    key_path.write_bytes(b"\x00" * 16)
    with pytest.raises(VoiceError):
        AuditLogger(
            path=tmp_path / "audit.jsonl",
            key_path=key_path,
            set_append_only=False,
        )


# ---------- deterministic HMAC reproducibility ----------


def test_hmac_recompute_matches_on_disk(
    logger: AuditLogger, audit_key: bytes
) -> None:
    """Must-pass: recomputing HMAC from the on-disk canonical body
    matches the stored hmac field — this is the invariant verify_chain
    relies on, tested directly."""
    from lib.hal.voice.audit import _canonical_bytes, _compute_hmac, _HMAC_FIELDS

    logger.append(event="voice.listen", payload={"turn": 0})
    rec = _read_lines(logger.path)[0]
    body = {k: rec[k] for k in _HMAC_FIELDS}
    recomputed = _compute_hmac(audit_key, _canonical_bytes(body))
    assert recomputed == rec["hmac"]
