"""Tests for scripts/vps-secrets-backup.sh.

The script's contract: encrypt ~/.hal/secrets.env + ~/.hal/state to a gpg
blob, verify it by test-decrypting before trusting it, record a
``.last-success`` timestamp+size, prune backups past retention, and expose
``--check`` (fresh/stale) and ``--emit-latest`` (the forced-SSH-command body)
without ever putting secret content on stdout/stderr/the log.

Real ``tar``/``gpg`` are used throughout (no fakes) because the script's own
value is "does the encrypted round-trip actually work", which a fake gpg
cannot answer. The one exception is ``test_min_bytes_floor_rejects_tiny_output``,
which shims a fake ``gpg`` on PATH to force the empty-output case
deterministically, mirroring the PATH-shim technique already used in
tests/test_vps_service_watchdog.py.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from tests._public_tree import folder_absent, reason

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "vps-secrets-backup.sh"
# Public HAL leaves scripts/ out, so on a public checkout every test here is
# skipped with that reason (HAL#859); in the full repository they all run.
pytestmark = pytest.mark.skipif(folder_absent("scripts"), reason=reason("scripts"))

SECRET_MARKER = "FAKE_SECRET_DO_NOT_LEAK_7f3a"

FAKE_EMPTY_GPG = r"""#!/usr/bin/env bash
# Stands in for gpg only in test_min_bytes_floor_rejects_tiny_output: writes
# an (almost) empty file regardless of its arguments, to force the backup
# script's own byte-floor check deterministically.
for a in "$@"; do
    :
done
prev=""
for a in "$@"; do
    [ "$prev" = "-o" ] && printf 'x' > "$a"
    prev="$a"
done
exit 0
"""


class Harness:
    """A throwaway HAL_HOME with a real secrets.env + state dir."""

    def __init__(self, tmp: Path):
        self.hal_home = tmp / "hal"
        (self.hal_home / "state").mkdir(parents=True)
        (self.hal_home / "secrets.env").write_text(f"{SECRET_MARKER}=1\n")
        self.passphrase_file = self.hal_home / ".backup-passphrase"
        self.passphrase_file.write_text("test-only-passphrase-not-real\n")
        self.backup_dir = self.hal_home / "backups"
        self.log = self.hal_home / "logs" / "vps-secrets-backup.log"

    def env(self, **overrides: str) -> dict:
        e = dict(os.environ)
        e.update(
            HAL_HOME=str(self.hal_home),
            BACKUP_DIR=str(self.backup_dir),
            BACKUP_PASSPHRASE_FILE=str(self.passphrase_file),
            BACKUP_LOG=str(self.log),
        )
        e.update(overrides)
        return e

    def run(self, *args: str, **env_overrides: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            env=self.env(**env_overrides),
            capture_output=True,
            text=False,  # stdout can be binary (the ciphertext channel)
            timeout=30,
        )

    def last_success(self) -> tuple[int, int] | None:
        p = self.backup_dir / ".last-success"
        if not p.exists():
            return None
        ts_s, bytes_s = p.read_text().split()
        return int(ts_s), int(bytes_s)

    def write_last_success(self, ts: int, nbytes: int) -> None:
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        (self.backup_dir / ".last-success").write_text(f"{ts} {nbytes}\n")

    def backup_files(self) -> list[Path]:
        if not self.backup_dir.exists():
            return []
        return sorted(self.backup_dir.glob("secrets-*.tar.gpg"))


@pytest.fixture()
def hb(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def _assert_no_secret_leak(*blobs: bytes, log_path: Path | None = None) -> None:
    needle = SECRET_MARKER.encode()
    for b in blobs:
        assert needle not in b, "secret content leaked into script output"
    if log_path is not None and log_path.exists():
        assert needle not in log_path.read_bytes(), "secret content leaked into the log"


def test_check_fails_before_any_backup(hb: Harness) -> None:
    r = hb.run("--check")
    assert r.returncode == 1
    _assert_no_secret_leak(r.stdout, r.stderr)


def test_backup_creates_last_success_and_tar_gpg(hb: Harness) -> None:
    r = hb.run()
    assert r.returncode == 0, r.stderr
    assert hb.last_success() is not None
    files = hb.backup_files()
    assert len(files) == 1
    assert files[0].stat().st_size > 64
    _assert_no_secret_leak(r.stdout, r.stderr, log_path=hb.log)


def test_check_passes_right_after_a_fresh_backup(hb: Harness) -> None:
    hb.run()
    r = hb.run("--check")
    assert r.returncode == 0, r.stderr


def test_check_fails_when_last_success_is_stale(hb: Harness) -> None:
    # 30 hours ago: past the default 26h floor without touching real time.
    hb.write_last_success(int(time.time()) - 30 * 3600, 1234)
    r = hb.run("--check")
    assert r.returncode == 1


def test_check_respects_custom_stale_window(hb: Harness) -> None:
    hb.write_last_success(int(time.time()) - 3 * 3600, 1234)
    assert hb.run("--check", BACKUP_STALE_HOURS="2").returncode == 1
    assert hb.run("--check", BACKUP_STALE_HOURS="4").returncode == 0


def test_non_numeric_setting_falls_back_to_the_default_with_a_warning(hb: Harness) -> None:
    # AGORA on HAL#717: "7d" used to become 26 hours in silence. The default
    # still applies (3h old is fresh under 26h), but the substitution is now
    # named on stderr AND in the log, naming the variable, the value and the
    # default, so an operator can see why their setting had no effect.
    hb.write_last_success(int(time.time()) - 3 * 3600, 1234)
    r = hb.run("--check", BACKUP_STALE_HOURS="7d")
    assert r.returncode == 0, r.stderr
    warning = b"WARN BACKUP_STALE_HOURS is not a whole number (got '7d'); using the default 26"
    assert warning in r.stderr
    assert warning.decode() in hb.log.read_text()
    # A whole number is taken as given, with no warning at all.
    quiet = hb.run("--check", BACKUP_STALE_HOURS="4")
    assert quiet.returncode == 0
    assert b"WARN" not in quiet.stderr


def test_emit_latest_header_then_body_reconstructs_exactly(hb: Harness) -> None:
    hb.run()
    r = hb.run("--emit-latest")
    assert r.returncode == 0, r.stderr
    assert r.stderr == b""
    header, _, body = r.stdout.partition(b"\n")
    ts_s, bytes_s = header.decode().split()
    assert int(ts_s) > 0
    assert len(body) == int(bytes_s)
    # The body is exactly the on-disk ciphertext -- no re-framing in transit.
    on_disk = hb.backup_files()[0].read_bytes()
    assert body == on_disk
    _assert_no_secret_leak(r.stdout)


def test_emit_latest_fails_with_no_backup_yet(hb: Harness) -> None:
    r = hb.run("--emit-latest")
    assert r.returncode == 1
    assert r.stdout == b""  # the puller must never mistake stderr for payload


def test_retention_prunes_backups_older_than_the_window(hb: Harness) -> None:
    hb.run()  # today's backup
    old = hb.backup_dir / "secrets-20250101T000000Z.tar.gpg"
    old.write_bytes(b"old-ciphertext-not-real")
    old_epoch = time.time() - 20 * 86400  # 20 days old
    os.utime(old, (old_epoch, old_epoch))
    hb.run(BACKUP_RETENTION_DAYS="14")  # a second run prunes on its way out
    names = [f.name for f in hb.backup_files()]
    assert old.name not in names
    assert any(n != old.name for n in names)  # today's two backups remain


def test_backup_fails_without_a_passphrase_file(hb: Harness) -> None:
    hb.passphrase_file.unlink()
    r = hb.run()
    assert r.returncode == 2
    assert hb.last_success() is None
    assert hb.backup_files() == []


def test_backup_fails_without_secrets_env(hb: Harness) -> None:
    (hb.hal_home / "secrets.env").unlink()
    r = hb.run()
    assert r.returncode == 2
    assert hb.last_success() is None


def test_min_bytes_floor_rejects_tiny_output(hb: Harness, tmp_path: Path) -> None:
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    fake_gpg = fake_bin / "gpg"
    fake_gpg.write_text(FAKE_EMPTY_GPG)
    fake_gpg.chmod(0o755)
    env = hb.env(PATH=f"{fake_bin}:{os.environ['PATH']}")
    r = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, timeout=30
    )
    assert r.returncode == 1
    assert hb.last_success() is None
    assert hb.backup_files() == []  # the undersized tmp file was never promoted


def test_a_second_concurrent_backup_skips_instead_of_racing(hb: Harness) -> None:
    hb.backup_dir.mkdir(parents=True)
    lock = hb.backup_dir / ".backup.lock"
    lock.mkdir()
    try:
        r = hb.run()
        assert r.returncode == 0  # a held lock is a designed skip, not a failure
        assert hb.last_success() is None  # nothing ran underneath the lock
    finally:
        lock.rmdir()


def test_unknown_flag_is_rejected() -> None:
    r = subprocess.run(
        ["bash", str(SCRIPT), "--nonsense"], capture_output=True, timeout=10
    )
    assert r.returncode == 64


def test_script_syntax_is_valid_bash() -> None:
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
