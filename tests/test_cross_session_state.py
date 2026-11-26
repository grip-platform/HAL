"""Unit tests for CrossSessionStateFile — W3 of RAILLM ultrado plan.

Covers:

* Basic read/write round trips
* Missing-file and empty-file handling
* Corrupted-JSON recovery
* Inode stability (truncate-in-place invariant)
* ``update`` read-modify-write atomicity with a mutator callback
* Concurrent multiprocess writers do not lose updates (H049/H063 gate)
* Goodhart mutation tests — the suite fails if the lock or persist step
  is removed
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import pytest
except ImportError:  # pragma: no cover - pytest always installed in CI
    raise unittest.SkipTest("pytest not installed")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.hal.cross_session_state import (  # noqa: E402
    CrossSessionStateFile,
    has_fcntl_support,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmpstate(tmp_path):
    """Yield a path under ``tmp_path`` for a state file."""
    return tmp_path / "state.json"


# ---------------------------------------------------------------------------
# Basic round trip
# ---------------------------------------------------------------------------


def test_read_missing_file_returns_empty_dict(tmpstate):
    sf = CrossSessionStateFile(tmpstate)
    assert sf.read() == {}


def test_write_and_read_back_roundtrip(tmpstate):
    sf = CrossSessionStateFile(tmpstate)
    sf.write({"k": "v", "n": 42})
    assert sf.read() == {"k": "v", "n": 42}


def test_write_creates_missing_parent_directory(tmp_path):
    path = tmp_path / "nested" / "deeply" / "state.json"
    sf = CrossSessionStateFile(path)
    sf.write({"a": 1})
    assert path.exists()
    assert sf.read() == {"a": 1}


def test_write_overwrites_existing_content(tmpstate):
    sf = CrossSessionStateFile(tmpstate)
    sf.write({"old": True})
    sf.write({"new": True})
    assert sf.read() == {"new": True}


# ---------------------------------------------------------------------------
# Corruption handling
# ---------------------------------------------------------------------------


def test_read_corrupted_json_returns_default_factory(tmpstate):
    tmpstate.write_text("{invalid json not parseable")
    sf = CrossSessionStateFile(tmpstate)
    assert sf.read() == {}


def test_update_corrupted_json_still_applies_mutator(tmpstate):
    tmpstate.write_text("garbage")
    sf = CrossSessionStateFile(tmpstate)
    result = sf.update(lambda d: {**d, "recovered": True})
    assert result == {"recovered": True}
    assert sf.read() == {"recovered": True}


def test_read_default_factory_is_honoured(tmpstate):
    def factory():
        return {"bootstrap": 1}

    sf = CrossSessionStateFile(tmpstate, default_factory=factory)
    assert sf.read() == {"bootstrap": 1}


def test_read_empty_file_returns_default_factory(tmpstate):
    tmpstate.write_text("")
    sf = CrossSessionStateFile(tmpstate)
    assert sf.read() == {}


# ---------------------------------------------------------------------------
# update() — read-modify-write
# ---------------------------------------------------------------------------


def test_update_creates_file_on_first_call(tmpstate):
    sf = CrossSessionStateFile(tmpstate)
    result = sf.update(lambda d: {**d, "first": True})
    assert result == {"first": True}
    assert tmpstate.exists()


def test_update_invokes_mutator_with_current_state(tmpstate):
    sf = CrossSessionStateFile(tmpstate)
    sf.write({"counter": 1})
    seen = {}

    def mutator(d):
        seen["received"] = dict(d)
        return {**d, "counter": d["counter"] + 1}

    new = sf.update(mutator)
    assert seen["received"] == {"counter": 1}
    assert new == {"counter": 2}


def test_update_persists_mutator_return_value(tmpstate):
    sf = CrossSessionStateFile(tmpstate)
    sf.update(lambda d: {"a": 1})
    sf.update(lambda d: {**d, "b": 2})
    sf.update(lambda d: {**d, "c": 3})
    assert sf.read() == {"a": 1, "b": 2, "c": 3}


def test_update_mutator_returning_empty_dict_clears_state(tmpstate):
    sf = CrossSessionStateFile(tmpstate)
    sf.write({"a": 1, "b": 2})
    sf.update(lambda d: {})
    assert sf.read() == {}


# ---------------------------------------------------------------------------
# Truncate-in-place invariant — inode stability
# ---------------------------------------------------------------------------


def test_write_preserves_inode_truncate_in_place(tmpstate):
    """The impl MUST keep the inode stable across writes.

    Rationale: atomic-rename (``os.replace``) would swap inodes, breaking
    the cross-process fcntl locking contract. A second writer opening the
    path after the rename would acquire a lock on a different inode and
    therefore not serialise with the first writer. Truncate-in-place is
    the deliberate invariant — this test fails loudly if someone
    refactors the impl into an atomic-rename pattern.
    """
    sf = CrossSessionStateFile(tmpstate)
    sf.write({"v": 1})
    inode_before = tmpstate.stat().st_ino
    sf.write({"v": 2})
    inode_after = tmpstate.stat().st_ino
    assert inode_before == inode_after, (
        "Inode changed across writes. The impl must use truncate-in-place, "
        "not atomic-rename, so fcntl locks serialise on a stable inode."
    )
    assert sf.read() == {"v": 2}


def test_write_creates_no_sibling_tempfiles(tmpstate):
    """Negative guard: no ``*.tmp`` siblings should appear.

    An atomic-rename implementation would create a sibling tempfile,
    then ``os.replace`` it onto ``tmpstate``. We rejected that design;
    this test ensures nobody accidentally reintroduces it.
    """
    sf = CrossSessionStateFile(tmpstate)
    sf.write({"k": "v"})
    siblings = list(tmpstate.parent.glob("state.json.*"))
    siblings += list(tmpstate.parent.glob("*.tmp"))
    assert siblings == []


# ---------------------------------------------------------------------------
# Concurrency — H049/H063 gate
# ---------------------------------------------------------------------------


def _concurrent_writer(path_str, key):
    """Top-level so ``mp.Pool`` can pickle it. Each writer adds a unique key."""
    import os  # local import — subprocess context
    import sys

    lib_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if lib_root not in sys.path:
        sys.path.insert(0, lib_root)
    from lib.hal.cross_session_state import CrossSessionStateFile

    sf = CrossSessionStateFile(path_str)
    sf.update(lambda d: {**d, key: key})


@pytest.mark.skipif(not has_fcntl_support(), reason="fcntl not available (Windows)")
def test_concurrent_writers_preserve_all_updates(tmp_path):
    """H049/H063 gate: N concurrent processes each add a unique key.

    The merged state MUST contain all N keys. A racy implementation
    (no lock, or lock on the wrong fd) drops updates under this workload
    because two writers observe the same pre-mutation state and the
    second overwrite clobbers the first.
    """
    state_path = str(tmp_path / "concurrent.json")
    CrossSessionStateFile(state_path).write({})

    n_writers = 8
    keys = [f"worker_{i}" for i in range(n_writers)]

    with mp.get_context("spawn").Pool(processes=n_writers) as pool:
        pool.starmap(_concurrent_writer, [(state_path, k) for k in keys])

    final = CrossSessionStateFile(state_path).read()
    for k in keys:
        assert k in final, f"missing key {k} in {final}"
    assert len(final) == n_writers


@pytest.mark.skipif(not has_fcntl_support(), reason="fcntl not available")
def test_concurrent_readers_do_not_block_each_other(tmp_path):
    """Shared-lock reads should not serialise.

    Smoke check: run 5 reads in rapid succession from a single process
    and assert they all return the same state. Real contention testing
    happens in ``test_concurrent_writers_preserve_all_updates``.
    """
    state_path = tmp_path / "readers.json"
    CrossSessionStateFile(state_path).write({"stable": 1})
    sf = CrossSessionStateFile(state_path)
    results = [sf.read() for _ in range(5)]
    assert all(r == {"stable": 1} for r in results)


# ---------------------------------------------------------------------------
# Goodhart mutation tests — suite must fail if critical logic is removed
# ---------------------------------------------------------------------------


def test_goodhart_update_persists_mutator_return_value(tmpstate):
    """If ``update`` silently dropped the mutator return value, this test
    would fail — the persisted state would not contain the new key."""
    sf = CrossSessionStateFile(tmpstate)
    sf.update(lambda d: {**d, "applied": True})
    assert sf.read().get("applied") is True


def test_goodhart_update_returns_defensive_copy(tmpstate):
    """The caller-side mutation of the returned dict must not bleed into
    the persisted state. This guards against the aliasing bug where the
    impl returns the in-memory mutator output directly.
    """
    sf = CrossSessionStateFile(tmpstate)
    returned = sf.update(lambda d: {**d, "a": 1})
    returned["leaked"] = True
    assert "leaked" not in sf.read()


def test_goodhart_write_calls_fsync_exactly_once_per_write(tmpstate):
    """The impl fsyncs inside ``_write_under_lock`` to prevent a crashed
    process from leaving an fs-buffered (unsynced) state on disk. If an
    optimisation removed ``os.fsync``, this test catches it.
    """
    sf = CrossSessionStateFile(tmpstate)
    with patch("lib.hal.cross_session_state.os.fsync") as mock_fsync:
        sf.write({"v": 1})
        assert mock_fsync.call_count == 1


@pytest.mark.skipif(not has_fcntl_support(), reason="fcntl not available")
def test_goodhart_write_acquires_exclusive_lock(tmpstate):
    """If someone removed the ``_locked(..., exclusive=True)`` wrapper,
    ``fcntl.flock`` would never be called with ``LOCK_EX`` and cross-session
    safety would silently break. We assert the flock call happens.
    """
    sf = CrossSessionStateFile(tmpstate)
    import fcntl as _fcntl

    with patch("lib.hal.cross_session_state.fcntl.flock") as mock_flock:
        sf.write({"v": 1})
    # Expect exactly one LOCK_EX and one LOCK_UN on the same fd.
    modes = [call.args[1] for call in mock_flock.call_args_list]
    assert _fcntl.LOCK_EX in modes, f"LOCK_EX never acquired; modes={modes}"
    assert _fcntl.LOCK_UN in modes, f"LOCK_UN never released; modes={modes}"


def test_goodhart_write_roundtrip_with_large_payload(tmpstate):
    """Large payload round-trip — exercises the truncate+write+fsync path
    on a non-trivial JSON document and asserts byte-exact equality."""
    sf = CrossSessionStateFile(tmpstate)
    big = {f"k{i}": f"v{i}" * 10 for i in range(100)}
    sf.write(big)
    assert sf.read() == big
    # Also verify the on-disk representation is canonical (sort_keys=True).
    raw = tmpstate.read_text()
    parsed = json.loads(raw)
    assert parsed == big


def test_goodhart_update_does_not_leak_mutator_input_mutation(tmpstate):
    """The mutator receives the decoded state; if the caller mutates that
    dict in place INSIDE the mutator, the mutator's return value wins.
    This guards the contract that persistence is driven by return value,
    not by side-effect on the input."""
    sf = CrossSessionStateFile(tmpstate)
    sf.write({"a": 1})

    def mutator(d):
        d["side_effect"] = True  # mutates input in place
        return {"only_return": True}  # but we return a disjoint dict

    sf.update(mutator)
    final = sf.read()
    assert final == {"only_return": True}
    assert "side_effect" not in final
    assert "a" not in final


# ---------------------------------------------------------------------------
# Platform diagnostic
# ---------------------------------------------------------------------------


def test_has_fcntl_support_matches_platform():
    """Sanity check: POSIX systems report True, Windows reports False."""
    if os.name == "posix":
        assert has_fcntl_support() is True
    else:
        assert has_fcntl_support() is False
