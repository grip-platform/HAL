"""limits.json's writers (fleet finding f96): whole files, no lost entries.

``state/hal/limits.json`` is shared state: hal-server's ``LimitSentinel``
writes it, and any second process writing it must go through the same door,
``lib/hal/locked_json.py``. Until 2026-10-05 each writer rewrote the file with
a plain ``open(..., "w")``: a reader could catch it half-written, the sentinel
put back its stale copy of entries the other writer had written since it
loaded, and two writers could lose each other's entries. (The second writer
this was measured against, the gateway's key failover, was removed by the
spend guard, Step 5h; the door it forgot to share is what these tests pin.)
Here the second writer is ``locked_json`` used directly, which is exactly what
any future limits.json writer must do. The tests run them at once, in separate
processes, on a temporary file.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

from lib.hal import locked_json, sentinel

ROUNDS = 30


def _sentinel_writer(path_text: str, provider: str, rounds: int) -> str:
    """Run in a spawned process: a sentinel saving one provider ``rounds`` times."""
    sentinel._STATE_PATH = path_text
    watch = sentinel.LimitSentinel(registry=None)
    for value in range(rounds):
        state = sentinel.ProviderLimitState()
        state.utilisation = float(value)
        watch._state[provider] = state
        watch._save_state()
    return provider


def _locked_writer(path_text: str, service: str, rounds: int) -> str:
    """Run in a spawned process: ``rounds`` cooldown bumps on one service,
    written the way every limits.json writer must: one locked, atomic
    read-modify-write through ``locked_json``."""
    for _ in range(rounds):
        locked_json.update_json(
            path_text, lambda states: _with_cooldown(states, service), mode=0o644)
    return service


def _with_cooldown(states: dict, service: str) -> dict:
    """``states`` with ``service``'s consecutive-429 count bumped by one."""
    state = sentinel.ProviderLimitState()
    state.consecutive_429s = states.get(service, {}).get("consecutive_429s", 0) + 1
    state.last_429_at = time.time()
    states[service] = state.to_dict()
    return states


def _watch(path_text: str, stop_text: str) -> tuple:
    """Run in a spawned process: read the file until told to stop; count torn reads."""
    path, stop = Path(path_text), Path(stop_text)
    torn = reads = 0
    deadline = time.monotonic() + 120
    while not stop.exists() and time.monotonic() < deadline:
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            continue
        reads += 1
        try:
            json.loads(raw)
        except ValueError:
            torn += 1
    return torn, reads


def test_concurrent_writers_lose_nothing_and_never_tear_the_file(tmp_path):
    path, stop = tmp_path / "state" / "hal" / "limits.json", tmp_path / "stop"
    providers = [f"provider-{number}" for number in range(3)]
    services = [f"service-{number}" for number in range(3)]
    with mp.get_context("spawn").Pool(len(providers) + len(services) + 1) as pool:
        watcher = pool.apply_async(_watch, (str(path), str(stop)))
        jobs = [pool.apply_async(_sentinel_writer, (str(path), name, ROUNDS))
                for name in providers]
        jobs += [pool.apply_async(_locked_writer, (str(path), name, ROUNDS)) for name in services]
        finished = sorted(job.get(timeout=180) for job in jobs)
        stop.touch()
        torn, reads = watcher.get(timeout=180)
    states = json.loads(path.read_text())
    assert finished == sorted(providers + services)
    assert {name: states[name]["utilisation"] for name in providers} == dict.fromkeys(
        providers, float(ROUNDS - 1))
    assert {name: states[name]["consecutive_429s"] for name in services} == dict.fromkeys(
        services, ROUNDS)
    assert (torn, reads > 0) == (0, True)


def test_the_sentinel_never_puts_back_a_stale_copy_of_another_writer_entry(
        tmp_path, monkeypatch):
    path = tmp_path / "limits.json"
    monkeypatch.setattr(sentinel, "_STATE_PATH", str(path))
    locked_json.update_json(str(path), lambda states: _with_cooldown(states, "service-a"),
                            mode=0o644)  # 1, before the sentinel loads
    watch = sentinel.LimitSentinel(registry=None)        # it loads service-a at 1
    locked_json.update_json(str(path), lambda states: _with_cooldown(states, "service-a"),
                            mode=0o644)  # 2, after it loaded
    watch._on_error("provider-x", RuntimeError("429 rate limited"))
    states = json.loads(path.read_text())
    assert states["service-a"]["consecutive_429s"] == 2  # not put back to 1
    assert states["provider-x"]["consecutive_429s"] == 1
