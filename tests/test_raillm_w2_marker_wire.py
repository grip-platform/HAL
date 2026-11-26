"""Goodhart-proof P3 W2 — the per-turn active-llm marker wire in
grip_infer is opt-in and fail-safe-to-the-fixed-ladder.

Fully hermetic: the GRIP active_llm loader and the happi runtime
boundary (_build_envelope/_drive/_collect) are monkeypatched, so no
real marker file, no real LLM, no ~/.claude dependency.

Mutation-survival (each fails a test if the invariant breaks):
  - marker ignored / not pinned          -> test_valid_solo_marker_pins FAILS
  - non-opted path not byte-identical    -> test_absent_marker_fixed_default FAILS
  - staleness gate dropped               -> test_stale_marker_ignored FAILS
  - ladder-membership gate dropped       -> test_addr_not_in_ladder_ignored FAILS
  - show_active error propagates         -> test_show_active_raises_failsafe FAILS
  - marker overrides an explicit model   -> test_explicit_model_wins FAILS
  - solo-once not consumed / wrong order -> test_solo_once_consumed_after_dispatch FAILS
  - any non-solo mode accepted as a pin  -> test_non_solo_mode_ignored FAILS
"""
from __future__ import annotations

import time

import lib.hal.grip_runtime as gr
from lib.hal.raillm_ladder import GRIP_LADDER_ADDRESSES

_DEFAULT = GRIP_LADDER_ADDRESSES[0]
_OTHER = GRIP_LADDER_ADDRESSES[1]          # a valid non-default ladder addr
_FRESH = lambda: int(time.time())          # noqa: E731


class _FakeActiveLLM:
    def __init__(self, rec, *, raises=False):
        self._rec = rec
        self._raises = raises
        self.consumed = 0

    def show_active(self):
        if self._raises:
            raise RuntimeError("marker read boom")
        return self._rec

    def consume_once_marker(self):
        self.consumed += 1
        return self._rec


def _wire(monkeypatch, fake):
    """Bypass the cached cross-repo loader entirely + capture the address
    grip_infer hands the runtime; record call ordering for the once test."""
    order: list[str] = []
    monkeypatch.setattr(gr, "_grip_active_llm", lambda: fake)
    captured: dict = {}

    # The fakes take the real call shape: grip_infer passes content_parts,
    # pin and max_tokens to _build_envelope and media_attached to _collect.
    # Fixed signatures went stale as those keywords arrived, and every test
    # here died with TypeError before reaching the marker logic.
    def fake_build(prompt, model, audit, **_flags):
        captured["requested"] = model
        return {"v": "happi/1.1", "model": model}

    def fake_drive(envelope):
        order.append("drive")
        return []

    monkeypatch.setattr(gr, "_build_envelope", fake_build)
    monkeypatch.setattr(gr, "_drive", fake_drive)
    monkeypatch.setattr(gr, "_collect", lambda events, requested, **_: "OK")
    return captured, order


def test_valid_solo_marker_pins(monkeypatch):
    fake = _FakeActiveLLM({"mode": "solo", "address": _OTHER,
                           "ts": _FRESH(), "ttl_seconds": 1800})
    cap, _ = _wire(monkeypatch, fake)
    assert gr.grip_infer("hi") == "OK"
    assert cap["requested"] == _OTHER          # pinned, not the default


def test_absent_marker_fixed_default(monkeypatch):
    cap, _ = _wire(monkeypatch, None)          # loader -> None (HAL standalone)
    monkeypatch.setattr(gr, "_grip_active_llm", lambda: None)
    gr.grip_infer("hi")
    assert cap["requested"] == _DEFAULT        # byte-identical to pre-W2


def test_stale_marker_ignored(monkeypatch):
    fake = _FakeActiveLLM({"mode": "solo", "address": _OTHER,
                           "ts": 0, "ttl_seconds": 1})   # ts+ttl << now
    cap, _ = _wire(monkeypatch, fake)
    gr.grip_infer("hi")
    assert cap["requested"] == _DEFAULT        # stale -> fixed default


def test_addr_not_in_ladder_ignored(monkeypatch):
    fake = _FakeActiveLLM({"mode": "solo", "address": "bogus/not-a-rung",
                           "ts": _FRESH(), "ttl_seconds": 1800})
    cap, _ = _wire(monkeypatch, fake)
    gr.grip_infer("hi")
    assert cap["requested"] == _DEFAULT


def test_show_active_raises_failsafe(monkeypatch):
    fake = _FakeActiveLLM(None, raises=True)
    cap, _ = _wire(monkeypatch, fake)
    assert gr.grip_infer("hi") == "OK"         # did NOT raise
    assert cap["requested"] == _DEFAULT


def test_explicit_model_wins(monkeypatch):
    fake = _FakeActiveLLM({"mode": "solo", "address": _OTHER,
                           "ts": _FRESH(), "ttl_seconds": 1800})
    cap, _ = _wire(monkeypatch, fake)
    gr.grip_infer("hi", model=GRIP_LADDER_ADDRESSES[2])
    assert cap["requested"] == GRIP_LADDER_ADDRESSES[2]   # explicit beats marker
    assert fake.consumed == 0                  # marker untouched when model given


def test_solo_once_consumed_after_dispatch(monkeypatch):
    fake = _FakeActiveLLM({"mode": "solo-once", "address": _OTHER,
                           "ts": _FRESH(), "ttl_seconds": 1800})
    cap, order = _wire(monkeypatch, fake)
    orig_consume = fake.consume_once_marker

    def tracked():
        order.append("consume")
        return orig_consume()

    fake.consume_once_marker = tracked
    gr.grip_infer("hi")
    assert cap["requested"] == _OTHER
    assert fake.consumed == 1                   # consumed exactly once
    assert order == ["drive", "consume"]        # AFTER dispatch, not before


def test_non_solo_mode_ignored(monkeypatch):
    fake = _FakeActiveLLM({"mode": "council", "address": _OTHER,
                           "ts": _FRESH(), "ttl_seconds": 1800})
    cap, _ = _wire(monkeypatch, fake)
    gr.grip_infer("hi")
    assert cap["requested"] == _DEFAULT        # only solo/solo-once pin
    assert fake.consumed == 0
