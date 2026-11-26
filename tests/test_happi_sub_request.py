"""HAPPI γ-3 fractal sub_request expansion tests.

Exercises the runtime's recursive sub_request expansion + the new
`compose` command that triggers it. Covers:

  - compose emits one sub_request per child envelope
  - each sub_request is followed by the child's full event block
    (started -> ... -> completed / error) inlined into the parent stream
  - depth cap prevents runaway recursion
  - malformed child payloads emit sub_request_parse_error and DO NOT halt
  - non-list flags.envelopes emits invalid_compose
  - non-dict child entries emit invalid_child
  - dispatching via public `_dispatch` vs `_dispatch_core` — the former
    expands sub_requests, the latter does not (pure leaf)
  - transports (HTTP, socket, runner.py) inherit expansion automatically
    because they all call `_dispatch`
"""
from __future__ import annotations

import json
from typing import Iterator

import pytest

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.events import emit_sub_request
from lib.hal.happi.runtime import (
    GAMMA3_COMPOSE_COMMANDS,
    MAX_SUB_REQUEST_DEPTH,
    SUPPORTED_COMMANDS,
    _dispatch,
    _dispatch_compose,
    _dispatch_core,
    _expand_sub_requests,
)


def _parse(lines: Iterator[str]) -> list[dict]:
    return [json.loads(line) for line in lines]


def _envelope(cmd: str = "compose", **overrides) -> HappiEnvelope:
    payload = {
        "v": "happi/1.0",
        "id": "root",
        "cmd": cmd,
        "flags": {},
    }
    payload.update(overrides)
    return HappiEnvelope.parse(payload)


# --- compose command --------------------------------------------------------

def test_compose_is_supported_command():
    assert "compose" in GAMMA3_COMPOSE_COMMANDS
    assert "compose" in SUPPORTED_COMMANDS


def test_compose_empty_list_emits_zero_sub_requests():
    env = _envelope(flags={"envelopes": []})
    events = _parse(_dispatch(env))
    types = [e["type"] for e in events]
    assert types == ["started", "completed"]
    assert events[-1]["usage"] == {"children": 0}


def test_compose_with_one_child_emits_sub_request_then_child_events():
    env = _envelope(flags={"envelopes": [
        {"v": "happi/1.0", "id": "child-1", "cmd": "noop"},
    ]})
    events = _parse(_dispatch(env))
    types = [e["type"] for e in events]
    # parent: started, sub_request, <child events>, completed
    assert types == ["started", "sub_request", "started", "completed", "completed"]
    # IDs trace the tree: root, root(sub_request pointing at child), child, child, root
    ids = [e["id"] for e in events]
    assert ids == ["root", "root", "child-1", "child-1", "root"]


def test_compose_preserves_child_order():
    env = _envelope(flags={"envelopes": [
        {"v": "happi/1.0", "id": "c-1", "cmd": "noop"},
        {"v": "happi/1.0", "id": "c-2", "cmd": "noop"},
        {"v": "happi/1.0", "id": "c-3", "cmd": "noop"},
    ]})
    events = _parse(_dispatch(env))
    ids = [e["id"] for e in events]
    # started(root) sr(root,->c-1) started(c-1) completed(c-1)
    # sr(root,->c-2) started(c-2) completed(c-2)
    # sr(root,->c-3) started(c-3) completed(c-3)
    # completed(root)
    assert ids == [
        "root",
        "root", "c-1", "c-1",
        "root", "c-2", "c-2",
        "root", "c-3", "c-3",
        "root",
    ]


def test_compose_rejects_non_list_envelopes_flag():
    env = _envelope(flags={"envelopes": "not-a-list"})
    events = _parse(_dispatch(env))
    types = [e["type"] for e in events]
    assert "error" in types
    err = next(e for e in events if e["type"] == "error")
    assert err["code"] == "invalid_compose"


def test_compose_with_non_dict_child_emits_invalid_child_but_continues():
    env = _envelope(flags={"envelopes": [
        {"v": "happi/1.0", "id": "ok-1", "cmd": "noop"},
        "not-a-dict",
        {"v": "happi/1.0", "id": "ok-2", "cmd": "noop"},
    ]})
    events = _parse(_dispatch(env))
    types = [e["type"] for e in events]
    # started, sub_request(ok-1), ok-1 events, error(invalid_child),
    # sub_request(ok-2), ok-2 events, completed
    # = started, sub_request, started, completed, error, sub_request, started, completed, completed
    assert types.count("error") == 1
    err = next(e for e in events if e["type"] == "error")
    assert err["code"] == "invalid_child"
    # Both good children still processed:
    assert any(e.get("id") == "ok-1" for e in events)
    assert any(e.get("id") == "ok-2" for e in events)


# --- nested recursion ------------------------------------------------------

def test_compose_with_nested_compose_recurses():
    """A child that is itself a `compose` envelope triggers deeper expansion."""
    env = _envelope(flags={"envelopes": [
        {
            "v": "happi/1.0", "id": "inner-compose", "cmd": "compose",
            "flags": {"envelopes": [
                {"v": "happi/1.0", "id": "grand-1", "cmd": "noop"},
                {"v": "happi/1.0", "id": "grand-2", "cmd": "echo",
                 "args": ["hi"]},
            ]},
        },
    ]})
    events = _parse(_dispatch(env))
    ids = [e["id"] for e in events]
    types = [e["type"] for e in events]

    # Grandchildren events must appear inside the inner-compose's block,
    # which is itself inside root's sub_request expansion.
    assert "grand-1" in ids
    assert "grand-2" in ids
    # grand-2 is an echo -> 3 events (started, delta, completed)
    grand_2_types = [t for i, t in zip(ids, types) if i == "grand-2"]
    assert grand_2_types == ["started", "delta", "completed"]


# --- depth cap -------------------------------------------------------------

def _build_compose_chain(depth: int, leaf_cmd: str = "noop") -> dict:
    """Return a compose envelope whose one child recurses `depth` levels."""
    envelope = {"v": "happi/1.0", "id": f"leaf", "cmd": leaf_cmd}
    for level in range(depth):
        envelope = {
            "v": "happi/1.0",
            "id": f"compose-level-{depth - level}",
            "cmd": "compose",
            "flags": {"envelopes": [envelope]},
        }
    return envelope


def test_depth_cap_emits_error_beyond_max_depth():
    # Build a chain that exceeds the cap. Depth of the public dispatch
    # starts at 0, so the leaf sits at depth `chain_length`.
    over = _build_compose_chain(depth=MAX_SUB_REQUEST_DEPTH + 2)
    env = HappiEnvelope.parse(over)
    events = _parse(_dispatch(env))
    codes = [e.get("code") for e in events if e["type"] == "error"]
    assert "sub_request_depth_cap" in codes


def test_at_cap_boundary_the_leaf_still_dispatches():
    """Exactly MAX_SUB_REQUEST_DEPTH deep should succeed — the cap is the
    first depth at which recursion REFUSES to descend further."""
    at_cap = _build_compose_chain(depth=MAX_SUB_REQUEST_DEPTH - 1)
    env = HappiEnvelope.parse(at_cap)
    events = _parse(_dispatch(env))
    ids = [e["id"] for e in events]
    # The leaf must have dispatched — its id appears in the stream.
    assert "leaf" in ids
    # And NO depth_cap error triggered.
    codes = [e.get("code") for e in events if e["type"] == "error"]
    assert "sub_request_depth_cap" not in codes


# --- expander isolation ----------------------------------------------------

def test_expand_sub_requests_passes_non_subrequest_events_through():
    """Non-sub_request events flow through the expander verbatim."""
    env = _envelope(cmd="echo", args=["hi"])
    # Use core (no expansion) as baseline.
    core_events = list(_dispatch_core(env))
    expanded = list(_expand_sub_requests(iter(core_events)))
    assert expanded == core_events  # byte-identical


def test_expand_handles_malformed_json_line_gracefully():
    """Expander must not crash on a non-JSON line (pass through untouched)."""
    events = list(_expand_sub_requests(iter(["not-json\n", '{"type":"noop"}'])))
    assert events == ["not-json\n", '{"type":"noop"}']


def test_sub_request_with_invalid_child_payload_yields_parse_error():
    # Craft a sub_request with a `child` that's not a dict.
    bad_line = json.dumps({
        "v": "happi/1.0", "id": "parent", "type": "sub_request",
        "child": "not-a-dict", "ts": 123,
    })
    events = _parse(_expand_sub_requests(iter([bad_line])))
    # Original sub_request plus a parse_error event.
    types = [e["type"] for e in events]
    assert types == ["sub_request", "error"]
    assert events[1]["code"] == "sub_request_parse_error"


def test_sub_request_with_invalid_envelope_yields_parse_error():
    # Child field is a dict but fails envelope parse (wrong version).
    bad_line = json.dumps({
        "v": "happi/1.0", "id": "parent", "type": "sub_request",
        "child": {"v": "happi/9.9", "id": "bad-child", "cmd": "noop"},
        "ts": 123,
    })
    events = _parse(_expand_sub_requests(iter([bad_line])))
    types = [e["type"] for e in events]
    assert types == ["sub_request", "error"]
    assert events[1]["code"] == "sub_request_parse_error"


# --- core vs public dispatch -----------------------------------------------

def test_dispatch_core_does_not_expand_sub_requests():
    """_dispatch_core is the leaf — sub_request events leave un-expanded."""
    env = _envelope(flags={"envelopes": [
        {"v": "happi/1.0", "id": "c-1", "cmd": "noop"},
    ]})
    core_events = _parse(_dispatch_core(env))
    types = [e["type"] for e in core_events]
    # Only root's own events: started, sub_request, completed. No child
    # dispatch interleaved.
    assert types == ["started", "sub_request", "completed"]


def test_public_dispatch_wraps_core_with_expansion():
    env = _envelope(flags={"envelopes": [
        {"v": "happi/1.0", "id": "c-1", "cmd": "noop"},
    ]})
    core_types = [json.loads(l)["type"] for l in _dispatch_core(env)]
    public_types = [json.loads(l)["type"] for l in _dispatch(env)]
    # Public expands the sub_request into started/completed of child.
    assert len(public_types) > len(core_types)
    assert public_types == ["started", "sub_request", "started", "completed", "completed"]


# --- Compose dispatch helper directly --------------------------------------

def test_dispatch_compose_emits_only_sub_requests_plus_completed():
    env = _envelope(flags={"envelopes": [
        {"v": "happi/1.0", "id": "a", "cmd": "noop"},
        {"v": "happi/1.0", "id": "b", "cmd": "noop"},
    ]})
    events = _parse(_dispatch_compose(env))
    types = [e["type"] for e in events]
    # _dispatch_compose is a leaf — does NOT include started (runtime's
    # wrapper adds that) and does NOT expand sub_requests.
    assert types == ["sub_request", "sub_request", "completed"]
    assert events[-1]["usage"]["children"] == 2


# --- spec field name: the marker carries the child under `envelope` ---------
#
# happi.md's event table is the contract (axiom 4):
#
#     | `sub_request` | `envelope` | Child HAPPI envelope dispatched |
#
# The polyglot reference runtime follows that table. This library emitted
# `child` instead, which PR #523 identified as "a real divergence from the
# spec" and deliberately left alone rather than change silently. These tests
# are the anchor for closing it: they fail against the pre-fix library.

def test_sub_request_marker_uses_spec_field_name_envelope():
    """The wire field is `envelope`, per the happi.md event table."""
    marker = json.loads(emit_sub_request(
        "parent", {"v": "happi/1.0", "id": "kid", "cmd": "version"}))
    assert "envelope" in marker, f"sub_request fields: {sorted(marker)}"
    assert marker["envelope"]["id"] == "kid"
    assert marker["envelope"]["cmd"] == "version"


def test_sub_request_marker_does_not_carry_the_legacy_child_field():
    """`child` is gone from the wire — parity with the polyglot runtime."""
    marker = json.loads(emit_sub_request(
        "parent", {"v": "happi/1.0", "id": "kid", "cmd": "version"}))
    assert "child" not in marker, f"sub_request fields: {sorted(marker)}"


def test_compose_stream_marker_carries_envelope():
    """End-to-end: a compose stream's marker uses the spec field name."""
    env = _envelope(flags={"envelopes": [
        {"v": "happi/1.0", "id": "c-1", "cmd": "noop"},
    ]})
    markers = [e for e in _parse(_dispatch_compose(env))
               if e["type"] == "sub_request"]
    assert len(markers) == 1
    assert markers[0]["envelope"]["id"] == "c-1"


def test_expander_still_reads_a_legacy_child_keyed_marker():
    """Streams emitted before the rename must still replay.

    The reader is deliberately more tolerant than the writer: it accepts the
    canonical `envelope` and the legacy `child`, so historical NDJSON and
    audit replays do not break. Only the emitted form is spec-tightened.
    """
    legacy = json.dumps({
        "v": "happi/1.0", "id": "parent", "type": "sub_request",
        "child": {"v": "happi/1.0", "id": "legacy-kid", "cmd": "noop"},
        "ts": 123,
    })
    events = _parse(_expand_sub_requests(iter([legacy])))
    types = [e["type"] for e in events]
    assert "error" not in types, f"legacy marker failed to expand: {events}"
    assert any(e.get("id") == "legacy-kid" for e in events), (
        f"child envelope was not dispatched: {events}")
