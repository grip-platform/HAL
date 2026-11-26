"""Axiom 3 exercised against the embedded reference runtime.

HAPPI.md axiom 3: "`sub_request` recurses through the same runtime. The protocol
is fractal; no privileged inner interface."

Until now that was the only axiom with no executable check, and it is the one
most easily satisfied in appearance only: a runtime can emit a `sub_request`
event and then serve the child from a private inner code path, which looks
identical on the wire for a single level and is not fractal at all.

These tests are written against the property rather than the appearance:

  * a child is dispatched THROUGH the same validator and the same handler
    table, proven by making a child fail in ways only the shared path
    produces (out-of-range version, unknown cmd) and asserting the SAME
    error contract a top-level envelope gets;
  * nesting works to arbitrary depth, proven by a compose inside a compose;
  * the recursion is bounded, proven by driving it past the cap.

THE falsifier: give the child a privileged path — a relaxed validator, a
second handler table, or a hard-coded reply — and
`test_child_with_bad_version_fails_exactly_as_a_top_level_envelope_does`
or `test_nested_compose_recurses` fails.
"""
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HAPPI = REPO / "happi.md"


def dispatch(envelope):
    """Send one envelope through `bash happi.md run`; return (rc, events)."""
    proc = subprocess.run(
        ["bash", str(HAPPI), "run"],
        input=json.dumps(envelope), capture_output=True, text=True,
    )
    events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    return proc.returncode, events


def compose(*children, envelope_id="parent"):
    return {"v": "happi/1.3", "id": envelope_id, "cmd": "compose",
            "flags": {"envelopes": list(children)}}


def assert_recursed(events):
    """Fail loudly when a runtime never recursed, instead of passing vacuously."""
    assert any(e.get("type") == "sub_request" for e in events), (
        "no sub_request in the stream — this runtime did not recurse, so the "
        "property under test was never exercised"
    )


def outcomes_by_id(events):
    """Map envelope id -> list of its outcome event types."""
    seen = {}
    for e in events:
        if e.get("type") in ("completed", "error"):
            seen.setdefault(e.get("id"), []).append(e["type"])
    return seen


def test_child_events_are_inlined_into_the_parent_stream():
    rc, events = dispatch(compose(
        {"v": "happi/1.3", "id": "kid", "cmd": "version"}))
    assert rc == 0
    types = [(e["id"], e["type"]) for e in events]
    assert types == [
        ("parent", "started"),
        ("parent", "sub_request"),
        ("kid", "started"),
        ("kid", "delta"),
        ("kid", "completed"),
        ("parent", "completed"),
    ]


def test_sub_request_carries_the_child_under_the_spec_field_name():
    """The event table in happi.md names the payload field `envelope`."""
    _, events = dispatch(compose(
        {"v": "happi/1.3", "id": "kid", "cmd": "version"}))
    marker = next(e for e in events if e["type"] == "sub_request")
    assert "envelope" in marker, f"sub_request fields: {sorted(marker)}"
    assert marker["envelope"]["id"] == "kid"
    assert marker["envelope"]["cmd"] == "version"


def test_every_envelope_id_terminates_exactly_once():
    """A recursive stream holds many outcomes; each ID still ends exactly once.

    The `assert_recursed` guard is load-bearing. Without it this test passes on
    a runtime that has no `compose` at all: that runtime answers the envelope
    with a single `unsupported_cmd` error, which trivially satisfies
    one-outcome-per-id while testing nothing. A check that passes BECAUSE the
    feature is missing reads as assurance and is worse than no check.
    """
    _, events = dispatch(compose(
        {"v": "happi/1.3", "id": "a", "cmd": "version"},
        {"v": "happi/1.3", "id": "b", "cmd": "echo", "args": ["x"]},
    ))
    assert_recursed(events)
    for env_id, got in outcomes_by_id(events).items():
        assert len(got) == 1, f"{env_id} terminated {len(got)} times: {got}"
    assert {"parent", "a", "b"} <= set(outcomes_by_id(events))


def test_child_reports_the_runtime_version_not_its_own():
    """Back-compat holds THROUGH recursion: a v1.0 child is stamped with the
    RUNTIME version, whatever that version currently is.

    Asserted as a PROPERTY, not a version literal. The literal form
    (`== {"happi/1.3"}`) pinned a spelling rather than the behaviour this test
    names: it went red on the v1.4 bump while the behaviour it describes was
    still correct, and it would re-stale on every future bump. The property
    below still kills the real mutant — a runtime echoing each envelope's own
    version would stamp the child `happi/1.0`, which both differs from the
    parent's stamp and trips the second assertion directly.
    """
    _, events = dispatch(compose(
        {"v": "happi/1.0", "id": "old", "cmd": "echo", "args": ["hi"]}))
    child = [e for e in events if e["id"] == "old"]
    parent = [e for e in events if e["id"] == "parent"]
    assert child, "child produced no events"
    assert parent, "parent produced no events"
    child_versions = {e["v"] for e in child}
    assert child_versions == {e["v"] for e in parent}, (
        "child events must carry the same runtime stamp as the parent's")
    assert child_versions != {"happi/1.0"}, (
        "child was stamped with its OWN envelope version, not the runtime's")


def test_child_with_unknown_cmd_gets_the_same_error_contract():
    """Proof the child reaches the shared handler table, not a private one."""
    rc, events = dispatch(compose(
        {"v": "happi/1.3", "id": "u", "cmd": "definitely-not-a-cmd"}))
    child_err = next(e for e in events if e["id"] == "u" and e["type"] == "error")
    assert child_err["code"] == "unsupported_cmd"
    assert rc != 0
    # The parent still completes — a child failure is the child's outcome.
    assert outcomes_by_id(events)["parent"] == ["completed"]


def test_child_with_bad_version_fails_exactly_as_a_top_level_envelope_does():
    """Proof the child reaches the shared validator, not a relaxed one."""
    bad = {"v": "happi/9.9", "id": "bad", "cmd": "version"}
    top_rc, top_events = dispatch(bad)
    top_err = next(e for e in top_events if e["type"] == "error")

    child_rc, child_events = dispatch(compose(bad))
    child_err = next(e for e in child_events if e["type"] == "error")

    assert top_err["code"] == child_err["code"] == "parse_error"
    assert "unsupported version" in top_err["message"]
    assert "unsupported version" in child_err["message"]
    assert top_rc != 0 and child_rc != 0


def test_malformed_child_terminates_the_parent_once():
    """An unparseable child is a defect in the parent request: one outcome.

    Guarded against the same false pass as the test above: a runtime with no
    `compose` also answers with exactly one parent error, so the error CODE is
    asserted too. `unsupported_cmd` means the feature is absent, not that the
    property holds.
    """
    _, events = dispatch(compose({"v": "happi/9.9", "id": "bad", "cmd": "version"}))
    assert outcomes_by_id(events)["parent"] == ["error"]
    err = next(e for e in events if e["type"] == "error")
    assert err["code"] != "unsupported_cmd", "runtime has no compose; nothing was tested"
    assert err["code"] == "parse_error"


def test_nested_compose_recurses():
    """A compose inside a compose — the fractal claim at depth 2."""
    rc, events = dispatch(compose(
        compose({"v": "happi/1.3", "id": "g2", "cmd": "version"},
                envelope_id="g1"),
        envelope_id="g0"))
    assert rc == 0
    ids = [e["id"] for e in events]
    assert ids.index("g0") < ids.index("g1") < ids.index("g2")
    assert len([e for e in events if e["type"] == "sub_request"]) == 2
    for env_id in ("g0", "g1", "g2"):
        assert outcomes_by_id(events)[env_id] == ["completed"]


def test_recursion_is_bounded_by_a_depth_cap():
    """Unbounded fractal recursion is a hang; the cap makes it an error."""
    env = {"v": "happi/1.3", "id": "d10", "cmd": "version"}
    for i in range(9, -1, -1):
        env = compose(env, envelope_id=f"d{i}")
    rc, events = dispatch(env)
    caps = [e for e in events if e.get("code") == "sub_request_depth_cap"]
    assert caps, "deep recursion was not capped"
    assert rc != 0


def test_envelopes_flag_must_be_a_list():
    rc, events = dispatch(
        {"v": "happi/1.3", "id": "p", "cmd": "compose",
         "flags": {"envelopes": "not-a-list"}})
    err = next(e for e in events if e["type"] == "error")
    assert err["code"] == "parse_error"
    assert rc != 0
    assert outcomes_by_id(events)["p"] == ["error"]


def test_compose_is_advertised_in_the_runtime_self_description():
    """spec.describe must not misreport the runtime's own capabilities."""
    _, events = dispatch({"v": "happi/1.3", "id": "s", "cmd": "spec.describe"})
    text = " ".join(e.get("text", "") for e in events if e["type"] == "delta")
    assert "compose" in text
