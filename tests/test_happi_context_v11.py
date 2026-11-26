"""HAPPI v1.1 context / memory-chain terminator tests (P1).

Covers the ``context`` event — the structural twin of ``idr``: the
``context.append`` cmd and the ``flags.context=true`` terminator. The
load-bearing invariant is the CONTENT ADDRESS: ``context_ref.sha256`` =
``"sha256:" + sha256`` of canonical JSON (sorted keys, ``ensure_ascii=False``)
EXCLUDING the volatile ``id``/``ts``/``audit`` fields — byte-identical to GRIP
``lib/precog/idr.py::content_addr`` for the same body.

Goodhart-proof — asserts content (event types, sha256 equality, exclusion
semantics), never call counts. A mutation of ``_content_addr`` or the
terminator wiring MUST fail at least one test. Tests drive the canonical
polyglot (``bash happi.md run``), mirroring ``test_happi_audit_v11.py``'s
polyglot-parity pattern.

Design: GRIP ``drafts/happi-context-event-memory-chain-design.md`` §3 (P1).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

def _resolve_polyglot() -> Path:
    """Locate the happi.md under test, preferring THIS checkout.

    Previously this was hardcoded to ``Path.home() / ".hal" / "happi.md"``, which
    made the suite test whatever happened to be in the developer's home directory
    rather than the repository it ships in. That is not a theoretical concern: it
    meant a checkout containing a FIXED runtime could still be graded against an
    UNFIXED one sitting in ~/.hal, so the tests passed while the change under test
    was never executed. Resolution order:

      1. $HAPPI_MD          -- explicit override, and what CI should set
      2. <repo>/happi.md    -- the file this test file ships alongside
      3. ~/.hal/happi.md    -- legacy fallback, kept so existing setups still run
    """
    from os import environ
    override = environ.get("HAPPI_MD")
    if override:
        return Path(override)
    in_repo = Path(__file__).resolve().parent.parent / "happi.md"
    if in_repo.exists():
        return in_repo
    return Path.home() / ".hal" / "happi.md"


POLYGLOT = _resolve_polyglot()

pytestmark = pytest.mark.skipif(
    not POLYGLOT.exists(),
    reason=f"happi.md not found (looked at {POLYGLOT}); set $HAPPI_MD to override",
)

_EXCLUDE = ("id", "ts", "audit")


# ---------- helpers ------------------------------------------------------

def _run(env: dict) -> list[dict]:
    """Dispatch an envelope through the canonical polyglot runtime."""
    proc = subprocess.run(
        ["bash", str(POLYGLOT), "run"],
        input=json.dumps(env), capture_output=True, text=True, timeout=20,
    )
    return [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]


def _canonical_content_addr(body: dict) -> str:
    """Reference content address — the documented canonical algorithm, identical
    to GRIP ``lib/precog/idr.py::content_addr``. Reimplemented here so the
    cross-runtime invariant is pinned WITHOUT a cross-repo import (HAL CI stays
    self-contained). A divergence in happi.md's ``_content_addr`` fails the
    ``test_content_addr_matches_canonical_algorithm`` assertion below."""
    addressed = {k: v for k, v in body.items() if k not in _EXCLUDE}
    canonical = json.dumps(addressed, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _append(body: dict, **flags) -> list[dict]:
    return _run({"v": "happi/1.1", "id": "ctx-test", "cmd": "context.append",
                 "args": [json.dumps(body)], "flags": flags})


def _context_event(events: list[dict]) -> dict:
    ctxs = [e for e in events if e["type"] == "context"]
    assert len(ctxs) == 1, (
        f"expected exactly one context event; got {[e['type'] for e in events]}"
    )
    return ctxs[0]


# ---------- context.append: emission + shape -----------------------------

def test_context_append_emits_context_event():
    types = [e["type"] for e in _append({"kind": "context-delta", "decision": {"learned": "X"}})]
    assert types == ["started", "completed", "context"], types


def test_context_event_follows_the_outcome():
    """context is a TERMINATOR: it must arrive after completed/error.

    The spec defines a context event as "a non-streaming terminator emitted after
    completed/error", and its falsification clause (b) names the reverse ordering
    -- "the context event arrives before completed/error" -- as a condition
    meaning the design is wrong. The runtime emitted it FIRST, in violation.

    This is the regression anchor for that fix. It deliberately mirrors
    ``test_audit_terminator_independent_of_context_flag`` below, which pins the
    twin case (idr after completed) the same way -- the codebase already knew the
    correct shape for one terminator while getting the other backwards.

    An index comparison rather than a whole-list equality, so it keeps holding if
    additional event types are ever introduced between started and the outcome.
    """
    types = [e["type"] for e in _append({"kind": "context-delta", "decision": {"learned": "X"}})]
    assert "context" in types and "completed" in types, types
    assert types.index("context") > types.index("completed"), types


def test_context_ref_has_required_fields():
    ev = _context_event(_append({"kind": "context-delta", "decision": {"a": 1}},
                                 predecessor_context="precog-0", model_versions=["m1"]))
    ref = ev["context_ref"]
    assert isinstance(ref["sha256"], str)
    assert ref["sha256"].startswith("sha256:") and len(ref["sha256"]) == 7 + 64
    assert ref["predecessor_context"] == "precog-0"
    assert ref["snapshot_ref"] is None
    assert ref["kind"] == "context-delta"
    assert ref["model_versions"] == ["m1"]


def test_kind_defaults_to_context_delta():
    ev = _context_event(_append({"decision": {"a": 1}}))
    assert ev["context_ref"]["kind"] == "context-delta"


def test_snapshot_ref_plumbs_through():
    ev = _context_event(_append({"decision": {"a": 1}}, snapshot_ref="sha256:abc",
                                 kind="context-snapshot"))
    assert ev["context_ref"]["snapshot_ref"] == "sha256:abc"
    assert ev["context_ref"]["kind"] == "context-snapshot"


# ---------- THE content-address invariant --------------------------------

def test_content_addr_matches_canonical_algorithm():
    """context_ref.sha256 == the documented canonical content address
    (= GRIP content_addr). A mutation of _content_addr fails here."""
    body = {"kind": "context-delta", "decision": {"learned": "X"},
            "predecessor_idr": "precog-0-zzzz"}
    ev = _context_event(_append(body))
    assert ev["context_ref"]["sha256"] == _canonical_content_addr(body)


def test_id_ts_audit_excluded_from_address():
    """THE Goodhart anchor: a body differing ONLY in id/ts/audit yields the SAME
    content address. Mutation killed: include id/ts/audit in the address."""
    base = {"kind": "context-delta", "decision": {"learned": "X"}}
    a = _context_event(_append({**base, "id": "p-1", "ts": "2026-01-01T00:00:00Z",
                                "audit": {"signature": "sig-a"}}))
    b = _context_event(_append({**base, "id": "p-999", "ts": "2099-12-31T23:59:59Z",
                                "audit": {"signature": "sig-b"}}))
    assert a["context_ref"]["sha256"] == b["context_ref"]["sha256"]


def test_semantic_change_flips_address():
    """Not constant: a changed semantic field flips the address."""
    a = _context_event(_append({"kind": "context-delta", "decision": {"learned": "X"}}))
    b = _context_event(_append({"kind": "context-delta", "decision": {"learned": "Y"}}))
    assert a["context_ref"]["sha256"] != b["context_ref"]["sha256"]


def test_unknown_top_level_key_is_addressed():
    """A key NOT in (id,ts,audit) is part of the address — adding one flips it.
    Fail-safe by construction: new semantic fields are addressed by default,
    never silently dropped."""
    a = _context_event(_append({"kind": "context-delta", "decision": {"a": 1}}))
    b = _context_event(_append({"kind": "context-delta", "decision": {"a": 1}, "extra": "v"}))
    assert a["context_ref"]["sha256"] != b["context_ref"]["sha256"]


# ---------- flags.context is NOT a stdlib auto-terminator (HAL #429) ------

def test_flags_context_does_not_auto_emit_in_stdlib():
    """The stdlib reference runtime does NOT auto-emit `context` on flags.context:
    a generic dispatch has no designated decision body, and addressing the whole
    envelope would fold dispatch flags into the content address (HAL #429). Only
    `context.append` emits context here. Mutation killed: re-add the flags.context
    auto-path."""
    types = [e["type"] for e in _run({"v": "happi/1.1", "id": "fc-1", "cmd": "echo",
                                      "args": ["hi"], "flags": {"context": True}})]
    assert "context" not in types
    assert types == ["started", "delta", "completed"]


def test_audit_terminator_independent_of_context_flag():
    """flags.audit still emits its idr terminator; a co-present flags.context is
    inert (no context auto-emit). Confirms the two opt-ins do not entangle."""
    types = [e["type"] for e in _run({"v": "happi/1.1", "id": "both-1", "cmd": "echo",
                                      "args": ["x"], "flags": {"audit": True, "context": True}})]
    assert types[-1] == "idr"
    assert "context" not in types
    assert types.index("idr") > types.index("completed")


# ---------- backward / forward compat ------------------------------------

def test_v10_dispatch_unchanged_no_context():
    """v1.0 envelope dispatches identically (started→delta→completed); no context."""
    types = [e["type"] for e in _run({"v": "happi/1.0", "id": "bc-1", "cmd": "echo",
                                      "args": ["a"]})]
    assert types == ["started", "delta", "completed"]


def test_context_is_opt_in():
    """No flags.context, no context.append → zero behaviour change (no context)."""
    types = [e["type"] for e in _run({"v": "happi/1.1", "id": "opt-1", "cmd": "echo",
                                      "args": ["a"]})]
    assert "context" not in types


# ---------- error handling -----------------------------------------------

def test_context_append_missing_args_errors():
    events = _run({"v": "happi/1.1", "id": "err-1", "cmd": "context.append"})
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "parse_error"


def test_context_append_non_object_body_errors():
    events = _run({"v": "happi/1.1", "id": "err-2", "cmd": "context.append",
                   "args": [json.dumps([1, 2, 3])]})
    assert any(e["type"] == "error" and e["code"] == "parse_error" for e in events)


# ---------- self-bootstrap conformance -----------------------------------

def test_spec_describe_mentions_context():
    events = _run({"v": "happi/1.1", "id": "spec-1", "cmd": "spec.describe"})
    text = " ".join(e.get("text", "") for e in events)
    assert "context" in text.lower()


# ---------- cross-repo invariant (skipped if GRIP absent) ----------------

_GRIP_IDR = Path.home() / ".claude" / "lib" / "precog" / "idr.py"


@pytest.mark.skipif(not _GRIP_IDR.exists(), reason="GRIP lib/precog/idr.py not present")
def test_cross_runtime_matches_grip_content_addr():
    """happi.md context_ref.sha256 byte-equals GRIP content_addr for the same body.
    The cross-repo invariant the memory-chain depends on (design §3 falsifier c).

    Loads GRIP idr.py by file PATH (not ``import lib.precog.idr``) to dodge the
    two-repo ``lib/`` package-name collision: in a pytest session that already
    imported ``lib.hal.*``, ``sys.modules['lib']`` is HAL's lib, so the dotted
    import would resolve to the wrong package.

    Loading by path is necessary and was NOT sufficient. GRIP's idr.py opens with

        try:     from _compat import fcntl
        except:  from lib._compat import fcntl

    and under a bare exec BOTH branches fail — the first because GRIP's ``lib/``
    is not on ``sys.path``, the second because ``lib`` is HAL's package, which
    has no ``_compat``. The module then raises ModuleNotFoundError before the
    comparison below is ever reached.

    In the full suite it passed anyway, because some earlier test left the name
    resolvable — so this invariant was only ever verified as a SIDE EFFECT OF
    TEST ORDERING, and running this file alone errored out. A load-bearing
    cross-repo check that cannot be run on its own is not a check you can trust;
    it is one that happens to be green.

    The fix puts GRIP's ``lib/`` on ``sys.path`` for the duration of the exec
    ONLY, so the FIRST branch succeeds and the dotted fallback is never reached.
    The removal in ``finally`` is load-bearing, not tidiness: GRIP's ``lib/``
    contains top-level modules that also exist in HAL's (``agora_pr_review.py``
    among them), so leaving it on the path would shadow HAL's own imports for
    the rest of the session — trading this collision for a worse one."""
    import importlib.util
    import sys

    grip_lib_path = _GRIP_IDR.parent.parent  # ~/.claude/lib
    grip_lib = str(grip_lib_path)

    # Fail with a clear message if GRIP's layout is not what the insertion
    # assumes. Without this an unexpected checkout inserts a useless path, both
    # import branches fail as before, and the test errors with a
    # ModuleNotFoundError that says nothing about the layout being the cause.
    assert (grip_lib_path / "_compat.py").exists(), (
        f"expected GRIP's _compat.py at {grip_lib_path}; the sys.path insertion "
        "below only helps if _compat.py sits directly in GRIP's lib/"
    )

    spec = importlib.util.spec_from_file_location("grip_precog_idr", _GRIP_IDR)
    # spec_from_file_location returns None for an unreadable or unloadable file.
    # The skipif above only checks existence, so a file that exists but cannot be
    # loaded would reach here and fail as `AttributeError: 'NoneType' has no
    # attribute 'loader'` — an error that names Python's internals rather than
    # the file that could not be read. (AGORA HAL#586, Speaker C concern 4.)
    assert spec is not None and spec.loader is not None, (
        f"could not build an import spec for {_GRIP_IDR} — the file exists but is "
        "not loadable as Python"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # @dataclass needs cls.__module__ resolvable

    # BOTH sys.path AND sys.modules are restored. Restoring only sys.path moves
    # the shadowing rather than removing it: exec_module runs
    # `from _compat import fcntl`, which caches sys.modules["_compat"] = GRIP's
    # module, so any later `import _compat` in this session silently gets GRIP's
    # — the same cross-package collision, one layer down. Restoring by INDEX
    # rather than by value, because exec_module could itself append the same
    # string and a value-remove would then delete an entry this test did not add.
    _sentinel = object()
    _prior_compat = sys.modules.get("_compat", _sentinel)
    sys.path.insert(0, grip_lib)
    try:
        spec.loader.exec_module(mod)
        # Prove the import resolved through the path just inserted, not through a
        # module some earlier test left cached. Without this the fix closes
        # sys.path contamination while leaving the ORDERING DEPENDENCE it exists
        # to remove alive in sys.modules — the check would still be able to pass
        # for an accidental reason.
        resolved = getattr(sys.modules.get("_compat"), "__file__", "")
        assert resolved and Path(resolved).resolve() == (
            grip_lib_path / "_compat.py"
        ).resolve(), (
            f"_compat resolved to {resolved!r}, not GRIP's — the invariant would "
            "be running against a module left cached by an earlier test"
        )
    finally:
        if sys.path and sys.path[0] == grip_lib:
            del sys.path[0]
        elif grip_lib in sys.path:
            sys.path.remove(grip_lib)
        if _prior_compat is _sentinel:
            sys.modules.pop("_compat", None)
        else:
            sys.modules["_compat"] = _prior_compat

    body = {"kind": "context-delta", "decision": {"learned": "X"},
            "predecessor_idr": "precog-0-zzzz", "fingerprint": "f" * 64}
    ev = _context_event(_append(body))
    assert ev["context_ref"]["sha256"] == mod.content_addr(body)


@pytest.mark.skipif(not _GRIP_IDR.exists(), reason="GRIP lib/precog/idr.py not present")
def test_cross_runtime_load_leaves_no_module_cache_residue():
    """The regression anchor for AGORA HAL#586's blocker.

    Loading GRIP's idr.py runs `from _compat import fcntl`, which caches
    sys.modules["_compat"] = GRIP's module. An earlier version of the loader
    restored sys.path and not sys.modules, so the cross-package shadowing it
    claimed to prevent simply moved one layer down: any later `import _compat`
    in the session would have received GRIP's module instead of HAL's.

    Asserted from OUTSIDE the loader, because a function cannot check its own
    cleanup — the state has to be observed after it returns.

    The cache is CLEARED before the call, and that is load-bearing rather than
    tidiness. A first version of this anchor simply captured
    sys.modules["_compat"] before and after and compared them. It could never
    fail: the test above runs first in file order, so when the restore was
    removed it leaked GRIP's module, and this anchor then captured the
    ALREADY-LEAKED value as its own baseline and found it unchanged. An
    order-dependent test guarding against order dependence — confirmed by
    mutation: deleting the restore left all tests green. Clearing first makes
    the starting state known instead of inherited.
    """
    import sys

    saved = sys.modules.pop("_compat", None)
    try:
        test_cross_runtime_matches_grip_content_addr()
        assert "_compat" not in sys.modules, (
            "loading GRIP's idr.py left sys.modules['_compat'] behind; a later "
            "`import _compat` in this session would get the wrong repo's module"
        )
    finally:
        if saved is not None:
            sys.modules["_compat"] = saved
        else:
            sys.modules.pop("_compat", None)
