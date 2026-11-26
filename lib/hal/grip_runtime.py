"""GRIP's first-class default LLM runtime -- happi.md as the shim (W4).

This is THE entrypoint that makes "GRIP runs through HAL" real. One call
builds the GRIP-wired RAILLM sentinel (W1 subscription bridge + W2
CCH->Kimi->cheap->local ladder), drives a single happi ``infer`` envelope
through the REAL ``lib.hal.happi.runtime`` (W3), and folds the NDJSON event
stream into a normalised result. happi.md is the syscall surface; RAILLM is
the InfiniteMachine behind it; the caller never imports a provider SDK.

GRIP's canonical adapter (``~/.claude/lib/hal_llm_adapter.py``) is to be
rewired to call ``grip_infer`` in a SEPARATE cross-repo follow-up on
CodeTonight-SA/GRIP -- never from this HAL session (placement boundary,
plan section "Cross-repo placement boundary").

Why a facade and not "just call the sentinel": the syscall doctrine. The
default GRIP LLM path must be an envelope-in / events-out boundary so it is
provider-agnostic, audit-ready (``audit=True`` -> deterministic ``idr``),
and hot-swappable. ``grip_infer`` IS that boundary for GRIP.

H-RAILLM-4: a thin envelope facade over the W1+W2+W3 substrate yields a
context-window-decoupled default GRIP runtime with ZERO core changes.
Falsified if this module must modify ``RAILLMSentinel`` / the registry /
the happi runtime to function -- it does not; it only composes them.
"""
from __future__ import annotations

import io
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from lib.hal.media import MEDIA_PART_TYPES

from lib.hal.happi import runtime
from lib.hal.raillm_ladder import GRIP_LADDER_ADDRESSES, ladder_addresses


@dataclass
class GripInferResult:
    """Normalised result of one GRIP-via-HAL inference.

    ``ok`` is the single success predicate. ``text`` is "" on failure
    (never None) so a caller's ``.text`` access is total. ``failover`` is
    non-None only when a provider other than the one requested answered.
    """

    ok: bool
    text: str
    provider: str | None
    model: str | None
    requested: str
    failover: dict[str, str] | None = None
    idr_ref: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    # True only when the turn was dispatched WITH real non-text parts
    # (image/audio) — set at resolution time from the parts actually built.
    media_attached: bool = False
    # Token counts from the HAPPI completed event, under its own names "in" and
    # "out" (Gemini's "out" already includes its thinking tokens, HAL#619).
    # Empty when the provider reported none: "not reported", never "zero".
    usage: dict[str, int] = field(default_factory=dict)
    # The provider's finish_reason (OpenAI "length"/"stop"/"tool_calls",
    # Anthropic "max_tokens"/"end_turn"/...), carried verbatim from the
    # completed event so a caller can tell WHY a turn ended. Empty text with
    # "length"/"max_tokens" is the answer-budget-exhausted signature HAL#648
    # surfaces as ok=false rather than a silent empty success.
    finish_reason: str | None = None


def _token_counts(usage: dict[str, Any]) -> dict[str, int]:
    """The completed event's ``in`` and ``out`` counts, when they are real numbers.

    The same block carries provider, model and failover beside the counts; only
    the counts are kept. A boolean, negative, infinite or non-numeric value is
    dropped rather than turned into a number the provider never reported.
    """
    return {key: int(value) for key, value in usage.items()
            if key in ("in", "out") and isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value) and value >= 0}


def _build_envelope(prompt: Any, model: str, audit: bool,
                    content_parts: list[dict] | None = None,
                    pin: bool = False,
                    max_tokens: int | None = None) -> dict[str, Any]:
    """One happi ``infer`` envelope. ``id`` is a fresh uuid per call.

    ``prompt`` is passed through untouched -- envelope-level validation
    (non-empty string) is the runtime's single source of truth, not
    re-implemented here (DRY: the W3 ``dispatch_infer`` already emits
    ``parse_error`` for a missing/!str prompt WHEN no multimodal content is
    present).

    ``content_parts`` (optional) is a list of OpenAI-compatible ContentPart
    dicts (text + image_url + input_audio). When present it is carried in
    ``flags.content_parts`` and ``dispatch_infer`` routes the turn through the
    multimodal messages path instead of the string thread. ABSENT by default
    -> the envelope is byte-identical to the text-only contract.

    ``pin`` (optional) carries ``flags.pin`` so ``dispatch_infer`` refuses a
    silent cross-provider failover for an explicitly-pinned provider (the
    exogenous-council guarantee). Default False -> flag absent -> byte-identical
    failover behaviour.
    """
    flags: dict[str, Any] = {
        "prompt": prompt, "model": model, "audit": bool(audit),
    }
    if content_parts:
        flags["content_parts"] = content_parts
    if pin:
        flags["pin"] = True
    if max_tokens is not None:
        flags["max_tokens"] = max_tokens
    return {
        "v": "happi/1.1",
        "id": uuid.uuid4().hex,
        "cmd": "infer",
        "flags": flags,
    }


def _resolve_content_parts(
    prompt: str,
    media: list[str] | None,
    content_parts: list[dict] | None,
) -> list[dict] | None:
    """Resolve the multimodal ContentPart list for this turn, or None.

    Pre-built ``content_parts`` win (the caller already encoded). Else
    ``media`` file paths are encoded via the HAL media pipeline
    (``build_multimodal_content`` reads each path, base64-encodes it, and
    appends the text ``prompt`` as the trailing part). Returns None when
    neither is supplied OR when media was supplied but nothing could be
    loaded -- the caller then takes the byte-identical text-only path.

    The text prompt is embedded INSIDE the returned parts (as the trailing
    text part), so the multimodal dispatch carries the full user turn.
    """
    if content_parts:
        return list(content_parts)
    if media:
        from lib.hal.media import build_multimodal_content
        parts = build_multimodal_content(prompt, list(media))
        return parts or None
    return None


def _drive(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    """Run ONE envelope through the real happi runtime; return its events.

    ``runtime.run`` is the canonical happi entrypoint (stdin stream ->
    stdout NDJSON, int rc). We feed a single-envelope stream and parse the
    event lines back -- exactly the contract the W3 tests pin. The runtime
    is NOT mocked here: this facade is an integration boundary, not a stub.
    """
    out = io.StringIO()
    runtime.run(io.StringIO(json.dumps(envelope)), out)
    return [json.loads(ln) for ln in out.getvalue().splitlines() if ln.strip()]


def _has_non_text_parts(parts: list[dict] | None) -> bool:
    """True iff the parts carry media — the single shared type set from
    media.py, so a new provider part type updates every attestation site."""
    if not parts:
        return False
    from lib.hal.media import MEDIA_PART_TYPES
    return any(p.get("type") in MEDIA_PART_TYPES
               for p in parts if isinstance(p, dict))


def _first_event(events: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    """The first event of ``kind`` in the stream, or None."""
    return next((e for e in events if e.get("type") == kind), None)


def _collect(events: list[dict[str, Any]], requested: str,
             media_attached: bool = False) -> GripInferResult:
    """Fold the happi event stream into a normalised result (pure, no I/O)."""
    err = _first_event(events, "error")
    if err is not None:
        return GripInferResult(
            ok=False, text="", provider=None, model=None,
            requested=requested, error_code=err.get("code"),
            error_message=err.get("message"), events=events,
            # Preserve the caller's computed truth: attachment happened at
            # dispatch time; a provider error later does not un-attach it.
            media_attached=bool(media_attached),
        )
    text = "".join(
        e.get("text", "") for e in events if e.get("type") == "delta"
    )
    completed = _first_event(events, "completed")
    usage = (completed or {}).get("usage", {}) or {}
    finish_reason = (completed or {}).get("finish_reason")
    idr = _first_event(events, "idr")
    result = GripInferResult(
        ok=True, text=text,
        provider=usage.get("provider"), model=usage.get("model"),
        requested=requested, failover=usage.get("failover"),
        idr_ref=(idr or {}).get("idr_ref"), events=events,
        media_attached=bool(media_attached), usage=_token_counts(usage),
        finish_reason=finish_reason,
    )
    if not text and finish_reason in ("length", "max_tokens"):
        # HAL#648: the model spent its whole answer budget on reasoning and
        # never began the answer. Report it as a failure — ok=true with empty
        # text is a silent success that reads as "the model said nothing".
        # Derived from the success result, so an audited or failed-over call
        # keeps its idr_ref and failover.
        return replace(
            result, ok=False, error_code="answer_budget_exhausted",
            error_message=(
                f"model stopped with finish_reason={finish_reason} before "
                "producing any answer text — the answer budget ran out on "
                "reasoning; pass a larger max_tokens"
            ),
        )
    return result


def _grip_active_llm():
    """Cached cross-repo load of GRIP's ``lib/active_llm.py``.

    Copy-adapted from ``lib/hal/server.py:_grip_active_llm`` (this is the
    2nd instance — INTENTIONAL duplication, noted per syscall-doctrine:
    copy-adapt at N=2, extract a shared shim only at N=3). A normal
    ``from lib import active_llm`` cannot work: both ~/.hal and ~/.claude
    ship a ``lib/__init__.py``, so the import system resolves to ~/.hal's
    package and shadows GRIP's module entirely. Direct file load bypasses
    that. Returns None outside GRIP (HAL standalone) — caller fails safe.
    """
    cache = getattr(_grip_active_llm, "_cache", "unset")
    if cache != "unset":
        return cache
    module = None
    try:
        import importlib.util
        grip_home = os.environ.get("CLAUDE_DIR", os.path.expanduser("~/.claude"))
        path = os.path.join(grip_home, "lib", "active_llm.py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("grip_active_llm", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    except Exception:
        module = None
    _grip_active_llm._cache = module
    return module


def _usable_marker(rec: object) -> dict | None:
    """The 4-gate accept policy: dict · mode in {solo,solo-once} · address
    is on the ladder this process would use (``ladder_addresses()``, so the
    Astra rung counts only while paid OpenAI use is opted in) · not past
    ts+ttl. Single source of truth so the session path and the cross-session
    fallback apply identical rules (DRY; the fail-safe stays one policy, not
    scattered — H-LLMSW b)."""
    if not isinstance(rec, dict):
        return None
    if rec.get("mode") not in ("solo", "solo-once"):
        return None
    if rec.get("address") not in ladder_addresses():
        return None
    if rec.get("ts", 0) + rec.get("ttl_seconds", 0) < time.time():
        return None  # stale -> treat as absent
    return rec


def _scan_session_markers() -> dict | None:
    """Cross-session fallback. The long-lived hal-server daemon has no
    ``CLAUDE_CODE_SESSION_ID``/``GRIP_CC_PID``, so ``active_llm.show_active``
    keys to the stale GLOBAL file and never sees a CC session's ``/llm
    pin`` (root cause: ``_resolve_cc_pid`` session-keys the marker; a
    cross-session consumer cannot resolve it). Resolve operator intent by
    the MOST-RECENTLY-MODIFIED valid per-session marker in TMPDIR (same
    TMPDIR precedence as ``active_llm._state_file``). Any failure -> None
    (fixed-ladder primary; byte-identical fail-safe). Trade-off: two
    sessions pinning different models -> daemon follows the last write —
    acceptable for the single-operator model, strictly better than today
    (the daemon honoured NO pin at all)."""
    try:
        import glob as _glob
        import json as _json
        tmp = os.environ.get("GRIP_TMP_DIR",
                              os.environ.get("TMPDIR", "/tmp"))
        files = _glob.glob(os.path.join(
            tmp, "grip-session-active-llm-*.json"))
        # Stat each candidate defensively. A sibling /tmp marker can
        # vanish or become unreadable between glob and stat (a real
        # TOCTOU in a long-lived daemon scanning a shared tmpdir). A
        # single bad neighbour MUST NOT abort the whole scan and drop a
        # *valid* pin in another file. Skip the unstattable one; keep
        # scanning. (HAL-Council PR #360 round-2 instinct: the stated
        # "glob.glob unhandled" reasoning is false — the outer except
        # already fail-safes propagation — but the underlying
        # robustness gap it points at is real and fixed here.)
        stamped: list[tuple[float, str]] = []
        for path in files:
            try:
                stamped.append((os.stat(path).st_mtime, path))
            except Exception:
                continue  # vanished / permission / race -> skip, not abort
        for _mtime, path in sorted(stamped, reverse=True):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    rec = _usable_marker(_json.load(fh))
                if rec is not None:
                    return rec
            except Exception:
                continue
        return None
    except Exception:
        return None


def _active_marker() -> dict | None:
    """The GRIP active-llm marker IFF it is a usable per-turn pin, else
    None. Primary: the session-scoped ``show_active()`` (in-session =
    byte-identical to pre-fix).

    Cross-session fallback is gated on the ROOT-CAUSE AXIS —
    *session-resolvability* — NOT on marker usability. Only a consumer
    that cannot resolve its own session marker
    (``mod._resolve_cc_pid() is None`` -> the GLOBAL fallback, i.e. the
    nohup/pm2 hal-server daemon — the exact bug axis) scans sibling
    markers. An IN-SESSION process whose ``show_active()`` returns a
    present-but-non-pinning marker (e.g. ``round-robin``/council) made a
    DELIBERATE no-pin choice: honour it as fixed-ladder (return None) —
    never override it with a *different* session's pin. This restores
    the original pre-#360 in-session semantic the HAL-Council flagged on
    PR #360 (race-label is a mislabel — it is a deterministic semantic
    change, not a data race) WHILE preserving the daemon fix. Reuse
    GRIP's own ``_resolve_cc_pid`` (never re-proxy it — DRY); its absence
    /error degrades to ``_scan_session_markers`` (no worse than the
    interim PR state, never a crash). ANY failure -> None (single
    fail-safe branch, H-LLMSW b)."""
    try:
        mod = _grip_active_llm()
        if mod is not None:
            rec = _usable_marker(mod.show_active())
            if rec is not None:
                return rec
            try:
                if mod._resolve_cc_pid() is not None:
                    return None  # in-session: show_active() authoritative
            except Exception:
                pass             # resolver missing/error -> fail to scan
        return _scan_session_markers()
    except Exception:
        return None


def _ladder_addresses() -> tuple[str, ...]:
    """The ladder addresses grip_infer reads its fail-safe primary from.

    Default (``HAL_LEARNED_LATTICE`` unset / != "1"): the fixed
    ``GRIP_LADDER_ADDRESSES`` tuple, returned UNCHANGED — same object, no new
    import, byte-identical to pre-#420 (the load-bearing flag-off invariant).

    Flag on (``HAL_LEARNED_LATTICE == "1"``): the addresses reordered by the
    learned capability lattice (HAL #420). The import is guarded inside the
    flag-on branch so the flag-off executed path never pulls in
    ``learned_lattice`` and carries zero new import cost."""
    if os.environ.get("HAL_LEARNED_LATTICE") != "1":
        return GRIP_LADDER_ADDRESSES
    from lib.hal.learned_lattice import learned_ladder_order  # flag-on only
    return tuple(learned_ladder_order(GRIP_LADDER_ADDRESSES))


def _resolve_primary(model: str | None) -> str:
    """The per-turn primary ladder address. Explicit ``model`` always
    wins; else a fresh in-ladder solo/solo-once marker pins its address
    for THIS turn; else the fixed ladder primary (the non-opted /
    fail-safe path, byte-identical to pre-W2)."""
    if model:
        return model
    rec = _active_marker()
    return rec["address"] if rec else _ladder_addresses()[0]


def grip_infer(prompt: str, *, model: str | None = None,
                audit: bool = False,
                media: list[str] | None = None,
                content_parts: list[dict] | None = None,
                no_failover: bool = False,
                max_tokens: int | None = None) -> GripInferResult:
    """THE default GRIP-via-HAL LLM call. happi envelope in, result out.

    Coordinator: resolve content -> build envelope -> drive the real runtime
    -> fold events. ``model`` defaults to the per-turn primary: an explicit
    ``model`` arg, else a fresh GRIP active-llm solo/solo-once pin (P3 —
    "switch subscription/LLM" == "make rung X the primary"), else the fixed
    ladder primary (the CCH-Anthropic rung). The RAILLM InfiniteMachine
    cascades CCH->Kimi->cheap->local beneath whatever the primary is, so
    the answer is decoupled from any single provider's context window and
    HAL#329 thread-carry survives the switch. Marker absent/stale/unknown
    /error -> fixed ladder, byte-identical to pre-W2 (fail-safe). The
    long-lived sentinel/thread is owned by ``runtime.run``; nothing here
    reconstructs it, so the one-runtime-session invariant holds trivially.
    ``audit=True`` -> the runtime appends a deterministic ``idr`` terminator
    (emitted centrally by ``_stream_with_audit``, not here).

    Multimodal (optional, backward-compatible): ``media`` (a list of local
    file paths — image / video / audio) or pre-built ``content_parts``
    (OpenAI ContentPart dicts) route the turn through the messages path so a
    multimodal-capable provider (e.g. Gemini for audio) receives the bytes
    inline. ``media`` is the lighter path locally (the envelope carries
    encoded parts only when present). With NEITHER, behaviour is
    byte-identical to the text-only contract. NOTE: the messages path uses
    single-provider limit-aware failover (``LimitSentinel.dispatch``), not
    the string-thread RAILLM cascade — the cascade flattens to a string and
    cannot carry parts; pass an explicit ``model`` to pin the provider.
    """
    rec = None if model else _active_marker()
    requested = model or (rec["address"] if rec else _ladder_addresses()[0])
    parts = _resolve_content_parts(prompt, media, content_parts)
    events = _drive(_build_envelope(prompt, requested, audit,
                                    content_parts=parts, pin=no_failover,
                                    max_tokens=max_tokens))
    result = _collect(events, requested,
                      media_attached=_has_non_text_parts(parts))
    if rec is not None and rec.get("mode") == "solo-once":
        try:  # best-effort revert; a consume failure never breaks dispatch
            _grip_active_llm().consume_once_marker()
        except Exception:
            pass
    return result
