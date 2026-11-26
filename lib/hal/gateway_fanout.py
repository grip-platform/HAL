"""hal-fanout — the gateway's ensemble pseudo-model (GRIP B, free-inbound tier).

One inbound Anthropic-wire request whose model id is exactly ``hal-fanout``
fans out in PARALLEL across a roster of free/cheap rungs (local Ollama + the
cloud free tiers), applies a deterministic acceptability floor to each
buffered candidate, and answers with the FIRST rung that passes — losers are
abandoned. Design: GRIP ``drafts/hal-fanout-b/DESIGN.md``.

Why the rungs are always fanned out NON-streamed: the floor must judge a
COMPLETE candidate (a hung stream is indistinguishable from a slow one), and
the proven crosswire fallback already turns one buffered Anthropic message
into a valid SSE burst for a streaming client
(:func:`gateway_crosswire.anthropic_message_to_sse_chunks` — the same path a
stream-ignoring provider takes today).

Honesty about "cancellation": stdlib urllib has no request abort. Losing
rungs are ABANDONED — ``cancel_futures`` stops queued rungs, in-flight ones
run to their per-rung timeout in daemonised pool threads and their responses
are discarded. The winner is never blocked on losers.

Default-OFF by construction: this module only runs when a request names the
exact id ``hal-fanout``. No family route maps to it, no existing model id
enters it, and the gateway hook checks the literal id before any routing —
every other request takes the byte-identical existing path.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from concurrent.futures import as_completed
from typing import Callable, Optional, Sequence, Tuple

from lib.hal import gateway_egress as egress
from lib.hal.gateway import UpstreamResponse, build_default_opener

FANOUT_MODEL_ID = "hal-fanout"

# (provider, model) rungs. Providers MUST exist in gateway_egress's endpoint
# tables (cloud PROVIDER_ENDPOINTS / local LOCAL_PROVIDER_ENDPOINTS) — rungs
# are EXPLICIT pairs precisely because family routing cannot express them
# (e.g. "qwen2.5:3b" would family-route to dashscope, not local Ollama).
# Both Ollama rungs ship by default: qwen2.5:3b is the VPS-primary pull,
# qwen3:8b the Mac pull — whichever is present answers; a missing model just
# fails its own rung (Ollama 404) and the floor discards it.
DEFAULT_RUNGS: Tuple[Tuple[str, str], ...] = (
    ("ollama", "qwen2.5:3b"),
    ("ollama", "qwen3:8b"),
    ("groq", "openai/gpt-oss-120b"),   # llama-3.3-70b-versatile retired at Groq 2026-08-30 (404)
    ("gemini", "gemini-3.5-flash-lite"),
    ("deepseek", "deepseek-flash"),
    ("qwen", "qwen3.5-flash"),
    ("xai", "grok-4.6"),               # grok-3 no longer served (live /v1/models 2026-09-15)
    ("moonshot", "kimi-k2.6"),   # moonshot-v1-auto retired 2026-08-31 (HTTP 404)
)

_RUNGS_ENV = "HAL_FANOUT_RUNGS"                # "provider/model,provider/model,…"
_RUNG_TIMEOUT_ENV = "HAL_FANOUT_RUNG_TIMEOUT_S"   # per-rung HTTP timeout
_BUDGET_ENV = "HAL_FANOUT_BUDGET_S"               # total wall-clock budget
_DEFAULT_RUNG_TIMEOUT_S = 45.0
_DEFAULT_BUDGET_S = 90.0

_MAX_BODY_BYTES = 4_000_000    # a candidate larger than this is not a chat reply
_MAX_TEXT_CHARS = 200_000      # sane ceiling for a relay-tier answer


def _parse_rungs(spec: str) -> Tuple[Tuple[str, str], ...]:
    """``"groq/llama-x,ollama/qwen3:8b"`` → ``(("groq","llama-x"), …)``.

    Entries without a ``/`` are dropped; the model part may itself contain
    ``/`` (split once) or ``:`` (Ollama tags).
    """
    rungs = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry or "/" not in entry:
            continue
        provider, model = entry.split("/", 1)
        if provider.strip() and model.strip():
            rungs.append((provider.strip(), model.strip()))
    return tuple(rungs)


def resolve_rungs() -> Tuple[Tuple[str, str], ...]:
    """Active roster: ``HAL_FANOUT_RUNGS`` env override, else the default —
    filtered to providers the egress tables actually know (an unknown provider
    cannot be forwarded, so it is dropped rather than crashing the ensemble)."""
    spec = os.environ.get(_RUNGS_ENV, "").strip()
    rungs = _parse_rungs(spec) if spec else DEFAULT_RUNGS
    return tuple(
        (provider, model) for provider, model in rungs
        if provider in egress.PROVIDER_ENDPOINTS
        or provider in egress.LOCAL_PROVIDER_ENDPOINTS
    )


def _floor_reason(status: int, body: bytes) -> Optional[str]:
    """``None`` iff the candidate passes the deterministic acceptability floor;
    else a short human-readable reason. The floor judges a COMPLETE buffered
    Anthropic-shaped message (what ``cross_wire_forward`` returns for a
    non-streamed rung): real 2xx, parseable, not an error envelope, and a
    non-empty text answer within sane bounds."""
    if status != 200:
        return f"status {status}"
    if not body:
        return "empty body"
    if len(body) > _MAX_BODY_BYTES:
        return f"body too large ({len(body)} bytes)"
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return "non-JSON body"
    if not isinstance(data, dict):
        return "non-object body"
    if data.get("type") == "error":
        return "error envelope"
    text = "".join(
        block.get("text", "")
        for block in (data.get("content") or [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
    if not text.strip():
        return "empty text content"
    if len(text) > _MAX_TEXT_CHARS:
        return f"text too long ({len(text)} chars)"
    return None


def _passes_floor(status: int, body: bytes) -> bool:
    return _floor_reason(status, body) is None


def _run_rung(provider: str, model: str, payload: dict,
              per_rung_timeout: float) -> UpstreamResponse:
    """One rung = the existing single-model crosswire, non-streamed, with a
    per-rung opener timeout. Reuses :func:`gateway_egress.cross_wire_forward`
    end to end (translation, UA, key resolution, account failover, error
    envelopes) — the ensemble adds NO new wire logic."""
    rung_payload = dict(payload)
    rung_payload["model"] = model
    return egress.cross_wire_forward(
        rung_payload,
        provider=provider,
        want_stream=False,
        opener=build_default_opener(timeout=per_rung_timeout),
    )


Runner = Callable[[str, str, dict, float], UpstreamResponse]


def fanout_response(
    payload: dict,
    *,
    want_stream: bool,
    rungs: Optional[Sequence[Tuple[str, str]]] = None,
    runner: Optional[Runner] = None,
    per_rung_timeout: Optional[float] = None,
    total_budget: Optional[float] = None,
) -> UpstreamResponse:
    """Answer one ``hal-fanout`` request: hedged parallel race, first
    floor-passing rung wins.

    ``rungs``/``runner``/timeouts are DIP-injected for tests (the real runner
    is :func:`_run_rung`); production callers pass only ``payload`` +
    ``want_stream``. Never raises: every failure shape (all rungs failing,
    budget exhausted, empty roster) returns a clean Anthropic-shaped error
    envelope with status 502.
    """
    active = tuple(rungs) if rungs is not None else resolve_rungs()
    run = runner or _run_rung
    rung_timeout = float(
        per_rung_timeout if per_rung_timeout is not None
        else os.environ.get(_RUNG_TIMEOUT_ENV, _DEFAULT_RUNG_TIMEOUT_S))
    budget = float(
        total_budget if total_budget is not None
        else os.environ.get(_BUDGET_ENV, _DEFAULT_BUDGET_S))

    if not active:
        return _error_response("no usable rungs configured "
                               f"({_RUNGS_ENV} empty or all providers unknown)")

    failures: list[str] = []
    pool = ThreadPoolExecutor(max_workers=len(active),
                              thread_name_prefix="hal-fanout")
    try:
        pending = {
            pool.submit(_collect, run, provider, model, payload, rung_timeout):
                (provider, model)
            for provider, model in active
        }
        # One flat completion-ordered pass — O(n) over rungs: as_completed
        # enforces the TOTAL budget (raises when it elapses), each future is
        # popped exactly once, the first floor-passing candidate returns.
        for fut in as_completed(tuple(pending), timeout=budget):
            provider, model = pending.pop(fut)
            status, body, err = fut.result()
            if err is not None:
                failures.append(f"{provider}/{model}: {err}")
                continue
            reason = _floor_reason(status, body)
            if reason is not None:
                failures.append(f"{provider}/{model}: {reason}")
                continue
            return _winner_response(body, want_stream=want_stream)
    except FuturesTimeout:
        failures.extend(
            f"{p}/{m}: unfinished at budget ({budget:g}s)"
            for p, m in pending.values())
    finally:
        # Abandon losers: queued rungs are cancelled; in-flight ones run to
        # their per-rung timeout in pool threads and are discarded. The winner
        # (or the error below) is never blocked on them.
        pool.shutdown(wait=False, cancel_futures=True)

    detail = "; ".join(failures)[:900] or "no rung produced a result"
    return _error_response(
        f"all {len(active)} rungs failed — {detail}")


def _collect(run: Runner, provider: str, model: str, payload: dict,
             rung_timeout: float) -> Tuple[int, bytes, Optional[str]]:
    """Execute one rung and normalise EVERY outcome to (status, body, error) —
    a raising rung (transport failure, Ollama down) must count as that rung's
    failure, never escape into the pool."""
    try:
        resp = run(provider, model, payload, rung_timeout)
        body = resp.read_all()
        resp.close()
        return resp.status, body, None
    except Exception as exc:  # noqa: BLE001 — one rung may never sink the race
        return 0, b"", f"{type(exc).__name__}: {exc}"[:200]


def _winner_response(body: bytes, *, want_stream: bool) -> UpstreamResponse:
    """Reshape the winning buffered Anthropic message for the client — the
    SAME two shapes the single-model crosswire uses (JSON, or one valid SSE
    burst for a streaming client)."""
    if want_stream:
        from lib.hal.gateway_crosswire import anthropic_message_to_sse_chunks
        return egress._synthetic_sse_response(
            anthropic_message_to_sse_chunks(json.loads(body)))
    return egress._synthetic_json_response(200, body)


def _error_response(message: str) -> UpstreamResponse:
    """A clean Anthropic-shaped 502 (fail-loud, never a hang or a bare 200)."""
    body = egress._anthropic_error_envelope(
        502, f"hal-fanout: {message}".encode("utf-8"))
    return egress._synthetic_json_response(502, body)
