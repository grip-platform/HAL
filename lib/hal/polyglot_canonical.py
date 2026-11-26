"""HAL Polyglot Canonicalisation — the C5 fix (cross-model context_ref schema).

The surviving #1 weakest link in the GRIP/HAL/HAPPI/IDR moat, per the round-5
cross-provider IP council (Gemini-3.1-Pro judge, 2026-06-23):

    "C5 (Abstraction Leakage in HAL) — VALID (NEW #1 THREAT). A context_ref
     built on Claude's semantic formatting will read as misaligned gibberish or
     suffer abstraction leakage if evaluated by a local DeepSeek model in a
     later session."

The judge's prescribed fix, implemented here verbatim:

    "Polyglot Canonicalization Schema. Introduce a mandatory, provider-neutral
     Banach projector function at the HAL gateway. Before a context-delta is
     committed to the memory chain and hashed, HAL must strip proprietary token
     formatting (Claude's system prompts, Gemini's chain-of-thought structures)
     and project the knowledge into a standardized, minimal JSON schema. If
     translation is lossy, the envelope flags a context_degraded warning, but
     the core semantic hash remains strictly cross-compatible."

WHY this closes C5 (the load-bearing why): GRIP's `content_addr`
(`lib/precog/idr.py`) canonicalises JSON *structure* (sorted keys) but never the
*semantic content* of a context value. So the SAME belief — "X is true" —
recorded under Claude's `Human:/Assistant:` framing hashes differently from the
same belief recorded under Llama's `[INST]…[/INST]` framing. The memory chain
then fails to recognise them as the same belief across a provider switch, and
the polyglot promise breaks at the memory layer. This projector strips the
provider scaffolding BEFORE the hash, so the semantic hash is the SAME belief's
identity regardless of which model phrased it.

It is a BANACH PROJECTOR: idempotent on its own image — ``P(P(x)) == P(x)`` — the
same idempotent-no-op fixed point GRIP dogfoods in `scope_contamination`
(``clean iff sanitise is a no-op``). The canonical form is the fixed point;
re-projecting a canonical form is a lossless no-op (``degraded=False``).

Design invariants (each is a Goodhart-proof test in
``tests/test_polyglot_canonical.py``):

  1. Cross-model identity — same belief, any provider's scaffolding → SAME
     ``semantic_hash`` (the judge's C5-closure evidence: frontier ↔ open-weight).
  2. Idempotent fixed point — ``polyglot_canonical(r.canonical).canonical == r.canonical``.
  3. ``degraded`` is LOSSY-ONLY — set True iff projection dropped semantic-bearing
     content that has no provider-neutral form (e.g. a model-internal
     chain-of-thought block). Pure scaffolding stripping is lossless → False.
  4. Volatile-metadata exclusion — model/provider/usage/ts differ → SAME hash.
  5. Discrimination (the Goodhart guard) — genuinely DIFFERENT beliefs → DIFFERENT
     hash. The projector must not collapse everything to one constant.

Zero non-stdlib dependencies (json/hashlib/re/unicodedata) — matches HAL's venv
posture (cf. gateway_witness.py, which also avoids the `cryptography` import).
Fail-open-safe by construction: a value the projector cannot neutralise is
dropped with a recorded reason and ``degraded=True`` — it never raises into a
caller's hot path.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass

SCHEMA = "happi.ctx/1"

# --- Provider scaffolding tokens (LOSSLESS to strip — carry no belief) --------
# Llama/Mistral instruct framing, ChatML (Qwen/OpenAI), Llama-3 header tokens,
# Gemini turn markers, generic special tokens, and explicit role tags.
_ROLE_WORD = r"(?:system|user|assistant|model|tool|developer)"
_SCAFFOLD = re.compile(
    # ChatML role openers (Qwen/OpenAI) — consume the trailing role word + newline
    # so the bare role token does not survive into the belief.
    rf"<\|im_start\|>[ \t]*{_ROLE_WORD}?\n?"
    # Llama-3 header block — the role sits BETWEEN the markers; consume the block.
    rf"|<\|start_header_id\|>[ \t]*{_ROLE_WORD}?[ \t]*<\|end_header_id\|>\n?"
    # Gemini turn opener — consumes its optional trailing role.
    r"|<start_of_turn>[ \t]*(?:model|user)?\n?"
    # Bare special / sentinel tokens (no role payload).
    r"|<\|(?:im_end|begin_of_text|end_of_text|eot_id|endoftext)\|>"
    r"|<end_of_turn>"
    r"|<\|(?:system|user|assistant)\|>"
    # Llama/Mistral instruct + system markers.
    r"|<</?SYS>>"
    r"|\[/?INST\]"
    r"|</?s>"
)
# Leading role labels at line start: "Human:", "Assistant:", "System:", "model:".
_ROLE_LINE = re.compile(r"(?im)^[ \t]*(?:human|assistant|system|user|ai|model)[ \t]*:[ \t]*")

# --- Keys dropped from the semantic projection --------------------------------
# Volatile / evidence-of-record metadata: NOT belief content. Dropping them is
# LOSSLESS for the belief (degraded stays False) — they are the "what model,
# when, how fast" record, not the "what is believed".
_VOLATILE_KEYS = frozenset({
    "model", "provider", "request_id", "response_id", "id", "created", "ts",
    "timestamp", "latency_ms", "usage", "finish_reason", "stop_reason",
    "logprobs", "system_fingerprint", "tool_call_id", "seed", "temperature",
    "top_p", "_raw", "_raw_response", "object", "index",
})
# Provider-internal reasoning with NO provider-neutral form: dropping it IS
# lossy (it may carry belief-bearing content that cannot cross models) → flags
# degraded=True. This is the literal "abstraction leakage" C5 names.
_OPAQUE_KEYS = frozenset({
    "reasoning_content", "thinking", "chain_of_thought", "cot",
    "internal_monologue", "scratchpad", "redacted_thinking",
})

_MAX_DEPTH = 12  # belief contexts are shallow; bound recursion (fail-open).


@dataclass(frozen=True)
class CanonResult:
    """Provider-neutral projection of a context payload.

    canonical    — the minimal, provider-neutral dict (sorted-key-hashable).
    degraded     — True iff projection dropped semantic-bearing content with no
                   neutral form (the ``context_degraded`` warning). LOSSY ONLY.
    dropped      — plain-language audit of what was dropped and why.
    semantic_hash— "sha256:…" over the canonical form. The SAME belief's identity
                   across any provider's formatting (the C5-closure property).
    diagnostic   — the provider-internal reasoning that was dropped from the
                   canonical form, PRESERVED here as non-authoritative specimens
                   instead of discarded. Each entry is {"key", "value"}. NOT part
                   of ``canonical`` and EXCLUDED from ``semantic_hash`` — the
                   belief's cross-provider identity is byte-unchanged. It exists
                   so the reasoning can be EXAMINED, never trusted as cause
                   (diagnostic surface, never evidence; see GRIP
                   ``drafts/cc-paper/thinking-traces-essay.md`` and the trace
                   fence in GRIP ``lib/thinking_trace.py``). Empty when nothing
                   opaque was dropped.
    """
    canonical: dict
    degraded: bool
    dropped: tuple[str, ...]
    semantic_hash: str
    diagnostic: tuple[dict, ...] = ()


def _canonical_json(obj) -> str:
    """Byte-stable JSON — sorted keys, no whitespace. Matches GRIP's
    ``lib/precog/idr.py::_canonical_json`` so the two layers agree on structure."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalise_text(text: str) -> str:
    """Strip provider scaffolding, normalise unicode + whitespace. LOSSLESS for
    belief content (only formatting is removed). Idempotent: re-running on its
    own output is a no-op (the Banach fixed-point property).

    Whitespace LAYOUT is itself provider formatting — Claude frames turns with
    ``\\n\\n``, Llama with spaces inside ``[INST]…[/INST]`` — so the same belief
    arrives with different line structure. Layout is collapsed (all whitespace
    runs -> single space) AFTER line-anchored role labels are stripped, so the
    semantic token sequence (the actual belief) is what survives and hashes
    identically across providers. Role labels are stripped first, while the
    newlines that anchor them still exist."""
    text = unicodedata.normalize("NFC", text)
    text = _SCAFFOLD.sub(" ", text)
    text = _ROLE_LINE.sub(" ", text)            # while line anchors still present
    text = re.sub(r"\s+", " ", text).strip()    # layout is provider formatting
    return text


def _project(value, dropped: list[str], diagnostic: list[dict], depth: int = 0):
    """Recursively project a value onto its provider-neutral form.

    Returns the projected value; appends to ``dropped`` (which carries the
    degraded signal) when belief-bearing content is removed, and to
    ``diagnostic`` the raw provider-internal reasoning that is dropped from the
    canonical form but PRESERVED for examination. Returns the sentinel ``_DROP``
    when the value itself must be omitted by the caller.
    """
    if depth > _MAX_DEPTH:
        dropped.append(f"depth>{_MAX_DEPTH}: nested structure dropped")
        return _DROP
    if isinstance(value, str):
        return _normalise_text(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return _project_dict(value, dropped, diagnostic, depth)
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            p = _project(item, dropped, diagnostic, depth + 1)
            if p is not _DROP:
                out.append(p)
        return out
    # Unknown/opaque type with no neutral form → lossy drop.
    dropped.append(f"opaque-type {type(value).__name__}: dropped (no neutral form)")
    return _DROP


_DROP = object()  # sentinel: this value must be omitted from the projection.


def _project_dict(payload: dict, dropped: list[str], diagnostic: list[dict], depth: int) -> dict:
    out: dict = {}
    for key, value in payload.items():
        k = str(key)
        if k in _VOLATILE_KEYS or k.startswith("_") and k != "_schema":
            continue  # lossless: metadata, not belief.
        if k in _OPAQUE_KEYS:
            dropped.append(f"{k}: provider-internal reasoning dropped (no cross-model form)")
            diagnostic.append({"key": k, "value": value})  # PRESERVE for examination, not discard.
            continue  # LOSSY for the canonical form → degraded; recoverable via .diagnostic.
        p = _project(value, dropped, diagnostic, depth + 1)
        if p is not _DROP:
            out[k] = p
    return out


def polyglot_canonical(payload, provider: str | None = None) -> CanonResult:
    """Project a context payload onto the provider-neutral canonical schema.

    ``payload`` may be a str (raw model-formatted context) or a dict
    (a structured context-delta). ``provider`` is accepted for caller telemetry
    and forward-compatibility but the projection is deliberately provider-neutral
    BY CONSTRUCTION — the same belief must canonicalise identically regardless of
    which provider produced it (that is the whole point of C5).

    Never raises. A value that cannot be neutralised is dropped with a recorded
    reason and ``degraded=True``.
    """
    dropped: list[str] = []
    diagnostic: list[dict] = []
    if isinstance(payload, str):
        body = {"content": _normalise_text(payload)}
    elif isinstance(payload, dict):
        body = _project_dict(payload, dropped, diagnostic, depth=0)
    else:
        # Last-resort fail-open: stringify, flag degraded (structure lost).
        dropped.append(f"non-str/dict payload {type(payload).__name__}: stringified")
        body = {"content": _normalise_text(str(payload))}

    # semantic_hash is computed over ``canonical`` ONLY — the diagnostic sidecar
    # is deliberately NOT hashed, so preserving dropped reasoning leaves the
    # belief's cross-provider identity byte-identical to before this change.
    canonical = {"_schema": SCHEMA, **body}
    semantic_hash = "sha256:" + hashlib.sha256(
        _canonical_json(canonical).encode("utf-8")
    ).hexdigest()
    return CanonResult(
        canonical=canonical,
        degraded=bool(dropped),
        dropped=tuple(dropped),
        semantic_hash=semantic_hash,
        diagnostic=tuple(diagnostic),
    )


def polyglot_hash(payload, provider: str | None = None) -> str:
    """Convenience: the cross-provider-stable semantic hash of ``payload``."""
    return polyglot_canonical(payload, provider).semantic_hash
