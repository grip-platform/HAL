"""HAPPI runtime — headless stdin envelope → stdout NDJSON stream.

No version in this title, deliberately. The version this runtime speaks is
``PROTOCOL_VERSION`` (``events.py``); the canonical declaration is
``HAPPI_VERSION`` in the polyglot runtime.

Commands (by phase):
    α-1 echo-mode:
        echo     — emit `args` joined by space as a single delta
        noop     — started + completed only
        version  — delta with PROTOCOL_VERSION
    α-3 / α-3.5 / α-3.6 / α-3.7 provider dispatch:
        chat     — route to a real LLM provider by `flags.provider`.
                   Anthropic (α-3):       claude-*, anthropic/*, or empty default.
                   OpenAI-compat (α-3.5): openai/*, groq/*, mistral/*,
                                          deepseek/*, openrouter/*, moonshot/*,
                                          ollama/*.
                   Gemini (α-3.6):        gemini/* (or bare "gemini").
                   Extended (α-3.7):      qwen/*, zhipu/*, xai/*,
                                          soruxgpt/*, vulavula/*, lemonade/*,
                                          vllm/*, lmstudio/*, mlx_vlm/*.
                   Tool loop (W1b): `flags.tools`, `flags.tool_choice` and a
                   verbatim `flags.messages` history; the adapter emits
                   `thinking`, `tool_call` and `tool_result` events and never
                   executes a tool -- the caller does, and sends the result
                   back as a `role: tool` message on the next envelope.
    γ-3 fractal composition:
        compose  — emit one `sub_request` per child envelope in
                   `flags.envelopes=[{...}, {...}]`. The runtime's
                   sub_request expander recursively dispatches each child
                   and inlines its events into the parent stream.

Any other cmd yields an `error` event with code `unsupported_cmd`. The
PRE-CHECK gate (capability HMAC, replay ledger, surface fingerprint) is
owned by the peer session's format-side spec (`plans/happi-polyglot-spec.md`
§3) and lands at β-2 — runtime currently trusts the envelope wholesale.

Sub-request expansion (γ-3):
    `_dispatch` wraps `_dispatch_core` with recursive sub_request
    expansion. Any `sub_request` event whose `child` payload is a valid
    nested envelope is inlined: parent yields `sub_request`, then every
    event from the child's dispatch, then the parent continues. Depth
    is capped at `MAX_SUB_REQUEST_DEPTH` (default 8) — hitting the cap
    emits a `sub_request_depth_cap` error instead of recursing further.

Usage:
    echo '{"v":"happi/1.0","id":"r1","cmd":"echo","args":["hi"]}' | hal --happi-api

Exit codes:
    0 — envelope parsed and dispatched (dispatch itself may have emitted error)
    1 — envelope failed to parse (empty stdin, invalid JSON, schema violation)
"""
from __future__ import annotations

import json
import sys
from typing import IO, Iterator

from lib.hal.happi.audit import (
    audit_buffered_lines,
    audit_model_versions,
    audit_session,
    compute_idr_sha256,
    is_audit_enabled,
    record_event,
)
from lib.hal.happi.cmds_cache import CACHE_COMMANDS, CACHE_HANDLERS
from lib.hal.happi.cmds_decide import DECIDE_COMMANDS, DECIDE_HANDLERS
from lib.hal.happi.cmds_infer import INFER_COMMANDS, INFER_HANDLERS
from lib.hal.happi.cmds_v11 import V11_HANDLERS, V11_RUNTIME_COMMANDS
from lib.hal.happi.envelope import EnvelopeError, HappiEnvelope
from lib.hal.happi.events import (
    PROTOCOL_VERSION,
    emit_completed,
    emit_delta,
    emit_error,
    emit_idr,
    emit_started,
    emit_sub_request,
)


A1_SCAFFOLD_COMMANDS = frozenset({"echo", "noop", "version"})
A3_PROVIDER_COMMANDS = frozenset({"chat"})
GAMMA3_COMPOSE_COMMANDS = frozenset({"compose"})
SUPPORTED_COMMANDS = (
    A1_SCAFFOLD_COMMANDS
    | A3_PROVIDER_COMMANDS
    | GAMMA3_COMPOSE_COMMANDS
    | V11_RUNTIME_COMMANDS
    | INFER_COMMANDS
    | CACHE_COMMANDS
    | DECIDE_COMMANDS
)

MAX_SUB_REQUEST_DEPTH = 8


ANTHROPIC_FAMILY_PREFIXES = ("claude", "anthropic")


def _select_family(provider_spec: str) -> str:
    """Extract family portion from 'family/model' specs; bare strings return themselves."""
    spec = provider_spec.lower()
    return spec.split("/", 1)[0] if "/" in spec else spec


def _dispatch_chat(envelope: HappiEnvelope) -> Iterator[str]:
    """Route a `chat` envelope to its provider adapter.

    Family selection by `envelope.flags.provider`:
        claude-* / anthropic / anthropic-* / empty  → Anthropic /v1/messages
        gemini                                       → Gemini streamGenerateContent
        openai, groq, mistral, deepseek, openrouter → OpenAI-compat adapter
        moonshot, ollama                             → OpenAI-compat adapter
        qwen, zhipu, xai, soruxgpt, vulavula → OpenAI-compat adapter (cloud)
        lemonade, vllm, lmstudio, mlx_vlm            → OpenAI-compat adapter (local)
        (anything else)                              → unsupported_provider error

    The adapter yields its own `delta` / `completed` / `error` sequence;
    runtime does not emit terminal events for `chat` commands.
    """
    provider_spec = str(envelope.flags.get("provider") or "")
    family = _select_family(provider_spec)

    if family == "" or family.startswith(ANTHROPIC_FAMILY_PREFIXES):
        from lib.hal.happi.providers.anthropic import stream_anthropic
        yield from stream_anthropic(envelope)
        return

    if family == "gemini":
        from lib.hal.happi.providers.gemini import stream_gemini
        yield from stream_gemini(envelope)
        return

    from lib.hal.happi.providers.openai_compat import (
        SUPPORTED_PROVIDERS as OPENAI_COMPAT_FAMILY,
        stream_openai_compat,
    )
    if family in OPENAI_COMPAT_FAMILY:
        yield from stream_openai_compat(envelope)
        return

    yield emit_error(
        envelope.id,
        f"unsupported provider: {provider_spec!r} (family={family!r}; "
        f"supported: anthropic/*, gemini/*, "
        f"{', '.join(sorted(OPENAI_COMPAT_FAMILY))}/*)",
        code="unsupported_provider",
    )
    yield emit_completed(envelope.id, usage={})


def _read_envelope(stream: IO[str]) -> tuple[HappiEnvelope, bytes]:
    """Read one JSON envelope from `stream` (EOF-terminated).

    Returns ``(envelope, envelope_raw_bytes)`` — the literal UTF-8 encoded
    bytes of what was read (post-strip) so the audit terminator can hash
    the same byte sequence as the polyglot canonical runtime which calls
    ``sys.stdin.read().encode("utf-8")``. The strip mirrors the polyglot
    behaviour where leading/trailing whitespace is not part of the
    envelope-of-record.
    """
    raw_text = stream.read().strip()
    if not raw_text:
        raise EnvelopeError("empty stdin; expected a JSON envelope")
    return HappiEnvelope.parse(raw_text), raw_text.encode("utf-8")


def _dispatch_compose(envelope: HappiEnvelope) -> Iterator[str]:
    """Emit one sub_request per child envelope listed in flags.envelopes.

    The runtime's sub_request expander (`_expand_sub_requests`) recursively
    dispatches each child and inlines its events into the parent stream,
    so callers see a single interleaved event sequence.
    """
    children = envelope.flags.get("envelopes") or envelope.flags.get("children") or []
    if not isinstance(children, list):
        yield emit_error(
            envelope.id,
            f"compose.flags.envelopes must be a list, got {type(children).__name__}",
            code="invalid_compose",
        )
        yield emit_completed(envelope.id, usage={})
        return

    dispatched = 0
    for child in children:
        if not isinstance(child, dict):
            yield emit_error(
                envelope.id,
                f"compose child must be an envelope dict, got {type(child).__name__}",
                code="invalid_child",
            )
            continue
        yield emit_sub_request(envelope.id, child)
        dispatched += 1

    yield emit_completed(envelope.id, usage={"children": dispatched})


def _dispatch_core(envelope: HappiEnvelope) -> Iterator[str]:
    """Yield NDJSON event lines for one envelope WITHOUT sub_request expansion.

    This is the leaf dispatcher — the public `_dispatch` wraps this in
    `_expand_sub_requests` so recursion is applied uniformly across all
    commands (including `compose`).
    """
    yield emit_started(envelope.id)

    if envelope.cmd == "echo":
        text = " ".join(envelope.args)
        yield emit_delta(envelope.id, text)
        yield emit_completed(envelope.id, usage={
            "in_chars": len(text), "out_chars": len(text)})
        return

    if envelope.cmd == "noop":
        yield emit_completed(envelope.id)
        return

    if envelope.cmd == "version":
        yield emit_delta(envelope.id, PROTOCOL_VERSION)
        yield emit_completed(envelope.id)
        return

    if envelope.cmd == "chat":
        yield from _dispatch_chat(envelope)
        return

    if envelope.cmd == "compose":
        yield from _dispatch_compose(envelope)
        return

    handler = (V11_HANDLERS.get(envelope.cmd)
               or INFER_HANDLERS.get(envelope.cmd)
               or CACHE_HANDLERS.get(envelope.cmd)
               or DECIDE_HANDLERS.get(envelope.cmd))
    if handler is not None:
        yield from handler(envelope)
        return

    yield emit_error(
        envelope.id,
        f"unsupported command: {envelope.cmd!r} "
        f"(supported in this build: {sorted(SUPPORTED_COMMANDS)})",
        code="unsupported_cmd",
    )
    yield emit_completed(envelope.id, usage={})


def _expand_sub_requests(
    source: Iterator[str], depth: int = 0
) -> Iterator[str]:
    """Yield events from `source`, recursively expanding any sub_request.

    Protocol: any event with `type == "sub_request"` is passed through
    verbatim, then — if depth budget remains and the `child` payload parses
    as a valid envelope — the child is dispatched and its events inlined.
    Depth cap and parse errors emit a single diagnostic error event and
    skip the expansion for that one sub_request without halting the parent.
    """
    for line in source:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            yield line
            continue

        if event.get("type") != "sub_request":
            yield line
            continue

        # Always yield the sub_request marker itself.
        yield line

        parent_id = event.get("id") or "sub-unknown"
        # `envelope` is the spec field (happi.md event table). `child` is the
        # form this library emitted before that divergence was closed; it is
        # still READ so historical NDJSON and audit replays keep expanding.
        # Deliberately asymmetric: tolerant reader, spec-exact writer.
        child_payload = event.get("envelope")
        if child_payload is None:
            child_payload = event.get("child")
        if not isinstance(child_payload, dict):
            yield emit_error(
                parent_id,
                "sub_request.envelope missing or not an object",
                code="sub_request_parse_error",
            )
            continue

        if depth + 1 > MAX_SUB_REQUEST_DEPTH:
            yield emit_error(
                parent_id,
                f"sub_request depth cap reached ({MAX_SUB_REQUEST_DEPTH}); "
                f"child envelope not dispatched",
                code="sub_request_depth_cap",
            )
            continue

        try:
            child_env = HappiEnvelope.parse(child_payload)
        except EnvelopeError as exc:
            yield emit_error(
                parent_id,
                f"sub_request child envelope parse error: {exc}",
                code="sub_request_parse_error",
            )
            continue

        yield from _expand_sub_requests(_dispatch_core(child_env), depth + 1)


def _audit_enabled_for(envelope: HappiEnvelope) -> bool:
    """True iff the envelope's flags.audit is the literal boolean True.

    Mirrors the polyglot's ``bool(flags.get("audit", False))`` discipline
    in ``~/.hal/happi-sources/04-python-runtime.py:425`` — only the literal
    truthy value enables audit; ``None``, missing, ``"true"`` (string)
    leave audit OFF. This is intentional: opt-in is a typed contract.
    """
    return envelope.flags.get("audit") is True


def _stream_with_audit(
    envelope: HappiEnvelope, envelope_raw: bytes
) -> Iterator[str]:
    """Yield expanded events; record each into the active audit buffer.

    Trailing ``idr`` event chains sha256 over (envelope_raw + each event
    line UTF-8 encoded) and is yielded LAST — after ``completed`` /
    ``error`` from the inner dispatch.
    """
    model_versions = list(envelope.flags.get("model_versions") or [])
    with audit_session(model_versions=model_versions):
        for line in _expand_sub_requests(_dispatch_core(envelope), depth=0):
            record_event(line)
            yield line
        block_anchor = envelope.flags.get("block_anchor")
        if not isinstance(block_anchor, int):
            block_anchor = None
        yield emit_idr(
            envelope.id,
            sha256=compute_idr_sha256(envelope_raw),
            cid=None,
            model_versions=audit_model_versions(),
            block_anchor=block_anchor,
        )


def _dispatch(
    envelope: HappiEnvelope, envelope_raw: bytes = b""
) -> Iterator[str]:
    """Public entry: dispatch an envelope with recursive sub_request expansion.

    When ``envelope.flags.audit is True``, every emitted line is buffered
    in the per-dispatch audit frame and a final ``idr`` event is yielded
    after ``completed``/``error``. ``envelope_raw`` is the literal bytes
    the transport received; the polyglot canonical runtime hashes the
    same bytes (``sys.stdin.read()``) so cross-runtime parity holds when
    callers plumb the raw stdin/HTTP-body/socket-line through verbatim.

    Callers that do not have access to the raw bytes (e.g. queue runner
    iterating pre-parsed envelopes) pass the default empty ``b""`` and
    accept that audit (if enabled in such a context) would hash an empty
    envelope-prefix — best-effort, documented limitation.
    """
    if _audit_enabled_for(envelope):
        yield from _stream_with_audit(envelope, envelope_raw)
        return
    yield from _expand_sub_requests(_dispatch_core(envelope), depth=0)


def run(stream_in: IO[str] = sys.stdin, stream_out: IO[str] = sys.stdout) -> int:
    """Read one envelope, stream NDJSON events, return exit code.

    Separates IO from logic so tests can feed StringIO and assert the wire
    stream without subprocess overhead.
    """
    try:
        envelope, envelope_raw = _read_envelope(stream_in)
    except EnvelopeError as exc:
        stream_out.write(emit_error("req-invalid", str(exc), code="parse_error") + "\n")
        stream_out.flush()
        return 1

    for line in _dispatch(envelope, envelope_raw):
        stream_out.write(line + "\n")
        stream_out.flush()
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())
