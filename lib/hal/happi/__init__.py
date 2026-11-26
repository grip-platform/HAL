"""HAPPI — Harnessed-AI Polyglot Protocol Interface.

Two-session collaborative package. Format-side owns parsing; runtime-side
owns the stdin/stdout envelope dispatch exposed via `hal --happi-api`.
(Extraction and the PRE-CHECK gate were planned here and never shipped --
the modules named for them do not exist, so they are not claimed.)

α-phase contents:

    Format side (peer session)
        parser.py       — fenced-block extraction (α-2)

    Runtime side (this session)
        envelope.py     — request envelope dataclass + parse
        events.py       — NDJSON wire-event encoder (started | delta | ... | completed)
        runtime.py      — headless stdin → stdout dispatcher
        providers/      — per-LLM streaming adapters (α-3: anthropic; α-3.5: others)

Specs:
    ~/.hal/happi.md                            — the specification itself (axiom 4)
    ~/.hal/plans/happi-protocol-triphase.md    — runtime-side (to be drafted)
"""
from lib.hal.happi.audit import (
    audit_session,
    compute_idr_sha256,
    is_audit_enabled,
    record_event,
)
from lib.hal.happi.cmds_v11 import V11_HANDLERS, V11_RUNTIME_COMMANDS
from lib.hal.happi.context import (
    build_context_ref,
    compute_content_addr,
    context_head,
    validate_body_decision,
)
from lib.hal.happi.continuity import (
    ContinuityError,
    build_continuity_ref,
    validate_kind,
)
from lib.hal.happi.reasoning import (
    ReasoningError,
    build_reasoning_ref,
)
from lib.hal.happi.decision import (
    ADMISSIBLE_FIELDS,
    Decision,
    DecisionError,
    from_model_dict,
    is_predicate_shaped,
    reject_surface_fields,
    render_plain,
    validate as validate_decision,
)
from lib.hal.happi.envelope import EnvelopeError, HappiEnvelope, SUPPORTED_VERSIONS
from lib.hal.happi.events import (
    PROTOCOL_VERSION,
    emit_completed,
    emit_context,
    emit_continuity,
    emit_delta,
    emit_error,
    emit_event,
    emit_idr,
    emit_reasoning,
    emit_started,
    emit_sub_request,
    emit_thinking,
    emit_tool_call,
    emit_tool_result,
)
from lib.hal.happi.parser import (
    CANONICAL_LABELS,
    HappiDocument,
    parse_happi,
    parse_happi_file,
)
from lib.hal.happi.runner import (
    QUEUE_LABEL,
    run_envelope_line,
    run_queue_file,
    run_queue_lines,
)
from lib.hal.happi.runtime import (
    A1_SCAFFOLD_COMMANDS,
    A3_PROVIDER_COMMANDS,
    SUPPORTED_COMMANDS,
    main,
    run,
)

__all__ = [
    "A1_SCAFFOLD_COMMANDS",
    "A3_PROVIDER_COMMANDS",
    "ADMISSIBLE_FIELDS",
    "CANONICAL_LABELS",
    "ContinuityError",
    "Decision",
    "DecisionError",
    "EnvelopeError",
    "HappiDocument",
    "HappiEnvelope",
    "PROTOCOL_VERSION",
    "QUEUE_LABEL",
    "ReasoningError",
    "SUPPORTED_COMMANDS",
    "SUPPORTED_VERSIONS",
    "V11_HANDLERS",
    "V11_RUNTIME_COMMANDS",
    "audit_session",
    "build_context_ref",
    "build_continuity_ref",
    "build_reasoning_ref",
    "compute_content_addr",
    "compute_idr_sha256",
    "context_head",
    "from_model_dict",
    "is_predicate_shaped",
    "reject_surface_fields",
    "render_plain",
    "validate_body_decision",
    "validate_kind",
    "validate_decision",
    "emit_completed",
    "emit_context",
    "emit_continuity",
    "emit_delta",
    "emit_error",
    "emit_event",
    "emit_idr",
    "emit_reasoning",
    "emit_started",
    "emit_sub_request",
    "emit_thinking",
    "emit_tool_call",
    "emit_tool_result",
    "is_audit_enabled",
    "main",
    "parse_happi",
    "parse_happi_file",
    "record_event",
    "run",
    "run_envelope_line",
    "run_queue_file",
    "run_queue_lines",
]
