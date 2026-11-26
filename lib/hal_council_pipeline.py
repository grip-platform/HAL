"""HAL Council pipeline — 14-step review orchestrator (Day 0 W5-E).

Composes the Day 0 foundation modules into the end-to-end review flow
specified in `plans/fluffy-twirling-waffle.md` §Architecture (lines 104-159):

    1. Setup + circuit breaker (cc_rate_limiter)
    2. Idempotency key (pr_number + head_sha + prompt_hash)
    3. Scope extraction (pr_scope_extractor)
    4. AST diff (pr_ast_differ)
    5. Mermaid diagram (pr_mermaid_adapter)
    6. Hypothesis auto-register (pr_hypothesis_autoregister)
    7. Goodhart pre-flight (goodhart_guard)
    8. 4-LLM MoE council (HAL_COUNCIL_REVIEW via HALCouncil)
    9. Adversarial pairing pass
   10. Permanent Gemini judge (judge_permanent)
   11. Final score with outsider multiplier + goodhart flag
   12. Unified sticky PR comment (post-review.py)
   13. Side-effects: hypothesis stored, council-health metric
   14. Appeal path: @halcouncil override <reason>

Every step is a pure function taking a `PipelineContext` and returning
an updated context. The orchestrator runs them in order. Step failures
become `context.errors` entries and short-circuit the pipeline to step
12 (post a structured error comment) instead of raising.

This module is an orchestrator only — all heavy logic lives in the
component modules it imports. Tests inject fakes for every side-effect
(LLM calls, goodhart check, idempotency store, comment poster) so the
whole pipeline runs hermetically.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Optional


# Step names — used for error reporting and step enable/disable.
STEP_SETUP = "setup"
STEP_IDEMPOTENCY = "idempotency"
STEP_SCOPE = "scope"
STEP_AST_DIFF = "ast_diff"
STEP_MERMAID = "mermaid"
STEP_HYPOTHESIS = "hypothesis"
STEP_GOODHART = "goodhart"
STEP_COUNCIL = "council"
STEP_ADVERSARIAL = "adversarial"
STEP_JUDGE = "judge"
STEP_FINAL_SCORE = "final_score"
STEP_COMMENT = "comment"
STEP_SIDE_EFFECTS = "side_effects"
STEP_APPEAL = "appeal"

PIPELINE_STEPS: tuple[str, ...] = (
    STEP_SETUP, STEP_IDEMPOTENCY, STEP_SCOPE, STEP_AST_DIFF, STEP_MERMAID,
    STEP_HYPOTHESIS, STEP_GOODHART, STEP_COUNCIL, STEP_ADVERSARIAL,
    STEP_JUDGE, STEP_FINAL_SCORE, STEP_COMMENT, STEP_SIDE_EFFECTS, STEP_APPEAL,
)


@dataclass
class PipelineContext:
    """Mutable context threaded through every pipeline step.

    Callers start with the PR identifiers; each step enriches the context
    with its output. Errors accumulate — a step can fail without aborting
    the whole pipeline (the `comment` step always fires so the reviewer
    leaves a trace).
    """

    pr_number: int = 0
    repo: str = ""
    head_sha: str = ""
    diff_text: str = ""
    pr_title: str = ""
    pr_body: str = ""

    # Populated by steps
    idempotency_key: str = ""
    scope: object = None
    ast_deltas: dict = field(default_factory=dict)
    mermaid_diagram: object = None
    hypothesis_result: object = None
    goodhart_pass: bool = True
    goodhart_reason: str = ""
    council_verdicts: list = field(default_factory=list)
    adversarial_no_error_rate: float = 0.0
    judge_verdict: object = None
    final_score: float = 0.0
    outsider_multiplier: float = 1.0
    comment_posted: bool = False

    # Binary council verdict (set by step_council via hal_council_dispatch)
    council_pass: bool = True
    council_blockers: list = field(default_factory=list)
    council_warnings: list = field(default_factory=list)
    council_providers_responded: int = 0

    # Control state
    short_circuited_at: Optional[str] = None
    errors: list = field(default_factory=list)
    skipped_steps: frozenset = frozenset()

    def record_error(self, step: str, reason: str) -> None:
        self.errors.append({"step": step, "reason": reason})

    def short_circuit(self, step: str) -> None:
        if self.short_circuited_at is None:
            self.short_circuited_at = step


@dataclass
class PipelineHooks:
    """Injectable side-effects — every step that touches the world goes through here.

    Tests construct this with fakes; production wires the real implementations
    (imports from the component modules are lazy inside the hook functions).
    """

    scope_extractor: Optional[Callable] = None
    ast_differ: Optional[Callable] = None
    mermaid_builder: Optional[Callable] = None
    autoregister: Optional[Callable] = None
    goodhart_check: Optional[Callable] = None
    council_runner: Optional[Callable] = None
    judge_caller: Optional[Callable] = None
    comment_poster: Optional[Callable] = None
    idempotency_store: Optional[Callable] = None
    metrics_writer: Optional[Callable] = None


# ---------- step implementations ----------


def step_setup(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    if not ctx.pr_number or not ctx.repo or not ctx.head_sha:
        ctx.record_error(STEP_SETUP, "missing_pr_identifiers")
        ctx.short_circuit(STEP_SETUP)
    return ctx


def _compute_key(pr_number: int, head_sha: str, prompt_hash: str = "v1") -> str:
    payload = f"{pr_number}|{head_sha}|{prompt_hash}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def step_idempotency(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    ctx.idempotency_key = _compute_key(ctx.pr_number, ctx.head_sha)
    if hooks.idempotency_store is not None:
        try:
            already_posted = hooks.idempotency_store(ctx.idempotency_key)
            if already_posted:
                ctx.short_circuit(STEP_IDEMPOTENCY)
        except Exception as exc:  # noqa: BLE001
            ctx.record_error(STEP_IDEMPOTENCY, f"{type(exc).__name__}:{exc}")
    return ctx


def step_scope(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    if hooks.scope_extractor is None:
        return ctx
    try:
        ctx.scope = hooks.scope_extractor(ctx.diff_text)
    except Exception as exc:  # noqa: BLE001
        ctx.record_error(STEP_SCOPE, f"{type(exc).__name__}:{exc}")
    return ctx


def step_ast_diff(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    if hooks.ast_differ is None or ctx.scope is None:
        return ctx
    try:
        ctx.ast_deltas = hooks.ast_differ(ctx.scope) or {}
    except Exception as exc:  # noqa: BLE001
        ctx.record_error(STEP_AST_DIFF, f"{type(exc).__name__}:{exc}")
    return ctx


def step_mermaid(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    if hooks.mermaid_builder is None or ctx.scope is None:
        return ctx
    try:
        ctx.mermaid_diagram = hooks.mermaid_builder(ctx.scope, ctx.ast_deltas)
    except Exception as exc:  # noqa: BLE001
        ctx.record_error(STEP_MERMAID, f"{type(exc).__name__}:{exc}")
    return ctx


def step_hypothesis(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    if hooks.autoregister is None:
        return ctx
    try:
        ctx.hypothesis_result = hooks.autoregister(ctx.pr_title, ctx.pr_body, ctx.pr_number)
    except Exception as exc:  # noqa: BLE001
        ctx.record_error(STEP_HYPOTHESIS, f"{type(exc).__name__}:{exc}")
    return ctx


def step_goodhart(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    if hooks.goodhart_check is None:
        return ctx
    try:
        passed, reason = hooks.goodhart_check(ctx)
    except Exception as exc:  # noqa: BLE001
        ctx.record_error(STEP_GOODHART, f"{type(exc).__name__}:{exc}")
        return ctx
    ctx.goodhart_pass = bool(passed)
    ctx.goodhart_reason = reason or ""
    if not ctx.goodhart_pass:
        ctx.short_circuit(STEP_GOODHART)
    return ctx


def step_council(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    if hooks.council_runner is None:
        return ctx
    try:
        ctx.council_verdicts = list(hooks.council_runner(ctx) or [])
    except Exception as exc:  # noqa: BLE001
        ctx.record_error(STEP_COUNCIL, f"{type(exc).__name__}:{exc}")
    return ctx


def _compute_no_error_rate(verdicts: list) -> float:
    if not verdicts:
        return 0.0
    no_errors = sum(1 for v in verdicts if getattr(v, "adversarial_no_error", False))
    return no_errors / len(verdicts)


def step_adversarial(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    ctx.adversarial_no_error_rate = _compute_no_error_rate(ctx.council_verdicts)
    return ctx


def step_judge(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    if hooks.judge_caller is None:
        return ctx
    try:
        ctx.judge_verdict = hooks.judge_caller(ctx)
    except Exception as exc:  # noqa: BLE001
        ctx.record_error(STEP_JUDGE, f"{type(exc).__name__}:{exc}")
    return ctx


def _compute_outsider_multiplier(verdicts: list) -> float:
    """Linear decay from 1.0 at |diff|<=10 to 0.7 at |diff|>=25."""
    if len(verdicts) < 2:
        return 1.0
    scores = [float(getattr(v, "score", 0)) for v in verdicts]
    outsider = next(
        (s for v, s in zip(verdicts, scores) if getattr(v, "is_outsider", False)),
        None,
    )
    if outsider is None:
        return 1.0
    council_scores = [s for v, s in zip(verdicts, scores) if not getattr(v, "is_outsider", False)]
    if not council_scores:
        return 1.0
    council_mean = sum(council_scores) / len(council_scores)
    diff = abs(outsider - council_mean)
    if diff <= 10:
        return 1.0
    if diff >= 25:
        return 0.7
    return 1.0 - 0.3 * (diff - 10) / 15


def step_final_score(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    goodhart_flag = 1.0 if ctx.goodhart_pass else 0.0
    base = float(getattr(ctx.judge_verdict, "confidence", 0)) / 100.0 if ctx.judge_verdict else 0.0
    ctx.outsider_multiplier = _compute_outsider_multiplier(ctx.council_verdicts)
    ctx.final_score = base * ctx.outsider_multiplier * goodhart_flag
    return ctx


def step_comment(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    if hooks.comment_poster is None:
        return ctx
    try:
        hooks.comment_poster(ctx)
        ctx.comment_posted = True
    except Exception as exc:  # noqa: BLE001
        ctx.record_error(STEP_COMMENT, f"{type(exc).__name__}:{exc}")
    return ctx


def step_side_effects(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    if hooks.metrics_writer is None:
        return ctx
    try:
        hooks.metrics_writer(ctx)
    except Exception as exc:  # noqa: BLE001
        ctx.record_error(STEP_SIDE_EFFECTS, f"{type(exc).__name__}:{exc}")
    return ctx


def step_appeal(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    # Appeal path is passive — hook invoked by out-of-band comment handler.
    # This step exists only to keep the step count at 14 and to allow
    # future in-pipeline appeal logic without renumbering.
    return ctx


STEP_FUNCTIONS: dict[str, Callable] = {
    STEP_SETUP: step_setup,
    STEP_IDEMPOTENCY: step_idempotency,
    STEP_SCOPE: step_scope,
    STEP_AST_DIFF: step_ast_diff,
    STEP_MERMAID: step_mermaid,
    STEP_HYPOTHESIS: step_hypothesis,
    STEP_GOODHART: step_goodhart,
    STEP_COUNCIL: step_council,
    STEP_ADVERSARIAL: step_adversarial,
    STEP_JUDGE: step_judge,
    STEP_FINAL_SCORE: step_final_score,
    STEP_COMMENT: step_comment,
    STEP_SIDE_EFFECTS: step_side_effects,
    STEP_APPEAL: step_appeal,
}


def _should_skip(step: str, ctx: PipelineContext) -> bool:
    """Decide whether the step should run given short-circuit and skip state."""
    if step in ctx.skipped_steps:
        return True
    # Step 12 (comment) always runs, even after short-circuit, so the reviewer
    # leaves a trace. All other steps skip after short-circuit.
    if ctx.short_circuited_at is not None and step != STEP_COMMENT:
        return True
    return False


def run_pipeline(ctx: PipelineContext, hooks: PipelineHooks) -> PipelineContext:
    """Execute all 14 steps in order. Never raises."""
    for step in PIPELINE_STEPS:
        if _should_skip(step, ctx):
            continue
        fn = STEP_FUNCTIONS[step]
        ctx = fn(ctx, hooks)
    return ctx


__all__ = [
    "PIPELINE_STEPS",
    "PipelineContext",
    "PipelineHooks",
    "run_pipeline",
    "STEP_SETUP", "STEP_IDEMPOTENCY", "STEP_SCOPE", "STEP_AST_DIFF",
    "STEP_MERMAID", "STEP_HYPOTHESIS", "STEP_GOODHART", "STEP_COUNCIL",
    "STEP_ADVERSARIAL", "STEP_JUDGE", "STEP_FINAL_SCORE", "STEP_COMMENT",
    "STEP_SIDE_EFFECTS", "STEP_APPEAL",
]
