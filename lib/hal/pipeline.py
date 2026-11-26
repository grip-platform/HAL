"""HAL Request Processing Pipeline — composable staged message handling.

Decomposes Conversation.send() into 7 ordered stages, each doing one thing.
Inspired by compiler pipeline architecture: input flows through stages,
each transforms the context and passes it forward.

Stages:
    MEDIA    — detect and prepare media attachments
    INTENT   — classify intent (multi-weighted)
    GUARD    — run pre-action hooks (placeholder for future wiring)
    ROUTE    — select model via cost router
    CONTEXT  — assemble system prompt + history
    DISPATCH — send to LLM via sentinel
    RECORD   — track cost + emit events

Coverage note: Only 7 of a possible 12 stages are implemented. The 5
post-processing stages (response caching, deduplication, formatting,
optimisation, emission) remain as future work.

Usage:
    from lib.hal.pipeline import Pipeline, PipelineContext

    pipeline = Pipeline([MediaStage(), IntentStage(), ...])
    ctx = pipeline.run(PipelineContext(message="Fix the auth bug"))
    print(ctx.response.text)
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
@dataclass
class PipelineContext:
    """Mutable context flowing through pipeline stages."""
    message: str
    model: str = ""
    system_prompt: str = ""
    multimodal: object = None
    media_paths: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    response: object = None
    metadata: dict = field(default_factory=dict)
    stage_timings: dict = field(default_factory=dict)
    error: str = ""


class PipelineStage(ABC):
    """Abstract stage in the request processing pipeline."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Transform the context. Return it for the next stage."""


class Pipeline:
    """Ordered sequence of stages with per-stage timing."""

    def __init__(self, stages: list[PipelineStage]):
        self._stages = stages

    @property
    def stages(self) -> list[PipelineStage]:
        return list(self._stages)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute all stages in order. Stops on error."""
        for stage in self._stages:
            start = time.monotonic()
            try:
                ctx = stage.process(ctx)
            except Exception as exc:
                ctx.error = f"{stage.name}: {exc}"
                ctx.stage_timings[stage.name] = round(
                    (time.monotonic() - start) * 1000, 2,
                )
                break
            elapsed_ms = round((time.monotonic() - start) * 1000, 2)
            ctx.stage_timings[stage.name] = elapsed_ms
        return ctx


# === Concrete stages ===


class MediaStage(PipelineStage):
    """Detect and prepare media attachments from the message."""

    def __init__(self, prepare_fn=None):
        self._prepare = prepare_fn

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if self._prepare:
            ctx.multimodal, ctx.media_paths = self._prepare(ctx.message)
        return ctx


class IntentStage(PipelineStage):
    """Classify user intent using IntentDetector (multi-weighted)."""

    def __init__(self, detector=None):
        self._detector = detector

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if self._detector:
            intent = self._detector.classify(ctx.message)
            ctx.metadata["intent"] = intent.category
            ctx.metadata["intent_confidence"] = intent.confidence
            if intent.weights:
                ctx.metadata["intent_weights"] = intent.weights.to_dict()
        return ctx


class GuardStage(PipelineStage):
    """Pre-action guard gate — rule engine + content scanning.

    When a :class:`~lib.hal.guard.GuardEngine` is supplied, the stage routes
    through :func:`lib.hal.gates.evaluate_chat` so secret-shaped content is
    refused in addition to whatever the rule engine decides. A missing
    engine is allowed for tests that need to bypass enforcement.
    """

    def __init__(self, guard_engine=None):
        self._guard = guard_engine

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if not self._guard:
            ctx.metadata["guard_passed"] = True
            return ctx
        # Local import to avoid a circular dep between pipeline and gates.
        from lib.hal.gates import evaluate_chat
        verdict = evaluate_chat(self._guard, ctx.message, ctx.model)
        ctx.metadata["guard_passed"] = verdict.allowed or verdict.needs_approval
        ctx.metadata["guard_decision"] = verdict.decision
        if verdict.denied:
            ctx.error = f"Guard denied: {verdict.reason}"
        return ctx


class RouteStage(PipelineStage):
    """Select target model via cost router or explicit override."""

    def __init__(self, select_fn=None):
        self._select = select_fn

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if self._select:
            ctx.model = self._select(ctx.message, ctx.model or None)
        return ctx


class ContextStage(PipelineStage):
    """Assemble system prompt + KONO recall + user preferences."""

    def __init__(self, build_fn=None):
        self._build = build_fn

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if self._build:
            ctx.system_prompt = self._build(ctx.message)
        return ctx


class DispatchStage(PipelineStage):
    """Send to LLM via Sentinel. The actual LLM call."""

    def __init__(self, sentinel=None):
        self._sentinel = sentinel

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if not self._sentinel:
            ctx.error = "No sentinel configured"
            return ctx
        ctx.response = self._sentinel.dispatch(
            ctx.model, ctx.message,
            system=ctx.system_prompt, messages=ctx.messages,
        )
        return ctx


class RecordStage(PipelineStage):
    """Track cost + emit events after LLM response."""

    def __init__(self, cost_tracker=None, emitter=None):
        self._cost = cost_tracker
        self._emitter = emitter

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.response and self._cost:
            self._cost.record_response(
                ctx.response.model, ctx.response.provider, ctx.response.usage,
            )
        if ctx.response and self._emitter:
            usage = ctx.response.usage
            self._emitter.emit("span.model_request_end", {
                "model": ctx.response.model,
                "provider": ctx.response.provider,
                "model_usage": {
                    "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                },
            })
        return ctx


class QualityStage(PipelineStage):
    """Post-processing: flag dismissive patterns in LLM response.

    First of the 5 outer (Mersenne) stages — response quality filtering.
    Advisory only: annotates pipeline metadata with warnings, does not
    modify the response text.
    """

    def __init__(self, checker=None):
        self._checker = checker

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if not self._checker or not ctx.response:
            return ctx
        text = getattr(ctx.response, "text", "")
        if not text:
            return ctx
        warnings = self._checker.check(text)
        if warnings:
            ctx.metadata["quality_warnings"] = [w.to_dict() for w in warnings]
        return ctx
