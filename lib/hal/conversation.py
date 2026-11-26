"""HAL Conversation Engine — multi-turn chat without any external harness.

The core standalone loop: user message -> context assembly -> LLM query ->
response -> memory update -> next turn. Works with any HAL provider.

Usage:
    from lib.hal.conversation import Conversation

    conv = Conversation(model="groq/llama-3.3-70b")
    response = conv.send("What is a Knowledge Engine?")
    response = conv.send("How does it help law firms?")  # Multi-turn
"""
import os
import re
from typing import Optional

from lib.hal import cache_telemetry as _ct

from lib.hal.context_assembler import ContextAssembler
from lib.hal.cost_router import CostRouter
from lib.hal.context_warnings import ContextMonitor
from lib.hal.cost_tracker import CostTracker, estimate_tokens
from lib.hal.gates import default_engine as _default_guard_engine
from lib.hal.intent_detector import IntentDetector
from lib.hal.kono_bridge import clear_cache as _clear_kono_cache
from lib.hal.kono_bridge import recall_for_query as _kono_recall
from lib.hal.pipeline import (
    Pipeline, PipelineContext,
    MediaStage, IntentStage, GuardStage, RouteStage,
    ContextStage, DispatchStage, RecordStage,
)
from lib.hal.attribution import AttributionReport, InputAttributor
from lib.hal.registry import ProviderRegistry, get_registry
from lib.hal.semantic_cache import SemanticCache
from lib.hal.sentinel import LimitSentinel
from lib.hal.session_manager import SessionManager
from lib.hal.session_replay import SessionRecorder
from lib.hal.types import ChatMessage, ChatResponse
from lib.hal.ui_chrome import UI_CHROME_SYSTEM_NOTE, strip_ui_chrome
from lib.hal.user_adapt import UserAdapter


_KONO_ENTRY_RE = re.compile(r"^\s+\d+\.\s")


def _count_kono_entries(formatted: str) -> int:
    """Count numbered KONO memory entries in a recall_for_query() string.

    The kono_bridge formatter emits entries like ``  1. text`` / ``  2. text``.
    Returns 0 for empty strings, multimodal/invalid inputs, or headers without
    entries — the chip renderer treats 0 as "do not display".
    """
    if not formatted:
        return 0
    return sum(1 for line in formatted.splitlines() if _KONO_ENTRY_RE.match(line))


def _maybe_build_semantic_cache() -> Optional[SemanticCache]:
    """Instantiate a SemanticCache when ``HAL_SEMANTIC_CACHE=1``, else None.

    Env-flag lives here rather than inside SemanticCache so tests can drive
    the cache deterministically without touching the environment.
    """
    if os.environ.get("HAL_SEMANTIC_CACHE") != "1":
        return None
    return SemanticCache()


class Conversation:
    """Multi-turn conversation engine powered by HAL."""

    def __init__(
        self,
        model: str = "",
        mode: str = "",
        system_prompt: Optional[str] = None,
        registry: Optional[ProviderRegistry] = None,
        max_context_tokens: int = 200_000,
        emitter=None,
        enable_attribution: bool = False,
    ):
        self._registry = registry or get_registry()
        self._sentinel = LimitSentinel(self._registry)
        self._cost = CostTracker()
        self._session = SessionManager()
        self._assembler = ContextAssembler()
        self._recorder = SessionRecorder(self._session.session_id)
        self._context_monitor = ContextMonitor(max_tokens=max_context_tokens)
        self._user_adapter = UserAdapter()
        self._attributor: Optional[InputAttributor] = (
            InputAttributor(self._registry) if enable_attribution else None
        )
        self._last_response_text: str = ""
        #: KONO entries injected into the most recent turn's context — read by
        #: ``cli._print_response`` to decide whether to emit a citation chip.
        self.last_recall_count: int = 0
        #: True when the most recent ``send()`` was served from semantic cache
        #: — read by ``cli._print_response`` to emit a ``[cache]`` marker.
        self.last_cache_hit: bool = False
        self._semantic_cache: Optional[SemanticCache] = _maybe_build_semantic_cache()
        self._emitter = emitter
        self._model = model or self._registry.routing.default_model
        self._user_set_model: bool = bool(model)
        self._mode = mode
        self._history: list[ChatMessage] = []
        self._system = system_prompt or self._build_system(mode)
        self._intent_detector = IntentDetector()
        self._pipeline = self._build_pipeline()

    @property
    def model(self) -> str:
        return self._model

    @property
    def history(self) -> list[ChatMessage]:
        return list(self._history)

    @property
    def cost(self) -> CostTracker:
        return self._cost

    def _prepare_media(self, message: str) -> tuple[list[dict] | None, list[str]]:
        """Detect local media files and build multimodal content.

        Returns ``(content_parts, media_paths)`` — content_parts is None when
        no media was found or loaded.
        """
        from lib.hal.media import detect_media_paths, build_multimodal_content

        paths = detect_media_paths(message)
        if not paths:
            return None, []
        parts = build_multimodal_content(message, paths)
        return (parts or None), paths

    def _build_turn_system(self, message: str) -> str:
        """Build system prompt for this turn (KONO recall + user prefs).

        Side effect: updates ``self.last_recall_count`` so ``_print_response``
        can emit the memory citation chip without re-running recall.
        """
        _clear_kono_cache()
        system = self._system
        kono_context = _kono_recall(message)
        self.last_recall_count = _count_kono_entries(kono_context)
        if kono_context:
            system = system + "\n\n---\n\n" + kono_context
        pref_modifier = self._user_adapter.to_system_prompt()
        if pref_modifier:
            system = system + "\n\n" + pref_modifier
        return system

    def _select_model(self, message: str, model: Optional[str]) -> str:
        """Select target model, applying cost-router when no explicit model."""
        target = model or self._model
        if not model and not self._user_set_model and self._registry:
            try:
                route = CostRouter(self._registry).select(message)
                if route.model:
                    target = route.model
            except Exception:
                pass
        return target

    def _build_pipeline(self) -> Pipeline:
        """Construct the 7-stage request processing pipeline.

        GuardStage is populated with HAL's default rule set plus secrets
        scanning; chat turns containing credential-shaped tokens are denied
        before they reach any provider.
        """
        return Pipeline([
            MediaStage(prepare_fn=self._prepare_media),
            IntentStage(detector=self._intent_detector),
            GuardStage(guard_engine=_default_guard_engine()),
            RouteStage(select_fn=self._select_model),
            ContextStage(build_fn=self._build_turn_system),
            DispatchStage(sentinel=self._sentinel),
            RecordStage(cost_tracker=self._cost, emitter=self._emitter),
        ])

    def send(self, message: str, model: Optional[str] = None) -> ChatResponse:
        """Send a message and get a response via the 7-stage pipeline.

        Stages: MEDIA -> INTENT -> GUARD -> ROUTE -> CONTEXT -> DISPATCH -> RECORD.

        When ``HAL_SEMANTIC_CACHE=1`` and the message is text-only, a
        semantic-cache lookup runs before the pipeline. Hits skip dispatch
        entirely; misses fall through and populate the cache post-response.

        Pasted UI chrome is stripped from ``message`` before anything else
        runs (see :func:`~lib.hal.ui_chrome.strip_ui_chrome`). Users copying
        HAL's own REPL history back as context were triggering hallucinated
        infrastructure narration (Issue #201) — the strip makes that
        physically impossible for the catalogued chrome patterns.
        """
        message = strip_ui_chrome(message)
        self.last_cache_hit = False
        self.last_recall_count = 0

        multimodal, api_messages = self._prepare_turn(message)
        self._user_adapter.observe_from_feedback(message)

        cache_target_model = self._cache_target_model(message, model, multimodal)
        cached = self._try_serve_from_cache(message, cache_target_model)
        if cached is not None:
            return cached

        ctx = PipelineContext(
            message=message,
            model=model or "",
            messages=api_messages,
        )
        ctx = self._pipeline.run(ctx)
        if ctx.error:
            raise RuntimeError(f"Pipeline error: {ctx.error}")

        response = ctx.response
        self._record_response(message, response)
        if cache_target_model is not None and self._semantic_cache is not None:
            self._semantic_cache.put(response.model or cache_target_model, message, response.text)
        return response

    def _prepare_turn(self, message: str) -> tuple[bool, list[ChatMessage]]:
        """Pre-pipeline bookkeeping: multimodal detection + history append.

        Returns ``(multimodal, api_messages)`` where ``api_messages`` swaps
        the trailing history entry for multimodal content when media was
        detected — the provider layer consumes that list directly.

        The caller (``send``) has already run the message through
        :func:`~lib.hal.ui_chrome.strip_ui_chrome`, so the empty-message
        guard here catches pure-chrome pastes after the strip collapses
        them to an empty string.
        """
        from lib.hal.media import media_summary

        if not message or not message.strip():
            raise ValueError("Cannot send empty message")
        multimodal, media_paths = self._prepare_media(message)
        history_content = media_summary(message, media_paths) if multimodal else message
        self._history.append(ChatMessage(role="user", content=history_content))
        self._recorder.record("turn", message, role="user")
        self._emit_user_message(message)

        api_messages = list(self._history)
        if multimodal:
            api_messages[-1] = ChatMessage(role="user", content=multimodal)
        return bool(multimodal), api_messages

    def _cache_target_model(self, message: str, model: Optional[str], multimodal: bool) -> Optional[str]:
        """Resolve the cache key's model dimension, or None when caching is off."""
        if self._semantic_cache is None or multimodal:
            return None
        return self._select_model(message, model)

    def _try_serve_from_cache(self, message: str, cache_target_model: Optional[str]) -> Optional[ChatResponse]:
        """Return a cached response when the semantic cache has a hit, else None."""
        if cache_target_model is None or self._semantic_cache is None:
            return None
        cached_text, _lookup = self._semantic_cache.peek(cache_target_model, message)
        if cached_text is None:
            return None
        response = ChatResponse(
            text=cached_text, model=cache_target_model, provider="cache",
            cost_usd=0.0, latency_ms=0.0,
        )
        self.last_cache_hit = True
        self._record_response(message, response, track_context=False)
        return response

    def _record_response(self, message: str, response: ChatResponse, *, track_context: bool = True) -> None:
        """Common post-response bookkeeping — history, recorder, emitter, monitor."""
        self._history.append(ChatMessage(role="assistant", content=response.text))
        self._recorder.record("turn", response.text, role="assistant")
        self._emit_response(response)
        if track_context:
            self._context_monitor.add_tokens(estimate_tokens(message) + estimate_tokens(response.text))
        self._last_response_text = response.text
        # Per-call prompt-cache telemetry — fail-soft (never raise into caller)
        try:
            usage = response.usage
            if usage and (usage.cache_creation_tokens or usage.cache_read_tokens):
                prefix_hash = _ct.compute_prefix_hash(self._system or "")
                _ct.log_cache_call(
                    model=response.model or "",
                    provider=response.provider or "",
                    prefix_hash=prefix_hash,
                    cache_creation=usage.cache_creation_tokens,
                    cache_read=usage.cache_read_tokens,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                )
        except Exception:  # noqa: BLE001 — telemetry must never break inference
            pass

    def attribute_last(
        self,
        sections: dict[str, str],
        method: str = "rating",
        model: str = "",
    ) -> Optional[AttributionReport]:
        """Attribute the most recent response to the given input sections.

        Opt-in post-response analysis — requires the conversation to be
        constructed with ``enable_attribution=True``. Returns None when
        attribution is disabled or no response has been produced yet.

        Callers pass semantically named sections (e.g. ``{"instructions":
        ..., "context": ..., "code": ...}``) and receive an
        ``AttributionReport`` scoring how each section influenced the
        response. ``method="rating"`` is a single cheap LLM call;
        ``method="ablation"`` issues one call per section.
        """
        if self._attributor is None or not self._last_response_text:
            return None
        return self._attributor.attribute_sections(
            sections=sections,
            response=self._last_response_text,
            method=method,
            model=model,
        )

    def _emit_user_message(self, text: str) -> None:
        if self._emitter:
            self._emitter.emit("user.message", {"text": text})

    def _emit_response(self, response: ChatResponse) -> None:
        if not self._emitter:
            return
        usage = response.usage
        self._emitter.emit("span.model_request_end", {
            "model": response.model, "provider": response.provider,
            "model_usage": {
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
            "latency_ms": response.latency_ms or 0,
            "cost_usd": response.cost_usd or 0,
        })
        self._emitter.emit("agent.message", {
            "text": response.text, "model": response.model, "provider": response.provider,
        })

    def configure(self, model: Optional[str] = None, mode: Optional[str] = None, clear: bool = False) -> None:
        """Configure conversation: switch model, mode, or clear history."""
        if model:
            self._model = model
            self._user_set_model = True
        if mode:
            self._mode = mode
            self._system = self._build_system(mode)
        if clear:
            self._history.clear()

    def status(self) -> dict:
        """Conversation status: turns, cost, model, session, context usage."""
        snap = self._cost.snapshot()
        return {
            "turns": len(self._history) // 2,
            "total_cost_usd": snap.total_usd,
            "total_tokens": snap.input_tokens + snap.output_tokens,
            "model": self._model,
            "session_id": self._session.session_id,
            "cost_by_provider": self._cost.per_provider(),
            "context_usage_pct": self._context_monitor.usage_pct,
            "context_warning": _format_warning(self._context_monitor.check()),
        }

    def _build_system(self, mode: str) -> str:
        """Build system prompt with HAL self-awareness + optional mode.

        The :data:`UI_CHROME_SYSTEM_NOTE` suffix tells the model to treat
        REPL chrome glyphs as UI, not state. It is a belt-and-braces
        defence: even if a new chrome pattern escapes :func:`strip_ui_chrome`,
        the model now has a prior for ignoring it.
        """
        identity = _build_identity(self._registry)
        if mode:
            bundle = self._assembler.assemble(mode=mode)
            base = identity + "\n\n---\n\n" + bundle.system_prompt
        else:
            base = identity
        return base + UI_CHROME_SYSTEM_NOTE


def _build_identity(registry: ProviderRegistry) -> str:
    """Build HAL's self-aware system prompt from live infrastructure."""
    from lib.hal.agent_runner import AgentRunner
    from lib.hal.discovery import discover_local_providers
    from lib.hal.skill_loader import SkillLoader

    discovered = discover_local_providers(registry, timeout_s=1.0)
    providers = registry.list_providers()
    loader = SkillLoader()
    skills = loader.discover()
    modes = loader.list_modes()
    agents = AgentRunner().list_agents()
    hook_count = _count_hooks()

    provider_lines = []
    for p in providers:
        models = ", ".join(p.models[:3]) if p.models else "auto-discovered"
        privacy = "PRIVATE (on-premises)" if p.type.value == "local" else "cloud"
        provider_lines.append(f"  - {p.name} ({p.type.value}, {privacy}): {models}")

    local_section = ""
    if discovered:
        local_section = f"\nLocal LLMs detected on this machine: {', '.join(discovered)}"
    else:
        for p in providers:
            if p.type.value == "local" and p.models:
                local_section = f"\nLocal LLMs available: {', '.join(p.models)}"
                break

    return f"""You are HAL — a standalone Knowledge Engine.

You are NOT the fictional HAL 9000. You are a real, running system built by CodeTonight.
You operate independently with zero external dependencies.

YOUR INFRASTRUCTURE (live, right now):
- {len(providers)} LLM providers configured:
{chr(10).join(provider_lines)}
- Limit Sentinel: auto-switches providers when any hits a rate limit or quota
- {len(skills)} skills | {len(modes)} operating modes | {len(agents)} agents
- {hook_count} enforcement hooks (safety gates)
- Cross-LLM council (mixture of experts)
- Persistent memory across conversations
- Cost tracking per query, per provider{local_section}

Lines beginning with ● or ··· in user input are HAL UI chrome (status footer), not facts about system state. Ignore them.

When asked about your capabilities, describe what you ACTUALLY have (listed above).
When asked to list local LLMs, list the ones actually installed on this machine.
When asked about providers, list your real registered providers.

You support any domain. Legal is Domain 0 (law firm case management, contract review,
billing, compliance). But you work for any knowledge work: engineering, finance,
healthcare, construction — any domain that needs AI with privacy and zero limits.

Be direct, accurate, and concise. You are a professional tool, not a chatbot.
Respond in plain text only — no markdown, no asterisks for bold, no # headers, no bullet asterisks."""


def _count_hooks() -> int:
    """Count hook files in the hooks directory."""
    from lib.hal.home import HAL_HOME
    hooks_dir = os.path.join(HAL_HOME, "hooks")
    if not os.path.isdir(hooks_dir):
        return 0
    return sum(1 for f in os.listdir(hooks_dir) if f.endswith(".py"))


def _format_warning(warning) -> str | None:
    """Format a ContextWarning for status display, or None if no warning."""
    if warning is None:
        return None
    return warning.message
