"""HAL Limit Sentinel — Zero limits through proactive detection and auto-failover.

Monitors provider utilisation via response headers, tracks limit state per
provider, and auto-routes to the next available provider before limits hit.

The user NEVER sees "You've hit your limit."

Usage:
    from lib.hal.sentinel import LimitSentinel

    sentinel = LimitSentinel(registry)
    response = sentinel.dispatch("anthropic/claude-opus-4-7", "Hello")
    # If Anthropic is near limit, transparently routes to Groq/Gemini/Ollama
"""
import json
import os
import threading
import time
from typing import Optional

from lib.hal.types import ChatResponse

from lib.hal.home import HAL_HOME as _HAL_HOME
_STATE_PATH = os.path.join(_HAL_HOME, "state", "hal", "limits.json")

# Default thresholds
_PROACTIVE_THRESHOLD = 0.85   # Auto-switch at 85% utilisation
_WARNING_THRESHOLD = 0.70     # Alert at 70%
_COOLDOWN_BASE_S = 60.0       # Base cooldown after 429
_COOLDOWN_MAX_S = 3600.0      # Max cooldown (1 hour)
_COOLDOWN_MULTIPLIER = 2.0    # Exponential backoff


class LimitSentinel:
    """Monitors limits across all providers and auto-routes around them."""

    def __init__(
        self,
        registry: "ProviderRegistry",
        proactive_threshold: float = _PROACTIVE_THRESHOLD,
        warning_threshold: float = _WARNING_THRESHOLD,
    ):
        self._registry = registry
        self._proactive = proactive_threshold
        self._warning = warning_threshold
        self._state: dict[str, ProviderLimitState] = {}
        # Each entry as this sentinel last read or wrote it: _save_state writes only
        # the entries that differ, so it never puts back a stale copy of an entry
        # another process wrote since.
        self._written: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load_state()

    def dispatch(
        self,
        model_address: str,
        prompt: str,
        fallback_chain: Optional[list[str]] = None,
        no_failover: bool = False,
        **kwargs,
    ) -> ChatResponse:
        """Route query with limit-aware failover.

        1. Pre-check: is the target provider near its limit?
        2. If yes: route to best available alternative
        3. On 429/402: catch, record, retry on next provider
        4. Post-response: update utilisation from headers

        ``no_failover=True`` tries the requested provider alone, so a pinned
        turn never reaches a provider the caller did not name (HAL#673).
        """
        from lib.hal.providers.openai_compat import OpenAICompatError
        from lib.hal.registry import parse_model_address

        provider_name, model_name = parse_model_address(model_address)

        # Build attempt chain: requested provider first, then fallbacks
        chain = ([provider_name] if no_failover
                 else self._build_chain(provider_name, fallback_chain))
        errors = []

        for attempt_provider in chain:
            provider = self._registry.get(attempt_provider)
            if not provider:
                # Recorded, as the RAILLM cascade does, so a pinned turn whose
                # provider is unregistered is refused with its reason.
                errors.append(f"{attempt_provider}: provider not registered")
                continue

            # Pre-check: skip providers near their limit
            if self._is_near_limit(attempt_provider):
                errors.append(f"{attempt_provider}: near limit, skipped")
                continue

            # Resolve model for this provider
            attempt_model = model_name if attempt_provider == provider_name else self._default_model(attempt_provider)
            if not attempt_model:
                continue

            try:
                response = provider.chat(prompt, model=attempt_model, **kwargs)
                self._post_response(attempt_provider, response)
                # H338+H347: signal fallback via typed ChatResponse.fallback_info
                # field (canonical) plus a best-effort mirror in response.raw
                # for backward-compat callers. AGORA Council #311 surfaced that
                # the magic dict key alone is fragile when raw is MappingProxyType
                # or any non-mutable mapping — typed field survives those shapes.
                if attempt_provider != provider_name:
                    from dataclasses import replace as _replace
                    from lib.hal.types import FallbackInfo
                    info = FallbackInfo(
                        requested_provider=provider_name,
                        requested_model=model_name,
                        actual_provider=attempt_provider,
                        actual_model=attempt_model,
                        errors_along_way=tuple(errors),
                    )
                    response = _replace(response, fallback_info=info)
                    # Best-effort raw mirror — skip if raw is non-mutable.
                    if isinstance(response.raw, dict):
                        try:
                            response.raw["_hal_fallback"] = {
                                "requested_provider": provider_name,
                                "requested_model": model_name,
                                "actual_provider": attempt_provider,
                                "actual_model": attempt_model,
                                "errors_along_way": list(errors),
                            }
                        except (TypeError, AttributeError):
                            # Non-mutable mapping — typed field still set above.
                            pass
                return response
            except OpenAICompatError as e:
                self._on_error(attempt_provider, e)
                errors.append(f"{attempt_provider}: {e}")
                continue
            except (ConnectionError, TimeoutError, OSError) as e:
                errors.append(f"{attempt_provider}: {type(e).__name__}: {e}")
                continue

        raise RuntimeError(
            f"All providers exhausted or at limit: {'; '.join(errors)}"
        )

    def get_rate_limit_info(self, provider: str) -> "RateLimitInfo":
        """Return structured RateLimitInfo for a provider (CC SDKRateLimitInfo pattern)."""
        from lib.hal.rate_limiter import RateLimitInfo
        with self._lock:
            state = self._state.get(provider)
            if not state:
                return RateLimitInfo(provider=provider)
            return RateLimitInfo(
                utilization=state.utilisation,
                reset_time_ms=int((state.resets_at or 0) * 1000),
                overage=state.status == "rejected" or state.rpm_remaining == 0,
                provider=provider,
            )

    def get_status(self) -> dict[str, dict]:
        """Return current limit state for all tracked providers."""
        with self._lock:
            return {
                name: state.to_dict()
                for name, state in self._state.items()
            }

    def is_available(self, provider_name: str) -> bool:
        """Check if a provider is currently available (not at limit)."""
        return not self._is_near_limit(provider_name)

    def reset(self, provider_name: str) -> None:
        """Manually reset a provider's limit state (e.g., after credit top-up)."""
        with self._lock:
            if provider_name in self._state:
                self._state[provider_name] = ProviderLimitState()
            self._save_state()

    def _build_chain(self, primary: str, fallback: Optional[list[str]]) -> list[str]:
        """Build ordered provider chain: primary -> fallbacks -> all others."""
        chain = []
        if primary:
            chain.append(primary)

        if fallback:
            chain.extend(p.split("/")[0] for p in fallback if p.split("/")[0] not in chain)
        else:
            routing = self._registry.routing
            raw_chain = routing.fallback_chain
            if not isinstance(raw_chain, list):
                raw_chain = []
            chain.extend(
                p.split("/")[0] for p in raw_chain
                if p.split("/")[0] not in chain
            )

        # Add any registered provider not yet in chain
        for info in self._registry.list_providers():
            if info.name not in chain:
                chain.append(info.name)

        return chain

    def _is_near_limit(self, provider_name: str) -> bool:
        """Check if provider is near its limit (proactive detection)."""
        with self._lock:
            state = self._state.get(provider_name)
            if not state:
                return False

            now = time.time()

            # Check cooldown from previous 429
            if state.cooldown_until and now < state.cooldown_until:
                return True

            # Check hard rejection
            if state.status == "rejected":
                # A rejection is held by whichever expiry evidence is still live:
                # resets_at (the provider's own reset header) or cooldown_until
                # (HAL's backoff). Reaching this line means cooldown_until has
                # lapsed or was never set, so only a FUTURE resets_at can still
                # hold it. Before HAL#612 the release was gated on resets_at
                # having PASSED -- a field only Anthropic's headers populate --
                # so every other provider stayed rejected for ever once its
                # cooldown lapsed (openrouter: 182h against a 1h maximum), while
                # anthropic recovered from the same file. Absence of a reset
                # header is not evidence of an ongoing limit.
                if state.resets_at and now < state.resets_at:
                    return True
                state.status = "allowed"
                state.utilisation = 0.0
                state.cooldown_until = None
                self._save_state()
                return False

            # Proactive: check utilisation threshold
            if state.utilisation >= self._proactive:
                return True

            return False

    def _post_response(self, provider_name: str, response: ChatResponse) -> None:
        """Extract limit info from response headers and update state."""
        raw = response.raw or {}
        headers = raw.get("_headers", {})
        if not headers:
            return

        with self._lock:
            state = self._get_or_create(provider_name)
            _update_from_headers(state, headers)
            self._save_state()

    def _on_error(self, provider_name: str, error: Exception) -> None:
        """Handle limit-related errors."""
        error_str = str(error).lower()
        # Structured billing-exhaustion check takes precedence over the string
        # heuristics below: openai_compat attaches error_code="insufficient_quota"
        # on a billing 429. That is a hard billing cap — never retryable — so
        # reject the provider for the MAX cooldown (it recovers on a
        # human-timescale billing top-up), NOT the exponential-backoff transient
        # path. Per rate-limit-never-blocks.md (usage-cap vs load-shed). Additive:
        # only fires when the attribute is present, so no other provider/error
        # path changes behaviour.
        with self._lock:
            # Read the structured code inside the lock too (harmless — it reads
            # the exception, not shared state — but keeps the whole rejected-state
            # assignment unambiguously lock-guarded).
            error_code = getattr(error, "error_code", None)
            state = self._get_or_create(provider_name)

            if error_code == "insufficient_quota":
                state.status = "rejected"
                state.cooldown_until = time.time() + _COOLDOWN_MAX_S

            elif "429" in error_str or "rate" in error_str:
                state.consecutive_429s += 1
                state.last_429_at = time.time()
                backoff = min(
                    _COOLDOWN_BASE_S * (_COOLDOWN_MULTIPLIER ** (state.consecutive_429s - 1)),
                    _COOLDOWN_MAX_S,
                )
                state.cooldown_until = time.time() + backoff

            elif "402" in error_str or "credit" in error_str or "payment" in error_str:
                state.status = "rejected"
                state.cooldown_until = time.time() + _COOLDOWN_MAX_S

            elif "quota" in error_str or "limit" in error_str:
                state.status = "rejected"
                retry_after = _extract_retry_after(error_str)
                state.cooldown_until = time.time() + (retry_after or _COOLDOWN_BASE_S)

            self._save_state()

    def _default_model(self, provider_name: str) -> str:
        """Get default model for a provider from config."""
        config = self._registry._configs.get(provider_name)
        if config and config.models:
            return config.models[0]
        return ""

    def _get_or_create(self, name: str) -> "ProviderLimitState":
        if name not in self._state:
            self._state[name] = ProviderLimitState()
        return self._state[name]

    def _load_state(self) -> None:
        """Load persistent limit state from disk."""
        if not os.path.exists(_STATE_PATH):
            return
        try:
            with open(_STATE_PATH, "r") as f:
                data = json.load(f)
            with self._lock:
                for name, raw in data.items():
                    self._state[name] = ProviderLimitState.from_dict(raw)
                    self._written[name] = self._state[name].to_dict()
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    def _save_state(self) -> None:
        """Persist the entries this sentinel changed: one locked, atomic upsert.

        ``limits.json`` may have more than one writer -- this sentinel
        (hal-server) plus any future state writer, each with its own entries.
        (A second writer, the gateway's key failover, existed until the spend
        guard removed it, Step 5h.) This used to rewrite the whole file from
        memory with a plain ``open(..., "w")``: a reader could catch it
        half-written, and the other writer's newer entries were put back as
        this process's stale copies (fleet finding f96). Now only changed
        entries are merged in, under ``locked_json``'s lock, and the file is
        replaced whole.
        """
        from lib.hal.locked_json import update_json  # lazy: sentinel imports stay light
        changed = {name: state.to_dict() for name, state in self._state.items()
                   if state.to_dict() != self._written.get(name)}
        if not changed:
            return
        try:
            update_json(_STATE_PATH, lambda current: {**current, **changed}, mode=0o644)
        except OSError:
            return
        self._written.update(changed)


class ProviderLimitState:
    """Mutable limit state for a single provider."""

    def __init__(self) -> None:
        self.utilisation: float = 0.0
        self.status: str = "allowed"
        self.resets_at: Optional[float] = None
        self.cooldown_until: Optional[float] = None
        self.consecutive_429s: int = 0
        self.last_429_at: Optional[float] = None
        self.rpm_remaining: Optional[int] = None
        self.tpm_remaining: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "utilisation": self.utilisation,
            "status": self.status,
            "resets_at": self.resets_at,
            "cooldown_until": self.cooldown_until,
            "consecutive_429s": self.consecutive_429s,
            "last_429_at": self.last_429_at,
            "rpm_remaining": self.rpm_remaining,
            "tpm_remaining": self.tpm_remaining,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProviderLimitState":
        state = cls()
        state.utilisation = data.get("utilisation", 0.0)
        state.status = data.get("status", "allowed")
        state.resets_at = data.get("resets_at")
        state.cooldown_until = data.get("cooldown_until")
        state.consecutive_429s = data.get("consecutive_429s", 0)
        state.last_429_at = data.get("last_429_at")
        state.rpm_remaining = data.get("rpm_remaining")
        state.tpm_remaining = data.get("tpm_remaining")
        return state


def _update_from_headers(state: ProviderLimitState, headers: dict) -> None:
    """Update limit state from response headers."""
    # Anthropic utilisation headers (verified from CC source)
    for window in ("5h", "7d"):
        util_key = f"anthropic-ratelimit-unified-{window}-utilization"
        reset_key = f"anthropic-ratelimit-unified-{window}-reset"

        if util_key in headers:
            util = float(headers[util_key])
            state.utilisation = max(state.utilisation, util)

        if reset_key in headers:
            state.resets_at = float(headers[reset_key])

    # Standard rate limit headers (OpenAI format, used by most providers)
    if "x-ratelimit-remaining-requests" in headers:
        state.rpm_remaining = int(headers["x-ratelimit-remaining-requests"])
    if "x-ratelimit-remaining-tokens" in headers:
        state.tpm_remaining = int(headers["x-ratelimit-remaining-tokens"])

    # Retry-After header (RFC 7231) — server tells us when to retry
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after:
        try:
            state.cooldown_until = time.time() + float(retry_after)
            return  # Don't reset cooldown if server says wait
        except ValueError:
            pass  # Non-numeric Retry-After (date format) — ignore for now

    # Reset consecutive 429 count on successful response
    state.consecutive_429s = 0
    state.cooldown_until = None
    state.status = "allowed"


def _extract_retry_after(error_text: str) -> Optional[float]:
    """Try to extract a retry-after value from error text."""
    import re
    match = re.search(r"retry.?after[:\s]+(\d+)", error_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None
