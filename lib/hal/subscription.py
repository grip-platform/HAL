"""Subscription-based authentication for LLM providers.

HAL treats LLM access as one of two economic models:

1. **API key** (per-token billing) — the classic "sk-..." style key. Handled
   directly by each provider (:class:`~lib.hal.providers.anthropic_native.AnthropicProvider`,
   :class:`~lib.hal.providers.openai_compat.OpenAICompatProvider`, etc.) and
   the cost is accounted per-request by :mod:`lib.hal.cost_tracker`.

2. **Subscription** (flat-fee, quota-metered) — a Pro/Max plan the user
   already pays for. Examples: Claude Code subscription (via CCH header
   signing), the Kimi Code membership (Moonshot, GRIP Pool D), Alibaba Qwen
   Pro, Z.ai GLM coding plan. Cost is zero per request (already prepaid),
   but quota is finite.

This module exposes a thin :class:`SubscriptionAuth` protocol that providers
can accept as an alternative transport. When a provider has both an API key
and a subscription configured, it picks one per request based on the ``prefer``
argument — ``"auto"`` (subscription first, fall back to api_key), ``"api_key"``
(force), or ``"subscription"`` (force).

Design constraint: a subscription OWNS the entire request lifecycle (auth +
send + parse). It must return a normalised :class:`~lib.hal.types.ChatResponse`
so callers upstream of the provider do not need to branch on transport.

Usage:
    from lib.hal.subscription import CCHSubscription
    from lib.hal.providers.anthropic_native import AnthropicProvider

    sub = CCHSubscription.from_env()
    provider = AnthropicProvider(subscription=sub)
    response = provider.chat("Hello", model="claude-opus-4-7")
"""
import os
import re
import subprocess
from types import MappingProxyType
from typing import Optional, Protocol, runtime_checkable

from lib.hal.types import ChatMessage, ChatResponse


@runtime_checkable
class SubscriptionAuth(Protocol):
    """Protocol every subscription transport must implement."""

    name: str
    provider_id: str

    def chat(
        self,
        prompt: str,
        *,
        model: str,
        messages: Optional[list[ChatMessage]] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 16384,
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        ...

    def health_check(self) -> bool:
        ...


_WHITESPACE_RE = re.compile(r"\s+")


def _scrub(raw: str) -> str:
    """Remove ALL whitespace from a token string.

    Embedded newlines are a real-world failure mode: clipboard word-wrap,
    environment variables set via multi-line shell heredocs, and keychain
    reads can all inject ``\\n`` into a token string. HTTP header values
    containing ``\\n`` are rejected by ``http.client.putheader`` with
    ``ValueError: Invalid header value`` — so scrubbing defensively here
    turns the corruption into a valid request rather than a late crash.
    """
    return _WHITESPACE_RE.sub("", raw) if raw else raw


def _fetch_keychain_secret(service: str) -> str:
    """Return password stored under service in macOS Keychain, or empty string.

    ``security find-generic-password -w`` hex-encodes the password when it
    contains bytes that stringify ambiguously (observed on macOS 15+ for
    tokens containing certain character classes, no ``0x`` prefix). Decode
    if the output looks like a pure even-length hex string — real tokens
    contain dashes and underscores that can't appear in a hex string.
    All returned values are passed through :func:`_scrub` to strip any
    embedded whitespace from the keychain write.
    """
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-w", "-s", service],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return ""
        raw = result.stdout.strip()
        if raw and len(raw) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in raw):
            try:
                return _scrub(bytes.fromhex(raw).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return _scrub(raw)
        return _scrub(raw)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def _fetch_token(env_vars: tuple[str, ...], keychain_service: str) -> str:
    """Read a secret from the first matching env var, falling back to macOS Keychain.

    ``env_vars`` is tried in order — this lets callers prefer a canonical
    upstream name (``CLAUDE_CODE_OAUTH_TOKEN``) while still honouring the
    legacy HAL-internal name (``CLAUDE_OAUTH_TOKEN``) for back-compat.
    All returned values are whitespace-scrubbed via :func:`_scrub`.
    """
    for name in env_vars:
        token = _scrub(os.environ.get(name, ""))
        if token:
            return token
    return _fetch_keychain_secret(keychain_service)


class CCHSubscription:
    """Anthropic Messages API via Claude Code subscription (CCH signing).

    Every caller of :meth:`chat` is one of HAL's own calls, so when HAL's API-key
    route is open (``lib/hal/anthropic_key_route.py``: opted in, a daily cap with
    room left, a priced model and the keychain item) the call goes on the API key,
    with prompt caching, instead of spending the subscription windows the
    operator's Claude Code sessions need. Otherwise it is signed with CCH exactly
    as before. The route ships closed.
    """

    provider_id = "anthropic"
    name = "cc-subscription"

    def __init__(self, oauth_token: str, *, cc_version: Optional[str] = None):
        from lib.hal.utils.cch import CCHSigner
        self._token = oauth_token
        self._signer = CCHSigner(cc_version=cc_version)

    @classmethod
    def from_env(cls, *, env_var: Optional[str] = None,
                 keychain_service: str = "grip-claude-oauth",
                 cc_version: Optional[str] = None) -> Optional["CCHSubscription"]:
        """Construct from env var or macOS Keychain. Returns None if no token.

        Lookup order:
        1. ``CLAUDE_CODE_OAUTH_TOKEN`` (canonical Anthropic name — also used
           by the official ``claude-code-action`` GitHub Action, so the same
           secret works in CI with no extra mapping).
        2. ``CLAUDE_OAUTH_TOKEN`` (legacy HAL-internal name, kept for
           back-compat with any operator rc files or scripts that still
           export it).
        3. macOS Keychain service ``grip-claude-oauth``.

        Pass ``env_var=`` to override the lookup order with a single
        explicit variable name.
        """
        env_vars = (env_var,) if env_var else ("CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_OAUTH_TOKEN")
        token = _fetch_token(env_vars, keychain_service)
        return cls(token, cc_version=cc_version) if token else None

    def chat(
        self,
        prompt: str = "",
        *,
        model: str,
        messages: Optional[list[ChatMessage]] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 16384,
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """Send a single- or multi-turn request through the CCH signer.

        Multi-turn: pass ``messages=[ChatMessage(...), ...]``; the signer
        serialises each message's ``content`` field verbatim so assistant
        thinking blocks with signatures round-trip correctly. Single-turn:
        pass ``prompt="..."`` (legacy form) and the signer synthesises a
        one-element messages array internally.

        ``tools`` (optional): Anthropic tool schemas from
        :func:`~lib.hal.tool_adapter.to_anthropic_tools`. When non-empty,
        the signer includes them in the body so the model can emit
        ``tool_use`` content blocks for :func:`run_tool_loop`. The
        returned :class:`ChatResponse`'s ``raw`` attribute carries the
        full Anthropic response dict including any ``tool_use`` blocks —
        use that as the chat_fn return path when driving a tool loop.

        HAL's API-key route is asked first (class docstring); it returns None
        whenever the call should be signed with CCH, the default.
        """
        from lib.hal import anthropic_key_route  # lazy: it imports the native provider, which imports this module
        routed = anthropic_key_route.try_api_key(
            prompt, model=model, messages=messages, system=system,
            max_tokens=max_tokens, tools=tools,
        )
        if routed is not None:
            return routed
        return self._signer.call(
            prompt,
            oauth_token=self._token,
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            tools=tools,
        )

    def health_check(self) -> bool:
        return bool(self._token)


class OpenAICompatSubscription:
    """OpenAI-compatible subscription transport.

    Wraps an :class:`~lib.hal.providers.openai_compat.OpenAICompatProvider`
    configured with the subscription's long-lived token as the Bearer
    credential. Gives us a real ``.chat()`` (POST /v1/chat/completions)
    and a real ``.health_check()`` (GET /v1/models) without duplicating
    HTTP retry / key scrubbing / error translation — all that logic
    already lives in :class:`OpenAICompatProvider`.

    Concrete subclasses declare:
        base_url           endpoint root (e.g. https://api.moonshot.ai/v1)
        provider_id        stable id consumed by :class:`ProviderRegistry`
        name               human-readable transport label
        env_vars           tuple of env var names to try in order
        keychain_service   macOS Keychain service name for the token

    This matches the :class:`SubscriptionAuth` protocol so it can be
    used anywhere a subscription transport is accepted.
    """

    provider_id: str = ""
    name: str = ""
    base_url: str = ""
    env_vars: tuple[str, ...] = ()
    keychain_service: str = ""

    def __init__(self, token: str):
        self._token = token
        # Deferred import keeps subscription.py free of network-adapter
        # imports at module load time (tests import this module a lot).
        self._provider = None

    @classmethod
    def from_env(cls) -> Optional["OpenAICompatSubscription"]:
        """Resolve a token from env vars or macOS Keychain.

        Each subclass specifies ``env_vars`` as a **tuple** (not a bare
        string — that iterates characters, a latent bug fixed here).
        """
        token = _fetch_token(cls.env_vars, cls.keychain_service)
        return cls(token) if token else None

    def _get_provider(self):
        """Lazy-construct the underlying OpenAICompatProvider."""
        if self._provider is None:
            from lib.hal.providers.openai_compat import OpenAICompatProvider
            self._provider = OpenAICompatProvider(
                name=self.name,
                base_url=self.base_url,
                api_key=self._token,
            )
        return self._provider

    def chat(self, prompt: str = "", **kwargs) -> ChatResponse:
        """Pure forwarding to the underlying OpenAI-compat provider.

        Keeps the :class:`SubscriptionAuth` protocol contract (``model``
        is required as a keyword) without re-declaring every chat arg —
        :class:`OpenAICompatProvider.chat` already validates them.
        """
        return self._get_provider().chat(prompt, **kwargs)

    def health_check(self) -> bool:
        if not self._token:
            return False
        return self._get_provider().health_check()


class KimiCodingSubscription(OpenAICompatSubscription):
    """Kimi Code — Moonshot's Kimi membership coding service. GRIP Pool D.

    Kimi Code is a SEPARATE product from the pay-per-token Kimi API platform
    (``providers.yaml`` provider ``moonshot``, keychain ``grip-moonshot``,
    GRIP Pool B): "the Kimi Code API is independent from the API service on
    this platform" (platform.kimi.ai/docs/guide/product-plans). Its own host,
    credential, model ids and bill (kimi.com/code/docs/en/kimi-code/models,
    /kimi-code/membership, /third-party-tools/claude-code, /opencode):

        OpenAI-compatible base     https://api.kimi.com/coding/v1
        Anthropic-compatible base  https://api.kimi.com/coding/
        model ids                  k3 (1M on Allegretto and above, else 262144),
                                   k3-256k (262144, image only),
                                   kimi-for-coding (262144, output 32768),
                                   kimi-for-coding-highspeed (262144,
                                   "6× speed, 3× quota usage")
        credential                 a Kimi Code Console API key ("up to 5 keys;
                                   each key is shown only once at creation");
                                   Claude Code reads it from ANTHROPIC_API_KEY
                                   on this endpoint, NOT ANTHROPIC_AUTH_TOKEN
        bill                       membership quota, not per-token: "refreshes
                                   automatically every 7 days ... a rolling
                                   5-hour rate window ... Requests from the
                                   CLI, VS Code, and third-party tools all
                                   count toward that quota"; overage is
                                   "close to the official API pricing"
        401                        can mean "the requested capability exceeds
                                   your plan's entitlements", not a bad key

    Pool D per the sovereign council of 2026-09-07 (seal ``e49c00c214c500be``,
    IDR leaf ``precog-1788805766-c58539b1``): pools are defined by who bills.
    The metered Kimi API stays Pool B; neither is Pool C (the ChatGPT
    subscription).

    UNARMED BY CONSTRUCTION. :meth:`from_env` resolves ``KIMI_CODING_API_KEY``
    and then macOS keychain ``grip-kimi-coding`` — the same two names the
    GRIP side uses (``lib/provider_secret.resolve_provider_key("kimi-coding")``,
    GRIP#5336), so both repos arm off one credential. Neither exists on any
    machine today, so :func:`auto_detect_subscriptions` never registers this
    transport and :func:`lib.hal.raillm_ladder.ladder_addresses` never adds
    its rung. Arming is the operator's step::

        security add-generic-password -s grip-kimi-coding -a hal -w

    Replaces ``KimiK2Subscription`` (keychain ``grip-moonshot-subscription``,
    env ``MOONSHOT_SUBSCRIPTION_TOKEN``), which pointed a "subscription" at
    the pay-per-token host — a model of a product that does not exist. The
    ``MOONSHOT_BASE_URL`` .cn/.ai escape hatch went with it: it belonged to
    the metered platform, and Kimi Code has one host. The Kimi
    locked-temperature quirk stays at the adapter seam
    (``lib/hal/model_quirks.py``); ``chat()`` is inherited unchanged.
    """

    provider_id = "kimi-coding"
    name = "kimi-coding-subscription"
    base_url = "https://api.kimi.com/coding/v1"
    #: The vendor's Anthropic-compatible base for the SAME credential — what a
    #: Claude Code launch sets as ANTHROPIC_BASE_URL. Not used by this
    #: OpenAI-compat transport; recorded here so the gateway's Anthropic
    #: same-wire table has one source to mirror.
    anthropic_base_url = "https://api.kimi.com/coding/"
    env_vars = ("KIMI_CODING_API_KEY",)
    keychain_service = "grip-kimi-coding"
    #: GRIP spend pool this transport bills (rules/pool-a-offload-routing.md).
    pool = "D"
    #: The vendor's stated quota shape (membership page). Read-only so a caller
    #: cannot quietly redefine the window it is budgeting against.
    quota = MappingProxyType({"window_days": 7, "rolling_hours": 5})


class QwenSubscription(OpenAICompatSubscription):
    """Alibaba Qwen subscription over the DashScope OpenAI-compatible API."""

    provider_id = "alibaba"
    name = "qwen-subscription"
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    env_vars = ("DASHSCOPE_SUBSCRIPTION_TOKEN", "DASHSCOPE_API_KEY")
    keychain_service = "hal-dashscope-subscription"


def auto_detect_subscriptions() -> dict[str, SubscriptionAuth]:
    """Discover subscriptions from env vars and keychain."""
    detected: dict[str, SubscriptionAuth] = {}
    cch = CCHSubscription.from_env()
    if cch and cch.health_check():
        detected[cch.provider_id] = cch
    kimi_code = KimiCodingSubscription.from_env()
    if kimi_code and kimi_code.health_check():
        detected[kimi_code.provider_id] = kimi_code
    qwen = QwenSubscription.from_env()
    if qwen and qwen.health_check():
        detected[qwen.provider_id] = qwen
    return detected
