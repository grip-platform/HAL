"""Goodhart-proof tests for the happi `infer` cmd (H-RAILLM-3).

Driven through the REAL ``runtime.run`` (StringIO in/out) -- behaviour, not
mocks of the runtime. HERMETIC: a 2-rung test equivalence map (fake
anthropic + moonshot only) so the cascade can never fall through to a real
network provider (deterministic; no keychain/network).

Verifies:
  * infer cascades CCH -> Kimi over the ladder, streams happi events, and
    reports the failover (derived from provider mismatch -- RAILLM does NOT
    set response.fallback_info)
  * flags.audit=true yields a DETERMINISTIC idr terminator (time pinned)
  * full exhaustion -> runtime_error ; missing prompt -> parse_error
"""
import io
import json

import lib.hal.happi.cmds_infer as cmds_infer
from lib.hal.happi import runtime
from lib.hal.providers.openai_compat import OpenAICompatError
from lib.hal.raillm import ModelEquivalenceClass, ModelTier, RAILLMSentinel
from lib.hal.raillm_subscription_bridge import register_subscriptions
from lib.hal.registry import ProviderRegistry
from lib.hal.types import ChatResponse, TokenUsage

_A = "anthropic/claude-opus-4-7"
_K = "moonshot/kimi-k2"
_O = "openai/gpt-5.5"            # the misrouted exogenous-council member
_G = "gemini/gemini-2.5-flash"   # the rung that silently served it


class _Sub:
    def __init__(self, pid, *, reply=None, raise_exc=None):
        self.provider_id = pid
        self._reply = reply
        self._raise = raise_exc

    def chat(self, prompt="", *, model, system=None, temperature=None,
             max_tokens=2048, messages=None, tools=None):
        if self._raise is not None:
            raise self._raise
        return ChatResponse(text=self._reply, model=model,
                            provider=self.provider_id, usage=TokenUsage())

    def health_check(self):
        return True


def _emap():
    """2-rung map: cascade is EXACTLY [anthropic, moonshot] -- hermetic."""
    return {
        ModelTier.FRONTIER: ModelEquivalenceClass(
            tier=ModelTier.FRONTIER, members=[_A, _K],
            context_window=200_000),
        ModelTier.FAST: ModelEquivalenceClass(ModelTier.FAST, [], 128_000),
        ModelTier.FREE_TIER: ModelEquivalenceClass(
            ModelTier.FREE_TIER, [], 128_000),
        ModelTier.LOCAL: ModelEquivalenceClass(ModelTier.LOCAL, [], 32_000),
    }


def _sentinel(tmp_path, monkeypatch, *, all_fail=False):
    monkeypatch.setattr(
        "lib.hal.sentinel._STATE_PATH", str(tmp_path / "limits.json")
    )
    reg = ProviderRegistry()
    cch = _Sub("anthropic", raise_exc=OpenAICompatError("cch limit"))
    if all_fail:
        kimi = _Sub("moonshot", raise_exc=OpenAICompatError("kimi down"))
    else:
        kimi = _Sub("moonshot", reply="answer from kimi")
    register_subscriptions(reg, {"anthropic": cch, "moonshot": kimi})
    return RAILLMSentinel(registry=reg, equivalence_map=_emap())


def _run(envelope: dict) -> list[dict]:
    out = io.StringIO()
    rc = runtime.run(io.StringIO(json.dumps(envelope)), out)
    assert rc == 0
    return [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]


def test_infer_cascades_cch_to_kimi_through_real_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _sentinel(tmp_path, monkeypatch),
    )
    events = _run({
        "v": "happi/1.1", "id": "i1", "cmd": "infer",
        "flags": {"model": _A, "prompt": "2+2?"},
    })
    types = [e["type"] for e in events]
    assert types == ["started", "delta", "completed"]
    assert events[1]["text"] == "answer from kimi"          # failed over to Kimi
    usage = events[2].get("usage", {})
    assert usage.get("provider") == "moonshot"
    assert usage.get("requested") == _A
    assert usage.get("failover") == {                       # derived from mismatch
        "requested": "anthropic", "served_by": "moonshot"}


def test_infer_audit_idr_is_deterministic(tmp_path, monkeypatch):
    """H-RAILLM-3: identical envelope + event sequence -> identical idr
    sha256. Time pinned so the only variable is the (identical) content."""
    monkeypatch.setattr("lib.hal.happi.events._now_ms", lambda: 1234)
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _sentinel(tmp_path, monkeypatch),
    )
    env = {
        "v": "happi/1.1", "id": "aud", "cmd": "infer",
        "flags": {"model": _A, "prompt": "hello", "audit": True},
    }
    first = _run(env)
    second = _run(env)

    for stream in (first, second):
        assert stream[-1]["type"] == "idr"                  # audit terminator
        assert stream[-2]["type"] == "completed"            # idr AFTER completed
    sha1 = first[-1]["idr_ref"]["sha256"]
    sha2 = second[-1]["idr_ref"]["sha256"]
    assert sha1 == sha2 and len(sha1) == 64                 # deterministic


def test_infer_full_exhaustion_is_runtime_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _sentinel(tmp_path, monkeypatch, all_fail=True),
    )
    events = _run({
        "v": "happi/1.1", "id": "x", "cmd": "infer",
        "flags": {"model": _A, "prompt": "hi"},
    })
    err = [e for e in events if e["type"] == "error"]
    assert err and err[0]["code"] == "runtime_error"
    assert "exhausted" in err[0]["message"]


def test_infer_missing_prompt_is_parse_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _sentinel(tmp_path, monkeypatch),
    )
    events = _run({"v": "happi/1.1", "id": "np", "cmd": "infer", "flags": {}})
    err = [e for e in events if e["type"] == "error"]
    assert err and err[0]["code"] == "parse_error"


def test_infer_multimodal_routes_content_parts_to_provider(tmp_path, monkeypatch):
    """flags.content_parts -> the messages path: the provider receives the
    parts AS-IS (audio not flattened/dropped), and an EMPTY prompt is accepted
    because the parts carry the turn (contrast: the no-parts case above is a
    parse_error)."""
    captured: dict = {}

    class _Cap:
        provider_id = "anthropic"

        def chat(self, prompt="", *, model, system=None, temperature=None,
                 max_tokens=2048, messages=None, tools=None):
            captured["prompt"] = prompt
            captured["messages"] = messages
            return ChatResponse(text="heard", model=model,
                                provider="anthropic", usage=TokenUsage())

        def health_check(self):
            return True

    def _mk():
        monkeypatch.setattr(
            "lib.hal.sentinel._STATE_PATH", str(tmp_path / "limits.json")
        )
        reg = ProviderRegistry()
        register_subscriptions(reg, {"anthropic": _Cap()})
        return RAILLMSentinel(registry=reg, equivalence_map=_emap())

    monkeypatch.setattr(cmds_infer, "make_grip_sentinel", _mk)

    parts = [
        {"type": "input_audio", "input_audio": {"data": "AAA=", "format": "wav"}},
        {"type": "text", "text": "describe"},
    ]
    events = _run({
        "v": "happi/1.1", "id": "mm", "cmd": "infer",
        "flags": {"model": _A, "prompt": "", "content_parts": parts},
    })
    assert [e["type"] for e in events] == ["started", "delta", "completed"]
    assert events[1]["text"] == "heard"
    assert not any(e["type"] == "error" for e in events)    # empty prompt OK w/ parts
    assert captured["messages"] is not None                 # messages transport
    assert captured["messages"][0].content == parts         # parts reached provider as-is


# ---------------------------------------------------------------------------
# Provider-pin tests — the openai/gpt-5.5 -> gemini misroute (bug #2)
#
# Hermetic reproduction of the LIVE misroute: an explicit ``openai/gpt-5.5``
# whose openai rung FAILS is silently failed-over to gemini by the RAILLM
# cascade. A 2-rung FAST map [openai, gemini] with the openai rung raising
# makes the cascade EXACTLY openai -> gemini (no network, no keychain).
# ---------------------------------------------------------------------------


def _pin_emap(fast_members):
    """Hermetic map: the FAST tier is the whole cascade; all others empty."""
    return {
        ModelTier.FRONTIER: ModelEquivalenceClass(ModelTier.FRONTIER, [], 200_000),
        ModelTier.FAST: ModelEquivalenceClass(ModelTier.FAST, fast_members, 128_000),
        ModelTier.FREE_TIER: ModelEquivalenceClass(ModelTier.FREE_TIER, [], 128_000),
        ModelTier.LOCAL: ModelEquivalenceClass(ModelTier.LOCAL, [], 32_000),
    }


def _pin_sentinel(tmp_path, monkeypatch, subs, fast_members):
    monkeypatch.setattr(
        "lib.hal.sentinel._STATE_PATH", str(tmp_path / "limits.json")
    )
    reg = ProviderRegistry()
    register_subscriptions(reg, subs)
    return RAILLMSentinel(registry=reg, equivalence_map=_pin_emap(fast_members))


def test_infer_unpinned_silently_fails_over_openai_to_gemini(tmp_path, monkeypatch):
    """Regression anchor for the BUG: without pin, an explicit openai/gpt-5.5
    whose openai rung 404s is silently served by gemini and reported ok=true.
    This is the exact live misroute — documented here so a fix that changes it
    is a deliberate, visible choice (the `failover` field already records it)."""
    subs = {
        "openai": _Sub("openai", raise_exc=OpenAICompatError(
            "openai: HTTP 404 — model 'gpt-5.5' does not exist")),
        "gemini": _Sub("gemini", reply="answer from gemini"),
    }
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _pin_sentinel(tmp_path, monkeypatch, subs, [_O, _G]),
    )
    events = _run({
        "v": "happi/1.1", "id": "unp", "cmd": "infer",
        "flags": {"model": _O, "prompt": "hi"},
    })
    assert [e["type"] for e in events] == ["started", "delta", "completed"]
    usage = events[2].get("usage", {})
    assert usage.get("provider") == "gemini"            # served by the WRONG provider
    assert usage.get("requested") == _O
    assert usage.get("failover") == {                   # recorded, but ok=true
        "requested": "openai", "served_by": "gemini"}


def test_infer_pin_refuses_silent_cross_provider_failover(tmp_path, monkeypatch):
    """THE FIX: flags.pin=true turns the silent openai->gemini failover into a
    loud ``provider_pin_violation`` error — never a mislabelled gemini success.
    Goodhart guard: assert NO delta/completed leaked past the pin gate.

    Inverted for HAL#673: this used to require the refusal to NAME gemini,
    which it could only do because the cascade had already sent gemini the
    prompt. The pinned rung is now tried alone, so gemini is never named."""
    subs = {
        "openai": _Sub("openai", raise_exc=OpenAICompatError(
            "openai: HTTP 404 — model 'gpt-5.5' does not exist")),
        "gemini": _Sub("gemini", reply="answer from gemini"),
    }
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _pin_sentinel(tmp_path, monkeypatch, subs, [_O, _G]),
    )
    events = _run({
        "v": "happi/1.1", "id": "pin", "cmd": "infer",
        "flags": {"model": _O, "prompt": "hi", "pin": True},
    })
    err = [e for e in events if e["type"] == "error"]
    assert err and err[0]["code"] == "provider_pin_violation"
    assert "openai" in err[0]["message"] and "gemini" not in err[0]["message"]
    assert not any(e["type"] in ("delta", "completed") for e in events)


def test_infer_pin_allows_matching_provider(tmp_path, monkeypatch):
    """Goodhart / no-over-block: pin must NOT block a pinned request that IS
    served by the requested provider. openai rung SUCCEEDS -> provider=openai
    -> normal success, no violation. (Without this, a mutation that always
    raises provider_pin_violation would still pass the two tests above.)"""
    subs = {"openai": _Sub("openai", reply="answer from openai")}
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _pin_sentinel(tmp_path, monkeypatch, subs, [_O]),
    )
    events = _run({
        "v": "happi/1.1", "id": "pm", "cmd": "infer",
        "flags": {"model": _O, "prompt": "hi", "pin": True},
    })
    assert [e["type"] for e in events] == ["started", "delta", "completed"]
    assert events[1]["text"] == "answer from openai"
    assert events[2]["usage"]["provider"] == "openai"   # pin satisfied -> success
    assert not any(e["type"] == "error" for e in events)


def test_infer_pin_refuses_bare_unknown_model_id(tmp_path, monkeypatch):
    """HAL#649: a pinned request with a BARE id (no provider prefix) is
    refused as unknown_model BEFORE dispatch — it can never silently become
    another vendor's answer wearing a success code. (Measured live: bare
    'deepseek-v4-pro' was served by gemini under ok=true.)"""
    subs = {"gemini": _Sub("gemini", reply="answer from gemini")}
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _pin_sentinel(tmp_path, monkeypatch, subs, [_G]),
    )
    events = _run({
        "v": "happi/1.1", "id": "bare", "cmd": "infer",
        "flags": {"model": "deepseek-v4-pro", "prompt": "hi", "pin": True},
    })
    err = [e for e in events if e["type"] == "error"]
    assert err and err[0]["code"] == "unknown_model"
    assert "deepseek-v4-pro" in err[0]["message"]
    assert not any(e["type"] in ("delta", "completed") for e in events)


def test_infer_unpinned_bare_id_still_resolves(tmp_path, monkeypatch):
    """Goodhart / no-over-block: a bare id WITHOUT pin keeps the pre-#649
    behaviour (the ladder resolves it). Only pin opts into the strict guard."""
    subs = {"gemini": _Sub("gemini", reply="answer from gemini")}
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _pin_sentinel(tmp_path, monkeypatch, subs, [_G]),
    )
    events = _run({
        "v": "happi/1.1", "id": "bareu", "cmd": "infer",
        "flags": {"model": "deepseek-v4-pro", "prompt": "hi"},
    })
    assert not any(e["type"] == "error" for e in events)
    assert events[-1]["type"] == "completed"


_D_WANT = "deepseek/deepseek-v99"        # qualified, but no such model
_D_PEER = "deepseek/deepseek-v4-flash"   # a same-provider tier peer


class _ModelSub(_Sub):
    """A provider that refuses model ids it does not serve, as a real API
    answers 'model not found', and that can echo back a model id other than
    the one requested, as an API answering with a dated snapshot id does."""

    def __init__(self, pid, *, reply, refuse=(), echo=None):
        super().__init__(pid, reply=reply)
        self._refuse = set(refuse)
        self._echo = echo

    def chat(self, prompt="", *, model, **kwargs):
        if model in self._refuse:
            raise OpenAICompatError(
                f"{self.provider_id}: HTTP 400 — model '{model}' not found")
        return ChatResponse(text=self._reply, model=self._echo or model,
                            provider=self.provider_id, usage=TokenUsage())


def _sibling_run(tmp_path, monkeypatch, *, pin, model=_D_WANT, **sub):
    subs = {"deepseek": _ModelSub("deepseek", **sub)}
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _pin_sentinel(tmp_path, monkeypatch, subs, [model, _D_PEER]),
    )
    flags = {"model": model, "prompt": "hi"}
    if pin:
        flags["pin"] = True
    return _run({"v": "happi/1.1", "id": "sib", "cmd": "infer", "flags": flags})


def test_infer_pin_refuses_a_same_provider_model_substitution(tmp_path, monkeypatch):
    """HAL#649 asks for "this model or nothing". A qualified id the provider
    does not serve fails its rung, and the cascade's next rung can be a sibling
    model from the SAME provider, which a provider-only check lets through. It
    is refused, and nothing streams past the gate.

    Inverted for HAL#673: the sibling used to be named because it had answered.
    The pinned rung is now tried alone, so the sibling never sees the prompt."""
    events = _sibling_run(tmp_path, monkeypatch, pin=True,
                          reply="answer from v4-flash", refuse={"deepseek-v99"})
    err = [e for e in events if e["type"] == "error"]
    assert err and err[0]["code"] == "provider_pin_violation"
    assert "deepseek-v99" in err[0]["message"]
    assert "no other model or provider was tried" in err[0]["message"]
    assert "deepseek-v4-flash" not in err[0]["message"]
    assert not any(e["type"] in ("delta", "completed") for e in events)


def test_infer_pin_judges_the_rung_called_not_the_model_id_echoed(tmp_path, monkeypatch):
    """No over-block: the served rung is the one the ladder called. An API that
    echoes a dated snapshot id answered the requested model; that is not a
    substitution."""
    events = _sibling_run(tmp_path, monkeypatch, pin=True,
                          model="deepseek/deepseek-v4-pro",
                          reply="answer from v4-pro",
                          echo="deepseek-v4-pro-2026-09-01")
    assert not any(e["type"] == "error" for e in events)
    assert events[1]["text"] == "answer from v4-pro"


def test_infer_unpinned_same_provider_substitution_still_succeeds(tmp_path, monkeypatch):
    """Only pin opts into the strict check: an unpinned request keeps the
    ladder's failover to a sibling model."""
    events = _sibling_run(tmp_path, monkeypatch, pin=False,
                          reply="answer from v4-flash", refuse={"deepseek-v99"})
    assert not any(e["type"] == "error" for e in events)
    assert events[1]["text"] == "answer from v4-flash"


# ---------------------------------------------------------------------------
# Multimodal capability guard tests (Goodhart-safe — guard MUST raise)
# ---------------------------------------------------------------------------

import pytest
from lib.hal.happi.cmds_infer import (
    _assert_multimodal_capable,
    _has_non_text_parts,
    _MULTIMODAL_PREFIXES,
)


@pytest.fixture(autouse=True)
def _no_real_providers(tmp_path, monkeypatch):
    """Every ProviderRegistry in this file starts empty.

    Applies to the whole module, wherever this sits in it. ProviderRegistry()
    otherwise loads the real provider config, and LimitSentinel.dispatch falls
    back to every registered provider, so a failing fake can hand its prompt to
    a real one. HAL#673's first controls did exactly that before this fence.
    """
    monkeypatch.setattr("lib.hal.registry._CONFIG_PATH",
                        str(tmp_path / "no-providers.yaml"))

_AUDIO_PART = {"type": "audio", "data": "AAA=", "format": "wav"}
_IMAGE_PART = {"type": "image", "url": "data:image/png;base64,AA=="}
_TEXT_PART  = {"type": "text", "text": "hello"}


class TestHasNonTextParts:
    def test_audio_detected(self):
        assert _has_non_text_parts([_AUDIO_PART])

    def test_image_detected(self):
        assert _has_non_text_parts([_IMAGE_PART])

    def test_image_url_detected(self):
        assert _has_non_text_parts([{"type": "image_url", "url": "http://x"}])

    def test_text_only_not_flagged(self):
        assert not _has_non_text_parts([_TEXT_PART])

    def test_empty_not_flagged(self):
        assert not _has_non_text_parts([])


class TestAssertMultimodalCapable:
    """Guard must raise on unknown model + non-text parts (fail-CLOSED)."""

    def test_text_only_always_passes_unknown_model(self):
        # Text-only parts must NEVER be blocked regardless of model.
        _assert_multimodal_capable("some-unknown-model-xyz", [_TEXT_PART])

    def test_audio_on_unknown_model_raises(self):
        with pytest.raises(ValueError, match="not known to support"):
            _assert_multimodal_capable("some-unknown-model-xyz", [_AUDIO_PART])

    def test_image_on_unknown_model_raises(self):
        with pytest.raises(ValueError, match="not known to support"):
            _assert_multimodal_capable("some-unknown-model-xyz", [_IMAGE_PART])

    def test_audio_on_known_gemini_passes(self):
        _assert_multimodal_capable("gemini-1.5-pro", [_AUDIO_PART])

    def test_audio_on_known_gpt4o_passes(self):
        _assert_multimodal_capable("gpt-4o-2024-05-13", [_AUDIO_PART])

    def test_image_on_known_claude3_passes(self):
        _assert_multimodal_capable("claude-3-opus-20240229", [_IMAGE_PART])

    def test_image_on_known_claude_opus4_passes(self):
        _assert_multimodal_capable("claude-opus-4-5", [_IMAGE_PART])

    def test_image_on_known_claude_sonnet4_passes(self):
        _assert_multimodal_capable("claude-sonnet-4-6", [_IMAGE_PART])

    def test_mixed_text_and_audio_on_unknown_raises(self):
        """Mixed parts containing audio must still raise on unknown model."""
        with pytest.raises(ValueError, match="not known to support"):
            _assert_multimodal_capable("llama-3-local", [_TEXT_PART, _AUDIO_PART])

    def test_error_message_is_actionable(self):
        """Error must name the model and tell the user what to do."""
        with pytest.raises(ValueError) as exc_info:
            _assert_multimodal_capable("my-custom-model", [_AUDIO_PART])
        msg = str(exc_info.value)
        assert "my-custom-model" in msg
        assert "_MULTIMODAL_PREFIXES" in msg
        assert "Failing closed" in msg

    def test_allowlist_prefixes_are_non_empty(self):
        """Regression anchor: allowlist must never be accidentally emptied."""
        assert len(_MULTIMODAL_PREFIXES) >= 5


# --- W1: the live Kimi lineup is image-capable ---------------------------------
#
# GET https://api.moonshot.ai/v1/models (2026-09-07) reports
# supports_image_in: true and supports_video_in: true for all four current ids.
# Qualified entries match the FULL address exactly (never by prefix), so the
# entries are the four live addresses, not a blanket "moonshot/" (a retired or
# future id must still fail closed).

_W1_IMAGE_PART = {"type": "image_url",
                  "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}


@pytest.mark.parametrize("address", [
    "moonshot/kimi-k3",
    "moonshot/kimi-k2.7-code",
    "moonshot/kimi-k2.7-code-highspeed",
    "moonshot/kimi-k2.6",
    "moonshot/KIMI-K3",
])
def test_live_kimi_addresses_accept_image_parts(address):
    _assert_multimodal_capable(address, [_W1_IMAGE_PART])


@pytest.mark.parametrize("address", [
    "moonshot/kimi-k2.5",        # retired 2026-08-31; never listed
    "moonshot/moonshot-v1-8k",   # retired, text-only
    "kimi-k3",                   # bare id is not the address the ladder uses
])
def test_non_listed_moonshot_addresses_still_fail_closed(address):
    with pytest.raises(ValueError, match="not known to support"):
        _assert_multimodal_capable(address, [_W1_IMAGE_PART])


# --- Provider-qualified addresses, and GPT-6 Astra ------------------------------
#
# The ladder passes PROVIDER-QUALIFIED addresses (GRIP_LADDER_ADDRESSES in
# lib/hal/raillm_ladder.py: "anthropic/claude-opus-4-7", "gemini/gemini-2.5-flash")
# and cmds_infer.py hands model_address to the guard UNPARSED. The bare family
# entries therefore never matched a real ladder address: "openai/gpt-4o" failed
# closed despite the allowlist's own comment calling it image-capable, and
# "gemini" passed only because that provider's name equals its model family.
# Probed against the pre-fix guard on 2026-09-13.
#
# openai/gpt-6-astra: vision verified live 2026-09-12 through the HAL gateway on
# :4010 — four-band colour image at 1-in-24 blind odds, with a no-image control
# that correctly refused to guess. Catalogued by PRs #558 and #559.
#
# Qualified entries are ADDITIVE: the bare forms stay, so callers passing a bare
# id (exercised above) keep working.

_QUALIFIED_IMAGE_PART = {"type": "image_url",
                         "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}


@pytest.mark.parametrize("address", [
    "openai/gpt-6-astra",
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
    "anthropic/claude-opus-4-7",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-3-opus-20240229",
    "OpenAI/GPT-6-Astra",              # the match is case-insensitive
])
def test_qualified_addresses_accept_image_parts(address):
    _assert_multimodal_capable(address, [_QUALIFIED_IMAGE_PART])


@pytest.mark.parametrize("address", [
    "openai/gpt-3.5-turbo",            # text-only
    "openai/o1-mini",                  # reasoning, no image input
    "anthropic/claude-instant-1",      # retired, text-only
    "openai/gpt-6",                    # bare family is not an allowlisted address
    "openai/gpt-6-astra-preview-xyz",  # NOT a real id; only exact prefixes pass
])
def test_qualified_text_only_addresses_still_fail_closed(address):
    with pytest.raises(ValueError, match="not known to support"):
        _assert_multimodal_capable(address, [_QUALIFIED_IMAGE_PART])


@pytest.mark.parametrize("value, expected", [
    (64000, 64000), (1, 1),
    (None, None), (0, None), (-5, None), (True, None), (False, None),
    (64000.0, None), ("64000", None), ([64000], None),
])
def test_answer_budget_accepts_only_a_positive_int(value, expected):
    """The one rule /api/infer and a raw envelope both use for max_tokens."""
    assert cmds_infer.answer_budget(value) == expected


# ---------------------------------------------------------------------------
# HAL#673: a pinned request tries ONLY its pinned rung. The pin used to be
# checked after the whole cascade had run, so the prompt had already reached
# every provider the cascade tried. Measured live on 2026-10-06: a pinned
# gemini/gemini-9-nonexistent was answered by anthropic-cch, then refused.
# Each case below has an unpinned control showing the same setup does fail
# over, so a green pinned case cannot come from a setup that never cascades.
# ---------------------------------------------------------------------------


class _Recording(_Sub):
    """A fake provider that records each call, to show which providers saw a prompt."""

    def __init__(self, pid, calls, **kwargs):
        super().__init__(pid, **kwargs)
        self._calls = calls

    def chat(self, prompt="", *, model, **kwargs):
        self._calls.append(self.provider_id)
        return super().chat(prompt, model=model, **kwargs)


def _with_default_model(sentinel, provider, model):
    """Give a fake provider a default model, so the multimodal failover can reach it.

    LimitSentinel.dispatch skips a fallback provider with no configured model,
    and register_subscriptions configures none; production providers carry one
    from config/providers.yaml.
    """
    from lib.hal.types import ProviderConfig, ProviderType

    sentinel._registry._configs[provider] = ProviderConfig(
        name=provider, type=ProviderType.CLOUD, base_url="", models=[model])
    return sentinel


def _openai_fails_gemini_answers(calls):
    return {
        "openai": _Recording("openai", calls, raise_exc=OpenAICompatError(
            "openai: HTTP 404 — model 'gpt-5.5' does not exist")),
        "gemini": _Recording("gemini", calls, reply="answer from gemini"),
    }


def test_dispatch_with_history_no_failover_tries_only_the_primary(tmp_path, monkeypatch):
    from lib.hal.raillm import RAILLMExhausted

    calls: list[str] = []
    control = _pin_sentinel(tmp_path, monkeypatch, _openai_fails_gemini_answers(calls), [_O, _G])
    assert control.dispatch_with_history(_O, "hi").text == "answer from gemini"
    assert calls == ["openai", "gemini"]

    calls.clear()
    (tmp_path / "pinned").mkdir()
    pinned = _pin_sentinel(tmp_path / "pinned", monkeypatch,
                           _openai_fails_gemini_answers(calls), [_O, _G])
    with pytest.raises(RAILLMExhausted):
        pinned.dispatch_with_history(_O, "hi", no_failover=True)
    assert calls == ["openai"]


def test_limit_sentinel_dispatch_no_failover_tries_only_the_primary(tmp_path, monkeypatch):
    calls: list[str] = []
    control = _with_default_model(
        _pin_sentinel(tmp_path, monkeypatch, _openai_fails_gemini_answers(calls), []),
        "gemini", "gemini-2.5-flash")
    assert control.dispatch(_O, "hi").text == "answer from gemini"
    assert calls == ["openai", "gemini"]

    calls.clear()
    (tmp_path / "pinned").mkdir()
    pinned = _with_default_model(
        _pin_sentinel(tmp_path / "pinned", monkeypatch, _openai_fails_gemini_answers(calls), []),
        "gemini", "gemini-2.5-flash")
    with pytest.raises(RuntimeError):
        pinned.dispatch(_O, "hi", no_failover=True)
    assert calls == ["openai"]


def test_infer_pin_tries_only_the_pinned_rung(tmp_path, monkeypatch):
    """The refusal keeps its code: GRIP's client sends provider_pin_violation on
    to HAL's gateway, which may still serve the pinned model."""
    calls: list[str] = []
    subs = _openai_fails_gemini_answers(calls)
    monkeypatch.setattr(cmds_infer, "make_grip_sentinel",
                        lambda: _pin_sentinel(tmp_path, monkeypatch, subs, [_O, _G]))
    events = _run({"v": "happi/1.1", "id": "p673", "cmd": "infer",
                   "flags": {"model": _O, "prompt": "hi", "pin": True}})
    err = [e for e in events if e["type"] == "error"]
    assert err and err[0]["code"] == "provider_pin_violation"
    assert "no other model or provider was tried" in err[0]["message"]
    assert calls == ["openai"]
    assert not any(e["type"] in ("delta", "completed") for e in events)


_AUDIO_PARTS = [
    {"type": "input_audio", "input_audio": {"data": "AAA=", "format": "wav"}},
    {"type": "text", "text": "describe"},
]


def _multimodal_run(tmp_path, monkeypatch, *, pin):
    calls: list[str] = []
    subs = {
        "anthropic": _Recording("anthropic", calls, raise_exc=OpenAICompatError("cch limit")),
        "moonshot": _Recording("moonshot", calls, reply="heard by kimi"),
    }
    monkeypatch.setattr(cmds_infer, "make_grip_sentinel", lambda: _with_default_model(
        _pin_sentinel(tmp_path, monkeypatch, subs, []), "moonshot", "kimi-k2"))
    flags = {"model": _A, "prompt": "", "content_parts": _AUDIO_PARTS}
    if pin:
        flags["pin"] = True
    return _run({"v": "happi/1.1", "id": "mm673", "cmd": "infer", "flags": flags}), calls


def test_infer_unpinned_multimodal_still_fails_over(tmp_path, monkeypatch):
    events, calls = _multimodal_run(tmp_path, monkeypatch, pin=False)
    assert events[1]["text"] == "heard by kimi"
    assert calls == ["anthropic", "moonshot"]


def test_infer_pin_multimodal_tries_only_the_pinned_provider(tmp_path, monkeypatch):
    events, calls = _multimodal_run(tmp_path, monkeypatch, pin=True)
    err = [e for e in events if e["type"] == "error"]
    assert err and err[0]["code"] == "provider_pin_violation"
    assert calls == ["anthropic"]
    assert not any(e["type"] in ("delta", "completed") for e in events)


def test_pin_backstop_refuses_an_answer_reported_from_another_provider():
    """A pinned turn tries its own rung alone now, so the cascade cannot hand
    it to another provider; an answer that still reports one is refused."""
    from types import SimpleNamespace

    envelope = SimpleNamespace(flags={"pin": True})
    response = SimpleNamespace(provider="gemini", fallback_info=None)
    message = cmds_infer._pin_violation(envelope, _O, response)
    assert message and "openai" in message and "gemini" in message


def test_pin_backstop_refuses_an_answer_reported_from_another_model():
    from types import SimpleNamespace

    envelope = SimpleNamespace(flags={"pin": True})
    info = SimpleNamespace(actual_provider="deepseek", actual_model="deepseek-v4-flash")
    response = SimpleNamespace(provider="deepseek", fallback_info=info)
    message = cmds_infer._pin_violation(envelope, _D_WANT, response)
    assert message and "deepseek-v4-flash" in message


@pytest.mark.parametrize("parts", [None, [_QUALIFIED_IMAGE_PART]], ids=["text", "multimodal"])
def test_pin_refusal_names_an_unregistered_provider(tmp_path, monkeypatch, parts):
    """A pinned provider HAL has not registered (openai/gpt-6-astra is served by
    the gateway, not /api/infer) is refused with the reason, on either path."""
    monkeypatch.setattr(cmds_infer, "make_grip_sentinel",
                        lambda: _pin_sentinel(tmp_path, monkeypatch, {}, ["openai/gpt-6-astra"]))
    flags = {"model": "openai/gpt-6-astra", "prompt": "hi", "pin": True}
    if parts:
        flags["content_parts"] = parts
    events = _run({"v": "happi/1.1", "id": "unreg", "cmd": "infer", "flags": flags})
    err = [e for e in events if e["type"] == "error"]
    assert err and err[0]["code"] == "provider_pin_violation"
    assert "provider not registered" in err[0]["message"]
