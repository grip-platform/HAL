"""Goodhart-proof tests for ``lib.hal.grip_runtime.grip_infer`` (H-RAILLM-4).

Driven through the REAL ``runtime.run`` -- only ``make_grip_sentinel`` is
faked (a 2-rung hermetic equivalence map: fake anthropic + moonshot only),
so the cascade can never reach a real network provider. These assert
concrete values (text, provider, failover dict, idr sha256) -- they fail if
the fold logic is wrong (mutation-test mindset), not call counts.

Pins, for H-RAILLM-4:
  * grip_infer cascades CCH -> Kimi and surfaces the failover in the
    normalised result (derived from provider mismatch -- RAILLM does NOT
    set response.fallback_info)
  * with NO model arg the requested address is the ladder PRIMARY
    (the "no default" gap -- CCH-Anthropic rung -- is closed)
  * audit=True yields a DETERMINISTIC idr_ref.sha256 through the facade
  * full exhaustion -> ok=False/runtime_error ; empty prompt ->
    ok=False/parse_error (validation delegated to the runtime, DRY)
"""
import uuid

import lib.hal.happi.cmds_infer as cmds_infer
from lib.hal.grip_runtime import GRIP_LADDER_ADDRESSES, grip_infer
from lib.hal.providers.openai_compat import OpenAICompatError
from lib.hal.raillm import ModelEquivalenceClass, ModelTier, RAILLMSentinel
from lib.hal.raillm_subscription_bridge import register_subscriptions
from lib.hal.registry import ProviderRegistry
from lib.hal.types import ChatResponse, TokenUsage

_A = "anthropic/claude-opus-4-7"
_K = "moonshot/kimi-k2"


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


def _sentinel(tmp_path, monkeypatch, *, all_fail=False, primary_ok=False):
    monkeypatch.setattr(
        "lib.hal.sentinel._STATE_PATH", str(tmp_path / "limits.json")
    )
    reg = ProviderRegistry()
    if primary_ok:
        cch = _Sub("anthropic", reply="answer from cch")
    else:
        cch = _Sub("anthropic", raise_exc=OpenAICompatError("cch limit"))
    if all_fail:
        kimi = _Sub("moonshot", raise_exc=OpenAICompatError("kimi down"))
    else:
        kimi = _Sub("moonshot", reply="answer from kimi")
    register_subscriptions(reg, {"anthropic": cch, "moonshot": kimi})
    return RAILLMSentinel(registry=reg, equivalence_map=_emap())


def test_grip_infer_cascades_cch_to_kimi_and_reports_failover(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _sentinel(tmp_path, monkeypatch),
    )
    r = grip_infer("2+2?", model=_A)
    assert r.ok is True
    assert r.text == "answer from kimi"          # failed over to Kimi
    assert r.provider == "moonshot"
    assert r.requested == _A
    assert r.failover == {"requested": "anthropic", "served_by": "moonshot"}
    # proves the REAL runtime was driven (not a stub of it):
    assert [e["type"] for e in r.events] == ["started", "delta", "completed"]


def test_grip_infer_defaults_to_ladder_primary(tmp_path, monkeypatch):
    """The W4 'no default' gap: with no model arg the call targets the
    ladder PRIMARY (CCH-Anthropic rung) and it answers."""
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _sentinel(tmp_path, monkeypatch, primary_ok=True),
    )
    r = grip_infer("hello")                       # NO model= arg
    assert r.requested == GRIP_LADDER_ADDRESSES[0]  # the ladder's primary, whatever it is named
    assert r.ok is True
    assert r.provider == "anthropic"
    assert r.text == "answer from cch"
    assert r.failover is None                     # primary answered


def test_grip_infer_audit_idr_is_deterministic(tmp_path, monkeypatch):
    """H-RAILLM-4 over H-RAILLM-3: the facade preserves the deterministic
    audit terminator -- identical envelope+events -> identical sha256."""
    monkeypatch.setattr("lib.hal.happi.events._now_ms", lambda: 1234)
    # Pin the envelope id too: grip_infer mints a fresh uuid per call (correct
    # for real audit -- each call is a distinct decision record). To test the
    # determinism PROPERTY, hold envelope+events identical so the only
    # remaining variable is the (identical) content.
    monkeypatch.setattr(
        "lib.hal.grip_runtime.uuid.uuid4", lambda: uuid.UUID(int=0)
    )
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _sentinel(tmp_path, monkeypatch),
    )
    first = grip_infer("hello", model=_A, audit=True)
    second = grip_infer("hello", model=_A, audit=True)
    assert first.idr_ref is not None and second.idr_ref is not None
    assert set(first.idr_ref) == {
        "sha256", "cid", "model_versions", "block_anchor"}
    s1, s2 = first.idr_ref["sha256"], second.idr_ref["sha256"]
    assert s1 == s2 and len(s1) == 64             # deterministic, real hash


def test_grip_infer_full_exhaustion_is_runtime_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _sentinel(tmp_path, monkeypatch, all_fail=True),
    )
    r = grip_infer("hi", model=_A)
    assert r.ok is False
    assert r.text == ""
    assert r.error_code == "runtime_error"
    assert "exhausted" in (r.error_message or "")


def test_grip_infer_empty_prompt_is_parse_error(tmp_path, monkeypatch):
    """Validation is the runtime's single source of truth (DRY): the empty
    prompt is rejected by dispatch_infer, surfaced through the facade."""
    monkeypatch.setattr(
        cmds_infer, "make_grip_sentinel",
        lambda: _sentinel(tmp_path, monkeypatch),
    )
    r = grip_infer("", model=_A)
    assert r.ok is False
    assert r.error_code == "parse_error"


def test_collect_empty_text_with_length_finish_is_budget_exhausted():
    """HAL#648 ask 1: empty text + finish_reason=length folds to ok=false with
    answer_budget_exhausted, not a silent ok=true empty success."""
    from lib.hal import grip_runtime
    events = [
        {"type": "completed", "id": "e1",
         "usage": {"provider": "deepseek", "model": "deepseek-v4-pro",
                   "in": 13000, "out": 16384},
         "finish_reason": "length"},
    ]
    r = grip_runtime._collect(events, "deepseek/deepseek-v4-pro")
    assert r.ok is False
    assert r.text == ""
    assert r.error_code == "answer_budget_exhausted"
    assert r.finish_reason == "length"
    assert r.provider == "deepseek"


def test_collect_empty_text_with_stop_finish_is_plain_empty_success():
    """Goodhart contrast: empty text + finish_reason=stop is NOT budget
    exhaustion — it is a genuinely empty answer, still ok=true."""
    from lib.hal import grip_runtime
    events = [
        {"type": "completed", "id": "e1",
         "usage": {"provider": "p", "model": "m"},
         "finish_reason": "stop"},
    ]
    r = grip_runtime._collect(events, "p/m")
    assert r.ok is True
    assert r.text == ""
    assert r.error_code is None
    assert r.finish_reason == "stop"


def test_collect_budget_exhausted_keeps_failover_and_idr_ref():
    """An audited, failed-over call that runs out of answer budget keeps the
    idr_ref and failover a successful call carries (AGORA on HAL#655): the
    runtime still emits the idr terminator, so dropping it would lose the
    deterministic record of the call."""
    from lib.hal import grip_runtime
    events = [
        {"type": "completed", "id": "e1",
         "usage": {"provider": "groq", "model": "openai/gpt-oss-120b",
                   "failover": {"requested": "deepseek", "served_by": "groq"}},
         "finish_reason": "length"},
        {"type": "idr", "id": "e1", "idr_ref": "idr-0123abcd"},
    ]
    r = grip_runtime._collect(events, "deepseek/deepseek-v4-pro")
    assert r.ok is False
    assert r.error_code == "answer_budget_exhausted"
    assert r.idr_ref == "idr-0123abcd"
    assert r.failover == {"requested": "deepseek", "served_by": "groq"}


def test_collect_partial_text_with_length_finish_stays_ok():
    """A reply cut off mid-answer is still an answer: some text plus
    finish_reason=length stays ok=true and reports the finish_reason, so the
    caller can see it was truncated. Only an EMPTY reply means the budget ran
    out (the falsifier AGORA's technical seat named on HAL#655)."""
    from lib.hal import grip_runtime
    events = [
        {"type": "delta", "id": "e1", "text": "The answer is"},
        {"type": "completed", "id": "e1",
         "usage": {"provider": "groq", "model": "openai/gpt-oss-120b"},
         "finish_reason": "length"},
    ]
    r = grip_runtime._collect(events, "groq/openai/gpt-oss-120b")
    assert r.ok is True
    assert r.text == "The answer is"
    assert r.error_code is None
    assert r.finish_reason == "length"


def _budget_recording_sub(tmp_path, monkeypatch):
    """A ladder whose rung records the max_tokens each chat() call receives;
    its own default, as a real provider's would, is 2048."""
    sentinel, sub = _capturing_sentinel(tmp_path, monkeypatch)
    monkeypatch.setattr(cmds_infer, "make_grip_sentinel", lambda: sentinel)

    def chat(self, prompt="", *, model, system=None, temperature=None,
             max_tokens=2048, messages=None, tools=None):
        self.calls.append({"prompt": prompt, "messages": messages,
                           "max_tokens": max_tokens})
        return ChatResponse(text="ok", model=model,
                            provider=self.provider_id, usage=TokenUsage())
    sub.chat = chat.__get__(sub, type(sub))
    return sub


def test_grip_infer_threads_max_tokens_to_provider(tmp_path, monkeypatch):
    """HAL#648 ask 2: max_tokens is threaded through the facade to the rung."""
    sub = _budget_recording_sub(tmp_path, monkeypatch)
    r = grip_infer("hi", model=_A, max_tokens=64000)
    assert r.ok is True
    assert sub.calls[0]["max_tokens"] == 64000


def test_grip_infer_without_max_tokens_keeps_the_provider_default(tmp_path, monkeypatch):
    """An unset budget is omitted, never passed as None, which would override
    the provider's own default."""
    sub = _budget_recording_sub(tmp_path, monkeypatch)
    r = grip_infer("hi", model=_A)
    assert r.ok is True
    assert sub.calls[0]["max_tokens"] == 2048


# --- multimodal (audio) path -------------------------------------------- #


class _CapturingSub:
    """Replying provider that RECORDS each chat() call, so a test can assert
    WHICH transport carried the turn: messages= (multimodal) vs prompt= (the
    string-thread cascade)."""

    def __init__(self, pid):
        self.provider_id = pid
        self.calls = []

    def chat(self, prompt="", *, model, system=None, temperature=None,
             max_tokens=2048, messages=None, tools=None):
        self.calls.append({"prompt": prompt, "messages": messages})
        return ChatResponse(text="heard the audio", model=model,
                            provider=self.provider_id, usage=TokenUsage())

    def health_check(self):
        return True


def _capturing_sentinel(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lib.hal.sentinel._STATE_PATH", str(tmp_path / "limits.json")
    )
    reg = ProviderRegistry()
    sub = _CapturingSub("anthropic")
    register_subscriptions(reg, {"anthropic": sub})
    return RAILLMSentinel(registry=reg, equivalence_map=_emap()), sub


def test_grip_infer_media_routes_audio_to_provider(tmp_path, monkeypatch):
    """The property that makes grip-hear routable through HAL: a ``media``
    audio path is encoded to an ``input_audio`` part and REACHES the provider
    via the messages transport — it is not silently dropped."""
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt \x01\x02\x03\x04")
    sentinel, sub = _capturing_sentinel(tmp_path, monkeypatch)
    monkeypatch.setattr(cmds_infer, "make_grip_sentinel", lambda: sentinel)

    r = grip_infer("describe this track", model=_A, media=[str(audio)])

    assert r.ok is True
    assert r.text == "heard the audio"
    assert r.provider == "anthropic"
    assert len(sub.calls) == 1
    messages = sub.calls[0]["messages"]
    assert messages is not None                              # messages transport
    audio_parts = [p for p in messages[0].content
                   if p.get("type") == "input_audio"]
    assert audio_parts and audio_parts[0]["input_audio"]["format"] == "wav"


def test_grip_infer_text_only_uses_string_thread_not_messages(tmp_path, monkeypatch):
    """Goodhart contrast: with NO media the text turn rides the string-thread
    cascade — prompt= is set and messages= is None. A mutation that always
    routed multimodal would set messages here and fail this test."""
    sentinel, sub = _capturing_sentinel(tmp_path, monkeypatch)
    monkeypatch.setattr(cmds_infer, "make_grip_sentinel", lambda: sentinel)

    r = grip_infer("just text", model=_A)

    assert r.ok is True
    assert len(sub.calls) == 1
    assert sub.calls[0]["messages"] is None                  # NOT the messages path
    assert "just text" in sub.calls[0]["prompt"]             # string thread carried it
