"""The gateway council's fail-open model is Gemini 4 Hydrogen (V>> 2026-11-26).

HAL#893 set the fail-open default to Gemini 4 Carbon, but
``_relay_council_failopen``'s own docstring asks for "ONE fast Pool-B model",
and Carbon is not fast: measured 2026-11-26 with short prompts, Hydrogen
answered in 2.6s against Neon's 6.7s and Carbon's 7.2s, and Carbon took over
100s on a larger prompt. V>> decided the fail-open model is Gemini 4 Hydrogen.

This test used to live in tests/test_gemini_carbon_seats.py, which pins
Carbon replacing 3.1 Pro in HAL's routing seats (the complex tier, the
premium chain, the RAILLM frontier class). The fail-open default is a
different seat and was never one of those -- it moved here so a reader
looking for what the fail-open model is finds it by the file's name.
"""
from __future__ import annotations

HYDROGEN = "gemini-4-hydrogen"


def test_the_council_fail_open_model_is_hydrogen_and_the_override_still_wins(monkeypatch):
    """Behaviour, not source text: the model the fail-open path asks the same-wire
    router for. Mutation: restore "gemini-4-carbon" or "gemini-3.1-pro-preview"
    as the default, or stop reading GRIP_COUNCIL_FAILOPEN_MODEL -> RED."""
    from lib.hal import gateway_egress, gateway_openai
    asked = []
    monkeypatch.setattr(gateway_egress, "same_wire_provider", lambda model: asked.append(model))
    monkeypatch.setattr(gateway_openai, "_send_openai_error", lambda *_args, **_kwargs: None)
    monkeypatch.delenv("GRIP_COUNCIL_FAILOPEN_MODEL", raising=False)
    gateway_openai._relay_council_failopen(None, {}, "council", 0.0, "down")
    monkeypatch.setenv("GRIP_COUNCIL_FAILOPEN_MODEL", "override-model")
    gateway_openai._relay_council_failopen(None, {}, "council", 0.0, "down")
    assert asked == [HYDROGEN, "override-model"]
