"""Kimi locked-temperature guard at the ADAPTER seam (PR #508).

Moonshot 400s on temperature != 1 for the whole current Kimi lineage
(live-verified 2026-07-26: kimi-k2.5, kimi-k2.6, kimi-k3; kimi-k2.7-code
per #2209). The gateway egress already dropped it, but the direct
registry/adapter path (OpenAICompatProvider.chat) still sent
temperature=0.7 by default, so every default-shaped call 400'd — the
model was catalogued yet unreachable.

Goodhart-safe: asserts the actual outbound request body at the
``_request`` seam (temperature gone for Kimi, present + caller-valued
for everything else), not call counts.
"""
from unittest.mock import patch

from lib.hal.providers.openai_compat import OpenAICompatProvider


def _capture_chat(model, **kw):
    """Drive chat() with a stubbed transport; return the outbound body."""
    captured = {}
    provider = OpenAICompatProvider(
        name="moonshot",
        base_url="https://api.moonshot.ai/v1",
        api_key="sk-test",
    )

    def fake_request(method, path, body=None):
        captured["body"] = body
        return {
            "id": "chatcmpl-stub",
            "model": model,
            "choices": [{
                "message": {"role": "assistant", "content": "stub"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    with patch.object(provider, "_request", side_effect=fake_request):
        provider.chat("hi", model=model, **kw)
    return captured["body"]


def test_kimi_k3_temperature_omitted_by_default():
    body = _capture_chat("kimi-k3")
    assert "temperature" not in body
    assert body["model"] == "kimi-k3"


def test_kimi_k3_temperature_omitted_even_when_caller_sets_one():
    # The lock wins over an explicit caller value — Moonshot rejects != 1,
    # and omission lets the provider apply its own default (1).
    body = _capture_chat("kimi-k3", temperature=0.3)
    assert "temperature" not in body


def test_kimi_k25_temperature_omitted():
    body = _capture_chat("kimi-k2.5", temperature=0.7)
    assert "temperature" not in body


def test_kimi_k26_temperature_omitted():
    body = _capture_chat("kimi-k2.6")
    assert "temperature" not in body


def test_non_kimi_keeps_caller_temperature():
    body = _capture_chat("llama-3.3-70b-versatile", temperature=0.3)
    assert body["temperature"] == 0.3


def test_moonshot_v1_keeps_default_temperature():
    # moonshot-v1-* are NOT locked — the adapter default must survive.
    body = _capture_chat("moonshot-v1-8k")
    assert body["temperature"] == 0.7


def test_max_tokens_survives_for_kimi():
    body = _capture_chat("kimi-k3", max_tokens=512)
    assert body["max_tokens"] == 512
    assert "temperature" not in body
