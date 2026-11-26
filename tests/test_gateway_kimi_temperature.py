"""Fix A (#2209): kimi-k2.7-code temperature=1-only guard in the gateway egress.

Moonshot's kimi-k2.7-code rejects temperature != 1 (HTTP 400 "invalid
temperature: only 1 is allowed for this model"). A coding harness routing
through the HAL gateway sends its own default temperature, which 400s for that
model. The egress drops `temperature` for the named variants so Moonshot applies
its own (1) — model-aware, never a global force.

Goodhart-safe: asserts the actual forwarded body (temperature gone for kimi,
present+unchanged for every other model), not call counts.
"""
import json

from lib.hal.gateway_egress import (
    _drop_temperature_for_fixed_temp_model,
    _fix_cloud_body_temperature,
    _model_requires_temperature_one,
)


def _body(model, **extra):
    return json.dumps({"model": model, "temperature": 0.7, **extra}).encode()


# --- _model_requires_temperature_one: only the named variants match ----------

def test_kimi_k27_code_requires_temperature_one():
    assert _model_requires_temperature_one("kimi-k2.7-code") is True


def test_kimi_k27_code_highspeed_variant_also_matches():
    # Prefix match so -highspeed is covered without a second literal.
    assert _model_requires_temperature_one("kimi-k2.7-code-highspeed") is True


def test_provider_prefixed_and_uppercase_kimi_matches():
    assert _model_requires_temperature_one("moonshot/KIMI-K2.7-CODE") is True


def test_kimi_k3_requires_temperature_one():
    # Live-verified 2026-07-26: "invalid temperature: only 1 is allowed".
    assert _model_requires_temperature_one("kimi-k3") is True
    assert _model_requires_temperature_one("moonshot/kimi-k3") is True


def test_kimi_k25_k26_also_require_temperature_one():
    # Live-verified 2026-07-26 against the current Moonshot API — refutes
    # this file's earlier "no such constraint" assertion for k2.5/k2.6
    # (lib/hal/subscription.py carried the 2026-05-19 evidence all along).
    assert _model_requires_temperature_one("kimi-k2.6") is True
    assert _model_requires_temperature_one("kimi-k2.5") is True


def test_moonshot_v1_models_do_not_match():
    # moonshot-v1-* are NOT locked — caller temperature must survive.
    assert _model_requires_temperature_one("moonshot-v1-8k") is False
    assert _model_requires_temperature_one("moonshot-v1-auto") is False


def test_unrelated_models_do_not_match():
    for m in ("openai/gpt-oss-120b", "llama-3.3-70b-versatile",
              "claude-opus-4-8", "gemini-2.5-flash", ""):
        assert _model_requires_temperature_one(m) is False


# --- _fix_cloud_body_temperature: verbatim same-wire body (Codex/Aider) -------

def test_cloud_body_drops_temperature_for_kimi():
    out = _fix_cloud_body_temperature(_body("kimi-k2.7-code"))
    obj = json.loads(out)
    assert "temperature" not in obj
    assert obj["model"] == "kimi-k2.7-code"  # everything else preserved


def test_cloud_body_preserves_temperature_for_non_kimi():
    raw = _body("openai/gpt-oss-120b")
    out = _fix_cloud_body_temperature(raw)
    assert out == raw  # byte-identical: a non-kimi request is untouched
    assert json.loads(out)["temperature"] == 0.7


def test_cloud_body_preserves_other_fields_for_kimi():
    out = _fix_cloud_body_temperature(_body("kimi-k2.7-code", top_p=0.9, max_tokens=512))
    obj = json.loads(out)
    assert "temperature" not in obj
    assert obj["top_p"] == 0.9 and obj["max_tokens"] == 512


def test_cloud_body_fail_open_on_garbage():
    # Non-JSON / missing / no-temperature bodies pass through unchanged.
    assert _fix_cloud_body_temperature(b"not json") == b"not json"
    assert _fix_cloud_body_temperature(None) is None
    no_temp = json.dumps({"model": "kimi-k2.7-code"}).encode()
    assert _fix_cloud_body_temperature(no_temp) == no_temp


# --- _drop_temperature_for_fixed_temp_model: cross-wire built dict -------------

def test_crosswire_dict_drops_temperature_for_kimi():
    obj = {"model": "kimi-k2.7-code", "temperature": 0.3, "stream": True}
    _drop_temperature_for_fixed_temp_model(obj)
    assert "temperature" not in obj
    assert obj["stream"] is True  # other fields intact


def test_crosswire_dict_preserves_temperature_for_non_kimi():
    obj = {"model": "openai/gpt-oss-120b", "temperature": 0.3}
    _drop_temperature_for_fixed_temp_model(obj)
    assert obj["temperature"] == 0.3
