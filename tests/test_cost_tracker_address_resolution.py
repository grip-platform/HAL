"""Forward-anchor tests for provider-qualified pricing resolution (issue #509).

Two silent budget-blindness shapes, both live before this fix:

1. "moonshot/kimi-k3" (address form) fell through exact + prefix matching to
   _DEFAULT_PRICING — wrong cache_write, and only right on in/out by luck.
2. "deepseek/deepseek-v4-pro" — the AGORA council's live reasoning speaker —
   matched the LOCAL-models substring heuristic ("deepseek") and metered at
   USD 0.00 despite being a paid cloud model.

Deterministic, no network.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib.hal.cost_tracker import (  # noqa: E402
    _DEFAULT_PRICING,
    _PRICING,
    _resolve_pricing,
    calculate_cost,
)


def test_address_form_kimi_resolves_real_pricing():
    assert _resolve_pricing("moonshot/kimi-k3") == _PRICING["kimi-k3"]


def test_address_form_deepseek_v4_pro_is_not_free():
    p = _resolve_pricing("deepseek/deepseek-v4-pro")
    assert p == _PRICING["deepseek-v4-pro"]
    assert p[0] > 0 and p[1] > 0, "paid council speaker must never meter at 0"


def test_address_form_calculate_cost_is_nonzero():
    usd = calculate_cost(
        "moonshot/kimi-k3", input_tokens=1_000_000, output_tokens=1_000_000)
    assert usd == 18.0


def test_unknown_address_still_falls_to_default():
    assert _resolve_pricing("wat/unknown-model-9") == _DEFAULT_PRICING


def test_bare_local_model_stays_free():
    assert _resolve_pricing("llama3:8b") == _PRICING["local"]


def test_unknown_bare_cloud_family_id_is_not_free():
    # "deepseek-unknown-9" has no pricing row; it must fall to the VISIBLE
    # default estimate, never the silent local zero (AGORA review of PR #512,
    # the unknown-non-qualified-model concern).
    assert _resolve_pricing("deepseek-unknown-9") == _DEFAULT_PRICING


def test_unknown_bare_model_falls_to_default():
    assert _resolve_pricing("wat-unknown-model-9") == _DEFAULT_PRICING


def test_ollama_style_tagged_model_stays_free():
    assert _resolve_pricing("qwen2.5:7b") == _PRICING["local"]
