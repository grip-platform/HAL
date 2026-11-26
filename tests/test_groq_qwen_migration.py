"""Fix B (#2213): groq/qwen3-32b -> groq/openai/gpt-oss-120b migration guard.

GroqCloud decommissions qwen3-32b on 2026-07-17. Every HAL roster / config /
cost-map entry that pointed at it must now resolve to the production
`openai/gpt-oss-120b`. These tests fail if any live HAL surface still routes to
qwen3-32b, OR if the replacement id is absent — Goodhart-safe (assert the actual
resolved ids in the canonical configs, not call counts).

Distinct preview models qwen/qwen3.5-flash and qwen/qwen3.6-* are NOT in scope
and must remain untouched (a separate test pins that they survive).
"""
from pathlib import Path

# Use HAL's OWN stdlib YAML parser (lib/hal/registry) — pyyaml is NOT guaranteed
# in the venv, and asserting via HAL's real loader tests what HAL actually parses.
from lib.hal.registry import _load_yaml_config, _parse_simple_yaml

_HAL = Path(__file__).resolve().parent.parent
_GPT_OSS = "openai/gpt-oss-120b"
_DECOMMISSIONED = "qwen3-32b"


def _read(rel):
    return (_HAL / rel).read_text(encoding="utf-8")


# --- providers.yaml (via HAL's own loader) -----------------------------------

def test_providers_yaml_no_qwen3_32b_in_rosters():
    configs, routing = _load_yaml_config(str(_HAL / "config/providers.yaml"))
    groq = next(c for c in configs if c.name == "groq")
    assert _GPT_OSS in groq.models
    assert not any(_DECOMMISSIONED in m for m in groq.models)
    premium = routing.tier_chains["premium"]
    assert f"groq/{_GPT_OSS}" in premium
    assert not any(_DECOMMISSIONED in m for m in premium)


def test_providers_yaml_preserves_other_qwen_models():
    # qwen/qwen3.5-flash is a DIFFERENT live model — migration must not touch it.
    raw = _parse_simple_yaml(str(_HAL / "config/providers.yaml"))
    council = raw["routing"]["council"]
    assert "qwen/qwen3.5-flash" in council


# --- model_catalog.yaml ------------------------------------------------------
# HAL's _parse_simple_yaml is tuned for providers.yaml, not the catalog's
# list-of-dicts-with-multiline-descriptions, so assert on the canonical address
# lines directly (still Goodhart-safe: new id present, old id absent).

def test_model_catalog_yaml_address_migrated():
    lines = [ln.strip() for ln in _read("config/model_catalog.yaml").splitlines()]
    assert f"- address: groq/{_GPT_OSS}" in lines
    assert f"- address: groq/{_DECOMMISSIONED}" not in lines


# --- python source surfaces --------------------------------------------------

def test_no_qwen3_32b_routing_in_python_sources():
    # The four code surfaces that routed to groq/qwen3-32b must now resolve to
    # the replacement. Comment lines naming the decommission are allowed; an
    # ACTIVE `groq/qwen3-32b` / `qwen/qwen3-32b` reference is not.
    for rel in ("lib/providers/groq.py", "lib/hal/rsi.py",
                "lib/hal/code_gen.py", "lib/hal/agent_spawn.py",
                "lib/hal/model_catalog.py", "lib/hal/cost_tracker.py"):
        src = _read(rel)
        for line in src.splitlines():
            code = line.split("#", 1)[0]  # strip trailing comment
            assert "qwen3-32b" not in code, f"{rel}: live qwen3-32b ref: {line!r}"


def test_groq_provider_reasoning_helper_uses_gpt_oss():
    from lib.providers.groq import _GPT_OSS_120B
    assert _GPT_OSS_120B == _GPT_OSS


def test_cost_tracker_has_gpt_oss_pricing():
    from lib.hal.cost_tracker import _PRICING  # type: ignore[attr-defined]
    assert _GPT_OSS in _PRICING
    # Verified Groq pricing 2026: $0.15/1M in, $0.60/1M out.
    in_price, out_price = _PRICING[_GPT_OSS][0], _PRICING[_GPT_OSS][1]
    assert (in_price, out_price) == (0.15, 0.60)
    assert _DECOMMISSIONED not in _PRICING


def test_model_catalog_registry_has_gpt_oss():
    from lib.hal.model_catalog import _CATALOG  # type: ignore[attr-defined]
    assert _GPT_OSS in _CATALOG
    assert _DECOMMISSIONED not in _CATALOG
