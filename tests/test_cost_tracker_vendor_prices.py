"""Vendor prices, pinned to the page each one was read from.

Every row below was re-read from the vendor's own pricing page on 2026-10-01,
unless its note names a later read date. A row that drifts from what its page said that day fails here, so changing a
price means touching this file too, with a new source and date. Two rows had
drifted without anyone noticing: gemini-3.1-pro carried gemini-2.5-pro's price,
and DeepSeek re-priced V4 (peak and off-peak rates) after the 2026-07-26 read.

Tuples are (input, output, cache read, cache write) in USD per 1M tokens.
"""
from pathlib import Path

import pytest

from lib.hal import cost_tracker as ct

REPO = Path(__file__).resolve().parent.parent

READ_ON = "2026-10-01"
GOOGLE = "https://ai.google.dev/gemini-api/docs/pricing"
DEEPSEEK = "https://api-docs.deepseek.com/quick_start/pricing"
XAI = "https://docs.x.ai/docs/models"
DASHSCOPE = "https://www.alibabacloud.com/help/en/model-studio/model-pricing"
ANTHROPIC = "https://platform.claude.com/docs/en/about-claude/pricing"
MISTRAL = "https://docs.mistral.ai/inference/pricing"

# model -> (expected row, source page, what the page said)
PINNED = {
    "gemini-3.1-pro": ((2.00, 12.00, 0.20, 2.00), GOOGLE,
                       "gemini-3.1-pro-preview: input $2.00, output (including thinking "
                       "tokens) $12.00, context caching $0.20, prompts up to 200k"),
    "gemini-4-argon": ((2.00, 10.00, 0.10, 2.00), GOOGLE,
                       "gemini-4-argon, read 2026-10-18: input '$2.00 through November 30, "
                       "2026.', output (including thinking tokens) '$10.00 through November "
                       "30, 2026.', context caching '$0.10 through November 30, 2026.'"),
    "gemini-3.5-flash": ((1.50, 9.00, 0.15, 1.50), GOOGLE,
                         "gemini-3.5-flash: input $1.50, output $9.00, caching $0.15 (the last "
                         "published rate; deprecated 2026-10-08 and off the page since, row kept "
                         "while the id is accepted)"),
    "gemini-3.6-flash": ((0.75, 3.75, 0.075, 0.75), GOOGLE,
                         "gemini-3.6-flash, read 2026-10-18: input '$0.75 through December 31, "
                         "2026.', output (including thinking) '$3.75 through December 31, 2026.', "
                         "caching '$0.075'"),
    "gemini-3.8-flash": ((0.75, 3.75, 0.075, 0.75), GOOGLE,
                         "gemini-3.8-flash, read 2026-10-18: the same row as gemini-3.6-flash"),
    "gemini-3.5-flash-lite": ((0.30, 2.50, 0.03, 0.30), GOOGLE,
                              "gemini-3.5-flash-lite: input $0.30, output $2.50, caching $0.03"),
    "gemini-3.1-flash-lite": ((0.25, 1.50, 0.025, 0.25), GOOGLE,
                              "gemini-3.1-flash-lite: input $0.25, output $1.50, caching $0.025"),
    "gemini-2.5-pro": ((1.25, 10.00, 0.125, 1.25), GOOGLE,
                       "gemini-2.5-pro: input $1.25, output $10.00, caching $0.125, "
                       "prompts up to 200k"),
    "gemini-2.5-flash": ((0.30, 2.50, 0.03, 0.30), GOOGLE,
                         "gemini-2.5-flash: input $0.30 (text / image / video), output "
                         "$2.50, caching $0.03"),
    "gemini-2.5-flash-lite": ((0.10, 0.40, 0.01, 0.10), GOOGLE,
                              "gemini-2.5-flash-lite: input $0.10 (text / image / video), "
                              "output $0.40, caching $0.01"),
    "deepseek-flash": ((0.30, 1.20, 0.006, 0.30), DEEPSEEK,
                       "deepseek-flash PEAK: cache hit $0.006, cache miss $0.3, output $1.2"),
    "deepseek-v4-flash": ((0.30, 1.20, 0.006, 0.30), DEEPSEEK,
                          "legacy name, 'billed at the Flash price'"),
    "deepseek-v4-pro": ((1.32, 3.96, 0.044, 1.32), DEEPSEEK,
                        "deepseek-v4-pro PEAK: cache hit $0.044, cache miss $1.32, "
                        "output $3.96"),
    "grok-4.5": ((2.00, 6.00, 0.30, 2.00), XAI,
                 "rendered card: $2.00 / 1M tokens in, $6.00 / 1M tokens out; "
                 "embedded data 20000 / 60000 / cached 3000"),
    "grok-4.3": ((1.25, 2.50, 0.20, 1.25), XAI,
                 "embedded data 12500 / 25000 / cached 2000, same unit as grok-4.5"),
    "grok-4.6": ((2.00, 6.00, 0.50, 2.00), XAI,
                 "read 2026-11-16: embedded data 20000 / 60000 / cached 5000 on the "
                 "global endpoint's clusters, same unit as grok-4.5"),
    "grok-4.7": ((2.00, 6.00, 0.50, 2.00), XAI,
                 "read 2026-11-24 (docs.x.ai/developers/models/grok-4.7): embedded data "
                 "20000 / 60000 / cached 5000, same unit as grok-4.5/4.6; release-notes "
                 "prose: '$2 / $0.50 / $6 per 1M tokens' below 200k prompt tokens"),
    "grok-4.20-0309-reasoning": ((1.25, 2.50, 0.20, 1.25), XAI,
                 "read 2026-11-24: embedded data 12500 / 25000 / cached 2000, the same "
                 "row as grok-4.3 -- this id was already in both providers.yaml files' "
                 "xai `models:` list, and in this file's own `premium` routing tier, "
                 "with no price row at all until this read"),
    "grok-4.20-0309-non-reasoning": ((1.25, 2.50, 0.20, 1.25), XAI,
                 "read 2026-11-24: the same embedded row as grok-4.20-0309-reasoning and "
                 "grok-4.3; PARITY -- GRIP's lib/broly_council.py and lib/council_seat.py "
                 "already call this id"),
    "qwen3.5-flash": ((0.10, 0.40, 0.10, 0.10), DASHSCOPE,
                      "qwen3.5-flash, International, 0<Token<=1M: $0.1 in, $0.4 out"),
    "qwen3.7-plus": ((0.40, 1.60, 0.08, 0.50), DASHSCOPE,
                     "qwen3.7-plus, read 2026-11-16, International, 0<Token<=256K: "
                     "'List price $0.4' in, 'List price $1.6' out; cache per "
                     "model-studio/context-cache: implicit hit 20%, explicit creation 125%"),
    "qwen3.8-max": ((2.00, 6.00, 2.00, 2.00), DASHSCOPE,
                    "qwen3.8-max, International, 0<Token<=1M: $2 in, $6 out"),
    "qwen3.7-flash": ((0.03, 0.13, 0.03, 0.03), DASHSCOPE,
                      "qwen3.7-flash, International, 0<Token<=32K: $0.030 in, $0.130 out"),
    "claude-haiku-5-5": ((0.10, 0.50, 0.01, 0.125), ANTHROPIC,
                         "Claude Haiku 5.5, read 2026-10-09: $0.10 / MTok for prompts up to "
                         "100,000 tokens, $0.50 out, $0.01 cache read, $0.125 5m cache write "
                         "(the $0.20 1h write rate lives in config/model_prices.yaml; these "
                         "rows have no 1h column)"),
    "claude-sonnet-5-5": ((2.00, 10.00, 0.10, 2.50), ANTHROPIC,
                          "Claude Sonnet 5.5, read 2026-10-12: $2 / MTok in, $2.50 5m cache "
                          "write ($4 for 1h, in config/model_prices.yaml), $0.10 cache hit "
                          "(footnote 2: 0.05x the base input price), $10 out"),
    "mistral-large-4": ((0.70, 2.10, 0.07, 0.70), MISTRAL,
                        "read 2026-11-24 from the page's own embedded pricing data (fetched "
                        "and the JSON read, not a search summary): the 'mistral-large-4' card "
                        "(weights huggingface.co/mistralai/Mistral-Large-4-1T-A52B) lists "
                        "priceUsd 0.7 input, 0.07 cached input, 2.1 output per million tokens; "
                        "no published cache-write rate, so that slot carries the input rate "
                        "as an upper bound"),
    "mistral-small-2603": ((0.15, 0.60, 0.15, 0.15), MISTRAL,
                           "read 2026-11-24, same page: the 'mistral-small-2603' card (weights "
                           "huggingface.co/mistralai/Mistral-Small-4-2603) lists priceUsd 0.15 "
                           "input, 0.6 output; no cached-input row at all, so both cache slots "
                           "carry the input rate as an upper bound"),
    # GRIP#9420: the parity gap this issue named. mistral-large-latest had no
    # catalogue or cost row at all; mistral-small-latest had a row, but with
    # fabricated (0.0, 0.0) cache figures rather than the vendor's own upper
    # bound. Both share a pricing CARD with a sibling already pinned above.
    "mistral-large-latest": ((0.70, 2.10, 0.07, 0.70), MISTRAL,
                            "read 2026-11-24: shares mistral-large-4's card "
                            "(modelIds mistral-large-4 / mistral-large-4-0 / "
                            "mistral-large-latest) -- the same row"),
    "mistral-small-latest": ((0.15, 0.60, 0.15, 0.15), MISTRAL,
                            "read 2026-11-24: shares mistral-small-2603's card "
                            "(modelIds mistral-small-2603 / mistral-small-latest) -- the "
                            "same row, not the (0.0, 0.0) cache pair the old row carried"),
}


@pytest.mark.parametrize("model", sorted(PINNED))
def test_row_matches_its_vendor_page(model):
    expected, source, said = PINNED[model]
    assert ct._PRICING[model] == expected, (
        f"{model} no longer matches {source} as read on the date its note names, "
        f"or on {READ_ON} when it names none ({said}). "
        "Re-read the page and update both the row and this pin.")


@pytest.mark.parametrize("address, row", [
    ("gemini/gemini-3.1-pro-preview", "gemini-3.1-pro"),   # AGORA's judge
    ("deepseek/deepseek-v4-flash", "deepseek-v4-flash"),   # AGORA's first speaker
    ("qwen/qwen3.5-flash", "qwen3.5-flash"),
    ("qwen/qwen3.8-max", "qwen3.8-max"),
    ("xai/grok-4.3", "grok-4.3"),
    ("xai/grok-4.6", "grok-4.6"),
    ("xai/grok-4.7", "grok-4.7"),
    ("xai/grok-4.20-0309-reasoning", "grok-4.20-0309-reasoning"),
    ("xai/grok-4.20-0309-non-reasoning", "grok-4.20-0309-non-reasoning"),
    ("qwen/qwen3.7-plus", "qwen3.7-plus"),
    ("qwen/qwen3.7-plus-2026-05-26", "qwen3.7-plus"),
    ("gemini/gemini-2.5-flash-lite", "gemini-2.5-flash-lite"),
])
def test_provider_addresses_price_at_their_row(address, row):
    assert ct._resolve_pricing(address) == PINNED[row][0]


def test_one_judge_sitting_costs_four_cents():
    # The Synthesis Ladder plan's estimate: ~8K tokens in and ~2K out per
    # judging at Google's $2 / $12. If this moves, so does the budget.
    cost = ct.calculate_cost("gemini/gemini-3.1-pro-preview",
                             input_tokens=8_000, output_tokens=2_000)
    assert cost == pytest.approx(0.040)


def test_flash_lite_no_longer_bills_as_flash():
    assert ct._resolve_pricing("gemini-2.5-flash-lite") != ct._resolve_pricing("gemini-2.5-flash")


MISTRAL_IDS = ["mistral-large-4", "mistral-small-2603",
               "mistral-large-latest", "mistral-small-latest"]


@pytest.mark.parametrize("model", MISTRAL_IDS)
def test_mistral_never_falls_to_the_free_local_pattern(model):
    """_LOCAL_PATTERNS below includes the "mistral" substring, so any Mistral id
    without its own row here bills $0 -- a priced id must never collapse to it."""
    assert ct._resolve_pricing(model) != ct._PRICING["local"]
    assert ct._resolve_pricing(f"mistral/{model}") != ct._PRICING["local"]


@pytest.mark.parametrize("model", MISTRAL_IDS)
def test_removing_the_row_now_bills_the_loud_default_not_free(model, monkeypatch):
    """GRIP#9420 fixed: _resolve_pricing's local-family check now requires a
    "/" in the model string, because every paid api.mistral.ai id is bare --
    this is what actually falsifies the bug for a BARE id. Before that fix,
    this exact deletion fell back to ct._PRICING["local"] (the prior version
    of this test, kept as the forward anchor it replaces): a bare id no longer
    collapsing to free local, even with its row gone, is the regression this
    narrowing closes."""
    monkeypatch.delitem(ct._PRICING, model)
    assert ct._resolve_pricing(model) == ct._DEFAULT_PRICING
    assert ct._resolve_pricing(model) != ct._PRICING["local"]


def test_a_provider_qualified_unpriced_mistral_id_is_a_known_residual_gap():
    """R0-honest boundary, not a claim of a complete fix: "mistral/<id>" supplies
    its OWN "/" and its OWN "mistral" substring from the PROVIDER prefix alone,
    independent of what <id> is. So an address-qualified call to a Mistral id
    with no row of its own still reaches the free local rate -- exactly as it
    did before this PR, for exactly the same reason, on a narrower surface
    (every id GRIP and HAL route today has its own row after this PR; only a
    FUTURE unrowed Mistral id, called in this address form, would hit it).
    Closing this fully needs cost_tracker to know which address prefixes name
    a cloud BILLING provider rather than a local HF-repo org, which couples
    this pure pricing module to config/providers.yaml -- left for a follow-up,
    named here rather than silently left uncovered."""
    assert "nonexistent-brand-new-mistral-id" not in ct._PRICING
    assert (ct._resolve_pricing("mistral/nonexistent-brand-new-mistral-id")
            == ct._PRICING["local"])


def test_a_bare_unknown_id_resembling_a_local_family_bills_the_default():
    """The positive case the fix exists for: a bare id that merely CONTAINS one
    of _LOCAL_PATTERNS' substrings, with no row and no "/", now bills the loud
    default rather than silently collapsing to free."""
    for fake in ("mistral-brand-new-2099", "phi-5-cloud-preview",
                 "llama3-enterprise-tier"):
        assert fake not in ct._PRICING
        assert ct._resolve_pricing(fake) == ct._DEFAULT_PRICING


def test_a_slash_qualified_local_address_still_resolves_free():
    """The narrowing must not cost the genuinely-local case: an HF-repo-style
    address with no row of its own, naming a local-only family, stays free."""
    for fake in ("TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                 "some-org/llama3-finetune-4bit"):
        assert fake not in ct._PRICING
        assert ct._resolve_pricing(fake) == ct._PRICING["local"]


def test_legacy_deepseek_vision_id_bills_at_the_flash_price():
    assert ct._resolve_pricing("deepseek-v4-flash-vision-exp") == PINNED["deepseek-flash"][0]


# Request tiers, read on the same day from the same pages. DashScope, qwen3.7-flash,
# International: "32K<Token<=256K $0.100 $0.400" and "256K<Token<=1M $0.200 $0.800".
# Google, gemini-3.1-pro-preview: "$4.00, prompts > 200k tokens" in and "$18.00,
# prompts > 200k" out (thinking included), "$0.40, prompts > 200k" cached. The
# whole request bills at the tier its prompt reaches.
@pytest.mark.parametrize("model, prompt, rates", [
    ("qwen3.7-flash", 32_000, (0.03, 0.13)),
    ("qwen3.7-flash", 32_001, (0.10, 0.40)),
    ("qwen/qwen3.7-flash", 256_001, (0.20, 0.80)),
    ("gemini/gemini-3.1-pro-preview", 200_000, (2.00, 12.00)),
    ("gemini/gemini-3.1-pro-preview", 200_001, (4.00, 18.00)),
    # Anthropic, Claude Haiku 5.5, read 2026-10-09: "$0.10 / MTok for prompts up to
    # 100,000 tokens" and "$0.50 / MTok for prompts over 100,000 tokens" in, $0.50
    # and $2.50 out.
    ("claude-haiku-5-5", 100_000, (0.10, 0.50)),
    ("claude-haiku-5-5", 100_001, (0.50, 2.50)),
    # Read 2026-11-16 (GRIP#9205). xAI's embedded data puts grok-4.6 at 40000 /
    # 120000 past its "longContextThreshold":"200000"; Alibaba lists qwen3.7-plus,
    # International, at $1.2 in and $4.8 out for "256K<Token<=1M".
    ("xai/grok-4.6", 200_000, (2.00, 6.00)),
    ("xai/grok-4.6", 200_001, (4.00, 12.00)),
    # Read 2026-11-24 (dogfood-domain family 5). grok-4.7 is the same tier as
    # grok-4.6; both grok-4.20-0309 variants are the same tier as grok-4.3.
    ("xai/grok-4.7", 200_000, (2.00, 6.00)),
    ("xai/grok-4.7", 200_001, (4.00, 12.00)),
    ("xai/grok-4.20-0309-reasoning", 200_000, (1.25, 2.50)),
    ("xai/grok-4.20-0309-reasoning", 200_001, (2.50, 5.00)),
    ("xai/grok-4.20-0309-non-reasoning", 200_000, (1.25, 2.50)),
    ("xai/grok-4.20-0309-non-reasoning", 200_001, (2.50, 5.00)),
    ("qwen/qwen3.7-plus", 256_000, (0.40, 1.60)),
    ("qwen/qwen3.7-plus", 256_001, (1.20, 4.80)),
    ("qwen/qwen3.7-plus-2026-05-26", 256_001, (1.20, 4.80)),
    # GRIP#9220, the same xAI data: grok-4.5 at 40000 / 120000 and grok-4.3 at
    # 25000 / 50000 past the same 200,000 tokens.
    ("xai/grok-4.5", 200_000, (2.00, 6.00)),
    ("xai/grok-4.5", 200_001, (4.00, 12.00)),
    ("xai/grok-4.3", 200_000, (1.25, 2.50)),
    ("xai/grok-4.3", 200_001, (2.50, 5.00)),
    ("xai/grok-4.3-latest", 200_001, (2.50, 5.00)),
])
def test_a_large_prompt_bills_the_whole_request_at_its_tier(model, prompt, rates):
    cost = ct.calculate_cost(model, input_tokens=prompt, output_tokens=1_000_000)
    assert cost == pytest.approx(prompt / 1_000_000 * rates[0] + rates[1])


def test_the_new_tier_rows_pin_their_cache_rates_too():
    """The test above checks input and output only (GRIP#9205). grok-4.6's long
    tier carries xAI's own cached rate; qwen3.7-plus's carries 20% of the tier's
    input for a cache read and 125% for a cache write, as its base row does."""
    assert ct._REQUEST_TIERS["grok-4.6"] == ((200_000, (4.00, 12.00, 1.00, 4.00)),)
    assert ct._REQUEST_TIERS["qwen3.7-plus"] == ((256_000, (1.20, 4.80, 0.24, 1.50)),)
    # GRIP#9220: the older xAI rows carry xAI's cached rate and the input rate for writes.
    assert ct._REQUEST_TIERS["grok-4.5"] == ((200_000, (4.00, 12.00, 0.60, 4.00)),)
    assert ct._REQUEST_TIERS["grok-4.3"] == ((200_000, (2.50, 5.00, 0.40, 2.50)),)
    # Read 2026-11-24 (dogfood-domain family 5).
    assert ct._REQUEST_TIERS["grok-4.7"] == ((200_000, (4.00, 12.00, 1.00, 4.00)),)
    assert ct._REQUEST_TIERS["grok-4.20-0309-reasoning"] == ((200_000, (2.50, 5.00, 0.40, 2.50)),)
    assert ct._REQUEST_TIERS["grok-4.20-0309-non-reasoning"] == ((200_000, (2.50, 5.00, 0.40, 2.50)),)


def test_cached_tokens_count_toward_the_prompt_tier():
    # 150K fresh plus 60K cached is a 210K prompt, over Gemini's 200K line.
    cost = ct.calculate_cost("gemini-3.1-pro", input_tokens=150_000, cache_read_tokens=60_000)
    assert cost == pytest.approx(0.150 * 4.00 + 0.060 * 0.40)


def test_an_untiered_model_ignores_prompt_size():
    cost = ct.calculate_cost("deepseek-v4-flash", input_tokens=900_000)
    assert cost == pytest.approx(0.9 * PINNED["deepseek-v4-flash"][0][0])


@pytest.mark.parametrize("model", sorted(ct._REQUEST_TIERS))
def test_tiers_are_ordered_and_only_ever_rise(model):
    # The lookup keeps the LAST tier a prompt passes, so the thresholds must
    # ascend; and a larger prompt must never bill cheaper per token.
    thresholds = [threshold for threshold, _ in ct._REQUEST_TIERS[model]]
    assert thresholds == sorted(thresholds)
    rows = [ct._PRICING[model]] + [rates for _, rates in ct._REQUEST_TIERS[model]]
    assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(rows, rows[1:]))


# Gemini 4 Argon's rate changes on a calendar date, not by prompt size: the same
# page, read 2026-10-18, says "$4.00 starting December 1, 2026." in, "$20.00
# starting December 1, 2026." out (thinking included), "$0.20 starting December
# 1, 2026." cached. lib/hal/cost_tracker._PRICE_SCHEDULE carries that change and
# ct._today is the seam these tests move the calendar through.
ARGON_INTRO = (2.00, 10.00, 0.10, 2.00)
ARGON_FROM_DECEMBER = (4.00, 20.00, 0.20, 4.00)


def test_argon_bills_the_introductory_rate_through_november(monkeypatch):
    monkeypatch.setattr(ct, "_today", lambda: "2026-11-30")
    assert ct._resolve_pricing("gemini-4-argon") == ARGON_INTRO
    assert ct._resolve_pricing("gemini/gemini-4-argon") == ARGON_INTRO


def test_argon_doubles_from_the_first_of_december(monkeypatch):
    monkeypatch.setattr(ct, "_today", lambda: "2026-12-01")
    assert ct._resolve_pricing("gemini-4-argon") == ARGON_FROM_DECEMBER
    assert ct._resolve_pricing("gemini/gemini-4-argon") == ARGON_FROM_DECEMBER


def test_argon_has_no_prompt_length_tier(monkeypatch):
    monkeypatch.setattr(ct, "_today", lambda: "2026-11-30")
    assert ct._request_rates("gemini-4-argon", 500_000) == ARGON_INTRO
    monkeypatch.setattr(ct, "_today", lambda: "2026-12-01")
    assert ct._request_rates("gemini-4-argon", 500_000) == ARGON_FROM_DECEMBER


def test_a_dated_change_never_touches_another_row(monkeypatch):
    monkeypatch.setattr(ct, "_today", lambda: "2027-01-01")
    assert ct._resolve_pricing("gemini-3.1-pro-preview") == PINNED["gemini-3.1-pro"][0]


FLASH_INTRO = (0.75, 3.75, 0.075, 0.75)
FLASH_FROM_2027 = (1.50, 7.50, 0.15, 1.50)


@pytest.mark.parametrize("model", ["gemini-3.6-flash", "gemini-3.8-flash"])
def test_the_flash_rows_double_on_new_years_day_2027(monkeypatch, model):
    monkeypatch.setattr(ct, "_today", lambda: "2026-12-31")
    assert ct._resolve_pricing(model) == FLASH_INTRO
    monkeypatch.setattr(ct, "_today", lambda: "2027-01-01")
    assert ct._resolve_pricing(model) == FLASH_FROM_2027
    assert ct._request_rates(f"gemini/{model}", 900_000) == FLASH_FROM_2027


# --- GRIP#9420: every paid id in the mistral/openai provider lists, swept ------

def _provider_models(provider_name: str) -> list:
    """The bare model ids THIS checkout's config/providers.yaml declares for one
    provider -- read from the live YAML, not copied into the test, so a model
    added to providers.yaml with no price row fails here on its own."""
    data = __import__("yaml").safe_load((REPO / "config" / "providers.yaml").read_text())
    for provider in data["providers"]:
        if provider["name"] == provider_name:
            return provider["models"]
    raise AssertionError(f"no provider named {provider_name!r} in providers.yaml")


@pytest.mark.parametrize("provider", ["mistral", "openai"])
def test_every_paid_id_in_the_provider_list_has_its_own_row(provider):
    """AGORA (HAL#837) asked for a stronger check than "not these two sentinel
    values": _pricing_key resolving to a real key is what actually means the
    id has its own row. It also asked to document the bare-id assumption this
    sweep and the "/" narrowing both rest on: every id providers.yaml lists
    directly (not through an org/repo address) is written WITHOUT a slash."""
    for model in _provider_models(provider):
        assert "/" not in model, f"{model} ({provider}) is not a bare id"
        assert ct._pricing_key(model) is not None, (
            f"{model} ({provider}) has no row of its own at all")
        assert ct._resolve_pricing(model) != ct._PRICING["local"], (
            f"{model} ({provider}) resolves to the free local rate")
        assert ct._resolve_pricing(model) != ct._DEFAULT_PRICING, (
            f"{model} ({provider}) has no row of its own -- it bills at the "
            "Sonnet-tier default")


# Gemini 4 Carbon and Hydrogen, onboarded 2026-11-24 (dogfood-domain phase 2
# family 6). aistudio.google.com/docs/pricing, read 2026-11-24: Carbon's
# Standard row is "$2.00" in / "$10.00" out / "$0.10" cached "through November
# 30, 2026", then "$4.00" / "$20.00" / "$0.20" "starting December 1, 2026" --
# the same calendar date Argon doubles. Hydrogen's row carries no dated
# change: "$1.50" in / "$7.50" out / "$0.075" cached, flat. Neither page lists
# a prompt-length tier.
CARBON_INTRO = (2.00, 10.00, 0.10, 2.00)
CARBON_FROM_DECEMBER = (4.00, 20.00, 0.20, 4.00)
HYDROGEN_FLAT = (1.50, 7.50, 0.075, 1.50)


def test_carbon_bills_the_introductory_rate_through_november(monkeypatch):
    monkeypatch.setattr(ct, "_today", lambda: "2026-11-30")
    assert ct._resolve_pricing("gemini-4-carbon") == CARBON_INTRO
    assert ct._resolve_pricing("gemini/gemini-4-carbon") == CARBON_INTRO


def test_carbon_doubles_from_the_first_of_december(monkeypatch):
    monkeypatch.setattr(ct, "_today", lambda: "2026-12-01")
    assert ct._resolve_pricing("gemini-4-carbon") == CARBON_FROM_DECEMBER
    assert ct._resolve_pricing("gemini/gemini-4-carbon") == CARBON_FROM_DECEMBER


def test_carbon_has_no_prompt_length_tier(monkeypatch):
    monkeypatch.setattr(ct, "_today", lambda: "2026-11-30")
    assert ct._request_rates("gemini-4-carbon", 500_000) == CARBON_INTRO
    monkeypatch.setattr(ct, "_today", lambda: "2026-12-01")
    assert ct._request_rates("gemini-4-carbon", 500_000) == CARBON_FROM_DECEMBER


def test_hydrogen_bills_a_flat_rate_with_no_dated_change(monkeypatch):
    """Unlike Carbon and Argon, Hydrogen's page lists no future rate, so the
    price never moves regardless of what day it is.
    Mutation: add a _PRICE_SCHEDULE row for gemini-4-hydrogen -> RED on the
    second assert, once 'today' crosses that row's date."""
    monkeypatch.setattr(ct, "_today", lambda: "2026-11-24")
    assert ct._resolve_pricing("gemini-4-hydrogen") == HYDROGEN_FLAT
    monkeypatch.setattr(ct, "_today", lambda: "2027-06-01")
    assert ct._resolve_pricing("gemini-4-hydrogen") == HYDROGEN_FLAT
    assert ct._request_rates("gemini/gemini-4-hydrogen", 900_000) == HYDROGEN_FLAT


def test_carbons_dated_change_never_touches_hydrogen_or_argon(monkeypatch):
    monkeypatch.setattr(ct, "_today", lambda: "2026-12-01")
    assert ct._resolve_pricing("gemini-4-hydrogen") == HYDROGEN_FLAT
    assert ct._resolve_pricing("gemini-4-argon") == ARGON_FROM_DECEMBER


# Gemini 4 Neon, onboarded 2026-11-26. ai.google.dev/gemini-api/docs/pricing,
# read 2026-11-26, Standard paid tier: "$3.00" in, "$15.00" out (thinking
# included), "$0.15" cached, with no dated change and no prompt-length tier.
NEON_FLAT = (3.00, 15.00, 0.15, 3.00)


def test_neon_bills_a_flat_rate_with_no_dated_change(monkeypatch):
    """Mutation: add a _PRICE_SCHEDULE row for gemini-4-neon, or price it at a
    sibling's rate -> RED."""
    monkeypatch.setattr(ct, "_today", lambda: "2026-11-26")
    assert ct._resolve_pricing("gemini-4-neon") == NEON_FLAT
    assert ct._resolve_pricing("gemini/gemini-4-neon") == NEON_FLAT
    monkeypatch.setattr(ct, "_today", lambda: "2027-06-01")
    assert ct._resolve_pricing("gemini-4-neon") == NEON_FLAT
    assert ct._request_rates("gemini/gemini-4-neon", 900_000) == NEON_FLAT
