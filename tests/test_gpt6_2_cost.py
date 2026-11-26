"""GPT-6.2 Astra and GPT-6.2 Sol cost rows -- budget-blindness anchors (2026-11-24).

Before this change ``lib/hal/cost_tracker.py`` had no row for either id, so
both would have billed at ``_DEFAULT_PRICING`` (the Sonnet-tier estimate)
silently -- the same trap gpt-6-astra and the gpt-5.6 tiers were anchored
against on 2026-09-08 (test_gpt6_astra_cost.py). Neither id collapses to a
sibling's row by accident either: "gpt-6.2-astra" does not start with
"gpt-6-astra" (the dot where the sibling has a hyphen), and vice versa, so the
point-release guard is not even needed here -- the two id strings simply never
prefix-match.

Prices read from developers.openai.com/api/docs/pricing on 2026-11-24
(Standard, short context <= 272K input): Astra $8 / $0.40 cached / $10
30-minute cache write / $40 output per 1M; Sol $2 / $0.10 / $2.50 / $10. The
>272K tier (input and cache 2x, output 1.5x, for the WHOLE request, quoted
verbatim from the vendor) is wired into `_REQUEST_TIERS`, closing the gap
AGORA's technical seat named on this PR: "A counter-test that forces a
273K-prompt request through calculate_cost and verifies it still resolves to
the short-context row rather than falling through to _DEFAULT_PRICING" --
`test_a_prompt_past_272k_tokens_bills_the_long_context_tier` below is that
test. The 12-hour cache write ($20 Astra, $5 Sol) is still not representable
in one 4-tuple and is documented in config/model_prices.yaml instead.
"""
from __future__ import annotations

import pytest

from lib.hal.cost_tracker import calculate_cost

MTOK = 1_000_000
UNKNOWN = "no-such-model-xyz"
# Well under the 272K-token cutoff, so these exercise the Standard (short-
# context) tier specifically -- an input_tokens of a full MTOK would itself
# cross the cutoff once _REQUEST_TIERS is wired (below), and silently assert
# the WRONG (long-context) row. output_tokens does not count toward the
# prompt-size trigger, so it is left at a full MTOK to keep the numbers round.
SHORT_INPUT = 100_000


@pytest.mark.parametrize("model,expected", [
    ("gpt-6.2-astra", 40.8),  # (100_000/1e6)*8 + (1_000_000/1e6)*40
    ("gpt-6.2-sol", 10.2),    # (100_000/1e6)*2 + (1_000_000/1e6)*10
])
def test_short_context_in_and_out_matches_the_published_standard_rate(model, expected):
    assert calculate_cost(model, input_tokens=SHORT_INPUT, output_tokens=MTOK) == pytest.approx(expected)


@pytest.mark.parametrize("model", ["gpt-6.2-astra", "gpt-6.2-sol"])
def test_row_is_not_the_unknown_model_default(model):
    priced = calculate_cost(model, input_tokens=SHORT_INPUT, output_tokens=MTOK)
    default = calculate_cost(UNKNOWN, input_tokens=SHORT_INPUT, output_tokens=MTOK)
    assert priced != pytest.approx(default), f"{model} still bills at the unknown-model default"


@pytest.mark.parametrize("model", ["gpt-6.2-astra", "gpt-6.2-sol"])
def test_neither_id_collapses_to_gpt_6_astras_row(model):
    """The point-release trap, inverted: prove the two id strings do not
    prefix-match each other, so the "never a sibling's price" guard is moot
    here but the OUTCOME it protects (a distinct row) still holds."""
    assert not model.startswith("gpt-6-astra")
    priced = calculate_cost(model, input_tokens=SHORT_INPUT, output_tokens=MTOK)
    astra_6 = calculate_cost("gpt-6-astra", input_tokens=SHORT_INPUT, output_tokens=MTOK)
    assert priced != pytest.approx(astra_6)


def test_address_form_resolves_to_the_same_row():
    for model, expected in (("gpt-6.2-astra", 40.8), ("gpt-6.2-sol", 10.2)):
        bare = calculate_cost(model, input_tokens=SHORT_INPUT, output_tokens=MTOK)
        addressed = calculate_cost(f"openai/{model}", input_tokens=SHORT_INPUT, output_tokens=MTOK)
        assert addressed == pytest.approx(bare) == pytest.approx(expected)


def test_cache_columns_follow_the_published_cached_and_write_rates():
    # 100K cached-read tokens + 100K cache-write tokens (short-context: the
    # 200K total prompt stays under the 272K cutoff).
    assert calculate_cost("gpt-6.2-astra", cache_read_tokens=SHORT_INPUT,
                          cache_creation_tokens=SHORT_INPUT) == pytest.approx(1.04)
    assert calculate_cost("gpt-6.2-sol", cache_read_tokens=SHORT_INPUT,
                          cache_creation_tokens=SHORT_INPUT) == pytest.approx(0.26)


@pytest.mark.parametrize("model,short_rate,long_rate", [
    ("gpt-6.2-astra", 8.0, 16.0),
    ("gpt-6.2-sol", 2.0, 4.0),
])
def test_a_prompt_past_272k_tokens_bills_the_long_context_tier(model, short_rate, long_rate):
    """AGORA's technical seat (xai/grok-4.3) on this PR: a 273K-token prompt
    must resolve to the long-context row, never silently fall through to the
    short-context rate or to _DEFAULT_PRICING. 272,000 exactly is still
    short-context (the vendor's cutoff is "more than 272K"); 273,000 is not."""
    at_cutoff = calculate_cost(model, input_tokens=272_000, output_tokens=0)
    past_cutoff = calculate_cost(model, input_tokens=273_000, output_tokens=0)
    assert at_cutoff == pytest.approx(272_000 / MTOK * short_rate)
    assert past_cutoff == pytest.approx(273_000 / MTOK * long_rate)


@pytest.mark.parametrize("model,long_input_rate,long_output_rate", [
    ("gpt-6.2-astra", 16.0, 60.0),
    ("gpt-6.2-sol", 4.0, 15.0),
])
def test_long_context_output_is_one_and_a_half_times_the_short_rate(model, long_input_rate, long_output_rate):
    """Vendor quote: "output costs 1.5 times the short-context rate for the
    full request" once input passes 272K. Checked two ways: the published
    long-context output figure is exactly 1.5x the short-context one (a fact
    about the vendor's own numbers), AND a real past-cutoff call actually
    bills the whole request at that figure (a fact about this module) --
    the first alone cannot fail if `long_input_rate`/`long_output_rate` above
    were simply copied from the short rates times 1.5 without the module
    agreeing, so `past_cutoff` must be asserted too."""
    short_output_rate = calculate_cost(model, input_tokens=0, output_tokens=MTOK)
    assert long_output_rate == pytest.approx(short_output_rate * 1.5)
    past_cutoff = calculate_cost(model, input_tokens=273_000, output_tokens=MTOK)
    expected_total = (273_000 / MTOK) * long_input_rate + long_output_rate
    assert past_cutoff == pytest.approx(expected_total)
