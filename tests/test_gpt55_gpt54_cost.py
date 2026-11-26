"""GPT-5.5 and GPT-5.4 cost rows — budget-blindness anchors (2026-09-09).

``config/providers.yaml`` declares ``gpt-5.5``, ``gpt-5.4`` and
``gpt-5.4-mini`` on the openai provider, but ``lib/hal/cost_tracker.py`` had
no row for any of them, so every call billed at ``_DEFAULT_PRICING`` — the
Sonnet-tier 3.0 / 15.0 estimate. That is not a small rounding error in either
direction: gpt-5.5 was UNDER-counted (true 5.0 / 30.0) and gpt-5.4-mini was
OVER-counted more than threefold (true 0.75 / 4.5).

Prices read from platform.openai.com/docs/pricing on 2026-09-09, the main
text-token table, Standard tier. Its columns are "Model | Input | Cached
input | Cache writes | Output", and the rows are, verbatim:

    gpt-5.5 (<272K context length)   5      0.5     -     30
    gpt-5.4 (<272K context length)   2.5    0.25    -     15
    gpt-5.4-mini                     0.75   0.075         4.5

A "-" or absent "Cache writes" cell is not a missing price. The column's own
tooltip reads "Input tokens are either Input, Cached Input, or Cache Write
and writes are not an additive fee" — with no separate write rate published,
a write is billed at the standard Input rate, so cache_write == input here.
That is the convention every existing OpenAI row in the table already
follows (gpt-4o 2.5/2.5, gpt-4o-mini 0.15/0.15, o1 15/15, o3-mini 1.1/1.1),
and those three rows independently confirm the column order above.

Not representable in one tuple, and deliberately left off the row: Batch and
Flex (0.5x), Fast mode (2x), and — for gpt-5.5 and gpt-5.4, whose published
rows are labelled "<272K context length" — any >272K rate.

Observed red before the rows landed: all three resolved to _DEFAULT_PRICING
and billed 18.0 for 1M in + 1M out, so every exact-rate assertion below
failed.
"""
from __future__ import annotations

import pytest

from lib.hal.cost_tracker import _DEFAULT_PRICING, _resolve_pricing, calculate_cost

MTOK = 1_000_000

# (model, input, output, cached_input) exactly as published.
PUBLISHED = [
    ("gpt-5.5", 5.0, 30.0, 0.5),
    ("gpt-5.4", 2.5, 15.0, 0.25),
    ("gpt-5.4-mini", 0.75, 4.5, 0.075),
]


@pytest.mark.parametrize("model,inp,out,cached", PUBLISHED)
def test_row_matches_the_published_standard_rate(model, inp, out, cached):
    assert calculate_cost(model, input_tokens=MTOK) == pytest.approx(inp)
    assert calculate_cost(model, output_tokens=MTOK) == pytest.approx(out)
    assert calculate_cost(model, cache_read_tokens=MTOK) == pytest.approx(cached)


@pytest.mark.parametrize("model,inp,_out,_cached", PUBLISHED)
def test_cache_write_bills_at_the_standard_input_rate(model, inp, _out, _cached):
    # OpenAI publishes no separate write rate for these ids; a write is an
    # Input token, not an additive fee.
    assert calculate_cost(model, cache_creation_tokens=MTOK) == pytest.approx(inp)


@pytest.mark.parametrize("model,_i,_o,_c", PUBLISHED)
def test_the_unknown_model_default_is_no_longer_reached(model, _i, _o, _c):
    assert _resolve_pricing(model) != _DEFAULT_PRICING, (
        f"{model} is declared in config/providers.yaml but still bills at the "
        "Sonnet-tier estimate for unknown models"
    )


@pytest.mark.parametrize("model,inp,out,_c", PUBLISHED)
def test_address_form_resolves_to_the_same_row(model, inp, out, _c):
    bare = calculate_cost(model, input_tokens=MTOK, output_tokens=MTOK)
    addressed = calculate_cost(f"openai/{model}", input_tokens=MTOK, output_tokens=MTOK)
    assert addressed == pytest.approx(bare) == pytest.approx(inp + out)


def test_mini_keeps_its_own_row_and_is_not_captured_by_the_gpt_5_4_prefix():
    # _resolve_pricing falls back to a startswith scan, so an exact row for
    # the mini is what keeps it off gpt-5.4's much larger rate.
    assert _resolve_pricing("gpt-5.4-mini") != _resolve_pricing("gpt-5.4")
    assert calculate_cost("gpt-5.4-mini", input_tokens=MTOK) == pytest.approx(0.75)
