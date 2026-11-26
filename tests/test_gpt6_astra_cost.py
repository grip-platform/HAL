"""GPT-6 Astra and GPT-5.6 cost rows — budget-blindness anchors (2026-09-08).

Before this change ``lib/hal/cost_tracker.py`` had no row for any gpt-5.x id,
so gpt-5.6-sol / -terra / -luna billed at ``_DEFAULT_PRICING`` (the
Sonnet-tier 3.0/15.0 estimate) silently, and gpt-6-astra would have too.
Prices are read from platform.openai.com/docs/pricing on 2026-09-08
(Standard, short context <= 272K input): Astra $10 / $1 cached / $12.50 cache
write / $50 output per 1M; Sol $4 / $0.40 / $5 / $20; Terra $2 / $0.20 /
$2.50 / $12; Luna $0.20 / $0.02 / $0.25 / $1.20. The >272K tier (2x input and
cache, 1.5x output), Batch/Flex (0.5x) and Fast mode (2x) are not
representable in one tuple and are documented on the row instead.

Observed red before the rows landed: every id below resolved to the default
18.0 per 2M tokens, so the inequality against an unknown id failed.
"""
from __future__ import annotations

import pytest

from lib.hal.cost_tracker import calculate_cost

MTOK = 1_000_000
UNKNOWN = "no-such-model-xyz"


@pytest.mark.parametrize("model,expected", [
    ("gpt-6-astra", 60.0),
    ("gpt-5.6-sol", 24.0),
    ("gpt-5.6-terra", 14.0),
    ("gpt-5.6-luna", 1.4),
])
def test_one_million_in_and_out_matches_the_published_standard_rate(model, expected):
    assert calculate_cost(model, input_tokens=MTOK, output_tokens=MTOK) == pytest.approx(expected)


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
def test_row_is_not_the_unknown_model_default(model):
    priced = calculate_cost(model, input_tokens=MTOK, output_tokens=MTOK)
    default = calculate_cost(UNKNOWN, input_tokens=MTOK, output_tokens=MTOK)
    assert priced != pytest.approx(default), f"{model} still bills at the unknown-model default"


def test_address_form_resolves_to_the_same_row():
    bare = calculate_cost("gpt-6-astra", input_tokens=MTOK, output_tokens=MTOK)
    addressed = calculate_cost("openai/gpt-6-astra", input_tokens=MTOK, output_tokens=MTOK)
    assert addressed == pytest.approx(bare) == pytest.approx(60.0)


def test_cache_columns_follow_the_published_cached_and_write_rates():
    # 1M cached-read tokens at $1 + 1M cache-write tokens at $12.50.
    assert calculate_cost("gpt-6-astra", cache_read_tokens=MTOK,
                          cache_creation_tokens=MTOK) == pytest.approx(13.5)
