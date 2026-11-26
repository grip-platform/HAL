"""HAL Cost Tracker — token-to-USD calculation for any provider.

Pricing data sourced from CC's modelCost.ts (TF++ analysis) and extended
with Groq, Gemini, OpenRouter, and local model costs.

Usage:
    from lib.hal.cost_tracker import CostTracker

    from lib.hal.cost_tracker import calculate_cost

    tracker = CostTracker()
    cost = calculate_cost("claude-opus-4-7", input_tokens=1000, output_tokens=500)
    tracker.record("claude-opus-4-7", input_tokens=1000, output_tokens=500)
    print(tracker.total_usd)
"""
import re
from datetime import datetime, timezone
import threading

from lib.hal.types import CostSnapshot, TokenUsage

# Pricing per million tokens (USD) — sourced from CC modelCost.ts + provider docs
# Format: (input_per_mtok, output_per_mtok, cache_read_per_mtok, cache_write_per_mtok)
_PRICING: dict[str, tuple[float, float, float, float]] = {
    # Anthropic (from TF++ src/utils/modelCost.ts)
    # Claude 5 generation and Opus 4.8 — platform.claude.com/docs/en/about-claude/pricing,
    # read live 2026-09-24; the tuple is (input, output, cache hits, 5m cache writes).
    # Before these rows every one of them fell through to _DEFAULT_PRICING, the
    # Sonnet 4.6 tier: Opus 5.5 billed 25% under, Fable 5.1 at under a third.
    # Opus 5.5 is priced BELOW Opus 5 ("$4 / MTok | $5 / MTok | $8 / MTok |
    # $0.20 / MTok | $20 / MTok"); Fable 5.1 cut cache hits to $0.25; Sonnet 5's
    # $2 / $10 "is now the standard price" (pricing footnote 3).
    "claude-opus-5-5":      (4.0,   20.0,  0.2,   5.0),
    "claude-opus-5":        (5.0,   25.0,  0.5,   6.25),
    "claude-opus-4-8":      (5.0,   25.0,  0.5,   6.25),
    "claude-sonnet-5":      (2.0,   10.0,  0.2,   2.5),
    # Sonnet 5.5 (2026-09-28) keeps Sonnet 5's $2 / $10 and halves cache hits:
    # "Claude Sonnet 5.5 | $2 / MTok | $2.50 / MTok | $4 / MTok | $0.10 / MTok |
    # $10 / MTok", footnote 2 "Cache hits and refreshes on Claude Opus 5.5 and
    # Claude Sonnet 5.5 are priced at 0.05x the base input price" (same page,
    # read 2026-10-12). _VERSION_CONTINUATION kept it from inheriting Sonnet 5,
    # so until this row it billed at the Sonnet 4.6 default.
    "claude-sonnet-5-5":    (2.0,   10.0,  0.1,   2.5),
    # Fable 5.2 is priced below Fable 5.1 on every column: "Claude Fable 5.2 |
    # $8 / MTok | $10 / MTok | $16 / MTok | $0.20 / MTok | $40 / MTok" (input,
    # 5m cache write, 1h cache write, cache hits, output; vendor pricing page,
    # read 2026-11-11). Without this row it would be logged as unpriced.
    "claude-fable-5-2":     (8.0,   40.0,  0.20,  10.0),
    "claude-fable-5-1":     (10.0,  50.0,  0.25,  12.5),
    "claude-fable-5":       (10.0,  50.0,  1.0,   12.5),
    # Opus 4.7 (April 2026); 4.6 retained for historical session lookup
    "claude-opus-4-7":      (5.0,   25.0,  0.5,   6.25),
    "claude-opus-4-7-fast": (30.0,  150.0, 3.0,   37.5),
    "claude-opus-4-6":      (5.0,   25.0,  0.5,   6.25),
    "claude-opus-4-6-fast": (30.0,  150.0, 3.0,   37.5),
    "claude-opus-4-5":      (5.0,   25.0,  0.5,   6.25),
    "claude-sonnet-4-6":    (3.0,   15.0,  0.3,   3.75),
    "claude-sonnet-4-5":    (3.0,   15.0,  0.3,   3.75),
    "claude-haiku-4-5":     (1.0,   5.0,   0.1,   1.25),
    # Haiku 5.5 -- platform.claude.com/docs/en/about-claude/pricing, read 2026-10-09:
    # priced by prompt length, "$0.10 / MTok for prompts up to 100,000 tokens" in,
    # $0.50 out, $0.01 cache read, $0.125 5m cache write ($0.20 for 1h); the tier
    # for prompts over 100,000 tokens is in _REQUEST_TIERS.
    "claude-haiku-5-5":     (0.1,   0.5,   0.01,  0.125),
    "claude-haiku-3-5":     (0.8,   4.0,   0.08,  1.0),
    # OpenAI
    "gpt-4o":               (2.5,   10.0,  1.25,  2.5),
    "gpt-4o-mini":          (0.15,  0.6,   0.075, 0.15),
    "o1":                   (15.0,  60.0,  7.5,   15.0),
    "o3-mini":              (1.1,   4.4,   0.55,  1.1),
    # GPT-6.2 Astra / Sol -- launched 2026-11-17, onboarded 2026-11-24
    # (dogfood-domain family 8). developers.openai.com/api/docs/pricing, read
    # 2026-11-24, Standard tier, short context (<=272K input). Field order is
    # THIS DICT'S OWN (input, output, cache_read, cache_write-30m) -- NOT the
    # order prose lists them in below: Astra $8 in / $40 out / $0.40 cached /
    # $10 30-minute cache write per 1M; Sol $2 / $10 / $0.10 / $2.50. A
    # 12-hour cache write ($20 Astra, $5 Sol) is not representable in this
    # 4-tuple and is documented in config/model_prices.yaml's `source` field
    # instead. The >272K tier (input+cache 2x, output 1.5x) IS representable
    # and lives in _REQUEST_TIERS below, keyed by this same dict's id.
    "gpt-6.2-astra":        (8.0,   40.0,  0.40,  10.0),
    "gpt-6.2-sol":          (2.0,   10.0,  0.10,  2.50),
    # OpenAI GPT-6 / GPT-5.6 — platform.openai.com/docs/pricing, Standard tier,
    # short context (<= 272K input), read 2026-09-08. One tuple cannot carry
    # Astra's >272K tier (2x input and cache, 1.5x output for the whole
    # request), Batch/Flex (0.5x) or Fast mode (2x); the standard short-context
    # row is recorded so a call is never a silent Sonnet-tier estimate.
    "gpt-6-astra":          (10.0,  50.0,  1.0,   12.5),
    "gpt-5.6-sol":          (4.0,   20.0,  0.4,   5.0),
    "gpt-5.6-terra":        (2.0,   12.0,  0.2,   2.5),
    "gpt-5.6-luna":         (0.2,   1.2,   0.02,  0.25),
    # gpt-5.5 / gpt-5.4 / gpt-5.4-mini — all three are declared on the openai
    # provider in config/providers.yaml and all three billed at
    # _DEFAULT_PRICING until now: gpt-5.5 UNDER-counted, gpt-5.4-mini
    # over-counted more than threefold. Read from the same pricing page on
    # 2026-09-09; an earlier note here said they had no published row, which
    # the page refutes. Their rows carry "-" (or no cell at all) under "Cache
    # writes", whose column tooltip reads "Input tokens are either Input,
    # Cached Input, or Cache Write and writes are not an additive fee" — with
    # no separate write rate published, a write bills at the standard Input
    # rate, so cache_write == input, as on every other OpenAI row here.
    # The mini is listed before gpt-5.4 so the startswith fallback in
    # _resolve_pricing cannot hand a dated mini snapshot gpt-5.4's rate.
    "gpt-5.5":              (5.0,   30.0,  0.5,   5.0),
    "gpt-5.4-mini":         (0.75,  4.5,   0.075, 0.75),
    "gpt-5.4":              (2.5,   15.0,  0.25,  2.5),
    # Groq (free tier — $0 for most, minimal for paid)
    "llama-3.3-70b-versatile": (0.59, 0.79, 0.0, 0.0),
    "llama-4-scout":           (0.11, 0.34, 0.0, 0.0),
    "openai/gpt-oss-120b":     (0.15, 0.60, 0.0, 0.0),
    "deepseek-r1-distill-llama-70b": (0.59, 0.79, 0.0, 0.0),
    # Gemini -- ai.google.dev/gemini-api/docs/pricing, paid tier, read live
    # 2026-10-01, for prompts up to 200K tokens. Output includes thinking tokens
    # ("Output price (including thinking tokens)"). Creating a context cache bills
    # the input rate plus storage per hour, and storage has no slot here, so
    # cache_write carries the input rate. Pro models bill more above 200K tokens
    # ($4 / $18 for 3.1 Pro), which one tuple cannot carry. The free tier is not
    # modelled: Google lists its content as "Used to improve our products: Yes".
    # The old gemini-3.1-pro row ($1.25 / $10) was gemini-2.5-pro's price, and
    # the old gemini-2.5-flash row ($0.15 / $0.60) a retired rate.
    "gemini-3.1-pro":        (2.00,  12.00, 0.20,  2.00),
    # Gemini 4 Argon -- the same page, read 2026-10-18. Introductory rate: "$2.00
    # through November 30, 2026." in, "$10.00 through November 30, 2026." out
    # (thinking included), "$0.10 through November 30, 2026." cached. The doubling
    # "starting December 1, 2026." sits in _PRICE_SCHEDULE below and as version 2
    # of config/model_prices.yaml. Caching is implicit only, so cache_write carries
    # the input rate like the rows above. No prompt-length tier.
    "gemini-4-argon":        (2.00,  10.00, 0.10,  2.00),
    "gemini-3.5-flash-lite": (0.30,  2.50,  0.03,  0.30),
    # gemini-3.5-flash left the pricing page after its 2026-10-08 deprecation (requests
    # are routed to gemini-3.6-flash); the row is the last published rate, kept while
    # the id is accepted. The vendor does not say how a routed request is billed.
    "gemini-3.5-flash":      (1.50,  9.00,  0.15,  1.50),
    # gemini-3.6-flash and gemini-3.8-flash -- the same page, read 2026-10-18: "$0.75
    # through December 31, 2026." in, "$3.75 through December 31, 2026." out (thinking
    # included), "$0.075" cached; both double on 2027-01-01 (_PRICE_SCHEDULE).
    "gemini-3.6-flash":      (0.75,  3.75,  0.075, 0.75),
    "gemini-3.8-flash":      (0.75,  3.75,  0.075, 0.75),
    "gemini-3.1-flash-lite": (0.25,  1.50,  0.025, 0.25),
    "gemini-2.5-pro":        (1.25,  10.00, 0.125, 1.25),
    "gemini-2.5-flash-lite": (0.10,  0.40,  0.01,  0.10),
    "gemini-2.5-flash":      (0.30,  2.50,  0.03,  0.30),
    # Gemini 4 Carbon and Hydrogen -- aistudio.google.com/docs/pricing, read
    # 2026-11-24 (paid/Standard tier). Carbon: introductory "$2.00 through
    # November 30, 2026." in, "$10.00 through November 30, 2026." out (thinking
    # included), "$0.10 through November 30, 2026." cached; the doubling
    # "starting December 1, 2026." sits in _PRICE_SCHEDULE below. Hydrogen's
    # page lists no dated change: "$1.50" in, "$7.50" out, "$0.075" cached,
    # flat. Both implicit-caching only, so cache_write carries the input rate;
    # neither page lists a prompt-length tier.
    "gemini-4-carbon":       (2.00,  10.00, 0.10,  2.00),
    "gemini-4-hydrogen":     (1.50,  7.50,  0.075, 1.50),
    # DeepSeek -- api-docs.deepseek.com/quick_start/pricing, re-read live
    # 2026-10-01 (the 2026-07-26 figures had been superseded). These are PEAK
    # rates, so an upper bound: off-peak is half, and peak is 01:00-04:00 and
    # 06:00-10:00 UTC on weekdays. "Use deepseek-flash as the model name. The
    # legacy names deepseek-v4-flash and deepseek-v4-flash-vision-exp are still
    # accepted ... billed at the Flash price", so the legacy row equals
    # deepseek-flash and the vision id inherits it by prefix. cache_write is the
    # cache-miss input rate (no write rate is published).
    "deepseek-flash":        (0.30,  1.20,  0.006, 0.30),
    "deepseek-v4-flash":     (0.30,  1.20,  0.006, 0.30),
    "deepseek-v4-pro":       (1.32,  3.96,  0.044, 1.32),
    # xAI -- the model data embedded in docs.x.ai/docs/models, read 2026-10-01.
    # Its unit (ten-thousandths of a dollar per 1M tokens) was calibrated against
    # the same page's rendered "$2.00 / 1M tokens" card for grok-4.5. Prompts over
    # 200,000 tokens bill double, in _REQUEST_TIERS (GRIP#9220).
    "grok-4.5":              (2.00,  6.00,  0.30,  2.00),
    "grok-4.3":              (1.25,  2.50,  0.20,  1.25),
    # grok-4.6, the same embedded data read 2026-11-16 (GRIP#9205): 20000 / 60000 /
    # cached 5000 on its us-east-1 and us-west-2 clusters, so $2.00 in, $6.00 out,
    # $0.50 cached. Its us-central-1 listing is 10% higher, which matches the US
    # regional endpoint: docs.x.ai/docs/key-information/regions says "Token usage on
    # the US endpoint costs 10% more than on the global endpoint". config/providers.yaml
    # sends xai to https://api.x.ai/v1, the global endpoint. Prompts over 200,000
    # tokens bill double (_REQUEST_TIERS).
    "grok-4.6":              (2.00,  6.00,  0.50,  2.00),
    # grok-4.7 -- the model data embedded in docs.x.ai/developers/models/grok-4.7,
    # read 2026-11-24 ("name":"grok-4.7"): promptTextTokenPrice 20000,
    # cachedPromptTokenPrice 5000, completionTextTokenPrice 60000, the same unit
    # as grok-4.5/4.6. Confirmed in prose by the vendor's own release-notes page
    # (same read): "Pricing is $2 / $0.50 / $6 per 1M tokens (input / cached
    # input / output) below 200k prompt tokens, and $4 / $1 / $12 above."
    "grok-4.7":                     (2.00,  6.00,  0.50,  2.00),
    # grok-4.20-0309-reasoning and -non-reasoning: the same embedded data, read
    # 2026-11-24 ("name":"grok-4.20-0309-reasoning" / "-non-reasoning"):
    # promptTextTokenPrice 12500, cachedPromptTokenPrice 2000,
    # completionTextTokenPrice 25000 for BOTH -- the same row as grok-4.3.
    # Both ids were already in this provider's `models:` list (reasoning is
    # even in this file's own `premium` routing tier) with no row here at all,
    # so every call billed at the Sonnet-tier default, $3 / $15 -- a latent
    # gap dogfood-domain family 5 closes. non-reasoning is also PARITY: GRIP's
    # lib/broly_council.py and lib/council_seat.py already call it.
    "grok-4.20-0309-reasoning":     (1.25,  2.50,  0.20,  1.25),
    "grok-4.20-0309-non-reasoning": (1.25,  2.50,  0.20,  1.25),
    # Provider-retired ids (deprecated 2026-07-24) — kept for historical lookups.
    "deepseek-chat":       (0.28,  0.42,  0.028, 0.0),
    "deepseek-reasoner":   (0.28,  0.42,  0.028, 0.0),
    # Qwen (DashScope). qwen3.5-flash re-read 2026-10-01 from
    # alibabacloud.com/help/en/model-studio/model-pricing (international, up to
    # 1M tokens): $0.1 in, $0.4 out. No cache rate is published beside it, so
    # both cache slots carry the input rate as an upper bound.
    "qwen3.5-flash":           (0.10,  0.40,  0.10, 0.10),
    # qwen3.8-max and qwen3.7-flash, read the same day from the same page
    # (International), cache slots at the input rate for the same reason.
    # qwen3.7-flash is priced by request size: this row is its first tier, up to
    # 32K tokens, and _REQUEST_TIERS carries the larger ones.
    "qwen3.8-max":             (2.00,  6.00,  2.00, 2.00),
    "qwen3.7-flash":           (0.03,  0.13,  0.03, 0.03),
    "qwen3.5-plus":            (0.30,  1.20,  0.0, 0.0),
    "qwen3.6-plus-preview":    (0.30,  1.20,  0.0, 0.0),
    # qwen3.7-plus, the same page read 2026-11-16 (GRIP#9205), Singapore table,
    # International scope, 0<Token<=256K: "List price $0.4 (Limited-time 20% off)"
    # in and "List price $1.6 (Limited-time 20% off)" out, thinking or not. The row
    # keeps the list price, an upper bound, because the discount names no end date.
    # The snapshot qwen3.7-plus-2026-05-26 lists the same figures and inherits this
    # row. Its cache rates are on alibabacloud.com/help/en/model-studio/context-cache,
    # read the same day: an implicit hit is "typically billed at 20% of the standard
    # input token price" and an explicit one at 10%, so cache read carries the 20%;
    # explicit cache creation "is billed at 125% of the standard input token price".
    # Prompts over 256K tokens bill $1.2 / $4.8 (_REQUEST_TIERS).
    "qwen3.7-plus":            (0.40,  1.60,  0.08, 0.50),
    # qwen4-max: alibabacloud.com/help/en/model-studio/qwen4-max +
    # .../billing-for-model-studio, both read 2026-11-24, Singapore/
    # International, 0<Token<=1M: "$1.5" in, "$4.5" out. GA 2026-11-16
    # (alibabacloud.com/press-room/alibaba-unveils-qwen4-max), which also
    # states "Input (Implicit Cache Hit) is priced at $0.075 per 1M tokens" --
    # the cache-read slot. No cache-WRITE rate is published, so that slot
    # carries the input rate as an upper bound, as for every other
    # unpublished-write Qwen/Kimi row in this table.
    "qwen4-max":               (1.50,  4.50,  0.075, 1.50),
    # qwen3.8-flash: the same billing page, same date, same region: "$0.15"
    # in, "$0.47" out, one tier up to 1M -- genuinely MORE than qwen3.5-flash's
    # $0.10/$0.40. No cache figures published; both cache slots carry the
    # input rate as an upper bound.
    "qwen3.8-flash":           (0.15,  0.47,  0.15, 0.15),
    # Moonshot (Kimi) — lineage per platform.kimi.ai/docs/pricing (chat-k26 /
    # chat-k27-code / chat-k3, re-verified 2026-09-07). cache_write =
    # input-miss upper bound for all (Moonshot publishes no distinct write rate).
    # kimi-k2.5 (0.60 / 3.00 / 0.10) was dropped with the vendor's 2026-08-31
    # retirement; a historical kimi-k2.5 row now prices like any unknown id.
    "kimi-k2.6":           (0.95,  4.00,  0.16,  0.95),
    "kimi-k2.7-code":      (0.95,  4.00,  0.19,  0.95),
    "kimi-k2.7-code-highspeed": (1.90, 8.00, 0.38, 1.90),
    "kimi-k3":             (3.00,  15.00, 0.30,  3.00),
    # Kimi K3.1 Pro and Highspeed: the price table embedded in
    # platform.kimi.ai/docs/pricing/chat, read 2026-11-24, flat at every context
    # length. Unlike K2, the K3 series publishes write rates: cache_write is the
    # 5-minute (default TTL) rate, and the 1-hour write is double it.
    "kimi-k3.1-pro":       (2.00,  10.00, 0.20,  2.00),
    "kimi-k3.1-pro-highspeed": (4.00, 20.00, 0.20, 4.00),
    # Mistral Large 4 and Small 2603 (/dogfood-domain, 2026-11-24): both read
    # from the "pricing" array embedded in docs.mistral.ai/inference/pricing's
    # own page data (fetched and the JSON read, not a search summary). Large 4
    # publishes a "Cached input" row at 0.1x input ($0.07); Small 2603 publishes
    # no cache row at all, so both cache slots carry the input rate as an upper
    # bound.
    "mistral-large-4":       (0.70,  2.10,  0.07,  0.70),
    "mistral-small-2603":    (0.15,  0.60,  0.15,  0.15),
    # mistral-large-latest (GRIP#9420, HAL's parity gap): the vendor's pricing
    # page groups it on the SAME card as mistral-large-4 (modelIds
    # ["mistral-large-4","mistral-large-4-0","mistral-large-latest"], re-read
    # live 2026-11-24), so it shares that row rather than carrying its own.
    # Before this it had no catalogue row and no cost row at all.
    "mistral-large-latest":  (0.70,  2.10,  0.07,  0.70),
    # Zhipu (GLM)
    "glm-5":               (0.65,  1.95,  0.0,   0.0),
    "glm-4.7":             (0.50,  1.50,  0.0,   0.0),
    "glm-4.7-flash":       (0.0,   0.0,   0.0,   0.0),
    # GLM-5.3: docs.z.ai/guides/overview/pricing.md, read 2026-11-24 -- "GLM-5.3 | $1.4 |
    # $0.26 | Limited-time Free | $4.4" (input, cached input, cached input storage,
    # output). No distinct cache-write rate is published; the 4th slot carries the (free)
    # storage rate, not a write premium. This is now the model _ZAI_MODEL_MAP's
    # ANTHROPIC_DEFAULT_OPUS_MODEL and PROVIDER_CONFIGS["zhipu"]["default_model"] both
    # name (GRIP#9431, 2026-11-24: Zhipu answers every glm-5.1 request with GLM-5.3), so
    # without this row every one of those calls would have silently inherited glm-5's
    # ($0.65/$1.95) -- a dot-separated version ("glm-5.3".startswith("glm-5")) is NOT
    # caught by _VERSION_CONTINUATION, which only recognises a dash before the
    # continuation digits, never a dot.
    "glm-5.3":             (1.4,   4.4,   0.26,  0.0),
    # GLM-5.3-Flash: docs.z.ai/guides/overview/pricing.md, read 2026-11-24 -- "GLM-5.3-Flash
    # | $0.15 | $0.03 | Limited-time Free | $0.50" (input, cached input, cached input
    # storage, output). No distinct cache-write rate is published; the 4th slot carries the
    # (free) storage rate, not a write premium. Was already in providers.yaml with no price
    # row -- without this row it would have silently inherited glm-5's ($0.65/$1.95): a
    # dot-separated version ("glm-5.3-flash".startswith("glm-5")) is NOT caught by
    # _VERSION_CONTINUATION, which only recognises a dash before the continuation digits.
    "glm-5.3-flash":       (0.15,  0.50,  0.03,  0.0),
    # GLM-5.4 has deliberately NO row here -- see _KNOWN_UNPRICED below, which is the
    # correct mechanism: this file's own stated policy for an unknown id in a paid
    # family (deepseek/qwen/glm) is the VISIBLE _DEFAULT_PRICING estimate, "never a
    # silent zero ... under-counting to zero is budget-blindness" (_resolve_pricing).
    # A literal (0.0, 0.0, 0.0, 0.0) row here would violate that policy outright.
    # Mistral
    # mistral-small-latest (GRIP#9420): shares its card with mistral-small-2603
    # (modelIds ["mistral-small-2603","mistral-small-latest"], re-read live
    # 2026-11-24) -- the SAME row as that id, not a separate $0.0/$0.0 cache
    # rate. No cached-input price is published for this card at all, so both
    # cache slots carry the $0.15 input rate as an upper bound, same as
    # mistral-small-2603 above; the previous (0.0, 0.0) pair under-counted
    # every cached token on this model to zero.
    "mistral-small-latest": (0.15, 0.60,  0.15,  0.15),
    # Google Gemma 4 (released 2 Apr 2026, Apache 2.0)
    "google/gemma-4-31b-it":     (0.14,  0.40,  0.0,   0.0),
    "google/gemma-4-26b-a4b-it": (0.10,  0.30,  0.0,   0.0),
    "gemma-4-27b":               (0.0,   0.0,   0.0,   0.0),
    # Meta Llama 4
    "llama-4-maverick":          (0.15,  0.60,  0.0,   0.0),
    "meta-llama/llama-4-maverick": (0.15, 0.60, 0.0,   0.0),
    # mlx-vlm v0.4.3 — local Apple Silicon inference, free tier
    # https://github.com/Blaizzy/mlx-vlm — vision, audio, OCR, detection, segmentation
    "mlx-community/Qwen2-VL-7B-Instruct-4bit":   (0.0, 0.0, 0.0, 0.0),
    "mlx-community/Qwen3-VL-7B-Instruct-4bit":   (0.0, 0.0, 0.0, 0.0),
    "mlx-community/phi-4-vision-instruct-4bit":   (0.0, 0.0, 0.0, 0.0),
    "mlx-community/gemma-4-vision-9b-4bit":       (0.0, 0.0, 0.0, 0.0),
    "falcon-ocr":                                 (0.0, 0.0, 0.0, 0.0),
    "falcon-perception":                          (0.0, 0.0, 0.0, 0.0),
    "granite-vision-3.2-2b":                      (0.0, 0.0, 0.0, 0.0),
    "sam-3.1":                                    (0.0, 0.0, 0.0, 0.0),
    "rf-detr":                                    (0.0, 0.0, 0.0, 0.0),
    # Local (free)
    "local":               (0.0,   0.0,   0.0,   0.0),
}

# Fallback for unknown models
_DEFAULT_PRICING = (3.0, 15.0, 0.3, 3.75)  # Sonnet-tier

# Rates that rise with the size of a request's prompt. The WHOLE request bills
# at the tier its prompt reaches, so a row lists (threshold, rates) pairs in
# ascending order: a prompt of more than ``threshold`` tokens bills at ``rates``.
# A prompt counts its cached tokens too, and "32K" is read as 32,000; both
# choices can only over-count, which keeps an estimate an upper bound.
_REQUEST_TIERS: dict[str, tuple[tuple[int, tuple[float, float, float, float]], ...]] = {
    # alibabacloud.com/help/en/model-studio/model-pricing, International, read
    # 2026-10-01: 32K-256K $0.100 / $0.400, 256K-1M $0.200 / $0.800.
    "qwen3.7-flash": ((32_000, (0.10, 0.40, 0.10, 0.10)),
                      (256_000, (0.20, 0.80, 0.20, 0.20))),
    # ai.google.dev/gemini-api/docs/pricing, read 2026-10-01: prompts over 200k
    # tokens bill $4.00 in, $18.00 out (thinking included), $0.40 cached.
    "gemini-3.1-pro": ((200_000, (4.00, 18.00, 0.40, 4.00)),),
    # platform.claude.com/docs/en/about-claude/pricing, read 2026-10-09: Haiku 5.5
    # "for prompts over 100,000 tokens" bills $0.50 in, $2.50 out, $0.05 cache read,
    # $0.625 5m cache write.
    "claude-haiku-5-5": ((100_000, (0.50, 2.50, 0.05, 0.625)),),
    # docs.x.ai/docs/models, read 2026-11-16: grok-4.6's long-context fields are
    # 40000 / 120000 / cached 10000 past "longContextThreshold":"200000", so $4.00
    # in, $12.00 out and $1.00 cached, on the same unit as its base row.
    "grok-4.6": ((200_000, (4.00, 12.00, 1.00, 4.00)),),
    # The same data, read 2026-11-16 (GRIP#9220): past the same threshold, grok-4.5
    # lists 40000 / 120000 / cached 6000 and grok-4.3 25000 / 50000 / cached 4000.
    # The cache-write slot carries the input rate, as in their base rows.
    "grok-4.5": ((200_000, (4.00, 12.00, 0.60, 4.00)),),
    "grok-4.3": ((200_000, (2.50, 5.00, 0.40, 2.50)),),
    # grok-4.7: docs.x.ai/developers/models/grok-4.7, read 2026-11-24, embedded
    # data past "longContextThreshold":"200000": promptTextTokenPriceLongContext
    # 40000, completionTokenPriceLongContext 120000,
    # cachedPromptTokenPriceLongContext 10000 -- the same tier as grok-4.6,
    # confirmed in prose on the same page/date ("$4 / $1 / $12 above" 200k
    # prompt tokens).
    "grok-4.7": ((200_000, (4.00, 12.00, 1.00, 4.00)),),
    # grok-4.20-0309-reasoning and -non-reasoning: the same embedded data, read
    # 2026-11-24: promptTextTokenPriceLongContext 25000,
    # completionTokenPriceLongContext 50000, cachedPromptTokenPriceLongContext
    # 4000 for BOTH -- the same tier as grok-4.3.
    "grok-4.20-0309-reasoning": ((200_000, (2.50, 5.00, 0.40, 2.50)),),
    "grok-4.20-0309-non-reasoning": ((200_000, (2.50, 5.00, 0.40, 2.50)),),
    # alibabacloud.com/help/en/model-studio/model-pricing, read 2026-11-16:
    # qwen3.7-plus, International, "256K<Token<=1M" lists $1.2 in and $4.8 out, and
    # the base row's cache ratios (20% read, 125% creation) apply to the $1.2.
    "qwen3.7-plus": ((256_000, (1.20, 4.80, 0.24, 1.50)),),
    # developers.openai.com/api/docs/pricing, read 2026-11-24: "For requests
    # with more than 272K input tokens, input and cache rates double" and
    # "output costs 1.5 times the short-context rate for the whole request" --
    # quoted verbatim, not inferred. Astra: $8->$16 in, $0.40->$0.80 cached,
    # $10->$20 cache write, $40->$60 out. Sol: $2->$4, $0.10->$0.20,
    # $2.50->$5, $10->$15. The vendor page confirms $16/$0.80/$60 and
    # $4/$0.20/$15 directly; the cache-write figures are the stated 2x applied
    # to the already-quoted short-context write rate (gpt-6-astra has the
    # SAME tier, undocumented here, since it predates this mechanism and this
    # PR does not touch its row).
    "gpt-6.2-astra": ((272_000, (16.0, 60.0, 0.80, 20.0)),),
    "gpt-6.2-sol": ((272_000, (4.0, 15.0, 0.20, 5.0)),),
}
# Rates that change on a CALENDAR DATE rather than by prompt size: ascending
# (ISO date, rates); from that UTC day on, the row bills at ``rates``. Argon is
# the first: ai.google.dev/gemini-api/docs/pricing, read 2026-10-18, "$4.00
# starting December 1, 2026." in, "$20.00 starting December 1, 2026." out
# (thinking included), "$0.20 starting December 1, 2026." cached. Without this the
# tracker would bill the introductory rate for ever, under-counting by half from
# that day. config/model_prices.yaml admits no future-dated row, so on the day a
# change here arrives, tests/test_infinite_cost.py fails until the ledger gains it.
_PRICE_SCHEDULE: dict[str, tuple[tuple[str, tuple[float, float, float, float]], ...]] = {
    "gemini-4-argon": (("2026-12-01", (4.00, 20.00, 0.20, 4.00)),),
    # aistudio.google.com/docs/pricing, read 2026-11-24: "$4.00 starting
    # December 1, 2026." in, "$20.00 starting December 1, 2026." out (thinking
    # included), "$0.20 starting December 1, 2026." cached -- Carbon's rate
    # doubles the same day as Argon's. Hydrogen carries no schedule: its page
    # lists no dated change. config/model_prices.yaml admits no future-dated
    # row, so on 2026-12-01 tests/test_infinite_cost.py fails until the
    # ledger gains Carbon's row, same as Argon.
    "gemini-4-carbon": (("2026-12-01", (4.00, 20.00, 0.20, 4.00)),),
    # The same page, read 2026-10-18: "$1.50 starting January 1, 2027." in, "$7.50
    # starting January 1, 2027." out, "$0.15 starting January 1, 2027." cached.
    "gemini-3.6-flash": (("2027-01-01", (1.50, 7.50, 0.15, 1.50)),),
    "gemini-3.8-flash": (("2027-01-01", (1.50, 7.50, 0.15, 1.50)),),
}


def _today() -> str:
    """Today's UTC date as ISO text; the seam tests move the calendar through."""
    return datetime.now(timezone.utc).date().isoformat()


def _scheduled(key: str, rates: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """The rates ``key`` bills at today, after every dated change that has arrived."""
    today = _today()
    for effective, dated in _PRICE_SCHEDULE.get(key, ()):
        if today >= effective:
            rates = dated
    return rates



def calculate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """Calculate USD cost for a single query."""
    prompt = input_tokens + cache_read_tokens + cache_creation_tokens
    pricing = _request_rates(model, prompt)
    return (
        (input_tokens / 1_000_000) * pricing[0]
        + (output_tokens / 1_000_000) * pricing[1]
        + (cache_read_tokens / 1_000_000) * pricing[2]
        + (cache_creation_tokens / 1_000_000) * pricing[3]
    )


def calculate_cost_from_usage(model: str, usage: TokenUsage) -> float:
    """Calculate USD cost from a TokenUsage object."""
    return calculate_cost(
        model,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
    )


class CostTracker:
    """Accumulates cost across a session. Thread-safe."""

    def __init__(self) -> None:
        self._total_input = 0
        self._total_output = 0
        self._total_cache_read = 0
        self._total_cache_creation = 0
        self._total_usd = 0.0
        self._per_model: dict[str, float] = {}
        self._per_provider: dict[str, float] = {}
        self._lock = threading.Lock()

    def record(
        self,
        model: str,
        provider: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> float:
        """Record usage and return the cost for this query."""
        cost = calculate_cost(model, input_tokens, output_tokens,
                              cache_read_tokens, cache_creation_tokens)
        with self._lock:
            self._total_input += input_tokens
            self._total_output += output_tokens
            self._total_cache_read += cache_read_tokens
            self._total_cache_creation += cache_creation_tokens
            self._total_usd += cost
            self._per_model[model] = self._per_model.get(model, 0.0) + cost
            if provider:
                self._per_provider[provider] = self._per_provider.get(provider, 0.0) + cost
        return cost

    def record_response(self, model: str, provider: str, usage: TokenUsage) -> float:
        """Record from a ChatResponse's TokenUsage."""
        return self.record(
            model, provider,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
        )

    @property
    def total_usd(self) -> float:
        return self._total_usd

    def snapshot(self) -> CostSnapshot:
        """Return current accumulated cost as immutable snapshot."""
        with self._lock:
            return CostSnapshot(
                input_tokens=self._total_input,
                output_tokens=self._total_output,
                cache_read_tokens=self._total_cache_read,
                cache_creation_tokens=self._total_cache_creation,
                total_usd=self._total_usd,
            )

    def per_model(self) -> dict[str, float]:
        """Cost breakdown by model."""
        with self._lock:
            return dict(self._per_model)

    def per_provider(self) -> dict[str, float]:
        """Cost breakdown by provider."""
        with self._lock:
            return dict(self._per_provider)

    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            self._total_input = 0
            self._total_output = 0
            self._total_cache_read = 0
            self._total_cache_creation = 0
            self._total_usd = 0.0
            self._per_model.clear()
            self._per_provider.clear()


def estimate_tokens(text: str) -> int:
    """Estimate token count from text. Uses word count / 0.75 heuristic.

    Accuracy: within 10-20% for English text. Sufficient for budget planning,
    not for billing.
    """
    if not text:
        return 0
    word_count = len(text.split())
    return int(word_count / 0.75)


def estimate_cost(model: str, input_text: str, estimated_output_tokens: int = 500) -> float:
    """Estimate cost for a query before sending it."""
    input_tokens = estimate_tokens(input_text)
    return calculate_cost(model, input_tokens=input_tokens, output_tokens=estimated_output_tokens)


# A point release is a DIFFERENT model, not a snapshot of its sibling:
# "claude-opus-5-5" starts with "claude-opus-5", yet bills $4 / $20 against Opus
# 5's $5 / $25. So a key may stand in for a longer id only when what follows the
# key is a snapshot date ("-20251001", "-2024-07-18", "-05-20") or a named variant
# ("-fast", "-preview"); a short numeric component ("-5", "-1") names another
# model. Ported from GRIP's lib/hal/cost_tracker.py (2026-09-24).
_VERSION_CONTINUATION = re.compile(r"-(?!\d{2}-\d{2}$)\d{1,2}(?:-|$)")


def _longest_inheritable_key(model: str) -> "str | None":
    """The MOST specific key whose row may price ``model``, never a sibling's.

    Longest, not first in insertion order: "claude-opus-4-7-fast-<date>" starts
    with both "claude-opus-4-7" and "claude-opus-4-7-fast", and the first-match
    loop this replaces billed it at the cheaper non-fast row.
    """
    keys = [k for k in _PRICING
            if model.startswith(k) and not _VERSION_CONTINUATION.match(model, len(k))]
    return max(keys, key=len, default=None)


def _request_rates(model: str, prompt_tokens: int) -> tuple[float, float, float, float]:
    """The rates a whole request bills at, given how many tokens its prompt holds."""
    rates = _resolve_pricing(model)
    for threshold, tier in _REQUEST_TIERS.get(_pricing_key(model), ()):
        if prompt_tokens > threshold:
            rates = tier
    return rates


# Vendor-documented models with NO published per-token price, excluded here so an id
# that happens to share a prefix with a cheaper, older sibling (the longest-prefix
# fallback this function implements) is never silently billed at that sibling's row.
# They fall through _resolve_pricing to the VISIBLE _DEFAULT_PRICING estimate instead
# -- this file's own existing policy for an unpriced id in a paid family, stated below:
# "over-counting is auditable, under-counting to zero is budget-blindness".
# GLM-5.4: docs.z.ai/guides/llm/glm-5.4.md, read 2026-11-24 -- "the Model API will be
# made available upon completion of the security assessment"; absent from the pricing
# page. Without this, "glm-5.4".startswith("glm-5") would silently inherit glm-5's
# $0.65/$1.95, a different, older model -- the same dot-vs-dash gap _VERSION_CONTINUATION
# has for glm-5.3-flash, which is protected instead by carrying its own real row above.
_KNOWN_UNPRICED: tuple[str, ...] = ("glm-5.4",)


def _is_known_unpriced(model: str) -> bool:
    bare = model.split("/")[-1] if "/" in model else model
    return model in _KNOWN_UNPRICED or bare in _KNOWN_UNPRICED


def _pricing_key(model: str) -> str | None:
    """The _PRICING row a model resolves to, or None when no row matches it."""
    if _is_known_unpriced(model):
        return None
    if model in _PRICING:
        return model

    # Provider-qualified address ("moonshot/kimi-k3", "deepseek/deepseek-v4-pro"):
    # resolve on the bare model id. Without this, paid address-form models fell
    # through — moonshot/* to _DEFAULT_PRICING, and deepseek/* (the live council
    # speaker) all the way to the LOCAL free heuristic below (issue #509 item 3).
    if "/" in model:
        bare = model.split("/")[-1]
        if bare in _PRICING:
            return bare
        known = _longest_inheritable_key(bare)
        if known is not None:
            return known

    # Try without version suffix (e.g. "claude-opus-4-7-20260416" -> "claude-opus-4-7")
    return _longest_inheritable_key(model)


def _resolve_pricing(model: str) -> tuple[float, float, float, float]:
    """Resolve pricing for a model, with fuzzy matching."""
    known = _pricing_key(model)
    if known is not None:
        return _scheduled(known, _PRICING[known])

    # Ollama-style locally-served tags ("llama3:8b", "qwen2.5:7b") are free.
    # Only for unprefixed ids — a provider-qualified miss must not land here.
    if ":" in model and "/" not in model:
        return _PRICING["local"]

    # Local model families are free. Paid-cloud families (deepseek/qwen/glm)
    # are deliberately NOT listed any more: an unknown id from a family that
    # has paid rows must fall to the VISIBLE _DEFAULT_PRICING estimate below,
    # never to a silent zero (AGORA review of PR #512 — the
    # unknown-non-qualified-model concern; over-counting is auditable,
    # under-counting to zero is budget-blindness).
    #
    # GRIP#9420: this used to match a BARE id too, so a paid api.mistral.ai
    # model whose name merely contains "mistral" -- "mistral-medium-3.5",
    # "codestral-latest", a ministral snapshot -- billed $0 unless it had
    # carried its own exact _PRICING row. api.mistral.ai never puts a slash in
    # its OWN model ids; every genuinely local id these five substrings are
    # meant to catch is either the Ollama "name:tag" form just above, or a
    # slash-qualified HF-repo-style address ("mlx-community/gemma-4-...",
    # "meta-llama/llama-4-maverick"), and every such address already carries
    # its own explicit $0 row in this table. Requiring "/" here keeps a bare
    # paid id off this free fallback without narrowing what a slash-qualified
    # local address still catches.
    #
    # Known residual (AGORA on HAL#837, not closed here): "mistral/<id>"
    # supplies its OWN "/" and its OWN "mistral" substring from the provider
    # prefix alone, so a FUTURE Mistral id with no row, called in that
    # provider-qualified address form, still reaches this free path -- see
    # test_a_provider_qualified_unpriced_mistral_id_is_a_known_residual_gap in
    # tests/test_cost_tracker_vendor_prices.py. Closing it needs this module to
    # know which address prefixes name a cloud BILLING provider (mistral,
    # qwen, xai, ...) rather than a local HF-repo org (mlx-community,
    # TheBloke, ...), which couples this pure pricing module to
    # config/providers.yaml.
    _LOCAL_PATTERNS = ("llama3", "llama4", "mistral", "phi", "gemma")
    if "/" in model and any(local in model.lower() for local in _LOCAL_PATTERNS):
        return _PRICING["local"]

    return _DEFAULT_PRICING
