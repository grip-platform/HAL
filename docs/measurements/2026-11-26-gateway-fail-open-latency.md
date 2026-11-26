# Gateway fail-open latency: Gemini 4 Hydrogen, Neon and Carbon (2026-11-26)

## Why this exists

HAL#923 makes Gemini 4 Hydrogen the gateway council's fail-open model, on V>>'s decision of
2026-11-26. The fail-open path's own docstring asks for a fast model. This file records the
measurement behind that choice, so it can be checked and run again (Codex P1 on HAL#923).

## Method

- 18 OpenAI-format chat requests through the live HAL gateway (`127.0.0.1:4010`, HAL main
  `b53c2315`). The gateway routes each Gemini 4 model over the same same-wire path that
  `_relay_council_failopen` uses.
- Two prompts: a short one (34 characters) and a council-sized one (10,055 characters, reported
  as 2,043 to 2,046 prompt tokens).
- `max_tokens` 1024 for every request.
- Three rounds, with the three models interleaved inside each round, so a slow minute does not
  land on one model.
- Wall-clock time per request, measured by the client. The raw rows are in
  `2026-11-26-gateway-fail-open-latency.json` beside this file.

## Results

Seconds: the median, then the range, of the requests that answered.

| Prompt | Hydrogen | Neon | Carbon |
|---|---|---|---|
| short | 2.56 (1.66 to 2.84), 3 of 3 | 14.64 (3.82 to 23.63), 3 of 3 | 6.0 (5.86 to 6.59), 3 of 3 |
| council-sized | 3.95 (3.69 to 4.06), 3 of 3 | 10.60 (9.96 to 11.23), 2 of 3; one request timed out at 200 s | 11.32 (10.81 to 35.91), 3 of 3 |

Hydrogen was the fastest model in every round, at both prompt sizes.

## Limits

- One machine, one afternoon and three rounds per cell: a small sample.
- Client-side timing includes the gateway's own overhead, which is the same for every model.
- The answers were short (2 to 95 completion tokens as reported), so a long answer takes longer
  for every model. This path does not report thinking tokens.

## Provenance

Measured by Claude Opus 5.5 (`claude-opus-5-5`) for V>>. Decision: V>>, 2026-11-26.

Primary Author: V>>
