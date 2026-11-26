# HAL decide: typed decisions from decision models

HAL can now ask a **decision model** a question and get back a typed answer instead of
text. The first decision model it reaches is **Jev**, from TypeSafe AI
(https://typesafe.ai). This page says what such a model is and is not, how HAL guards
every request to it, what was measured against the live API, and the rules for using it.

## What a decision model is

A decision model reads a `state` (a string, an object or an array) and a map of typed
questions, and returns one answer per question:

| Question | Answer |
|---|---|
| **Noul** (yes or no) | `noul`: the probability, from 0 to 1, that the answer is yes |
| **Choice** (one of up to 255 options) | `choice`, plus `probabilities` for every option and a `confidence` |
| **Score** (2 to 10 ordered levels) | `score`, a probability-weighted level that can land between levels, plus `probabilities`, a `legend` and a `confidence` |

It never writes text, so it is **not a chat model**. TypeSafe says so plainly: "Jev is
**not** a drop-in replacement for the LLM behind Claude Code ... It does not generate
text, write code, or hold a conversation"
(https://docs.typesafe.ai/introduction/coding-agents.md). That is why decision models
have their own catalogue, `config/decision_models.yaml`, and are kept out of
`config/model_catalog.yaml`, whose loader could route any entry there as a chat model.

The wire format is TypeSafe's (https://docs.typesafe.ai/api.md). Other vendors have copied
it, so the HAL interface (`lib/hal/decide/`) is vendor-neutral and TypeSafe is its first
provider.

## How to call it

```python
from lib.hal.decide import decide, noul, choice

result = decide("Payouts have failed for three days.",
                {"urgent": noul("Is this urgent?"),
                 "team": choice("Which team should handle it?",
                                {"billing": "Payments and refunds", "technical": None})})
if result.ok:
    result.answers["urgent"]["noul"]       # e.g. 0.95
else:
    result.reason                          # why not, from a closed set (below)
```

```bash
hal decide --state "Payouts have failed for three days." --noul urgent "Is this urgent?"
hal decide --request body.json             # a whole System One request body
```

Over HTTP, hal-server takes `POST /api/decide` with `{"state": ..., "questions": {...},
"model": optional}`. Like `/api/infer` it is fail-safe: every failure is HTTP 200 with
`ok: false` and a `reason`, never a 5xx. Remotely it needs a gateway token, like every
non-public `/api` route. The route is added by `lib/hal/decide/http.py` rather than in
`_make_handler`'s route table, because that function's complexity blocks edits until it
is decomposed.

**Ask every question about one state in one call.** A request carries a fixed overhead
of about 270 input tokens, and each further short yes/no question adds about 15
(measured 2026-10-10). Eight questions cost about 1,420 tokens in one call and about
10,500 in eight separate calls.

## What every request goes through

`lib/hal/decide/typesafe.py`, in order. The request stops at the first step that says no,
and the result names that step:

1. **Circuit breaker.** After three failed decisions in a row, the provider rests for 60
   seconds (`circuit_open`).
2. **Spend guard.** `typesafe-api-key` must be opted in and under its daily cap
   (`config/spend_guard.json`; the operator set $1 a day on 2026-10-10 and lowered it to
   $0.10 a day on 2026-10-17). Each reply is
   priced from its input tokens at $0.042 per million, and recorded. Output tokens are
   free, but the catalogue still states their price as 0: a missing price is refused as
   a configuration error rather than read as zero. A reply that carries no token usage
   closes the source until midnight UTC, because its cost would be unknown.

   An error reply, 4xx or 5xx, records nothing and leaves the source open. That is
   HAL's behaviour, pinned for a 400 and a 503 by
   `test_an_error_reply_records_nothing_and_leaves_the_source_open`. It rests on an
   assumption about TypeSafe: that an error is not billed. Probed on 2026-10-10, a 400
   reply carried no token usage; whether TypeSafe bills any error is unverified, because
   nobody has checked the console balance against one. Recording errors as unknown
   spend instead would shut the source for the rest of the day after one transient
   overload.
3. **Key.** From the macOS keychain item `grip-typesafe`, or the environment
   (`TYPESAFE_API_KEY`, `GRIP_TYPESAFE_API_KEY`) elsewhere. The key is a local variable
   for one request. It is never stored on the provider, and never put into a result, a log
   line or an error. TypeSafe's SDK is not used, partly because its versions before 0.7.1
   could put the key into logged exceptions.
4. **Redaction shield.** GRIP's `shield_decision_payload` (GRIP#7933) scrubs every string
   at every depth of the request. If a question id or a Choice option would itself need
   scrubbing, the request is refused rather than sent. If GRIP's shield is missing or too
   old to have that function, **nothing is sent** (`shield_unavailable`). There is no
   fallback to the older prose-only shield.
5. **Size.** A local ceiling of 512 KiB. The API enforces its own token budgets.
6. **Send.** Proxies are disabled, so no TLS-intercepting hop sees the key. Retries follow
   TypeSafe's own SDK: at most two, on a connection failure, 408, 429 or any 5xx, honouring
   `retry-after`, backing off from 0.5 s, doubling to 5 s, less up to 25% jitter. Each
   attempt has 10 s, and the whole request 30 s.
7. **Answers.** Every answer is checked against the question asked: the same ids, the
   right type, probabilities between 0 and 1, a Choice among the options offered, a Score
   along the levels given. A reply that fails the check is `bad_response`, never a decision.

## Measured against the live API, 2026-10-10

Documented and served are not the same thing, so these were tested with synthetic text
and recorded:

- `GET /v1/models` lists `jev-latest` and `jev-preview` (release date 2026-09-10). A
  decision answered by `jev-latest` names `jev-1.13.0`. That version is pinned in the
  catalogue, and a reply from any other version sets the result's `drifted` flag, so a
  threshold tuned on one version is not silently trusted on the next.
- **A malformed request is HTTP 400, not the documented 422.** An unknown model returns
  `{"detail": {"error_type": "api_usage_error", "message": "Unknown model: ..."}}`. A Noul
  with neither instructions nor criteria returns `"Noul question must have criteria or
  instructions"`, so a Noul with criteria alone is legal.
- **An over-budget request is refused, never truncated.** A state past 32k tokens, and a
  request past 64k, both returned HTTP 400 `{"error_type": "max_tokens_exceeded"}` with no
  usage. HAL reports that as `over_budget`.
- **Identical requests do NOT return identical answers.** The first check (2026-10-10:
  Noul, Choice and Score in one batch, sent twice) saw two equal replies, and this page
  said the answers were identical. Two requests were too few. On 2026-10-12 the same
  three-question request was sent six times to the same `jev-1.13.0` and came back as six
  different answer sets: the Noul ranged from 0.91 to 0.94, the Score from 1.82 to 1.95
  with single level probabilities moving by up to 0.08, and a Choice already at 1.0 did
  not move. HAL adds nothing random to the request body (only the retry delay is
  randomised). The vendor reports the same kind of variation between repeats of one
  request, and published it before our first check: its self-consistency cookbook
  (<https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook.md>) repeats one
  request 15 times and reports "`typesafe_noul` varies most on `covered` (`0.43` to
  `0.53`) and `exclusion` (`0.53` to `0.62`)", notes that "reasoning models and TypeSafe run
  without a temperature setting", and warns that "A value near either outer boundary
  can still move between `uncertain` and yes or no." The API reference (read on 2026-10-12, for
  `jev-1.13.0`) documents no temperature or seed field, so a caller has no way to switch
  the variation off. Check that again when the pinned version moves. What follows:
  - a reading near a threshold can fall on either side when it is asked again, so never
    repeat a request just to get a reading that passes; take several readings and use
    their mean and their spread;
  - a recorded answer is one reading, and replaying the request will not reproduce its
    numbers;
  - a loop that waits for two equal answers needs a tolerance, or it may never stop.
- End-to-end latency from South Africa was about 0.35 to 0.6 s per request. The vendor's
  0.1 s figure is server-side.
- No rate-limit headers were returned.

## Known weaknesses, and the rules that follow

TypeSafe's own list for `jev-1.13` (its "jaggedness" page) and an independent review
(the plan's research lane) agree on the shape:

- **It can be steered by text inside `state`.** One published adversarial benchmark
  reports that appending a single unverified opinion flipped 12.1% of decisions. So: **a
  decision model is never the sole gate, and never decides on untrusted text that could
  unlock a capability.** In a safety gate it may only tighten, never loosen.
- **First-listed bias in a Choice.** Before any Choice gates an action, ask it twice with
  the options reversed, and act only when both agree. Today the only caller is GRIP's
  `lib/infer_classify.py`, whose contract forbids use as a safety gate (its module
  docstring, "What this is NOT for"), so no Choice gates an action yet. Asking again in
  the same order does not remove the bias, because the lean belongs to the order.
  Whether the variation between answers (see above) weakens it has not been measured.
- **Literal reading; unreliable arithmetic and date comparison; weak multi-hop; distracted
  by large irrelevant state; confused by contradictory criteria.** Keep states small and
  questions single-step.
- **`confidence` is derived from the probabilities**, for a Choice `(p_max - 1/n) / (1 -
  1/n)`, and an independent evaluation found it adds no signal beyond the top probability.
  **No calibration is on file**, so every result carries `calibrated: false`, and nothing
  in HAL sets it to true yet. A probability here is the model's own number, not a measured
  error rate. Measure on your own labelled examples before tuning a threshold. Turning
  `calibrated` on needs two things that do not exist yet: a recorded measurement of one
  question configuration's error (its wording, options and model version) on a labelled
  set, and code that reads that record.

## What may be sent

**Internal and public text only**: GRIP and HAL code, docs and logs, and public legal
sources. **Never** personal matters, WhatsApp content, children's data, secrets, or client
documents (V>> 2026-10-10). The shield is a net, not a licence: callers decide what they
send. TypeSafe accepts no data from anyone under 18.

## Terms that bind us (TypeSafe's Master Customer Agreement)

- No uptime SLA; the service is provided "as is". The status page showed 99.836% from one
  region.
- Credits are prepaid, cannot be refunded, and expire within 12 months.
- **Section 2.3(b) forbids training a model to imitate Jev's outputs**, so Jev's labels are
  never used to train a GRIP classifier.
- TypeSafe may derive "Telemetry" from our data (4.1(c) and 4.3). Inputs are not used for
  training. Data is held in the US under EU standard contractual clauses. Zero data
  retention is for enterprise customers only.
- Liability is capped at 12 months' fees or $50. Breaching the credential terms is uncapped
  (12.3).

## Results and reasons

A result is `ok` only when every question has a checked answer. Otherwise `reason` is one
of:

| Reason | Meaning |
|---|---|
| `invalid_request` | the request breaks the format (checked before any send) |
| `unknown_model` | the model id is not in `config/decision_models.yaml` |
| `catalog_unavailable` | that file cannot be used |
| `spend_refused` | not opted in, no cap, or today's cap reached |
| `no_key` | no usable key in the keychain or the environment |
| `shield_unavailable` | GRIP's decision shield is missing, so nothing was sent |
| `shield_refused` | the shield could not scrub the body, or a key would have changed |
| `over_budget` | over the local byte ceiling or the model's token budget |
| `circuit_open` | recent decisions kept failing, so this one was not tried |
| `auth_failed` | the provider rejected the key |
| `rate_limited`, `overloaded`, `server_error` | 429, 529 or another 5xx after the retries |
| `timeout`, `unreachable` | no answer in time, or no connection |
| `bad_response` | a reply that does not answer the questions asked |
| `provider_error` | the provider raised where it should have answered |

Every caller keeps its own fallback for an unavailable result. `decide()` never guesses.

## Files

- `lib/hal/decide/`: `__init__.py` (the public names), `contract.py` (questions, answers,
  results, and `REASONS`, the closed set of reason codes), `catalog.py`,
  `shield.py` (loads GRIP's shield), `typesafe.py` (the provider), `dispatch.py`
  (`decide()`), `http.py` (the route), `__main__.py` (`hal decide`).
- `config/decision_models.yaml`, and `config/spend_guard.json` (`typesafe-api-key`).
- Tests: `tests/test_decide_contract.py`, `tests/test_decide_typesafe.py`,
  `tests/test_decide_surfaces.py`, `tests/test_decide_shield.py`. `tests/conftest.py`
  fences the keychain item and the environment fallback, so no test can read a real key.
