# The InfiniteMachine continuity contract

Status: adopted by V>> on 2026-10-04, as part of the approved InfiniteMachine v2
plan. This file is Step 1 of that plan. The decision record is GRIP
`docs/adr/0016-infinite-machine-v2.md`.

**In one line:** the thread survives; a turn may pause; nothing is faked.

## What this document governs

This contract governs HAL code. The wire (`lib/hal/wire_proxy.py`, Step 2), the
signed thread store (`lib/hal/infinite_thread.py`, Step 4) and failover (Step 5)
must keep to it. If HAL code and this contract disagree, the code is wrong,
unless this contract is changed first (see the last section).

It says what HAL must do. It does not describe what HAL does today. When the
plan was approved, none of it was built, and the inspection behind the plan
found that HAL's failover had never fired.

## The promise

No rate-limit, overload or usage-limit error reaches any harness while any
authorised capacity exists. One such error is a bug.

If literally everything is exhausted, the user sees one honest pause notice. It
names when work resumes. Work then resumes by itself, where the harness can do
that. A paid key is never used silently.

## The continuity table

| Event | User sees | Survives | May pause | Recoverable |
|---|---|---|---|---|
| One account hits its limit | nothing | thread, Remote Control, voice | the turn, up to `retry-after` | parked request, replayed once only if never ambiguously accepted |
| Anthropic overloaded | a non-fatal "holding" note after about 30 s | session, thread | the turn | parked request |
| Anthropic fully down | one honest pause notice | session, signed thread, checkpoint | inference | resume when any upstream returns, or a W4 switch at a turn boundary |
| Local wire dies mid-session | one visible, recoverable interruption | session (now direct, badged), Remote Control, chain to last signed row | -- | gap recorded, never backfilled; next launch re-wires |
| `/clear` cycle | a short reload, only after the dry run shows savings | thread id, sealed checkpoint, chain | -- | prior context from checkpoint |
| Host restart | a resume prompt | thread store, chain, cool-down state, CA key | -- | Remote Control re-pairs on relaunch |
| VPS down | nothing on local sessions | everything local | aggregation, shared cool-down sync (queued) | sync on return |
| All authorised capacity exhausted | one pause notice naming the earliest `retry-after` | session, thread | inference | parked request when capacity returns |

## Invariants

These hold on every route, on every host, and at every rung of the ladder.

1. An account-bound route never changes bearer.
2. HAL never splices two responses into one. Responses from two accounts are
   never spliced.
3. Headers are never logged. Exception dumps are scrubbed.
4. HAL switches account or provider only before any byte has reached the
   client.
5. HAL never replays a request that may already have been accepted.
6. Once bytes have reached the client, a failure is one visible, recoverable
   interruption. It is never hidden.

## Route classes

HAL sorts each request by its content, not by its path alone. On Claude Code's
wire, HAL decrypts `api.anthropic.com` only.

| Class | What it covers | HAL does |
|---|---|---|
| `inference` | real conversation turns on `/v1/messages` | witnesses it, prices it; failover may act on it |
| `aux` | quota checks, token counts and title calls on the same path | witnesses it; never swaps it |
| `account-bound` | `/v1/code/*`, `/v1/environments*`, `/api/oauth/*`, `/api/ws/speech_to_text/*`, settings | byte-exact pass-through |
| `unknown` | anything else | pass-through, plus an audit row |

## The ladder

When an `inference` request fails, HAL escalates in this order. The table under
"Error to action" says what each upstream signal triggers.

1. Retry on the same account, honouring `retry-after`.
2. Switch account.
3. Switch provider (the W4 provider cascade).
4. Give one honest pause notice, then resume by itself.

Every rung obeys the invariants. No rung acts on an `aux` or `account-bound`
route. A hold stays inside the tolerance that Step 1 measures for each harness
and route (H-HOLD-1).

## The per-account ledger

A per-account ledger decides before HAL answers. HAL builds it from the
`anthropic-ratelimit-unified-*` headers on every response, success or failure:
status, 5-hour and 7-day use, reset times, overage state, token health and
requests in flight.

- The ledger is written atomically.
- A cooldown lasts until the reset that the headers name.
- Tokens come from GRIP's token store.
- HAL uses one auth scheme: `Authorization`, only.

## Error to action

| Upstream signal | Scope | HAL does |
|---|---|---|
| 429 carrying `representative-claim` or `overage-status`, or overage showing on a 2xx | this account | mark it spent until its reset; switch account (see the account switch below) |
| 401 on a pool token | this account | mark the token dead (needs a re-login) and switch; with no account to switch to, answer the honest exhaustion 429 above if another pooled account is merely spent to a known future reset, else relay the 401 (f15; the split below). A 401 that names a revocation or a suspension is the enforcement row below instead |
| 403 on a pool token | this account | relay it unchanged: never switch, replay, cascade or pause it. A 403 alone does not mark the token dead: it is a permission answer (a model the account may not use, for one), which can also be an enforcement action, and switching account after one is what Anthropic's Usage Policy names. A 403 that names a revocation or a suspension is the enforcement row below, which does hold its token dead |
| 401 or 403 whose message names a revocation or a suspension, from any account: a turn, an aux call, a replacement's answer or an organisation lookup | every account | stop ALL account switching (the enforcement stop below), record why, tell the operator once, and relay the answer unchanged; a 403's token is held dead (a 401's already is) |
| 429 without quota headers | uncertain | short hold honouring `retry-after`, then switch after N attempts |
| 529 overloaded | everyone | hold and retry; switching account does not help |
| 5xx, connect or DNS failure before any byte | provider or network | hold and retry; the OpenAI-wire rails may switch provider at a request boundary |
| any failure after bytes reached the client | -- | no splice; one visible, recoverable interruption |
| 400, 413, context too long | the request itself | forward unchanged |
| `/v1/code/*`, `/api/oauth/*`, voice, settings | the signed-in account | byte-exact pass-through |

## HAL's own errors (Step 5a, built)

Holding or switching a request multiplies every error HAL makes itself, so
these rules land before any hold or switch. They hold in the gateway
(127.0.0.1:4010) and, where the same fault can arise, in the wire
(127.0.0.1:4011). Tests: `tests/test_gateway_error_source.py` and the Step 5a
tests at the end of `tests/test_wire_proxy.py`. These are faults HAL meets while
it is running. The continuity table's "Local wire dies mid-session" is the
other case: HAL itself stops, and nothing is left to send an answer.

| HAL meets | The client is sent |
|---|---|
| a connect, DNS, TLS, reset or timeout fault before any byte of the answer | a 503 with an error body in its own dialect, never a 400, which Codex, OpenCode and Hermes never retry (f48, f68). The wire says the request may have reached Anthropic. |
| an upstream 429 on `/v1/responses` or `/v1/chat/completions` | a 503, `Retry-After` kept: one 429 ends a Codex turn even with `Retry-After` set (f26). `/v1/messages` keeps its 429. Step 5c replaces the rename with a hold and an account switch. |
| an upstream error carrying `Retry-After` or `retry-after-ms` | the same headers, on every relay (f73) |
| a stream | each event as it arrives, within 100 ms (f79). The first comes with the response head, so its time also counts HAL's per-response work before the first byte. |
| a stream that fails after it has started | one in-band error event in its own dialect, then the end: never a finish chunk, `[DONE]`, `response.completed` or `message_stop` after it (f72). The Responses route sends an `error` event, never `response.failed`, whose codes Codex can treat as final. |
| more requests at once than the in-flight cap | a 503 with `Retry-After`, at once: never a 429 and never a wait (f82, f121). `/health`, Remote Control, voice and OAuth are never capped. |

One 429 is HAL's own and is not renamed. When the spend guard refuses a spent
subscription whose paid overage is not opted in, nothing goes upstream and
every rail, the OpenAI-wire ones included, is sent the guard's 429 with
`Retry-After` until the reset (`lib/hal/spend_guard.py`, Step 5h). The account
cannot serve before then, so a retry sooner would only be refused again;
absorbing it is the account failover the plan gives these rails (5g). Pinned
by `tests/test_spend_guard_paths.py`.

The wire's own 503 bodies are distinct and honest; both are defined in
`lib/hal/wire_proxy.py`. When `api.anthropic.com` cannot be reached at all,
the wire answers with `_UNREACHABLE`: "nothing was sent to it". When the
request went out but no answer came back, it answers with `_NO_ANSWER`: "the
request may have reached it; retry your request". `_NO_ANSWER` is written with
`_send_quietly`: a client that has already gone away while waiting is not an
error, so that send never raises. `_UNREACHABLE` is a plain send; if its client
has already gone, the failure ends that one connection and nothing else.

The gateway's upstream timeouts (f83). Each is read from the environment; a
value that is not a positive number keeps the default.

| Setting | Default | Covers |
|---|---|---|
| `HAL_GATEWAY_CONNECT_TIMEOUT_S` | 10 s | TCP connect, any proxy CONNECT and the TLS handshake |
| `HAL_GATEWAY_TOTAL_TIMEOUT_S` | 600 s | the whole exchange; a stream leaves it once its headers arrive |
| `HAL_GATEWAY_STREAM_IDLE_TIMEOUT_S` | 600 s | a stream: the longest silence between two reads |
| `HAL_GATEWAY_SHORT_TIMEOUT_S` | 60 s | the whole exchange for a GET or a token count |

The stream idle limit stays at the old 600 s so that no stream that passed
before is cut now; Step 5d tunes it against the measured hold budgets. The
wire's own limits (connect 10 s, TLS handshake 15 s, idle 900 s) were already
split and are unchanged.

The in-flight cap: `HAL_GATEWAY_MAX_INFLIGHT` (default 64) requests to `/v1/*`
in the gateway, and `HAL_WIRE_MAX_INFLIGHT` (default 64) requests on the
messages rail in the wire. The `Retry-After` sent past each cap is
`HAL_GATEWAY_BUSY_RETRY_AFTER_S` and `HAL_WIRE_BUSY_RETRY_AFTER_S` (default 5
seconds each).

## The account switch, and the thinking-signature rule

HAL switches account by replaying the same request bytes, with the bearer
replaced.

Anthropic's thinking docs set the boundary:

- "Required: within a tool-use turn, pass thinking blocks back"
- "Allowed: outside tool use, omit prior turns' thinking"
- "Modified thinking blocks are rejected with a 400 error."

So:

- **At a turn boundary outside a tool loop,** a switch does not depend on
  cross-account signatures. If H-INF-SIG fails, HAL omits prior turns'
  thinking, as the vendor allows.
- **Inside a tool loop,** the thinking blocks must travel unmodified, so the
  switch depends on H-INF-SIG. If it fails, HAL holds within the budget and
  then gives the pause notice.
- **Forbidden:** dropping thinking mid-turn. The API answers that by silently
  disabling thinking for that request. That would fake quality, and this
  contract forbids faking.

H-INF-SIG asks whether thinking blocks signed under account A are accepted on
account B. The vendor's thinking docs say that signature values work across
platforms. That is not a test of two subscription accounts, so the live test
stays.

**Usage headers.** HAL records the values of the `anthropic-ratelimit-unified-*`
headers in the ledger, then strips those headers from what Claude Code sees. So
Claude Code shows no usage warning for the wrong account. The one exception is
the final exhaustion 429. It carries `representative-claim` and the earliest
reset, so Claude Code shows its own usage-limit line, and auto-continue can
resume.

**Cache.** A switched conversation starts with a cold prompt cache on the other
account. Keeping a conversation on one account needs a conversation key, which
HAL lacks today.

**Off-switch.** Switching is on by default (Decision 6). One command turns it
off, or on again. HAL honours the change from the next request. The
enforcement stop below engages the same off-switch by itself; only the
operator turns it back on.

**Capacity by organisation (2026-10-08).** Accounts in one organisation share
limits. So an exhausted or a spent mark on one account counts for every pool
account whose known organisation is the same, wherever HAL reads a pool
account's capacity: the switch's choice, the hold's "all spent" check and its
pause reset, and the W4 status. An organisation HAL has not learned changes
nothing. HAL learns an organisation from the free `GET /v1/models` lookup the
switch already makes, and keeps it by a digest of the token.

How far that is documented: Anthropic's rate-limit page says the API "enforces
service-configured limits at the organization level" and "Limits are set at
the organization level" (platform.claude.com/docs/en/api/rate-limits, read
2026-10-08). It says nothing about the 5-hour and weekly windows of a
subscription. That those are shared within an organisation too is HAL's working
assumption, the same one its limit switch has made since HAL#663 (a
replacement must come from another organisation), not a documented fact. It is
wrong if two pool tokens that report one `anthropic-organization-id` are seen
with different `anthropic-ratelimit-unified-*` utilisation at the same moment;
the account ledger records those numbers per account, so that is checkable.

**The enforcement stop (2026-10-08).** Anthropic says it may enforce its terms
"without prior notice", and its Usage Policy names continuing activity across
accounts after an enforcement action. So the wire reads the body of every 401
and 403 it sees on a turn, an aux call, a replacement's answer or an
organisation lookup, puts it back byte for byte, and judges it
(`lib/hal/auth_signal.py`) from the JSON `error.type` and `error.message`
only. When the message names a revocation or a suspension (revoked, disabled,
suspended, banned, terminated, violation), on any account, the wire:

- engages the off-switch, so nothing is switched from that moment, the
  request in hand included;
- writes into the off-switch's marker why: the account's label, the status,
  the error type, the word that matched and when, never a token, a header
  value or the message;
- tells the operator once, through HAL's notification hub
  (`lib/hal/notification.py`), whose history is kept in
  `$HAL_HOME/state/hal/notifications/history.json`; nothing subscribes to that
  hub yet, so the reason is also on `/health` (`account_switch_stop`) and in
  `python -m lib.hal.account_switch status`;
- relays the answer to the client unchanged, and holds a 403's token dead in
  the ledger (a 401's already is).

Switching stays off until the operator runs `python -m lib.hal.account_switch
on`; `off` keeps the recorded reason. Anthropic documents the error types (401
`authentication_error`, 403 `permission_error`), not the wording an enforcement
action is sent with, so the word list is a guess, to be corrected from real
bodies. A message that names both an expiry and a revocation reads as
enforcement. HAL's gateway (127.0.0.1:4010) honours the stop through the same
switch but does not judge 401 or 403 bodies itself; it never switches on one.

**The dead-token split (2026-10-08).** Only a 401 on the presented token is a
dead token, and only when its message names no enforcement action (the stop
above). When no account can carry such a turn now (`choose()` found no
replacement, and the W4 provider cascade found none either), HAL asks what the
OTHER pooled accounts are doing. If at least one is merely spent -- cooling to a reset the
ledger or the account switch already knows, the same marks the usage-header
exhaustion answer above reads -- HAL answers that SAME honest exhaustion 429,
with that real reset, instead of the 401. Claude Code then shows its
usage-limit line once and its auto-continue resumes when the reset passes,
rather than retrying a dead key and then asking for a fresh login
(`tools/wire-matrix/RESULTS.md` scenario S4 recorded 11 requests over 175.9
seconds before "Failed to authenticate. API Error: 401 Invalid API key."; a
re-run of S4 through a scratch wire on 2026-10-08, with the same Claude Code
2.1.289, stopped after 2 requests in about 2 seconds with the same message --
why the two counts differ is not established). When no pooled account has a known future reset -- every
one of them is also dead, or none names a reset -- HAL relays the 401
unchanged, because only a re-login fixes that; no reset is ever invented. A
403 never takes this path, on either leg: it can be a permission or
enforcement decision the operator must see, and pausing or switching over it
would read as using another account after a ban. Since the pool hardening of
the same day a 403 on the presented token is never switched or replayed at
all, with or without a replacement: it is always relayed unchanged.

## The spend guard

- HAL alone chooses a billing source. The guard is fail-closed, and it lives in
  one place.
- HAL refuses every paid source without explicit opt-in (amended 2026-10-06). The 2026-10-11
  no-spend date was removed on 2026-10-06 at the operator's order: it refused
  an account whose extra usage he had just paid for, while the account switch
  it was meant to sit behind did not yet exist. Subscription capacity on
  another pool account is used before overage (see the account switch).
- A paid key never counts as capacity.
- Overage showing on a 2xx response counts as spent.
- Paths that get around the guard are removed, not merely tested for:
  - the auth-failover auto-engage, which counted HAL's own 502s as 429s and
    moved new shells onto the paid key for 8 hours on 2026-10-03;
  - `gg++` exporting `hal-anthropic-external` as `OPENAI_API_KEY` for OpenCode
    and Hermes (unchecked);
  - the paid rungs: `grip-codex-ladder` rung 2, `key_failover`'s funded keys,
    and the env fallback chain.

## The operator's decisions (2026-10-04)

1. **Wire by intercepting proxy** for Claude Code: `HTTPS_PROXY`, plus a
   per-host CA name-constrained to `api.anthropic.com`. Never by
   `ANTHROPIC_BASE_URL`. Other harnesses keep base-URL wiring.
2. **"Never dies" covers all of:** account failover; the W1 automatic cycle,
   gated by an `/ultraclear` dry run (clear only when it actually saves
   something); the W2/W3 signed thread store; W4 provider switching through
   RAILLM.
3. **The promise:** no rate-limit, overload or usage-limit error reaches any
   harness while any authorised capacity exists. One such error is a bug. If
   literally everything is exhausted: one honest pause notice naming when it
   resumes, then automatic resume. Never a silent paid key.
4. **Hosts:** a wire on every trusted host, any OS, plus a central VPS (its own
   sessions and the shared brain). If a host's own wire is down, its new
   session goes direct, with a visible "unwitnessed" badge. Decrypted traffic
   and each host's CA never leave that host.
5. **Records:** public-key signed, with checkpoints held outside the writer's
   control.
6. **Automatic account switching ships on by default.** This is the operator's
   decision, made at the fourth gate on 2026-10-04. Before it, the operator was
   shown Anthropic's Consumer Terms (effective 8 October 2025, read
   2026-10-04). They forbid access "through automated or non-human means,
   whether through a bot, script, or otherwise" except "via an Anthropic API
   Key or where we otherwise explicitly permit it". They also forbid "bypassing
   any of our systems or protective measures". The operator accepted the risk
   to both accounts. The switch keeps one off-switch, the brake, so the
   decision stays reversible in one command.

   *Note, 2026-10-15:* a third Max account, stored under the label `laurdev`,
   joined GRIP's pool on this date at the operator's request, so the switch
   now chooses among three accounts. The decision and the brake are
   unchanged. GRIP ADR-0016 carries the same note.

## How to change this document

- Every change rides a pull request.
- A change to the invariants is a bootstrap-authority change. The operator
  merges it. A review verdict alone never merges it.

## Sources

- The approved plan, `plans/ULTRAPLAN-2026-10-04-infinite-machine-v2.md` in
  GRIP. `plans/` is gitignored; a copy is in
  `PARAMOUNT_BACKUPS/2026-10-04/infinite-machine-v2-plan-approved`. The table
  comes from its "continuity contract"; the route classes from Step 2; the
  ledger, error to action, account switch and spend guard from Step 5 (5b, 5c,
  5e, 5h).
- `fN` is a finding of converge-fleet run `wf_5f3c7295-40f`. The evidence is in
  `PARAMOUNT_BACKUPS/2026-10-04/infinite-machine-v2-fleet-evidence`, outside
  this repository.
- Not restated here: the per-harness hold budgets and duties (plan Step 5: 5d,
  5g). They stay in the plan until Step 5 turns them into code and tests. Step
  5a is built and has its own section above.
- The in-band error frames follow what the T1 stub matrix measured on the real
  harnesses (`tools/wire-matrix/RESULTS.md`, scenario S7), and the Codex
  lane's runs of the fleet evidence (`response.failed` with
  `server_is_overloaded` ended a Codex turn outright).
- Vendor sources, read 2026-10-04: platform.claude.com "Thinking"; Anthropic's
  Consumer Terms, effective 8 October 2025.

## Provenance

V>> -- operator; decisions 1-6; plan approved 2026-10-04 17:03 UTC.

With assistance of:

- Claude Fable 5.1 (`claude-fable-5-1`) -- drafted the continuity table.
- Claude Opus 5.5 (`claude-opus-5-5[1m]`) -- plan synthesis; wrote this file
  from the approved plan.
- Converge-fleet run `wf_5f3c7295-40f`, and a mesh council (deepseek-v4-pro and
  grok-4.5, judged by gemini-3.1-pro-preview, seal `9c8a247fbb7ce8aa`) -- the
  evidence behind the error-to-action table and the spend guard.
- Claude Opus 5.5 (`claude-opus-5-5[1m]`) -- the pool-hardening amendments of
  2026-10-08 (the 403 rows, capacity by organisation, the enforcement stop),
  from GRIP `plans/ULTRAPLAN-2026-10-08-claude-capacity.md`, section 4, phase 1.
