# W4 provider cascade on the OpenAI wires

After every configured Anthropic gateway account is known spent, the gateway can
try another provider before returning the existing pause notice. This applies to
`/v1/responses` and `/v1/chat/completions` only. Claude Code's fresh-thread move
belongs to GRIP's launcher and signed thread store, not this gateway
([GRIP#7720](https://github.com/grip-platform/GRIP/issues/7720)).

## Policy and configuration

`config/w4_cascade.json` ships with `{"targets": []}`. `HAL_W4_CONFIG` can point to
another file. Legacy string entries are `provider/model` addresses from
`raillm_ladder.ladder_cascade_order()` and retain ladder order and paid or unknown
billing. A Token Plan subscription entry is an exact object binding; it does
not need ladder membership. Unknown or malformed objects are ignored. An
unreadable or malformed policy enables nothing. Production activation remains
an operator configuration decision.

The approved object has exactly these fields: `provider: qwen-token-plan`,
`product: token-plan-personal`, `model: qwen3.8-max` or `deepseek-v4-pro`,
`base_url: https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1`,
`credential: grip-qwen-token-plan`, `auth: bearer`, `billing: subscription`, and
`transport: openai-chat`. The credential is a selector, never a value. The
gateway checks this tuple against the trusted product definition and the
provider-qualified model catalog before resolving that selector. A changed
endpoint, selector, provider, model or billing claim cannot inherit subscription
status. The bound HTTPS opener refuses redirects.

Enabling a target does not authorize payment. Each cloud candidate asks the
existing spend guard for `<provider>-api-key` (OpenAI uses its canonical guard
source and separate opt-in). Unknown billing is treated as paid. The one
exception is an exact subscription binding: the Token Plan's two catalogued
models are bound as product `token-plan-personal` in `lib/hal/cascade_target.py`,
over chat completions only, and nothing else inherits that status. No target is
enabled by default. Do not populate targets merely to make a demonstration
switch work.

The legacy supported transports are Moonshot, Groq, Gemini and DeepSeek's
existing gateway endpoints. The validated Token Plan binding uses its separate
OpenAI-compatible endpoint and account. The other ladder
entries remain in policy but are refused if their window, transport or tool
support cannot be established. No local service is started or reconfigured.

## Eligibility and request preservation

Every candidate must pass the spend guard, the request's context bound, tool
support, and its credential-scoped account-ledger cooldown. The context ceiling
is the smaller of the model catalog's window and its RAILLM tier's window; no
fallback guesses a window for an unknown model. A text request is conservatively
bounded by its JSON UTF-8 byte count with ASCII escaping, plus 4096 framing tokens
and the requested output limit (4096 when absent). Byte-BPE and byte-fallback
text tokenizers cannot emit more tokens than the represented bytes. Images,
audio, files, server-side continuation IDs and unresolved input references are refused rather than assigned
an invented token estimate. This is a bound, not a measured tokenizer count.

Only ordinary function tools that the existing translators preserve are
accepted. Custom tools, hosted tools, `additional_tools` input items, unsupported
strict definitions, parallel-call constraints and forced Responses tool choices
are refused. In particular, Codex's GPT-6 built-ins must not disappear during
the Anthropic pivot. Tool schemas and argument data remain opaque; protocol IDs
are mapped consistently and provider reasoning/signatures/encrypted replay state
are stripped from switched requests. Nonempty continuation IDs are refused
because HAL cannot reconstruct their hidden history. Fresh provider reply IDs
are also normalized, including streamed tool calls. The chat stream translator
preserves function-call starts and argument fragments, with contiguous tool
indices even when Anthropic content-block indices include intervening text. The original payload is not modified.

The checked credential is pinned for that attempt. The account ledger observes
the raw target response before translation, including Retry-After cooldown.
On the Claude Code W4 wire, a quota-signalled 429 cools that credential until
Retry-After, the Alibaba Token Plan reset, or a 15-minute default, in that order.
While it cools, a fresh turn does not contact that target and `/w4/status` does
not advertise it as `fresh_thread_target`.
Also on the Claude Code W4 wire, a target's `drop` list never removes a
constraint. `context_management` is dropped only when it is `{}`,
`{"edits": []}` or exactly `{"edits": [{"type": "clear_thinking_20251015",
"keep": "all"}]}`, `safeguards` only when it is `[]`, and `output_config` only
when it is `{}` or holds nothing but `effort`. That clear_thinking edit keeps
every thinking block (Anthropic's context-editing page: `"all"` keeps all
thinking blocks, which is already the default on Opus 4.5 and later), so it
clears nothing; an edit that does clear blocks, or that edit beside another,
still refuses. Any other value in those three
fields refuses the move with the reason `fields`. That is the reason an
unexpected field already gets: the field names in the audit `detail` say which
it was, and the detail holds those names alone, never a value, a schema or a
nested key. `thinking`, `diagnostics` and any other field a target declares are
dropped whatever they hold: losing `thinking` and the effort setting is the
capability downgrade accepted when those targets were armed.
Probed 2026-10-12, Claude Code 2.1.296 (`claude -p`, API-key auth) sends one
`safeguards` entry of type `dangerous_tool_use` on every turn and, with
thinking on, the keep-all `clear_thinking_20251015` edit. Under this rule each
of the two shipped Token Plan targets refuses that headless request, and the
audit names `safeguards`. Probed 2026-10-15 (HAL#676), the same version in a
real terminal, through the wire with a subscription bearer, sends the keep-all
edit on every turn with its thinking setting on or off, with and without
`--safe-mode`, and no `safeguards` entry. Such a turn moves.
The `safeguards` entry has keys such as `user_identity` and `home_dir` (key
names only were recorded), so it is treated as naming the user and the machine
and is never forwarded to another vendor: the choice for it is refuse or drop.
Ordinary function declarations and the complete supplied history travel through
the gateway's existing translators, never through a flattened prompt.

## Failure and audit behavior

Known pre-send failures (credential/preparation failures, connection refused,
DNS resolution failure) decline that candidate. A timeout, reset, broken pipe
or other uncertain acceptance ends the cascade. Every non-2xx target response
is closed and declined before client headers, then the next candidate is tried.
The audit names rejected providers, closed-vocabulary reasons and HTTP status,
and names the provider/model that ultimately served. It includes the original
account-exhaustion reason and earliest Retry-After, never credentials or prompts.
For field and tool refusals, the audit detail names refused field names and tool
types only when they match `^[a-z][a-z0-9_]{0,39}$`, with at most 12 names per
list. It never records values or tool names.
The audit sink checks every detail again, whoever produced it: it keeps safe
names and counts any it drops, bounds the counters and the cooldown time, keeps
a cooldown source only beside a valid time, and drops every other detail value.
The existing witness writes the gateway audit, its legacy HMAC chain and the
public-key thread row (subject to the existing signer availability contract).
On both OpenAI rails, a signed `failover` decision row naming the target and its
declared billing is required before forwarding; if it cannot be written, the
audit records `unsigned` and the client receives the original exhaustion answer.

On the Claude Code wire, a first move is signed before it happens. A move
sends the conversation to another vendor, so a public-key signed `failover`
row recording that decision (target provider and model, its declared billing,
`status` null) is written before any byte reaches the target. If the thread
store cannot write it -- no signer, a damaged thread, a disk error -- the
move is refused with the audit reason `unsigned`, the target receives
nothing, and the client gets the honest exhaustion answer. A second signed
`failover` row then records the target's answer status. Every signed row of a
moved exchange, the witness row included, carries the billing the target
declares, never the billing the inbound Anthropic credential suggests.

When no target can serve, HAL#667 remains the last rung: Codex gets one final
message naming the original earliest reset; chat gets 429 and Retry-After.
Cascade failures never synthesize a 5xx. A chat client's native retry after that
reset remains possible; the cascade does not promise exactly-once execution.
After a selected stream begins, a failure is a visible stream interruption;
it never re-enters provider selection or splices another provider's response.

## Production reachability and account switching

`build_failover_proxy()` configures `anthropic-primary` by default. HAL#686 now
switches accounts inside that gateway proxy before the OpenAI handlers answer.
An Anthropic usage-limit 429 marks the account exhausted; the gateway can replay
once on another eligible subscription account. When the authorised pool is spent,
the proxy gives W4 an `AccountsSpent` refusal with the earliest observed reset.
If a configured Anthropic upstream (not a W4 target) needs paid usage and the
spend guard refuses it, that upstream is unavailable capacity. It is never sent
without explicit opt-in, and it cannot prevent W4 from starting after the
subscription accounts are spent. W4 starts only when at least one account
reported its limit: a pool made only of refused paid upstreams keeps its
original refusal and never starts W4. W4 targets themselves still pass the spend
guard as described under "Eligibility and request preservation".
A plain 429 without usage-limit headers is still an upstream throttle, so it
does not start W4. A third eligible account left after the one replay budget
also keeps W4 from starting until a later request checks that account.
HAL#676 remains open for Step 7 end-to-end cascade acceptance.

On that wire, a fresh bounded turn now offers W4 before a spend-guard exhaustion
429, a pre-header hold exhaustion answer, or a locally refused replacement. A
timeout, uncertain send, expired hold budget, timeout-derived fallback, or any
response whose headers or pings reached the client never starts a new move.

When the replacement account also answers its limit, the pool is spent and W4
is offered. If no target takes the turn, the client gets one pause notice that
names the earliest reset the wire knows, as `docs/infinite-machine-contract.md`
promises ("All authorised capacity exhausted"). The presented account's own
refusal is relayed unchanged when it already names that earliest reset, when no other
pooled account is known to come back, or when its own limit has passed during
the attempt. When a pooled subscription has capacity again by then, because its
limit passed while the fallback was tried, the 429 names the present as the
reset and says so, and Claude Code's retry is switched to that account (HAL#748
item 18). The test is the account switcher's own choice, which must also carry
no spent mark in the account ledger, so an account the switcher would not use,
or one another wire has seen spent, never counts; and it is never answered
after a fallback send that may have been accepted. Otherwise the wire sends the
honest exhaustion 429 with another pooled account's earlier reset, as a hold that finds the pool spent does; a
pool token held dead is never counted, whatever its old reset says. When the
replay began with a dead presented token, the client gets the dead-token pause
from the contract's 401 row; its 401 is relayed only when no other pooled
account is known to come back. Before this, the presented account's own answer
was always relayed, so Claude Code waited for that account's reset even when
another account came back sooner (found by HAL#676's weekly terminal run).

### What keeps one turn from being sent twice

Contract invariant 5 says: "HAL never replays a request that may already have
been accepted" (`docs/infinite-machine-contract.md`). So the wire offers a turn
to W4 only when it holds no record that anything may have accepted it.
`wire_cascade.Attempts` is that record. It is not complete: "Known limits"
below names where.

- **The key.** A turn is named by its session id and its content
  (`wire_hold.turn_key`), never by its credential, so an account switch keeps
  the same key. A turn with no session id is never offered. The content leaves
  out `max_tokens`, which Claude changes on its non-streaming fallback, so a turn
  and its fallback share one key. A one-token account probe is marked apart, by the
  classifier's own rule (`wire_classify.is_account_probe`: a request the gateway
  hint header names a turn is never a probe), so an unanswered probe's closure no
  longer closes the key of a turn with the same messages. One written before this
  change still does; see "Known limits".
- **Reservation.** An offer first creates the turn's reservation file
  (`O_EXCL`), so two requests for one turn cannot both be sent. It is released
  only on proof that nothing accepted the turn. It stays for good after a 2xx
  answer from a target, after an attempt that got no answer, and after any
  failure that came once request bytes may have left, whatever raised it.
- **Closure.** A second file, `<key>.closed`, is written when a send to
  Anthropic ended with no response head. It is permanent: releasing a
  reservation never removes it.
- **Sends on their way to Anthropic.** They are counted in memory, with no file
  work on a normal turn. A held or replayed send marks its turn. A streamed send
  marks its SESSION from before its first byte, because its turn's key is not
  known until its body has been read. A reservation is refused while either mark
  stands, and withdrawn when a send that began while its file was being created
  is still on its way. A send that began and ended in that window is caught only
  by its closure, when it ended unanswered; an answered one is the first known
  limit below. A refused reservation removes the file it made.
- **A lost closure.** When a closure cannot be written, every offer in that
  process is refused until restart, and an `UNHEALTHY` file is written when it
  can be. An operator checks the disk and then removes that file.
- **The deadline.** The hold deadline is checked at the offer, before the
  signature, after the address is resolved and before the send. The connection
  and the TLS handshake each get only what is left of the budget. A decision row
  signed just as the budget runs out, with no outcome row after it, is a decision
  that was abandoned; `_exchange()` still refuses to send. After the send, every
  wait on the target is cut to what is left of the budget, worked out again
  before each read, until the response head is in. A target that takes the
  request and then says nothing ends as an uncertain send when the budget ends,
  and the client gets the refusal it would have had with no move. A move with no
  hold behind it has no such budget and keeps the wire's ordinary idle timeout.
- **The marks are removed when a send is settled.** A send that got no answer is
  closed first and unmarked second, so no offer can pass between the two.
- **The in-flight cap.** Every offer runs inside a slot of the wire's in-flight
  cap, the terminal path included. With no slot free there is no offer, and the
  client gets the refusal it would have had with no move.
- If a target's response headers have reached the client, a later body failure
  ends that exchange. The original refusal is never appended.

### Known limits

Nobody has to switch anything on for these to matter. `config/w4_cascade.json`
already declares targets, and the wire offers them a first turn whenever the
pool is exhausted. They serve nothing only while the request-field rules and
the vendor's own quota refuse. When that quota returns, a first turn that
passes the field rules moves by itself, on whatever code the running wire has.
Settle these limits, or lock first moves, before then.

| Limit | What can happen | What guards it today | Tracked in |
|---|---|---|---|
| **A turn that Anthropic answered is not remembered.** | If the client later sends the same turn again and the pool is then exhausted, that request can be offered to W4. | The wire never re-sends a turn by itself; this is the client's own retry. | HAL#748 items 1 and 4. Counted: each offer of a turn this process saw answered 2xx writes an audit row `{"event": "w4.limit", "limit": "answered_turn_again"}`. The memory of answered turns is bounded and decides nothing. |
| **A restart forgets sends in flight.** | The count of sends on their way to Anthropic is in memory, so a turn that was in flight when the wire stopped is not marked afterwards. | Reservations and closures are files and survive a restart. | HAL#748 item 4. Counted: each account in the account ledger keeps `orphaned_in_flight`, the running total of requests found in flight for a process that had stopped. |
| **A streamed send that fails before its whole body has been handed over is not closed.** | The turn stays open to W4: a closure needs the turn's key, and the key needs the whole body. | Anthropic cannot accept an incomplete body. A send that fails on its last piece, or after it, is closed, because by then the wire holds the whole body. | Safe by construction |
| **The record never stops a send to Anthropic.** | It stops a second offer to W4 and nothing else. If a turn has been offered to a target, or may have reached one, and the client sends it again while Anthropic has capacity, that request goes to Anthropic, so one turn can be answered by a target and by Anthropic, on two requests of the client's. | HAL sends neither request twice, and an ordinary turn never waits on this record or fails because of it. | HAL#748 item 11 (a decision) |
| **A session that has moved is not checked against this record.** | Once a first move has bound a session to a target, every later request of that session goes there (`serve_bound`), the client's retries included. If the target's first answer is cut off after its 2xx head and the client sends that turn again, the target receives it a second time. | The record guards the first move only. Whether such a retry should be served or refused is open. | HAL#748 item 6 (a decision) |
| **A probe closure written before HAL#765 still closes a turn with the same messages.** | Closures are permanent. Before HAL#765 an unanswered probe's closure carried the key of a real turn with the same messages, so such a file keeps that turn from W4 for good. | The safe direction: the turn is never offered, never sent twice. Such a file cannot be told apart from a real turn's closure, so none is removed. | HAL#748 item 7 |
| **"Retry now" can draw a burst, and can be wrong once.** | Every session told to retry is switched to the same returned account, and nothing reserves its capacity, so it can meet its limit again at once. A ledger that cannot be read, the spend guard's or the account ledger, counts as holding no marks (both readers return nothing rather than fail), so a still-spent account can read as returned. | Either way the retried turn meets that account's limit and its mark is written again. "Retry now" is answered only after a move that found every account spent, so the next refused move names a reset again. | HAL#748 item 18 (accepted) |
| **A replacement the spend guard refuses at the send.** | If an account is marked spent, or the overage policy changes, between the switcher's choice and the send, the spend guard refuses that replacement. After a refused move the presented account's own refusal is relayed, as before W4, even when another pooled account has capacity. | The switcher already passes over spent accounts and forbidden overage, so this needs a mark or a policy change inside that window. The refused account is never presented, and nothing chooses again. | HAL#748 item 19 (left as it is) |

## Billing evidence checked 2026-10-07

Public product availability is not proof of this installation's credential tier.
No real credential, account endpoint or paid inference was queried for this work.

- Anthropic: CCH is the existing subscription path, already exhausted before W4;
  it is not a new cross-provider target. Overages remain spend-guard controlled.
- OpenAI: the gateway endpoint is `api.openai.com` with an API key, not a ChatGPT
  subscription. Its separate paid opt-in still applies.
- Moonshot: the ladder and model catalog explicitly identify `kimi-k3` as metered.
  Kimi Code is a separate membership service and endpoint. Its optional ladder
  rung requires a resolved subscription; W4 does not arm that credential path.
  [Kimi Code membership guide](https://www.kimi.com/en/help/kimi-code/membership-guide)
  distinguishes membership billing from Open Platform pay-as-you-go.
- Alibaba: a ladder comment says subscription, but no subscription binding for
  this gateway endpoint is established. Unknown is paid; `qwen-max` also lacks
  an entry in this model catalog and is refused.
- Groq: [billing FAQ](https://console.groq.com/docs/billing-faqs) and
  [rate limits](https://console.groq.com/docs/rate-limits) describe free and paid
  Developer plans. The installed credential's plan is unknown, so treated paid.
- Gemini: [billing documentation](https://ai.google.dev/gemini-api/docs/billing)
  distinguishes Free and Paid tiers. The installed project's tier is unknown,
  so treated paid; consumer subscriptions do not establish API billing.
- DeepSeek: the catalog has positive input/output prices; the old ladder
  “cheap/free” comment is not evidence of a free credential. Treated paid.
- Ollama/MLX: local computation has no provider token bill, but the ladder's
  specific models have no verified catalog window and tool/transport binding
  here. No default local target is enabled on an assumption of capacity.

This deliberately leaves the default target list empty. Measurements of the
actual Codex request shape and the falsification runs are reported with the PR.

## Real Codex measurement

`tools/wire-matrix/w4_measure.py --out <new-scratch-folder>` runs Codex against
an OS-assigned loopback gateway port and the wire-matrix S0 stub, under the
matrix's network sandbox. It captures counts at Responses ingress, before the
Anthropic pivot can drop declarations. It exports no request text or headers.
The report in `docs/w4-codex-measurement.json` records the successful run:

- Codex CLI 0.156.1, GPT-6 Astra, one completed turn, exit 0.
- 55,964 wire bytes; 56,515 bytes when JSON is serialized with ASCII escaping;
  a text-only size bound of 64,707 including 4096 framing and 4096 output tokens.
- No top-level `tools`; one `additional_tools` input item with three namespaces,
  containing ten function tools and one custom tool.
- The size bound fits Kimi K3's effective 200,000 window and Groq Llama 3.3,
  Gemini 2.5 Flash and DeepSeek Chat's effective 128,000 windows. This says
  nothing about billing authorization or tool compatibility.

**This observed Codex request cannot cascade.** Its declarations would be lost
by the existing pivot, so W4 refuses it as `unsupported_tool`. No approximate
custom-tool translation was added. The byte-bound number is illustrative;
`request_budget` itself refuses the unsupported item rather than certifying it.
This was a fresh isolated HOME without GRIP's preload: it does not establish that
a production GRIP conversation, a later turn or a long session fits any target.
Three earlier probes reached the ingress but ended in the stub's fake-credential
401; only the subsequent exit-0 run is the completed-turn measurement above.
