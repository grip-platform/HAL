# Harness in a harness: picking a non-Anthropic model from Claude Code's own `/model` picker

Point Claude Code at the HAL gateway and it will talk to any model HAL routes —
OpenAI's `gpt-6-astra`, Moonshot's `kimi-k3`, anything in `config/providers.yaml`.
This page is the working procedure for getting those models to appear in Claude
Code's interactive `/model` list, and what was measured while establishing it.

Everything below was executed on 2026-09-10 against a private gateway on
127.0.0.1:49841, run from a worktree on `origin/main`, with Claude Code 2.1.267.
Nothing here is inferred from the code alone.

## Two mechanisms, not one

Selecting a model and routing to one are separate, and only the first was broken:

- **Routing already worked.** `claude -p --model openai/gpt-6-astra` answered, and
  the gateway's own audit row recorded `provider: openai, status: 200, 7218 ms`.
  The same for `--model moonshot/kimi-k3`: `provider: moonshot, status: 200,
  12253 ms`. A model id typed or pinned on the command line reaches its provider
  whether or not the picker has ever heard of it.
- **The picker did not.** With discovery enabled, Claude Code fetched
  `GET /v1/models?limit=1000`, received 68 ids from the gateway, and wrote exactly
  **one** of them into its own picker cache. Neither astra nor kimi was in it.

The reason is a documented filter. Anthropic's
[gateway protocol reference](https://code.claude.com/docs/en/llm-gateway-protocol)
states it plainly:

> Claude Code keeps an entry when its `id` contains `claude` or `anthropic`
> anywhere in the string, matched case-insensitively, and ignores the rest.
> Provider-prefixed IDs such as `vertex_ai/claude-sonnet-4-6` or
> `bedrock/anthropic.claude-sonnet-4-5` pass the filter; an ID that contains
> neither substring doesn't.

HAL addresses its models `<provider>/<model>`, so 67 of its 68 ids contain
neither substring. The one survivor was `soruxgpt/claude-3-5-sonnet-20241022`.

## The rest of the contract, as documented

- Discovery is **off by default**. Turn it on with
  `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`.
- It does not run at all if any `CLAUDE_CODE_USE_*` provider variable is set, or
  if `ANTHROPIC_BASE_URL` is unset or points at `api.anthropic.com`.
- The request is `GET /v1/models?limit=1000` with a **3-second timeout**, and
  **any redirect is treated as failure** — including `http` to `https`. Serve the
  endpoint directly at the configured base URL.
- Claude Code needs a credential to send or it skips discovery entirely. Any
  non-empty `ANTHROPIC_AUTH_TOKEN` satisfies this; the gateway resolves its own
  upstream keys.
- Results cache to `$CLAUDE_CONFIG_DIR/cache/gateway-models.json` (default
  `~/.claude/cache/`) and refresh on each startup. **That file is the picker's
  own data, which makes it the honest way to check what the picker will show
  without driving the interactive UI.**

## Turning it on

```bash
# 1. Gateway: serve discovery-visible aliases for models the filter would drop.
export HAL_GATEWAY_PICKER_ALIASES=1

# 2. Claude Code: point it at the gateway and enable discovery.
export ANTHROPIC_BASE_URL=http://127.0.0.1:4010
export ANTHROPIC_AUTH_TOKEN=hal-local        # any non-empty value
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
```

Start a session and run `/model`. Astra and kimi now appear as rows named
`gpt-6-astra (openai)` and `kimi-k3 (moonshot)`.

`HAL_GATEWAY_PICKER_ALIASES` is **off by default** because turning it on roughly
doubles the length of `/v1/models` (68 ids became 135 in the measured run), and
other harnesses read that same list — Codex consumes the injected `models` array
off the same route.

## What the alias is, and why it is not a trick

An alias is the canonical id with `anthropic-wire/` in front:
`anthropic-wire/openai/gpt-6-astra`. The prefix contains `anthropic`, so Claude
Code keeps the entry.

The substring the filter looks for is a proxy for "this model speaks the
Anthropic Messages wire". For this gateway the proxy is wrong rather than
protective: HAL's cross-wire translates every one of these models onto the
Anthropic Messages format, which is the only reason Claude Code can talk to them
at all. The alias supplies the fact the substring stands in for, and says so in
the id. Anthropic's own documentation points the same direction for ids that miss
the filter — "developers can add those aliases manually with the model
configuration variables" — this just moves that work to the gateway, which
already knows the answer.

The row a human reads stays honest. The picker shows `display_name`, which names
the real provider and the real model and never says Claude or Anthropic. The
gateway resolves the alias back to the canonical id before anything routes, so
the provider receives the model it actually serves.

## Verifying it yourself

```bash
# What the gateway serves, and how much of it the picker would keep.
curl -s "http://127.0.0.1:4010/v1/models?limit=1000" \
  | python3 -c "import json,sys; ids=[m['id'] for m in json.load(sys.stdin)['data']]; \
      keep=[i for i in ids if 'claude' in i.lower() or 'anthropic' in i.lower()]; \
      print(len(ids),'served,',len(keep),'reach the picker')"

# What Claude Code itself decided to put in the picker.
cat "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/cache/gateway-models.json"

# That a picked model reaches its real provider.
claude -p 'Reply with the single word PONG' --model anthropic-wire/openai/gpt-6-astra
tail -1 logs/gateway-audit.jsonl        # provider must be openai, not anthropic
```

Measured before and after, same gateway, same session shape:

| | ids served | reach the picker | astra selectable | kimi selectable |
|---|---|---|---|---|
| aliases off | 68 | 1 | no | no |
| aliases on | 135 | 68 | yes | yes |

Both aliases were then run end to end. `anthropic-wire/openai/gpt-6-astra`
answered PONG with an audit row of `provider: openai, status: 200, 4492 ms`;
`anthropic-wire/moonshot/kimi-k3` answered PONG with `provider: moonshot,
status: 200, 18561 ms`.

That last check is not ceremony. The first live run of the alias produced a row
reading `provider: anthropic` — the audit rail re-reads the caller's body, which
still carries the alias, so it labelled an OpenAI-served request as Anthropic.
The routing was right and the witness was lying. It is fixed and anchored by
tests, but the lesson stands: read the audit row, do not assume it.

## Running a tool, not just answering

PONG proves the model answered. It does not prove Claude Code can *work* on a
model served this way, because the thing Claude Code does that a plain chat
client does not is call tools. That was run separately, with a nonce the model
could not have known:

```bash
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(8))')
echo "$NONCE" > /tmp/nonce.txt

ANTHROPIC_BASE_URL=http://127.0.0.1:<your-port> \
ANTHROPIC_API_KEY=gateway-doorstop \
CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 \
HAL_GATEWAY_PICKER_ALIASES=1 \
claude -p 'Read the file /tmp/nonce.txt using the Read tool and reply with only its contents.' \
  --model 'anthropic-wire/openai/gpt-6-astra' --allowedTools Read
```

It returned the nonce, in 17 seconds, exit 0. The value was sixteen random hex
characters generated moments earlier, so it could only have come from the tool
result — the loop closed rather than merely started. The audit showed five
`/v1/messages` rows for the one command, all `provider: openai, model:
openai/gpt-6-astra, status: 200`. One row would have meant the model answered
without calling anything and the run had proved nothing.

## All three ingress rails resolve the alias

An alias has to be turned back into the id that routes, and the gateway has
three doors a request can come in through. Two of them handled it from the
start. The third did not, and the failure was silent: `POST /v1/responses` read
the model straight out of the body, did not recognise the aliased id, found no
provider for it, and fell through to the plain Anthropic relay. A request that
named an OpenAI model was served by the wrong vendor with nothing reporting an
error.

It was measured with one key held constant across four calls, so the door is the
only thing that varies:

| rail | id | result |
|---|---|---|
| `/v1/messages` | `openai/gpt-6-astra` | 200 |
| `/v1/messages` | `anthropic-wire/openai/gpt-6-astra` | 200 |
| `/v1/responses` | `openai/gpt-6-astra` | 200 |
| `/v1/responses` | `anthropic-wire/openai/gpt-6-astra` | **401** |

This is reachable rather than theoretical, because the alias rows are mirrored
into the Codex-compatible `models` array that the same route serves — all 67 of
them — and Codex posts to `/v1/responses`. It is fixed, and the same call now
returns 200 with `model: gpt-6-astra`. The general lesson is the one the audit
row taught earlier in this document: a fix applied to the rail you were looking
at is not a fix applied to the others.

## The `[1m]` suffix belongs on your command line, not in the alias

Claude Code reads a trailing `[1m]` as "this model has a million-token window"
and strips it before the request goes anywhere. Across every run recorded here,
no audit row has ever contained a bracket — the gateway only ever sees
`openai/gpt-6-astra`. So the suffix is for the client, and the alias rows served
by `/v1/models` correctly do not carry it.

You do need it. That was left unresolved here because two earlier runs
disagreed, so it was isolated: three runs back to back against one gateway
process on one port, same prompt, same tool, same client, with the suffix as
the only thing that changed.

| # | model id | outcome |
|---|---|---|
| 1 | `anthropic-wire/openai/gpt-6-astra[1m]` | exit 0, nonce returned, 26.6s, **3** audit rows |
| 2 | `anthropic-wire/openai/gpt-6-astra` | **hung**, killed at 200s, **1** audit row |
| 3 | `anthropic-wire/openai/gpt-6-astra[1m]` (control) | exit 0, nonce returned, 51.6s |

Run 3 is the control, and it is the reason this reads as a finding rather than
a flake: the bare id failed and the suffixed id worked again straight
afterwards on the same process, so nothing had degraded between them.

The failure is not the one previously recorded, which matters more than the
verdict. The bare id did **not** refuse client-side with "Prompt is too long"
and send nothing. It sent the first turn, got a clean 200 back from OpenAI, and
then stalled — no second request, no error, no exit, until it was killed. From
the outside that looks like a hang rather than a rejected model, so anyone
debugging it will go looking at the gateway, which is working perfectly.

The likely mechanism is that an unrecognised id gets a context budget too small
to hold the transcript once a tool result is appended, so the loop cannot take
its second turn. That is inference from the row count, not measurement — what
is measured is one request then silence.

So: **append `[1m]` whenever you pin one of these ids by hand.** Claude Code
strips it before the wire, the gateway never sees it, and without it a tool loop
does not finish.

## Known limits

- The picker rows come from `config/providers.yaml`, so an id HAL does not
  declare will not appear no matter what discovery is set to.
- Discovery is refreshed only at startup. A model added to the gateway
  mid-session shows up in the next session, not this one.
- The interactive `/model` list itself was not driven in a TUI during this work.
  What was verified is Claude Code's own cache file, which is the data the picker
  is built from, plus end-to-end routing of the ids in it.
- A client that sends the window suffix itself puts it in the audit row. Claude
  Code strips `[1m]` before the wire, so its rows are clean, but a direct caller
  — curl, a script, Codex — that posts `openai/gpt-6-astra[1m]` gets a witness
  row recording exactly that string. Measured 2026-09-10: the same model then
  appears under two names in the audit, which would split any per-model total
  built on it, cost attribution included. Routing is unaffected; the request
  reaches OpenAI and returns 200. Left alone here deliberately, because the fix
  belongs in the witness rail rather than in the alias resolver.
- ~~Kimi has been run answering a prompt, not running a tool.~~ Closed
  2026-09-10: `anthropic-wire/moonshot/kimi-k3[1m]` completed the same nonce
  tool run, exit 0 in 38.4s, two audit rows both `provider: moonshot, model:
  moonshot/kimi-k3, status: 200`. This is worth more than a second green tick,
  because kimi reaches its provider by a different route inside the gateway and
  shares none of the cross-wire code astra uses — so the alias mechanism is now
  measured on both wires rather than on one. Each acceptance still only speaks
  for its own path.
- `HAL_GATEWAY_PICKER_ALIASES` has to be exported by whoever launches the
  gateway. Wiring it into GRIP's launcher (`bin/grip`) is a separate change in a
  separate repository and has not been done.

## Provenance

V>> (Lourens Cornelius "Laurie" Scheepers), Claude Fable 5.1, and
Claude Opus 5 (`claude-opus-5[1m]`, the three-rail and tool-run sections, and
the isolated `[1m]` measurement).
