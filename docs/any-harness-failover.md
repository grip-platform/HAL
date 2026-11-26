# Any-Harness Failover — Runbook

**Situation:** Anthropic is throttling your harness (Claude Code, Codex, Aider,
opencode, a relay `claude -p`, etc.). You see HTTP 429 or 529 ("Server is
temporarily limiting requests / not your usage limit"). Per
`rules/rate-limit-never-blocks.md`, this is a retry/failover signal, not a halt.

**Fix:** point the harness at the HAL gateway (`localhost:4010`). It absorbs the
429 and retries the next configured upstream — the harness sees a clean response.

---

## One-time setup (start the gateway)

```bash
# Start the HAL gateway (stays running; re-start after reboot)
hal gateway start          # or: python3 ~/.hal/lib/hal/gateway.py
```

Confirm it is up:

```bash
curl -s http://127.0.0.1:4010/v1/models | python3 -m json.tool | head -5
# Should return a JSON object with a "data" list of Anthropic models
```

---

## Point your harness at the gateway

Set ONE environment variable before launching the harness. The harness keeps
using its own credentials; the gateway forwards them verbatim.

| Harness | Variable to set | Value |
|---------|----------------|-------|
| **Claude Code** (`claude`) | `ANTHROPIC_BASE_URL` | `http://127.0.0.1:4010` |
| **Codex** | `ANTHROPIC_BASE_URL` | `http://127.0.0.1:4010` |
| **Aider** | `ANTHROPIC_BASE_URL` | `http://127.0.0.1:4010` |
| **opencode** | `ANTHROPIC_BASE_URL` | `http://127.0.0.1:4010` |
| **OpenAI-wire harness** (Codex Chat, etc.) | `OPENAI_BASE_URL` | `http://127.0.0.1:4010` |
| **relay `claude -p`** (VPS relay-dispatcher) | `ANTHROPIC_BASE_URL` | `http://127.0.0.1:4010` |

```bash
# Example — Claude Code
export ANTHROPIC_BASE_URL=http://127.0.0.1:4010
claude

# Example — Codex
export OPENAI_BASE_URL=http://127.0.0.1:4010
codex
```

The gateway is Anthropic-wire (`/v1/messages`) **and** OpenAI-wire
(`/v1/chat/completions`, `/v1/responses`, `GET /v1/models`) — both harness
types work with the same port.

---

## How the failover works

```
harness  →  POST /v1/messages  →  HAL gateway :4010
                                        │
                                  primary upstream
                                  (api.anthropic.com)
                                        │
                                   HTTP 429 / 529?
                                        │
                               yes → close response
                                        │
                                  fallback upstream
                                  (second credential or
                                   alternate base_url)
                                        │
                                   return to harness
```

Failover triggers on: `429, 500, 502, 503, 504, 529`.
The harness sees **one** response — the failover is invisible.
The gateway relays the body **verbatim** (byte-identical), so Anthropic
`thinking`/`redacted_thinking` block signatures are preserved across retries
(the fix for GRIP issue #3274).

---

## Configure a fallback credential

By default the gateway has one upstream (the harness's own credentials forwarded
through). To add a second Anthropic key as the fallback (e.g. a metered API key
when the Max subscription is throttled):

```bash
export HAL_GATEWAY_UPSTREAM=https://api.anthropic.com          # default; set explicitly if needed
export HAL_GATEWAY_FALLBACK_UPSTREAM=https://api.anthropic.com # same endpoint, different key
export HAL_GATEWAY_FALLBACK_KEY=sk-ant-...                     # the fallback API key

hal gateway start   # picks up the env vars at startup
```

The fallback key replaces the client's credential on every request sent to the
fallback upstream — so the harness never needs to know about the second
credential. Both of the client's credential headers (`Authorization` and
`x-api-key`) are removed and the fallback key is sent once, as `Authorization:
Bearer`; the gateway turns an API key into `x-api-key` on its way out, so exactly
one credential reaches Anthropic. (Until 2026-10-05 the key was injected as
`x-api-key` and replaced only a client `x-api-key`: a client signed in with an
OAuth token reached the fallback carrying its own bearer as well.)

---

## Read limits.json — what is cooled down?

After a 429 the sentinel records a cooldown so the flagged provider is skipped
until it recovers:

```bash
cat ~/.hal/state/hal/limits.json | python3 -m json.tool
```

Key fields per provider:

| Field | Meaning |
|-------|---------|
| `cooldown_until` | Unix timestamp; provider is skipped until `now > cooldown_until` |
| `consecutive_429s` | Backoff exponent: cooldown = 60 × 2^(n−1), capped at 3600 s |
| `status` | `"allowed"` or `"rejected"` (hard limit / 402 payment required) |
| `utilisation` | 0–1 from Anthropic's `anthropic-ratelimit-unified-*-utilization` header |

To manually reset a provider (e.g. after topping up credits):

```bash
python3 - <<'EOF'
import sys; sys.path.insert(0, "~/.hal")
from lib.hal.sentinel import LimitSentinel
from lib.hal.registry import ProviderRegistry
reg = ProviderRegistry.from_config()
s = LimitSentinel(reg)
s.reset("groq")   # replace with the provider name to reset
print("reset ok")
EOF
```

---

## Audit log

Every gateway request (upstream, status, latency, whether a failover fired) is
appended to:

```
~/.hal/logs/gateway-audit.jsonl
```

To see recent failovers:

```bash
python3 -c "
import json, sys
for line in open('$HOME/.hal/logs/gateway-audit.jsonl'):
    e = json.loads(line)
    if e.get('failed_over'):
        print(e['ts'], e['upstream'], e['status'], e.get('note',''))
" 2>/dev/null | tail -20
```

---

## Scope of each path

| Path | What the gateway does |
|------|-----------------------|
| `POST /v1/messages` | Verbatim Anthropic relay + same-format failover (P1+P3) |
| `POST /v1/chat/completions` | OpenAI→Anthropic translation + relay (P2) |
| `POST /v1/responses` | OpenAI Responses API→Anthropic translation + relay |
| `GET /v1/models` | Relayed + `models` array injected for Codex 0.136 compat |
| `HAL_GATEWAY_MULTIPROVIDER=1` | Enables cross-provider routing (model-name-based): non-Anthropic model names route to OpenAI-compatible provider endpoints (Groq, Gemini, xAI, DeepSeek, …) |

Cross-provider ladder (`HAL_GATEWAY_MULTIPROVIDER`) is **off by default** — the
`/v1/messages` path stays byte-identical Anthropic until you enable it.

---

## Falsification

This runbook is wrong if:
- A harness pointing at `localhost:4010` still sees 429 when a fallback is configured
  → check `gateway-audit.jsonl` for `"note":"failover"` events; if absent, the
  fallback env vars were not picked up at gateway start.
- `HAL_GATEWAY_FALLBACK_KEY` is set but the fallback ignores it → verify with
  `jq '.failed_over' ~/.hal/logs/gateway-audit.jsonl` — a `true` entry means it
  fired; `false` means the primary served.
