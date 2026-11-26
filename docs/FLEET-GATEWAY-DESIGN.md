# Fleet Command Multi-Platform Gateway (W4b absorb)

Status: DESIGN SPEC (no implementation this wave)
Grade: A (Pattern 2 from W4b research-council audit)
Hypothesis: H051 (design spec clarity)
Parent plan: `~/.claude/plans/ultrado-apr-15-multi-wave.md`
Source audit: `/tmp/hermes-agent-analysis.md` section 4, Pattern 2

## 1. Motivation

The W4b research-council audit of NousResearch/hermes-agent identified a
first-class multi-platform messaging gateway as the highest-verified novel
pattern (Grade A, survived falsification claim C3). See
`/tmp/hermes-agent-analysis.md` section 4, Pattern 2:

> "One gateway process routes a single agent across Telegram, Discord,
> Slack, WhatsApp, Signal, Email. Voice memo transcription on the inbound
> path. Conversation continuity preserved across platforms. HAL has
> `grip-channel` (GRIP-only) and the HAL apps layer, but multi-channel
> routing is ad-hoc today."

HAL currently treats each transport as a separate app. There is no
shared identity layer, no shared thread state, no routing fabric. The
HAL Fleet Command project (see `memory/project_hal_fleet_command.md`)
explicitly needs this: a provider-agnostic Managed Agents competitor
cannot ship with CLI-only reach.

## 2. Target Architecture

```
                    +--------------------------+
                    |   HAL Fleet Gateway      |
                    |   (single process)       |
                    |                          |
   Inbound -->      |  +--------------------+  |    --> Outbound
                    |  | transport adapters |  |
  [Telegram]  ----->|  | tg|dc|sl|wa|sg|em  |<-|---->  (same adapter
  [Discord]   ----->|  +---------+----------+  |        set used in
  [Slack]     ----->|            |             |        reverse)
  [WhatsApp]  ----->|            v             |
  [Signal]    ----->|  +--------------------+  |
  [Email]     ----->|  | Channel Router     |  |
  [Voice STT] ----->|  | (normalise envlp)  |  |
                    |  +---------+----------+  |
                    |            |             |
                    |            v             |
                    |  +--------------------+  |
                    |  | Identity Resolver  |<----  facts/identity.md
                    |  | (external_id -->   |  |    + identity_provider.py
                    |  |  canonical user)   |  |
                    |  +---------+----------+  |
                    |            |             |
                    |            v             |
                    |  +--------------------+  |
                    |  | Thread State Store |<----  session_manager.py
                    |  | (per canonical     |  |    + session_registry.py
                    |  |  user, cross       |  |
                    |  |  transport)        |  |
                    |  +---------+----------+  |
                    |            |             |
                    |            v             |
                    |  +--------------------+  |
                    |  | HAL Agent Core     |  |
                    |  | (agent_runner.py)  |  |
                    |  +--------------------+  |
                    +--------------------------+
```

One inference loop. Many transports. Shared identity. Shared thread state.

## 3. HAL Integration Points (5)

All paths verified to exist in `~/.hal`.

1. **`lib/hal/agent_runner.py`** — `AgentRunner` class is the synchronous
   loop that the gateway must feed. Gateway delivers normalised messages
   to this runner and streams outputs back. No change to runner shape,
   only new entry points.
2. **`lib/hal/session_manager.py` + `lib/hal/session_registry.py`** —
   `SessionManager` and session registry become the cross-transport
   thread store. Canonical session keys must stop assuming a single TUI
   caller.
3. **`lib/hal/identity_provider.py`** — `get_identity()` and the
   `Identity` dataclass must be extended with a resolver that maps
   external transport IDs (Telegram user_id, Slack user, WhatsApp MSISDN,
   email address) to a single canonical HAL user.
4. **`lib/hal/fleet.py`** — `FleetManager` already exists. The gateway
   router registers itself as a fleet participant so multi-agent and
   multi-transport share one aggregator (`FleetAggregator`).
5. **`apps/`** — new directory `apps/gateway/` with one subdirectory per
   adapter (`telegram/`, `discord/`, `slack/`, `whatsapp/`, `signal/`,
   `email/`, `voice/`). Each adapter implements a small interface
   (`recv`, `send`, `capabilities`). Existing apps (`copilot-api`,
   `grip-mobile`, `grip-web`, `mirror`) remain as-is; the gateway is
   additive, not a replacement.

A sixth integration point, `lib/hal/channels.py` (NEW), is the only
net-new module this wave would introduce. It is the thin router that
sits between adapters and the runner.

## 4. SMT Mapping Summary

From `/tmp/hermes-agent-analysis.md` section 2 (Structure Mapping Theory
table):

| hermes-agent concept | HAL concept | Mapping |
|---|---|---|
| Multi-platform gateway (Telegram/Discord/Slack/WhatsApp/Signal/Email) | HAL apps layer + grip-channel MCP | **partial** |
| `AIAgent` synchronous loop | HAL orchestration loop (`agent_runner.py`) | **deep** |
| Honcho dialectic user modelling | HAL facts/identity.md + USER.md | **surface** (rejected) |

The deep mapping on the loop means no restructuring is required. The
partial mapping on the gateway layer is exactly the gap this spec
closes. The surface mapping on Honcho is explicitly rejected: HAL
sticks with static identity facts.

## 5. Effort Estimate (5 sub-milestones)

| # | Milestone | Effort | Gate |
|---|---|---|---|
| M1 | `lib/hal/channels.py` router skeleton + adapter interface + Telegram adapter as reference | S (3 days) | Unit tests for router, one live Telegram round-trip |
| M2 | Identity resolver extension in `identity_provider.py` + cross-transport session keying in `session_manager.py` | M (5 days) | Integration test: same canonical user from Telegram + Slack hit same thread |
| M3 | Discord + Slack + Email adapters | M (5 days) | One live message per transport |
| M4 | WhatsApp + Signal adapters (harder, external service deps) | L (8 days) | At least one adapter live end-to-end |
| M5 | Voice STT inbound path + fleet.py integration + gateway as `hal home` registered app | M (5 days) | Voice memo -> text -> runner -> reply to source transport |

Total: ~26 engineering days. Scoped as a HAL Fleet Command design
programme, not a sprint. Must be phased to avoid breaking existing apps.

## 6. Falsification Criteria

This design is refuted if any of the following hold:
1. Two transports cannot share a single canonical user without per-user
   manual mapping (identity resolver is insufficient).
2. The router cannot maintain conversation continuity when the same user
   switches mid-thread from Telegram to Slack (state key design failure).
3. Adding a new transport requires changes to `agent_runner.py` (runner
   leakage — the adapter interface is too thin).
4. Voice STT latency pushes round-trip beyond conversational thresholds
   (>5s) and cannot be mitigated by streaming.
5. Fleet aggregator cannot account for gateway traffic alongside CLI
   traffic in the same snapshot (fleet integration failure).

Any single failure above triggers a re-scope AskUserQuestion before M2
begins.

## 7. Open Questions (3)

1. **Voice pipeline ownership.** Inbound STT: ship in-process (whisper.cpp)
   or delegate to an external service? Affects M5 scope materially.
2. **WhatsApp / Signal legal posture.** Both require business-account
   registration or bridge services (Matrix, Twilio). Who owns the
   upstream account for HAL?
3. **grip-channel MCP overlap.** Does the new gateway replace
   grip-channel, coexist, or fold grip-channel into adapter #0?
   Decision needed before M1.

## 8. Hypothesis Pointer

H051 (`design_spec_clarity_pass`): this document is sufficient for a
follow-up implementation wave to begin M1 without further operator
clarification. A fresh reader must be able to identify the 5 integration
points and 3 open questions in under 10 minutes. Prediction: 1.
Deadline: 2026-04-20. Registered via `lib/hypothesis_engine.py`.

## 9. Non-Goals

- No code changes to `agent_runner.py`, `session_manager.py`, or any
  existing HAL module in this wave.
- No `execute_code` programmatic tool calling (rejected by SMT as
  anti-mapping against converge).
- No cron scheduler, Honcho dialectic modelling, or Atropos RL
  trajectory capture — all rejected as surface mappings.

## 10. Cross-Links

- W4b audit: `/tmp/hermes-agent-analysis.md`
- Companion design note: `docs/SELF-NUDGING-SKILL-AUTHORING.md`
- HAL Fleet Command project memory: `memory/project_hal_fleet_command.md`
- Parent ultrado plan: `~/.claude/plans/ultrado-apr-15-multi-wave.md`
