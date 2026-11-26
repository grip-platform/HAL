# Fleet Gateway — Integration Guide

**Status**: scaffolded via #116. Core router + 7 transport adapters landed on `feat/fleet-command-p0`. Live transports are operator-provisioned post-merge.

## How the pieces fit

```
transport (Telegram, Slack, ...)
            │
            ▼
  ChannelAdapter subclass  ──────── (lib/hal/channel_adapter.py)
            │
            ▼
  ChannelRouter.deliver()  ──────── (lib/hal/channels.py)
            │
            ▼
      handler callback
            │
            ▼
  resolve canonical user   ──────── ExternalIdentityResolver
            │                       (lib/hal/identity_provider.py)
            ▼
  append to thread state   ──────── GatewayThreadStore
            │                       (lib/hal/gateway_threads.py)
            ▼
  agent_runner inference   ──────── (lib/hal/agent_runner.py)
            │
            ▼
  envelope.reply(body)  →  router.send()  →  adapter.send()
```

One inference loop. Many transports. Shared identity. Shared thread state. The runner is transport-agnostic — it works on `MessageEnvelope` objects and never sees HTTP payloads or platform-specific IDs.

## Minimum viable wiring

```python
from lib.hal.channels import ChannelRouter, MessageEnvelope
from lib.hal.identity_provider import ExternalIdentityResolver
from lib.hal.gateway_threads import GatewayThreadStore
from apps.gateway.telegram.adapter import TelegramAdapter
from apps.gateway.slack.adapter import SlackAdapter

router = ChannelRouter()
resolver = ExternalIdentityResolver()
threads = GatewayThreadStore()

# Operator binds platform IDs to canonical HAL users (once, at config time)
resolver.bind("telegram", "123456", "M>>")
resolver.bind("slack",    "U076LP", "M>>")

def on_message(envelope: MessageEnvelope):
    canonical = resolver.resolve(envelope.transport, envelope.external_id)
    if canonical is None:
        return  # unknown sender — drop or enqueue for human review

    threads.append(canonical, envelope)

    # Hand off to HAL's agent runner (see agent_runner_hook below)
    reply_body = handle_with_agent(envelope, threads.messages(canonical))

    router.send(envelope.reply(reply_body))

router.set_handler(on_message)

# Adapters need tokens/credentials injected — never read from environment
telegram = TelegramAdapter(router, token=load_token("telegram"))
slack    = SlackAdapter(router, token=load_token("slack"))

router.register("telegram", telegram)
router.register("slack",    slack)

telegram.start()   # long-polls Telegram getUpdates in background thread
slack.start()      # no-op in this wave (Events API recv is a follow-up)
```

## Milestone status

| Milestone | Wave | Status | What works | What's operator work |
|-----------|------|--------|------------|----------------------|
| M1 Router + Telegram | W1–W3 | shipped | unit tests green; Telegram send via injected HTTP transport | live bot token provisioning; register adapter against real long-poll |
| M2 Identity + session keying | W3–W4 | shipped | cross-transport resolver (`ExternalIdentityResolver`) + per-canonical thread store (`GatewayThreadStore`) | populate `facts/external_bindings.json` with real user IDs |
| M3 Discord + Slack + Email | W4–W5 | shipped (send path) | send via stdlib HTTP / SMTP | Discord Gateway WebSocket recv; Slack Events API recv; IMAP recv |
| M4 WhatsApp + Signal | W5 | stub | class surface; routing; capability declaration | provision Twilio client; install signal-cli; wire live send + recv |
| M5 Voice STT + fleet wiring | W5–W6 | stub | inbound-STT adapter surface; fleet-aware hooks documented | choose whisper-cpp or external service; compile / install; implement `_transcribe()` |

## Adding a new transport

1. Create `apps/gateway/<transport>/adapter.py`.
2. Subclass `ChannelAdapter`. Set `transport` class attribute.
3. Accept the router and all configuration (tokens, hosts, paths) via `__init__` — never `os.environ`.
4. Implement `send(envelope)` using stdlib where possible (urllib, smtplib, imaplib) or a library injected at construction.
5. Implement `capabilities()` returning a frozenset of `CAPABILITY_*` strings.
6. If inbound is needed, override `start()` to spawn a recv thread that normalises platform payloads into `MessageEnvelope` and calls `self.deliver(envelope)`. Override `stop()` to tear it down.
7. Register it: `router.register(adapter.transport, adapter)`.

No changes to `agent_runner.py` required — this is the falsification check for claim H058 (`adding a new transport requires zero changes to agent_runner.py`).

## Agent runner hook pattern

Rather than modifying `agent_runner.py`, callers compose a handler that bridges envelopes to the existing runner. The runner stays transport-agnostic:

```python
from lib.hal.agent_runner import AgentRunner

runner = AgentRunner()

def handle_with_agent(envelope: MessageEnvelope, history: tuple) -> str:
    agent_spec = runner.parse_agent("broly")
    prompt = f"{agent_spec.prompt}\n\nUser: {envelope.body}"
    # ... call out to the runner's execution backend (CC Task or standalone query)
    return "reply body"
```

The envelope's `external_id`, `thread_id`, `meta` carry platform nuance to the agent without the runner caring which transport produced it.

## Falsification criteria (from design doc §6)

Any of the following refute this scaffold and trigger a rescope via `AskUserQuestion`:

1. Two transports cannot share a single canonical user without per-user manual mapping — refuted by the resolver working for any (platform, external_id) pair.
2. Router cannot maintain conversation continuity when a user switches mid-thread from Telegram to Slack — addressed by keying threads on `canonical_user`, not `transport + thread_id`.
3. Adding a new transport requires changes to `agent_runner.py` — the envelope is the only coupling; adapters never touch the runner.
4. Voice STT latency pushes round-trip beyond conversational thresholds (>5s) — open; VoiceAdapter stubs both in-process and external STT paths.
5. Fleet aggregator cannot account for gateway traffic alongside CLI traffic — handled by registering the router as a fleet participant (follow-up wiring in `fleet.py`).

## Productionisation checklist

Before this programme is live end-to-end:

- [ ] Populate `facts/external_bindings.json` with real user IDs per transport.
- [ ] Provision bot tokens for Telegram, Discord, Slack (stored outside the repo).
- [ ] Wire real long-polling / Events API recv for Discord + Slack.
- [ ] Install `signal-cli` and start the daemon socket.
- [ ] Provision a Twilio WhatsApp business sender.
- [ ] Choose a voice STT backend (whisper-cpp in-process or external HTTP service) and implement `VoiceAdapter.start()`.
- [ ] Register the router as a fleet participant in `lib/hal/fleet.py` so `FleetAggregator` sees gateway traffic.
- [ ] Operator manually closes the five falsification criteria above against the live deployment.
- [ ] Add `hal home`-registered application entry for the gateway process.

## Cross-links

- Design spec: `docs/FLEET-GATEWAY-DESIGN.md`
- Plan file: `plans/hal-116-fleet-gateway-ultrado.md`
- Source audit: `/tmp/hermes-agent-analysis.md` section 4 pattern 2
