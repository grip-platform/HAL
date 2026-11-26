# GRIP-Guest

A **sandboxed conversational GRIP** a guest (e.g. an AI Craftspeople Guild member)
can talk to in a Discord channel. It answers in GRIP's voice but **cannot touch
the operator's substrate** — by construction, not by trust.

## Why it's safe (the core decision)

A read-only `claude` session cannot be sandboxed by its working directory:
`Read`/`Grep` take absolute paths and, running as the `grip` user, can read
`~/.claude` wherever you point it. So GRIP-Guest is given **no filesystem tools at
all** — it is a pure infer call. There is nothing to misuse, so there is nothing
to leak (operational context, memory, keychain names, client data). The guest
persona's hard-limits are a *second, independent* layer (defence-in-depth), never
the only one.

This is `rules/untrusted-content-capability-lock.md` at its strongest: untrusted
content gets the most-restricted surface — **talk, never touch**.

## Architecture

```
Discord guest message
  → grip-channel Discord bot (allowlist: guest tier)        [inbound — the switch]
  → lib.hal.grip_guest.handle(sender, text)
        ├─ RateLimiter.allow(sender)        per-sender sliding window (no flood/spend)
        ├─ guest_respond(text, infer)       grip_infer, NO tools, guest persona
        └─ discord_webhook_post(url, reply) no-bot channel webhook (#430), with <@id> mention
  → reply appears in the channel, mentioning the guest
```

Every external boundary (infer, post) is dependency-injected, so the whole path
is unit-tested with zero network and zero live messages (`tests/test_grip_guest.py`,
13 tests). Dogfood (real infer, faked post) confirmed: it answers a normal
question and **refuses** a substrate-probe, sending nothing.

## Safety gates (all must be ON for a live guest reply)

| Gate | Default | Purpose |
|------|---------|---------|
| `GRIP_GUEST_ENABLED=1` | **OFF** | The operator's conscious switch. Reactive-only; never initiates. |
| `HAL_CHANNEL_LIVE=1` | OFF | Webhook actually posts (shared with all live sends). |
| `HAL_DISCORD_WEBHOOK_URL` (or keychain `grip-discord-webhook`) | unset | Where the reply is posted. |
| `RateLimiter` | 8/60s, 2s min | Caps per-sender spend + channel flood. |
| No tools passed to `grip_infer` | — | The structural guarantee. Not configurable. |

## Activation (the one switch left for the operator)

The build + the HAL responder are done and tested. The single remaining step is
the **inbound cutover on the VPS** — routing a guest-tier Discord sender to the
responder. It is left OFF deliberately (don't expose a v1 guest bot to a chaotic
user unsupervised).

To go live:

1. On the VPS, in `~/.claude/.grip-channel.env` (or the hal env):
   ```
   HAL_DISCORD_WEBHOOK_URL=<the channel webhook URL>
   GRIP_GUEST_ENABLED=1
   HAL_CHANNEL_LIVE=1
   ```
2. Add the guest to the grip-channel allowlist at a **guest** tier (NOT operator,
   NOT team) — guest tier routes to `grip_guest.handle` instead of the full
   `claude -p` relay. Example entry: the guest's Discord id, e.g.
   `000000000000000000`.
3. Integration point (the one wiring line): for a guest-tier inbound, invoke
   ```
   echo "<guest text>" | python3 -m lib.hal.grip_guest --sender <discord-id>
   ```
   from the relay (text on STDIN, never argv). The CLI is inert unless
   `GRIP_GUEST_ENABLED=1`.

Reverse: unset `GRIP_GUEST_ENABLED` (or remove the guest from the allowlist).

## Limits (honest)

- **Outbound-only reach is a webhook**: the reply posts to the *channel* with a
  mention, not a private DM (the trade for dropping the bot). The guest sees it.
- **Conversational, not tool-using**: GRIP-Guest reasons + knows public GRIP/HAL/
  HAPPI; it cannot read curated docs (that would need a real jail — a separate
  VPS user/container — not a prompt).
- **No memory across messages** in v1 (each call is independent); add a per-sender
  short history later if wanted.
