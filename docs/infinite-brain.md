# Step 6, part 1: shared brain

The brain shares account cooldowns and keeps signed checkpoints from enrolled
hosts. It never serves inference. A separate host worker queues updates on disk;
when the VPS is down, it retains them and local sessions continue. Neither the
worker nor the server writes the host's account ledger. Shared observations are
returned through a separate cache for a future scheduler to consume.

## Service and enrolment

Run from HAL with the host's configured virtual-environment interpreter:

```sh
"$HAL_PYTHON" -m lib.hal.brain --port 7416 --bind 127.0.0.1 \
  --state /path/to/brain.sqlite --pins /path/to/public-pins.json
```

**7416** is the brain's port: the default in `lib/hal/brain.py` and in the deploy
renderer, and the row GRIP's port registry (`rules/port-registry.md`) holds for
the brain. Check that the port is free on the deployment host, or pass `--port`
to choose another. The brain exits at start on 4010, 4011 or 4012, which it
treats as reserved for other services, and the deploy renderer refuses those
three for the brain before it writes a unit. Tests bind port zero. The brain
never reclaims a port from another process.

For remote access bind the server to its own Tailscale address, set
`HAL_GATEWAY_TS_ACL_TAG` to the required ACL tag, and allow only enrolled nodes
through the tailnet ACL. `gateway_guard` authenticates the **socket peer** using
`tailscale whois --json` and `Node.Tags`; forwarding headers cannot assert a
local identity. Every GET and POST is gated. Unknown, untagged or unresolvable
remote peers receive 403. Loopback remains trusted. Do not put an unauthenticated
local reverse proxy in front of this trusted loopback boundary. Tailnet HTTP
uses Tailscale's encrypted transport; do not expose it on a public interface.
The optional existing gateway token/mTLS requirements remain enforced if set;
this client uses tag authentication and does not supply that optional token.

Pins are an operator-controlled JSON mapping of random public host aliases to
`{"ed25519": "<raw-public-key-hex>", "mldsa65": "<raw-public-key-hex>"}`.
The ML-DSA field may be omitted for an explicitly Ed25519-only enrolment. Export
public keys from the host's existing thread-store signing identity and provision
the pins out of band. A request cannot enrol itself. The service reuses
`infinite_thread.grip_verifier`; GRIP's signer module and its crypto dependencies
must be available. If a post-quantum key is pinned, inability to verify that limb
is a refusal, not a downgrade. Restart the brain after intentionally changing pins.

Endpoints:

- `GET /health`: `{"service":"hal-brain"}`.
- `POST /v1/cooldowns`: exactly `account`, `reset` (epoch seconds or null),
  `status`. The merge keeps the **maximum known reset** and strongest status:
  `allowed < allowed_warning < unknown < rejected`. Unknown resets remain null
  until a known reset arrives. Merge is commutative and idempotent; stale retries
  never relax it. This is historical shared evidence: consumers must compare the
  reset with current time, never treat an expired rejection as a permanent ban.
- `GET /v1/cooldowns`: merged entries under `accounts`.
- `POST /v1/heads`: `v: hal-brain-head/1`, `host`, `thread`, `seq`, `sha256`, `sig`.
  The signature covers canonical JSON of all fields except `sig`, using the
  existing thread-store signer. The version binds it to this purpose. Local
  `.head` files are unsigned summaries, so the worker signs an envelope around
  each summary. The brain verifies it before storage. Older heads are acknowledged
  without replacing newer ones; a different hash at the same sequence is refused.

SQLite transactions protect concurrent updates and survive restarts. The brain
stores no row bodies, prompts, credentials, token fingerprints or header values.
A verified head proves that the pinned host attested to that checkpoint; it does
not prove every earlier row is available or honest. Full thread replication and
verification remain separate work. Compromise of a host's signing key is outside
this guarantee; keep both pins and retained anchors outside that host's control.

## Host worker

```sh
"$HAL_PYTHON" -m lib.hal.brain_client --url http://brain-tailnet-alias:7416 \
  --queue /path/to/brain-queue.sqlite --ledger /path/to/account-ledger.json \
  --threads /path/to/infinite --host random-host-alias
```

This is a separate process; do not call `flush()` from a request handler.
`--once` performs one collection and sync attempt; otherwise it retries every
30 seconds. The worker reads `LedgerView` and `heads_snapshot`, signs envelopes
with **already provisioned** thread-store keys, then syncs at most 100 queued
updates per pass. Use non-identifying account and host aliases. Cooling (including
paid overage or headerless throttling) exports as rejected. No credential state
is exported. A local enqueue never makes a network call.
Collection disables the signer's first-use provisioning: missing private keys
fail collection without creating or replacing any local key material.

Transport ignores ambient HTTP proxies and refuses redirects. Each request has
a two-second timeout. An unreachable brain, invalid response or queue I/O failure
returns failure without raising into a local session. Durable entries survive
worker restart and are removed only after acknowledgement; delivery is at least
once, and server updates are idempotent. The separate shared cache changes only
after a validated response. A permanently rejected queued update remains queued
and blocks later delivery until the operator fixes enrolment or quarantines that
entry; it never blocks inference. Disk failure cannot guarantee queuing: callers
get `False`, and periodic collection can retry local state later.

## Timestamp job and independent verification

```sh
GRIP_TSA_URL=https://configured-tsa.example/timestamp \
  "$HAL_PYTHON" -m lib.hal.brain_anchor --state /path/to/brain.sqlite \
  --pins /path/to/public-pins.json --output /path/to/anchors \
  --tsa-ca /path/to/trusted-tsa-ca.pem
```

The job calls the existing `infinite_thread.grip_anchor` adapter, which loads
GRIP's `lib/rfc3161_anchor.py` from 2026-09-17. That implementation constructs
RFC 3161 DER requests with SHA-256 and `cert_req`, sending **only the digest**.
No authority is chosen by default. `GRIP_TSA_URL` enables anchoring; `asn1crypto`
and OpenSSL are required. Use a URL without embedded credentials.

The job re-verifies persisted heads with current pins, sorts them by host/thread,
and constructs a Merkle tree:

- leaf: SHA256(`0x00 || canonical signed envelope`);
- parent: SHA256(`0x01 || left digest || right digest`);
- an odd level duplicates its last node; the empty root is SHA256(empty bytes).

Canonical encoding is `infinite_thread.canonical` (sorted JSON keys, compact
separators, UTF-8, no floating-point values). Each unique output directory holds
`root.json` with the complete leaf set and root, and `timestamp.der` when a token
was received. Only an OpenSSL-verified signature, trust chain and matching message
imprint earns `status: anchored`. Missing configuration is `disabled`; TSA,
parsing or verification failure is `unavailable`, never success. Failed attempts
retain the snapshot and cannot overwrite successful anchors. The job exits 1
unless anchored, allowing a scheduler to report failure and retry next time.

Recompute `merkle_root(manifest["heads"])`, compare it to `manifest["root"]`,
check every head against independently retained host pins, then verify the token:

```sh
openssl ts -verify -digest "$MERKLE_ROOT_HEX" -token_in \
  -in timestamp.der -CAfile trusted-tsa-ca.pem
```

Retain the TSA certificate chain for later verification. The certificate must be
trusted independently, not simply accepted because it accompanied a token. A
timestamp proves the root existed by the TSA's time, not that the underlying
claims are true. Keep the anchor directories outside the writers' control.

## Service definitions

`deploy/hal-brain.service` is a systemd user template that
`lib.hal.deploy_render` fills. The brain is a Linux host service, so it has no
launchd or Task Scheduler definition and the renderer refuses those platforms.

```sh
"$HAL_PYTHON" -m lib.hal.deploy_render render --platform systemd --service brain \
  --python "$HAL_PYTHON" --hal-home /path/to/hal --log-dir /path/to/logs \
  --state-db /path/to/brain.sqlite --pins-file /path/to/public-pins.json \
  --bind <this-host-tailnet-ipv4> --tailscale-tag tag:example
```

Any caller, GRIP's installer included, must pass `--state-db` and `--pins-file`:
both are absolute paths with no safe default, and the renderer refuses to
render without them. `--bind` defaults to `127.0.0.1` and `--port` to 7416.
The bind address must be an IPv4 address on loopback or in the tailnet range
`100.64.0.0/10`. `0.0.0.0`, any IPv6 address (`::1` included, because the brain
listens on IPv4 only) and any hostname are refused. `--tailscale-tag` takes
Tailscale's `tag:name` form with its case kept, and is required for a tailnet
bind. It may be left out on loopback: the unit then sets
`HAL_GATEWAY_TS_ACL_TAG` to nothing, which the guard treats exactly as unset,
so loopback callers are served and every other peer gets 403. The renderer
writes the unit to stdout; it installs and starts nothing.

`hal-brain-anchor.service` and `hal-brain-anchor.timer` remain manual templates
whose placeholders must be filled explicitly. The timer schedules a checkpoint
every five minutes.
