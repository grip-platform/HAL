#!/usr/bin/env bash
: <<'HAPPI_DOC'

<!-- happi:label=frontmatter -->
# happi.md — HAPPI/1.6

> *"AI is a syscall. happi.md is the protocol."* — V>>--<<V

**Version**: `happi/1.6` — current runtime. `happi/1.0` through `happi/1.6` envelopes are all accepted; emitted events carry the runtime's version (`happi/1.6`), not the envelope's. v1.1 added the `idr` event type, the `flags.audit=true` opt-in, three runtime cmds (`pr.reference`, `hypothesis.register`, `quine.spawn`), the `context` event type, and the `context.append` cmd — a replayable signed memory-chain link content-addressing a decision body (twin of `idr`). v1.2 is the **memory-chain capstone**: it ratifies the v1.1 `context` event + `context.append` cmd as the stable surface for signed memory chains (GRIP context-chain P0–P6). v1.3 adds the **deterministic citation-provenance floor**: the `cite.verify` cmd checks, by exact string-match, that every cited quote is verbatim-present in its source — a fabricated citation can never verify. No new event types; full back-compat (accepts `1.0`–`1.3`). v1.4 adds the **`continuity` event type** + the `continuity.append` cmd: a pre-registered falsifiable commitment — the premise an agent is about to act on, with what would refute it, emitted BEFORE the dispatch it governs. Full back-compat (accepts `1.0`–`1.4`). v1.5 adds the **`reasoning` event type** + the `reasoning.append` cmd: a reasoning trace carried as a DIAGNOSTIC SPECIMEN — stamped `authoritative: false` and `role: "diagnostic"`, with the runtime refusing before emission any record that claims authority or tries to settle a commitment. Full back-compat (accepts `1.0`–`1.5`). v1.6 adds the **`decide` cmd**, library-only: served by the reference library runtime rather than this embedded one, it is a typed-decision call to a decision model (never a chat model) -- one `noul`/`choice`/`score` answer per question asked, never free text. No new event type. Full back-compat (accepts `1.0`–`1.6`).

**Changelog**:
- `happi/1.6` -- typed decisions, library-only. **This embedded runtime's own change is narrow: it widens version acceptance to `happi/1.6` and implements no cmd of its own** (it also splits `cmd_compose` into two helpers, `_compose_guard` and `_compose_dispatch_one`, with no change in behaviour) -- reaching a decision model needs a network call, a spend guard and a redaction shield, none of which belong in a stdlib-only polyglot, the same reason `chat` and `infer` stay out of it. The `decide` cmd itself ships in the reference LIBRARY runtime (`lib/hal/happi/`): a caller asks a **decision model** (never a chat model) a `state` plus a map of typed questions -- `noul` (yes/no), `choice` (up to 255 named options), or `score` (2–10 ordered levels) -- and gets back one checked, typed answer per question, never free text. The library runtime streams one `delta` per question, in the order asked, each carrying that question's `qid` and `answer`, then closes with a `completed.usage` record (`input_tokens`, `output_tokens`, `cost_usd`, `model`, `calibrated: false`). A failure is the existing `error` event carrying a reason `code` from the decide package's own closed set -- no new event type. **The protocol carries the reading; whether to act on it, and at what threshold, stays in caller code.** This (embedded) runtime emits `happi/1.6` and accepts `1.0`–`1.6` (full back-compat).
- `happi/1.5` — reasoning traces as diagnostic specimens. Adds the `reasoning` event type and the `reasoning.append` cmd. The governing constraint is that a trace is admissible as a specimen to be EXAMINED and inadmissible as EVIDENCE OF CAUSE, so the fence is on the wire rather than in prose: `authoritative` is stamped literally `false` and `role` literally `"diagnostic"`, neither caller-settable, and a record asserting authority or carrying `settles` is refused before it reaches the stream. `precedes` is a temporal-position link only and says nothing causal. Runtime emits `happi/1.5` and accepts `1.0`–`1.5` (full back-compat).
- `happi/1.4` — pre-registered falsifiable commitments. Adds the `continuity` event type and the `continuity.append` cmd: a `continuity-commit` names a load-bearing premise and the observation that would refute it, and is emitted BEFORE the dispatch whose reasoning depends on it; a `continuity-verdict` settles one (CONGRUENT / INCONGRUENT / UNVERIFIABLE). The ordering is the guarantee: a reader verifies from the stream alone that the reasoning predates the outcome. Runtime emits `happi/1.4` and accepts `1.0`–`1.4` (full back-compat). The protocol-level twin of GRIP `lib/continuity_chain.py`.
- `happi/1.3` — deterministic citation-provenance floor. Adds the `cite.verify` cmd: each citation's quote is verified verbatim against its cited source (exact → whitespace/typographic-flexible → not_found); `flags.strict=true` turns any not_found into an `error` (a build-gate any harness can fail on). No new event types. Runtime emits `happi/1.3` and accepts `1.0`/`1.1`/`1.2`/`1.3` envelopes (full back-compat). The protocol-level twin of GRIP `lib/prove_it.py` — the un-fakeable floor, available to any AI on any harness.
- `happi/1.2` — memory-chain capstone. No new event types or cmds vs v1.1. Ratifies the `context` event + `context.append` cmd (shipped in v1.1) as the stable signed-memory-chain capture/cutover surface; runtime emits `happi/1.2` and accepts `1.0`/`1.1`/`1.2` envelopes (full back-compat).
- `happi/1.1` — added the `idr` event type, `flags.audit=true`, the `context` event type + `context.append` cmd, and runtime cmds `pr.reference`, `hypothesis.register`, `quine.spawn`. Accepts `1.0`/`1.1` envelopes.
- `happi/1.0` — base protocol: seven streaming event types, the envelope contract, and the polyglot form.
**Format**: polyglot — one file, five parsers (Markdown, bash, embedded Python runtime, JSON envelope, OpenAPI YAML)
**Canonical reference**: this file (`happi.md`). Other implementations (e.g. `lib/hal/happi/` α-1 transport scaffold) MAY support a strict subset; this document is normative for the protocol.
**Watermark**: `V>>--<<V`
**Prose style (humans only, NOT the protocol)**: assistant prose *about* HAPPI follows GRIP's canonical output style (`rules/canonical-output-style.md`); HAPPI envelope/event payloads are governed by THIS spec, never by output style.

---

<!-- happi:label=quickstart -->
## Quickstart

```bash
# show identity + smoke
bash happi.md

# dispatch a HAPPI/1.0 envelope through the embedded runtime
echo '{"v":"happi/1.0","id":"hello","cmd":"version"}' | bash happi.md run

# v1.1 audit-traced run — emits an idr event after completed
echo '{"v":"happi/1.1","id":"audit-1","cmd":"echo","args":["hi"],"flags":{"audit":true}}' \
  | bash happi.md run

# install the symlink so `happi` is on $PATH
bash happi.md install
```

Self-bootstrapping: any machine with `bash` and `python3 >= 3.11` runs this
file as-is. No `git clone`, no `pip install`, no other files needed for
protocol mechanics. v1.0 envelopes dispatch unchanged; v1.1 envelopes
unlock `flags.audit` and the v1.1 cmd set.

---

<!-- happi:label=contract -->
## The Contract

HAPPI is a syscall for AI. One JSON envelope in, one NDJSON event stream out.
Any tool, any provider, any transport — one contract.

```text
stdin:  {"v":"happi/1.0","id":"req-001","cmd":"version"}
stdout: {"v":"happi/1.6","id":"req-001","type":"started","ts":0}
        {"v":"happi/1.6","id":"req-001","type":"delta","ts":1,"text":"happi/1.6"}
        {"v":"happi/1.6","id":"req-001","type":"completed","ts":2}
stderr: diagnostics only
exit:   0 on completed · non-zero on error
```

The runtime is a subprocess. Envelopes are arguments. Events are return
values. Composition is envelope trees.

> **Version semantics.** The envelope's `v` declares the *client's*
> protocol; the runtime accepts it if it appears in `_ACCEPTED_VERSIONS`,
> which is declared once, in the Python layer. The version history above
> lists the accepted set; this paragraph deliberately does not repeat it,
> because a second copy of that fact is exactly what let this sentence sit
> two releases stale — it still named `happi/1.3` as the current runtime
> after both v1.4 and v1.5 had shipped, while the envelope table below
> correctly listed all six. Emitted events carry the *runtime's* own
> `HAPPI_VERSION`, regardless of envelope `v`. Forward-compat (line below)
> means a v1.0 client safely ignores unknown event-level fields, so a newer
> runtime can reply with newer events to an older client. The `cmd version`
> delta echoes the runtime version for the same reason — it answers "what
> does this runtime speak?", not "what did the envelope say?".

---

<!-- happi:label=envelope-schema -->
## Envelope (stdin)

| Field   | Req | Type   | Notes |
|---------|-----|--------|-------|
| `v`     | yes | string | `"happi/1.0"`, `"happi/1.1"`, `"happi/1.2"`, `"happi/1.3"`, `"happi/1.4"`, `"happi/1.5"`, or `"happi/1.6"` (v1.1 adds `idr`/`context` events; v1.2 is the memory-chain capstone; v1.3 adds the `cite.verify` cmd; v1.4 adds the `continuity` event; v1.5 adds the `reasoning` event; v1.6 adds the `decide` cmd — all wire-compatible) |
| `id`    | yes | string | Caller-assigned, opaque, unique per call |
| `cmd`   | yes | string | Dot-path; full list in **Cmds** section below |
| `args`  |     | array  | Positional arguments |
| `flags` |     | object | Named arguments |
| `auth`  |     | object | Credentials — scrubbed from logs |

Forward-compatible: unknown fields are forwarded unchanged.

### Extension fields — provider dispatch (NOT canonical)

The canonical runtime in this file reads none of the following and has no
`chat` cmd. They are defined by the reference transport library
(`lib/hal/happi/envelope.py`) for provider dispatch, ride on
`additionalProperties: true`, and are listed here because a reader had no way
to discover them — not because the protocol requires them. An implementation
speaking only the table above is fully conforming.

| Field          | Type   | Notes |
|----------------|--------|-------|
| `content`      | string | The primary prompt-carrying field for `chat`. |
| `ctx`          | object | Passthrough context; `ctx.context_head` is the memory-chain pointer (the latest `context` node id). An older runtime forwards it unchanged. |
| `capabilities` | array  | Strings naming capabilities the caller claims. |

**Prompt resolution, for any adapter that implements `chat`.** One order, and
every adapter must use it rather than resolving the prompt itself:

1. `content`
2. `flags.message`
3. `" ".join(args)`

Empty string only when the envelope genuinely carries no prompt in any of the
three, so an empty input still fails loudly rather than becoming filler.

That order is written down because implementing it per-adapter went wrong in a
way nothing caught for a while. Three adapters resolved the prompt
independently: one honoured `content`, the other two looked only at
`flags.message` and `args`. An envelope carrying its prompt solely in
`content` — the documented primary field — therefore reached one provider with
the real prompt and every other with an **empty user message**. Those models
free-associated, and the failure surfaced as a council review failing its own
validity gate rather than as anything resembling a dispatch bug.


---

<!-- happi:label=events -->
## Events (stdout)

One compact JSON object per line. Always ends with `completed` or `error`.
Every event carries: `v`, `id` (echoes envelope `id`), `type`, `ts` (ms integer).

| `type`        | Additional required fields        | Meaning |
|---------------|-----------------------------------|---------|
| `started`     | —                                 | Dispatch began |
| `delta`       | `text`                            | Streamed content chunk |
| `thinking`    | `text`                            | Streamed reasoning chunk — the model thinking before it answers; never part of the answer (a consumer joining `delta.text` is unaffected). Additive, provider adapters only |
| `tool_call`   | `name`, `input`, `call_id`        | The model asked for a tool; the caller runs it |
| `tool_result` | `name`, `output`, `call_id`       | Tool returned |
| `sub_request` | `envelope`                        | Child HAPPI envelope dispatched |
| `completed`   | `usage` (optional), `idr_ref` (optional, v1.1), `stop_reason` + `message` (optional; a chat adapter sets both when the provider returned material the next turn must echo verbatim — `reasoning_content` or `tool_calls`) | Success |
| `error`       | `code`, `message`, `idr_ref` (optional, v1.1) | Failure — runtime exits non-zero |
| `idr` (v1.1)  | `idr_ref`                         | Audit receipt — content hash of envelope + event stream |
| `context` (v1.1) | `context_ref`                  | Memory-chain link — content address of the decision body, threading `predecessor_context` |
| `continuity` (v1.4) | `continuity_ref`            | Pre-registered falsifiable commitment, or its settlement — emitted BEFORE the dispatch it governs, not a terminator |
| `reasoning` (v1.5) | `reasoning_ref`              | A reasoning trace, content-addressed — a DIAGNOSTIC SPECIMEN to be examined, never evidence of cause. Stamped `authoritative: false`, `role: "diagnostic"`. Not a terminator |

Tool loop: a `tool_call` names what the model asked for (`call_id` is the provider's id); no runtime executes it — the caller runs the tool and returns its output as a `role: tool` message in the next envelope's `flags.messages`, which the adapter echoes as `tool_result` and forwards verbatim.

Standard `error.code`: `parse_error` · `unsupported_cmd` · `auth_error` · `runtime_error`.
Provider-specific fields live under `provider.*`.

---

<!-- happi:label=idr -->
## IDR — Intent Decision Record (v1.1)

An IDR is the audit receipt for a HAPPI dispatch. When a runtime is producing
audit-traceable output (typically because the envelope sets `flags.audit=true`),
it emits an `idr` event after `completed` or `error` carrying the content hash
of (envelope JSON + concatenated event NDJSON). Wire shape:
`{"v":"happi/1.6","id":...,"type":"idr","ts":...,"idr_ref":{...}}`.

`idr_ref` shape:

| Field            | Required | Notes |
|------------------|----------|-------|
| `sha256`         | yes      | Content hash of envelope + event stream — locally verifiable |
| `cid`            | no       | IPFS CID if pinned; `null` otherwise |
| `model_versions` | yes      | List of model identifiers consulted (audit chain) |
| `block_anchor`   | no       | On-chain block reference if anchored |

**Two emission patterns** (runtime chooses):

1. Standalone `idr` event AFTER `completed`/`error` (this runtime's pattern).
2. Inline `idr_ref` field on the `completed`/`error` event itself.

A given dispatch uses one pattern, never both.

**The cmd `idr.emit`** (stdlib runtime): given `args = [envelope_json, ndjson_events]`,
computes `sha256` and emits a single `idr` event. Used to produce IDRs after-the-fact
from recorded streams — e.g. for replay, conformance, or RWA on-chain anchoring.

**Backward-compat**: v1.0 envelopes dispatch unchanged on a v1.1 runtime. IDR is
opt-in via `flags.audit=true` or explicit `cmd: "idr.emit"`.

**Falsification**: this design is wrong if (a) audit-enabled runs produce a
non-deterministic `sha256` for the same envelope + event sequence, (b) the `idr`
event arrives before `completed`/`error` (it must be the terminator), or (c) a
single dispatch emits both an `idr` event AND an `idr_ref` field on the
`completed`/`error` event.
<!-- happi:label=idr:end -->

---

<!-- happi:label=context -->
## CONTEXT — Replayable memory-chain link (v1.1)

A `context` event is the structural twin of `idr`: a non-streaming terminator
emitted after `completed`/`error`, carrying a `context_ref` instead of an
`idr_ref`. Where `idr` is the audit *receipt* of one dispatch, `context`
threads a **signed memory-chain** — each event links its predecessor, so a
session's evolving knowledge is an append-only chain of content-addressed
deltas (design: GRIP `drafts/happi-context-event-memory-chain-design.md`).
Wire shape:
`{"v":"happi/1.6","id":...,"type":"context","ts":...,"context_ref":{...}}`.

`context_ref` shape:

| Field                 | Required | Notes |
|-----------------------|----------|-------|
| `sha256`              | yes      | **Content address** of the decision body — `"sha256:<hex>"` over canonical JSON (sorted keys) EXCLUDING the volatile `id`, `ts`, `audit` fields (git's tree-hash excludes committer-date). Two records with identical semantic content but different `id`/`ts` share ONE `sha256` — the dedup/merkle coordinate. |
| `predecessor_context` | no       | `id` of the prior `context` node — the backward chain link (`null` at genesis) |
| `snapshot_ref`        | no       | `sha256` of the snapshot this delta tail builds on; `null` if none |
| `kind`                | yes      | `context-delta` \| `context-snapshot` \| `context-supersede` |
| `model_versions`      | yes      | Model identifiers that produced the delta (audit chain) |

**Content address vs audit receipt.** `context_ref.sha256` is NOT the
`idr`-style hash of (envelope + event stream). It is the content address of
the *semantic body* (the dedup coordinate), excluding `id`/`ts`/`audit`. The
two hashes answer different questions: `idr` = "what exactly happened in this
run?"; `context` = "what is the identity of this belief, regardless of when it
was recorded?". A run may carry both an `idr` (via `flags.audit`) and a
`context` (via `context.append`) — independent terminators, each at most once.

---

<!-- happi:label=context:end -->

<!-- happi:label=continuity -->
## CONTINUITY — Pre-registered falsifiable commitment (v1.4)

`idr` records what was decided; `context` records what was believed. A
`continuity` event records what was **assumed** — the load-bearing premise an
agent is about to act on, named together with the observation that would refute
it. It completes the family: decision, belief, and the reasoning path between
them.

The event is **not a terminator**, and that is the whole design. A
`continuity-commit` is emitted BEFORE the dispatch whose reasoning depends on
it, so a reader checking the stream can see that the premise and its falsifier
predate the outcome. Reasoning recorded after the result can always be fitted
to it; reasoning recorded before it cannot. A later `continuity-verdict`
settles the commitment it names. Wire shape:
`{"v":"happi/1.6","id":...,"type":"continuity","ts":...,"continuity_ref":{...}}`.

`continuity_ref` shape:

| Field                    | Required | Notes |
|--------------------------|----------|-------|
| `sha256`                 | yes      | **Content address** of the commitment body — `"sha256:<hex>"` over canonical JSON (sorted keys) EXCLUDING the volatile `id`, `ts`, `audit` fields. Byte-identical to the `context` address algorithm, so one implementation serves both. |
| `kind`                   | yes      | `continuity-commit` \| `continuity-verdict` |
| `predecessor_continuity` | no       | `id` of the prior `continuity` node — the backward chain link (`null` at genesis) |
| `settles`                | verdicts | The `sha256` of the commitment this verdict settles. REQUIRED on a `continuity-verdict`: a settlement that names no commitment cannot be joined back to a premise, so it proves nothing. |
| `model_versions`         | yes      | Model identifiers that produced the record (audit chain) |

A commit body carries the premise (`claim`, `falsifier`, `deps`, `severity`); a
verdict body carries the settlement (`verdict` — `CONGRUENT` / `INCONGRUENT` /
`UNVERIFIABLE` — plus `axis`, `evidence`, `verifier`). The runtime rejects an
unrecognised `kind` and a verdict with no `settles` rather than writing a
record that cannot be checked.

**What stays runtime-side.** The protocol carries the RECORD, never the POLICY.
Dependency-contamination closure, settlement worklists, predictor stacks and
any braking or halting behaviour built on these records live in the harness —
exactly the division `idr` and `context` already keep.

**Emission** — the stdlib reference runtime emits a `continuity` event via the
explicit `cmd: "continuity.append"`, content-addressing a supplied commitment
body (`args[0]`). Chain metadata is supplied via `flags.{kind,
predecessor_continuity, settles, model_versions}` (default
`kind=continuity-commit`). There is no flags-driven auto-emission and there
cannot be one: a commitment is a claim the CALLER makes, and a runtime that
manufactured one from a dispatch envelope would be inventing a premise nobody
stated.

**Backward-compat**: v1.0 envelopes dispatch unchanged; in this runtime
`continuity` is opt-in via `cmd: "continuity.append"`. A consumer that does not
recognise the `continuity` event type forwards it inertly (forward-compat:
unknown event types/fields are ignored, never rejected).

**First conformant consumer**: GRIP `lib/continuity_chain.py`, whose commitment
bodies are addressed by `lib/precog/idr.py::content_addr` — which is why
falsifier (c) below pins byte-identity with that module.

**Falsification**: this design is wrong if (a) two bodies identical but for
`id`/`ts`/`audit` produce DIFFERENT `continuity_ref.sha256` (content-address
broken), (b) a `continuity-commit` arrives AFTER the `completed`/`error` of the
dispatch it governs — the ordering is the entire guarantee, and a commitment
that can be written after the outcome proves nothing — or (c)
`continuity_ref.sha256` is not byte-identical to GRIP's
`lib/precog/idr.py::content_addr` for the same body (cross-runtime divergence).
<!-- happi:label=continuity:end -->

---

<!-- happi:label=reasoning -->
## REASONING — A trace as a diagnostic specimen (v1.5)

`idr` records what was DECIDED, `context` what was BELIEVED, `continuity` what
was PREMISED. A `reasoning` event records what was THOUGHT — the trace itself,
content-addressed like any other body.

It is governed by one constraint, and the constraint is the only reason the
event type is safe to add:

> A reasoning trace is admissible as a **diagnostic specimen to be examined**,
> and inadmissible as **evidence of cause**.

A trace explains nothing on its own authority. Reasoning written alongside an
outcome can always have been fitted to it, and a self-report is the least
reliable claim class there is — so a trace may APPEAR and may NEVER be believed.
That is not advice to implementers; it is enforced on the wire. Every record is
stamped `authoritative: false` and `role: "diagnostic"`, neither of them
caller-settable, and the runtime REFUSES — before anything reaches the stream —
a record that claims authority or carries `settles`. Wire shape:
`{"v":"happi/1.6","id":...,"type":"reasoning","ts":...,"reasoning_ref":{...}}`.

`reasoning_ref` shape:

| Field             | Required | Notes |
|-------------------|----------|-------|
| `sha256`          | yes      | **Content address** of the trace body — `"sha256:<hex>"` over canonical JSON (sorted keys) EXCLUDING the volatile `id`, `ts`, `audit` fields. The same algorithm that addresses a `context` or `continuity` body, so one implementation serves all three. |
| `authoritative`   | yes      | ALWAYS literal `false`. Stamped, never supplied. A record asserting `true` is refused. |
| `role`            | yes      | ALWAYS `"diagnostic"`. Stamped, never supplied. |
| `precedes`        | no       | Address of the record this trace was written before. A TEMPORAL-POSITION link and nothing more — it does not assert that the trace caused what follows. |
| `model_versions`  | yes      | Model identifiers that produced the trace (audit chain) |
| `signature`       | no       | Provider thinking-signature, when one is supplied |

**Not a terminator**, like `continuity` and unlike `idr`/`context`: the record is
emitted before `completed`.

**Why `settles` is refused.** Settling is what a `continuity-verdict` does to a
pre-registered commitment, against stated evidence. A trace settles nothing, and
permitting the field would let a specimen pose as a settlement — the precise
confusion between examining reasoning and crediting it that this design exists
to prevent.

**What a stateful reader must still enforce.** This runtime holds no record
store and reads exactly one envelope, so it cannot resolve an address back to
the record that bears it. Two checks therefore belong to a reader that holds the
chain, not here: a `continuity-verdict` whose `settles` names a REASONING
address, and a `continuity-commit` naming one as a `dep`. Both would smuggle a
trace into the causal position the stamp denies it. Stating the gap is
deliberate — a guard that cannot fire would read as protection that is absent.

**Backward-compat — and why inert forwarding is the RIGHT behaviour here, not
merely a tolerated one.** v1.0 envelopes dispatch unchanged; in this runtime
`reasoning` is opt-in via `cmd: "reasoning.append"`. A consumer that does not
recognise the event type forwards it inertly (forward-compat: unknown event
types and fields are ignored, never rejected). For this event type that default
is exactly correct: **a reader that does not interpret a trace cannot credit
one.** A v1.4 consumer (the last version that predates `reasoning`) relaying a
current, v1.6 stream:

```text
in:   {"v":"happi/1.6","id":"r-1","type":"reasoning","ts":7,"reasoning_ref":{"sha256":"sha256:ab...","authoritative":false,"role":"diagnostic","precedes":"sha256:cd...","model_versions":["..."]}}
out:  {"v":"happi/1.6","id":"r-1","type":"reasoning","ts":7,"reasoning_ref":{"sha256":"sha256:ab...","authoritative":false,"role":"diagnostic","precedes":"sha256:cd...","model_versions":["..."]}}
```

It neither parses `reasoning_ref` nor drops the event, so the record reaches a
reader that does understand it, byte-identical and still carrying its stamps.
Dropping it would be its own conformance failure: the trace belongs to the
stream a later auditor replays, and a consumer is not entitled to decide that an
event it cannot read should not exist.

**The counter-example: what a NON-conforming consumer does.** Recognising the
type is where the obligation starts, not where it ends. A consumer that renders
the trace as "why the system did X", that presents it as a justification, or
that resolves `precedes` into a causal edge has treated a specimen as evidence
of cause — the precise confusion the stamps exist to prevent, committed by
software that read them and ignored what they meant. `authoritative: false` is
not a hint to be weighed against other signals; it is the record stating that no
weight is available. The ignorant consumer above is conformant and the informed
one here is not, which is the whole point: this fence is not enforced by
understanding the event, it is enforced by refusing to promote it.

**Falsification**: this design is wrong if (a) two bodies identical but for
`id`/`ts`/`audit` produce DIFFERENT `reasoning_ref.sha256`, (b) any emitted
record carries `authoritative: true` or a `role` other than `"diagnostic"`, or
(c) a consumer is ever observed treating `precedes` as a causal claim — in which
case the wire fence is insufficient and the link itself must go.
<!-- happi:label=reasoning:end -->

---

<!-- happi:label=cite-verify -->
## cite.verify — deterministic citation provenance (v1.3)

The un-fakeable floor of grounded AI, lifted to protocol level. A cited quote is
verbatim-present in its source, or it is not — a deterministic string-match, not a
judgement. `cite.verify` performs that check, so **any AI on any harness** that
speaks HAPPI can prove (or disprove) its own citations the same way.

```bash
echo '{"v":"happi/1.4","id":"cv-1","cmd":"cite.verify","flags":{
  "sources":   [{"id":"s1","text":"The Limitation Act 1980 bars the claim after six years."}],
  "citations": [
    {"id":"c1","source_id":"s1","quote":"bars the claim after six years"},
    {"id":"c2","source_id":"s1","quote":"bars the claim after three years"}
  ]}}' | bash happi.md run
# delta c1 verified ; delta c2 not_found (fabricated) ; completed.usage = the record
```

**Inputs** (`flags`):

- `sources` — `[{"id": str, "text": str}, ...]` (required). The documents a citation may quote.
- `citations` — `[{"id": str, "source_id": str, "quote": str}, ...]` (required). Each claim's exact quoted words and which source they come from.
- `strict` — `bool` (default `false`). When `true`, ANY `not_found` citation makes the runtime emit `error` (exit non-zero) — a build-gate any harness can fail on. Default emits `completed`; the caller reads `grounding_rate`.

**Behaviour** — for each citation the runtime locates `quote` in its source by a
deterministic ladder: exact substring → whitespace-and-typographic-flexible
(unicode dashes / smart-quotes / nbsp normalised, length-preserving so offsets
index the original) → `not_found`. It streams one `delta` per citation
(`"<id> <status>"`), then emits `completed` whose `usage` is the provenance record:

```json
{"sources": {"s1": {"sha256": "…", "chars": 55}},
 "citations": [{"id": "c1", "source_id": "s1", "status": "verified", "start": 24, "end": 54},
               {"id": "c2", "source_id": "s1", "status": "not_found", "start": -1, "end": -1}],
 "tally": {"verified": 1, "fuzzy": 0, "not_found": 1},
 "grounding_rate": 0.5}
```

The record has the same shape as GRIP `lib/prove_it.py::provenance`, so the two
engines interoperate; the per-source `sha256` + per-citation offsets are exactly
what an IDR / memory-chain leaf needs to sign.

**The guarantee** — a fabricated quote can never earn `verified` or `fuzzy`; it
resolves to `not_found`. The check is arithmetic, not opinion. It proves the quote
is verbatim in the *supplied* source — NOT that the source is authentic, nor that
the quote *supports* the claim (those remain the caller's, or a higher LLM layer's,
responsibility).

**Falsification**: wrong if (a) a quote absent from its source ever resolves to
anything but `not_found`; (b) a whitespace-or-typographic-only variant fails to
resolve (the normalisation is unsound); or (c) the same `(quote, source)` resolves
differently here than in GRIP `lib/prove_it.py::verify_quote` (cross-runtime
divergence — the two MUST agree).
<!-- happi:label=cite-verify:end -->

---

<!-- happi:label=decide -->
## decide -- typed decisions (v1.6)

A caller that needs an answer, not a completion, asks a **decision model**
instead of a chat model: a `state` (a string, an object or an array) plus a
map of typed questions, and gets back one checked, typed answer per question --
never free text. `decide` lifts that call to protocol level. It is served by
the reference library runtime (`lib/hal/happi/`), not this embedded one:
reaching a decision model needs a network call, a spend guard and a redaction
shield, none of which belong in a stdlib-only polyglot -- the same reason
`chat` and `infer` are library-only (see Extension fields, above). Piping the
example below through `bash happi.md run` therefore fails with
`unsupported_cmd`; it dispatches through the library runtime instead.

```bash
echo '{"v":"happi/1.6","id":"d-1","cmd":"decide","flags":{
  "state": "Payouts have failed for three days.",
  "questions": {
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "team":   {"type": "choice", "instructions": "Which team should handle it?",
               "criteria": {"billing": "Payments and refunds", "technical": null}}
  }}}' | hal --happi-api
# delta {"qid":"urgent","answer":{"type":"noul","noul":0.95}}
# delta {"qid":"team","answer":{"type":"choice","choice":"billing", ...}}
# completed.usage = {"input_tokens":287,"output_tokens":0,"cost_usd":"0.000012","model":"jev-1.13.0","calibrated":false}
```

**Inputs** (`flags`):

- `state` -- string, object or array (required). What the model is asked about.
- `questions` -- a non-empty map of question id to question (required). Each
  question is one of three kinds, exactly as `docs/decide.md` and
  `lib/hal/decide/contract.py` define them:
  - **noul** -- a yes/no question; the answer is the probability, 0 to 1, that
    it is yes.
  - **choice** -- one of up to 255 named options; the answer names the chosen
    option plus every option's probability and a confidence.
  - **score** -- 2 to 10 ordered levels; the answer is a probability-weighted
    level (which can land between levels), plus a confidence.
- `model` -- string (optional). A decision model id from
  `config/decision_models.yaml`; defaults to the catalogue's primary model.

**Behaviour** -- the handler checks `(state, questions)` against the format
before dispatching (`lib/hal/decide/contract.py::validate`); a request that
breaks the format never reaches a decision model. On success it streams one
`delta` per question, **in the order `questions` was given**, each carrying
that question's `qid` and its checked, typed `answer` -- the order is the only
correlation between a delta and the question it answers, so it is exact, not
advisory. It then emits `completed`, whose `usage` carries:

```json
{"input_tokens": 287, "output_tokens": 0, "cost_usd": "0.000012", "model": "jev-1.13.0", "calibrated": false}
```

`calibrated` is always `false` today: a probability here is the decision
model's own number, not a measured error rate, because no labelled-set
calibration is on file for any question yet (`docs/decide.md`, "Known
weaknesses, and the rules that follow").

**Failure** is the existing `error` event -- no new event type -- carrying a
reason `code` from the decide package's own closed set
(`lib/hal/decide/contract.py::REASONS`): `invalid_request`, `unknown_model`,
`catalog_unavailable`, `spend_refused`, `no_key`, `shield_unavailable`,
`shield_refused`, `over_budget`, `circuit_open`, `auth_failed`,
`rate_limited`, `overloaded`, `server_error`, `timeout`, `unreachable`,
`bad_response`, `provider_error`. An unrecognised `cmd` still errors exactly
as before (`unsupported_cmd`) -- `decide` narrows what already happens to an
unknown command; it does not change it.

**What this is NOT.** A decision model's answer is a probability, never a
verdict. Nothing in this cmd gates a merge, an approval, or any other action:
`confidence_threshold` and `confidence_gate` are refused everywhere HAPPI
already carries confidence (`lib/hal/happi/decision.py`), and `decide` does
not reopen that door. **The protocol carries the reading; whether to act on
it, and at what threshold, stays in caller code.** A decision model can also
be steered by text inside its own `state` -- one published benchmark measured
12.1% of decisions flipped by a single appended, unverified opinion -- so it
is never the sole gate on anything, and never decides on untrusted text that
could unlock a capability.

**Falsification**: wrong if (a) two calls with the same `state` and
`questions` sent to the same model ever leave a reader unable to tell which
`qid` a given `delta` answers from order alone; (b)
`completed.usage.calibrated` is ever stamped `true` without a recorded,
labelled-set measurement backing it; or (c) any caller reaches a decision
model without GRIP's redaction shield in front of it.
<!-- happi:label=decide:end -->

---

<!-- happi:label=axioms -->
## The Four Axioms

1. **CLI (stdio) is the canonical transport.** Every other transport is a
   shim that maps to this contract.
2. **Seven core event types are sufficient for streaming.** All known
   LLM-provider streaming semantics map onto them; provider-specific data lives
   in sub-fields. Sufficiency is the claim: a conforming stream can be
   expressed in the seven, and the canonical runtime emits only those.
   `thinking` is optional and additive — some provider adapters emit it, the
   canonical runtime never does, and nothing requires it, so its existence
   does not weaken the claim.

   Four types sit *outside* the streaming set, none carrying streaming
   content, each emitted at most once: `idr` (audit receipt) and `context`
   (memory-chain link), both v1.1 terminators emitted AFTER `completed`/
   `error`; and `continuity` (v1.4, a pre-registered premise) and `reasoning`
   (v1.5, a diagnostic specimen), which are deliberately **not** terminators —
   they precede the outcome, and that ordering is the guarantee they exist to
   make. For the complete wire vocabulary, read the `type` enum in the OpenAPI
   schema below rather than any count written in prose.
3. **`sub_request` recurses through the same runtime.** The protocol is
   fractal; no privileged inner interface.
4. **The polyglot form IS the specification.** This document is the
   contract — no separate authoritative schema repo.

Falsification conditions for each axiom are recorded in
`plans/happi-protocol-triphase.md` section 1.

---

<!-- happi:label=cmds -->
## Cmd handlers (this file)

### Bash invocations

| Bash invocation | Behaviour |
|-----------------|-----------|
| `bash happi.md` | identity (banner + spec summary + smoke) |
| `bash happi.md identity` | identity (explicit; same as no-arg) |
| `bash happi.md morning` | morning-boot routine (subsumed) |
| `bash happi.md run` | exec embedded runtime; reads envelope from stdin |
| `bash happi.md install` | symlink to `~/.local/bin/happi` |
| `bash happi.md extract <layer>` | dump `markdown`, `bash`, `python`, `envelope`, or `openapi` |
| `bash happi.md spec.describe` | recursive dogfood — dispatch via embedded runtime |
| `bash happi.md help` | alias for identity (also `-h`, `--help`) |

### Embedded runtime cmds (stdlib-only)

| Cmd | Since | Behaviour |
|-----|-------|-----------|
| `version` | 1.0 | Emit one delta with the runtime's protocol version |
| `echo` | 1.0 | Emit each `args[i]` as a separate delta |
| `spec.describe` | 1.0 | Emit canonical spec summary lines as deltas |
| `envelope.validate` | 1.0 | Confirm envelope passed schema validation; reach implies pass |
| `idr.emit` | 1.1 | Emit one `idr` event for a recorded `(envelope, ndjson_events)` pair |
| `context.append` | 1.1 | Emit one `context` event content-addressing a decision body (`args[0]` JSON, excludes `id`/`ts`/`audit`); chain metadata via `flags.{predecessor_context,snapshot_ref,kind,model_versions}`. Twin of `idr.emit`. |
| `pr.reference` | 1.1 | Reference a PR (informational; no side effects). Required: `flags.pr` (int), `flags.repo` (str). Used in fractal seeds. |
| `hypothesis.register` | 1.1 | Append a falsifiable hypothesis to NDJSON log (default `~/.hal/data/hypotheses.jsonl`; override `HAL_HYPOTHESES_PATH`). Required: `args[0]` ID; `flags.{claim,metric,prediction,deadline}` |
| `quine.spawn` | 1.1 | Spawn child HAPPI seed issue from parent (`#233` fractal pattern). Required: `flags.parent_issue` (int), `flags.repo` (str). DRY-RUN unless `flags.live=true`. Generation counter parsed from parent title (`generation N`); child title bumps to N+1. Capped by `flags.depth_limit` (default `16`; env override `HAPPI_QUINE_DEPTH_MAX`). LIVE mode requires `gh` CLI authenticated for the target repo. |
| `cite.verify` | 1.3 | Deterministically verify each citation's `quote` is verbatim in its cited source (the un-fakeable provenance floor). Required: `flags.sources` `[{id,text}]`, `flags.citations` `[{id,source_id,quote}]`. Optional `flags.strict=true` → any `not_found` emits `error` (build-gate). `completed.usage` carries the provenance record (per-source sha256, per-citation status+offsets, tally, grounding_rate). |
| `compose` | 1.3 | Dispatch each child envelope in `flags.envelopes` through this same runtime. Emits one `sub_request` per child, then inlines that child's own events into the parent stream (axiom 3). Children are validated and dispatched by the same `parse_envelope`/`dispatch` pair that serves stdin, so a child fails exactly as the same bytes would have failed at the top level. Recursion is capped at 8; exceeding it emits `error` with code `sub_request_depth_cap`. |
| `continuity.append` | 1.4 | Emit one `continuity` event content-addressing a commitment body (`args[0]` JSON, excludes `id`/`ts`/`audit`). Chain metadata via `flags.{kind,predecessor_continuity,settles,model_versions}` (default `kind=continuity-commit`). Fail-closed: an unrecognised `kind`, or a `continuity-verdict` with no `settles`, is refused before emission. NOT a terminator — emitted BEFORE the dispatch whose premise it registers. |
| `reasoning.append` | 1.5 | Emit one `reasoning` event content-addressing a trace body (`args[0]` JSON, excludes `id`/`ts`/`audit`). Optional `flags.{precedes,model_versions,signature}`. `authoritative: false` and `role: "diagnostic"` are STAMPED, not caller-settable; a record claiming authority, or carrying `settles`, is refused before emission. NOT a terminator. |

Audit terminator: setting `flags.audit=true` on the input envelope causes the runtime to emit an `idr` event after `completed`/`error`. **Context event:** `cmd: context.append` emits a `context` event content-addressing a supplied decision body (excluding `id`/`ts`/`audit`); the stdlib runtime does not auto-emit `context` on a `flags.context` flag (HAL #429), so a `flags.audit` `idr` and a `context.append` `context` are independent. Stdlib-only — no extra dependencies for any cmd (`quine.spawn` LIVE mode additionally requires the `gh` CLI; DRY-RUN does not).

**Embedded runtime — emission profile** — this stdlib runtime emits `started`, `delta`, `completed`, `error`, (when `cmd: compose`) `sub_request`, (when `flags.audit=true` or `cmd: idr.emit`) `idr`, (when `cmd: context.append`) `context`, (when `cmd: continuity.append`) `continuity`, and (when `cmd: reasoning.append`) `reasoning`. It does not emit `thinking`, `tool_call` or `tool_result` events; those are reserved for richer runtimes (e.g. `lib/hal/happi/` providers, future tool integrations) that conform to the same protocol surface.

**Recursion — one outcome per envelope, not per stream.** A `compose` stream carries the parent's events and every child's events interleaved, so it contains more than one `completed`/`error`. The invariant is per envelope `id`: each id that appears in a stream terminates exactly once. A client that counts outcomes per stream rather than per id will misread any recursive stream.

---

<!-- happi:label=openapi -->
## OpenAPI 3.1

```yaml
openapi: 3.1.0
info: {title: HAPPI, version: "1.6"}
paths:
  /v1/dispatch:
    post:
      summary: Dispatch one HAPPI envelope
      requestBody:
        required: true
        content:
          application/json:
            schema: {$ref: "#/components/schemas/Envelope"}
      responses:
        "200":
          description: NDJSON event stream
          content:
            application/x-ndjson:
              schema: {$ref: "#/components/schemas/Event"}
components:
  schemas:
    Envelope:
      type: object
      required: [v, id, cmd]
      properties:
        v:     {enum: ["happi/1.0", "happi/1.1", "happi/1.2", "happi/1.3", "happi/1.4", "happi/1.5", "happi/1.6"]}
        id:    {type: string, minLength: 1}
        cmd:   {type: string, minLength: 1}
        args:  {type: array}
        flags: {type: object}
        auth:  {type: object}
      additionalProperties: true
    Event:
      type: object
      required: [v, id, type, ts]
      properties:
        v:    {enum: ["happi/1.0", "happi/1.1", "happi/1.2", "happi/1.3", "happi/1.4", "happi/1.5", "happi/1.6"]}
        id:   {type: string}
        type: {enum: [started, delta, thinking, tool_call, tool_result, sub_request, completed, error, idr, context, continuity, reasoning]}
        ts:   {type: integer, minimum: 0}
      additionalProperties: true
```
<!-- happi:label=openapi:end -->

---

<!-- happi:label=envelope -->
## Dogfood envelope

This document IS a HAPPI/1.0 envelope:

```json
{
  "v": "happi/1.0",
  "id": "happi-md-canonical",
  "cmd": "spec.describe",
  "flags": {
    "version": "1.0",
    "format": "polyglot",
    "layers": ["markdown", "bash", "python", "envelope", "openapi"],
    "watermark": "V>>--<<V"
  }
}
```
<!-- happi:label=envelope:end -->

If `bash happi.md run < envelope.json` cannot dispatch this envelope and emit
a valid `completed` event, the file has failed its own first test.

---

<!-- happi:label=see-also -->
## See also

- `plans/happi-spec.md` — formal HAPPI/1.0 spec (extractable layers)
- `plans/happi-protocol-triphase.md` — alpha/beta/gamma roadmap (PR #218 shipped alpha-1)
- `docs/HAPPI-PROTOCOL-POSSIBILITIES.md` — external-audience white paper
- `plans/happi-md-canonical-v1.md` — the design doc for THIS file

---

<!-- happi:label=licence -->
## Licence

This specification and its embedded runtime are licensed under the Apache
License, version 2.0 (Apache-2.0). Anyone may implement, modify, extend or
embed them in any product, open or closed source. The licence includes an
express patent grant. Earlier publications of this specification were released
under AGPL-3.0-only. The full licence text is at
https://www.apache.org/licenses/LICENSE-2.0 (SPDX: Apache-2.0).

Copyright (C) 2024-2026 Lourens Cornelius Scheepers / CodeTonight (Pty) Ltd

---

*The protocol is stable; the possibilities are not.*
*Canonical reference: `happi.md` · V>>--<<V*

HAPPI_DOC

# ============================================================================
# BASH EXECUTION — content below renders as plain text in Markdown viewers
# ============================================================================

set -euo pipefail

HAPPI_VERSION="happi/1.6"
HAPPI_FILE="${BASH_SOURCE[0]:-$0}"
HAPPI_TMPDIR="${TMPDIR:-/tmp}/happi"
HAPPI_INSTALL_DIR="${HAPPI_INSTALL_DIR:-$HOME/.local/bin}"
mkdir -p "$HAPPI_TMPDIR"

# ----- Embedded HAPPI Python runtime (stdlib-only) ----------
# No version here: HAPPI_VERSION above is the single declaration.
# Captured into HAPPI_PY via single-quoted heredoc — bash performs no expansion
# inside, so the Python source survives byte-identical for `extract python`.
HAPPI_PY="$(cat <<'PYTHON_EOF'
#!/usr/bin/env python3
"""HAPPI/1.6 reference runtime — embedded in happi.md.

Reads exactly one HAPPI envelope (JSON) from stdin (accepts v1.0 through v1.6).
Emits NDJSON events to stdout: started -> (delta)* -> completed | error -> [idr].
v1.1 adds: idr event type, idr.emit cmd, flags.audit=true triggers auto-IDR.
v1.2: memory-chain capstone — no new event types or cmds vs v1.1; ratifies the
context event + context.append cmd as the stable signed-memory-chain surface and
widens version acceptance to include happi/1.2 (full back-compat).
v1.3: deterministic citation-provenance floor — adds the cite.verify cmd. A cited
quote is verbatim-present in its source, or it is not; a fabricated citation can
never verify. No new event types. Accepts v1.0..v1.3 (full back-compat).
v1.4: pre-registered falsifiable commitments — adds the continuity event type and
the continuity.append cmd. A continuity-commit names the premise an agent is about
to act on and what would refute it, and is emitted BEFORE the dispatch it governs;
a continuity-verdict settles one. Accepts v1.0..v1.4 (full back-compat).
v1.5: reasoning traces as DIAGNOSTIC SPECIMENS — adds the reasoning event type
and the reasoning.append cmd. A reasoning record content-addresses a trace and
carries it as something to be EXAMINED, never as evidence of cause. The fence is
on the wire, not merely in prose: authoritative is stamped literally false and
role literally "diagnostic" (neither caller-settable), a record asserting
authority is refused, and a reasoning record may not carry settles — settling is
what a continuity-verdict does, and a trace settles nothing. Accepts v1.0..v1.5
(full back-compat).
v1.6: typed decisions -- adds the decide cmd, served by the reference LIBRARY
runtime (lib/hal/happi/), never by this embedded one: reaching a decision model
needs a network call, a spend guard and a redaction shield, none of which belong
in a stdlib-only polyglot (the same reason chat and infer are library-only). No
cmd and no event type of this runtime's own. Widens version acceptance to
happi/1.6 (full back-compat), and splits cmd_compose into two helpers
(_compose_guard, _compose_dispatch_one) with no change in behaviour.
Diagnostics (if any) go to stderr only.
Stdlib-only: hashlib, json, sys, time. Compatible with Python >= 3.8.
Watermark: V>>--<<V
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HAPPI_VERSION = "happi/1.6"
_ACCEPTED_VERSIONS = ("happi/1.0", "happi/1.1", "happi/1.2", "happi/1.3",
                      "happi/1.4", "happi/1.5", "happi/1.6")
_START_NS = time.time_ns()

_AUDIT_ENABLED = False
_AUDIT_BUFFER = []
_AUDIT_MODEL_VERSIONS = []

# Recursion bound for `compose` (axiom 3). A child envelope may itself be a
# `compose`, so the fractal is unbounded by construction; this caps it. Mirrors
# lib/hal/happi/runtime.py::MAX_SUB_REQUEST_DEPTH so both runtimes refuse at the
# same depth with the same error code.
MAX_SUB_REQUEST_DEPTH = 8
_SUB_DEPTH = 0

# Volatile fields excluded from a content address (v1.1 context event) — mirrors
# GRIP lib/precog/idr.py::_CONTENT_ADDR_EXCLUDE (the cross-runtime invariant).
_CONTENT_ADDR_EXCLUDE = ("id", "ts", "audit")


def _ts_ms():
    return max(0, (time.time_ns() - _START_NS) // 1_000_000)


def _emit(event):
    line = json.dumps(event, separators=(",", ":"))
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    if _AUDIT_ENABLED:
        _AUDIT_BUFFER.append(line)


def emit_started(req_id):
    _emit({"v": HAPPI_VERSION, "id": req_id, "type": "started", "ts": _ts_ms()})


def emit_delta(req_id, text):
    _emit({"v": HAPPI_VERSION, "id": req_id, "type": "delta",
           "ts": _ts_ms(), "text": text})


def emit_completed(req_id, usage=None):
    event = {"v": HAPPI_VERSION, "id": req_id, "type": "completed", "ts": _ts_ms()}
    if usage is not None:
        event["usage"] = usage
    _emit(event)


def emit_error(req_id, code, message):
    _emit({"v": HAPPI_VERSION, "id": req_id, "type": "error",
           "ts": _ts_ms(), "code": code, "message": message})


def emit_sub_request(req_id, child):
    """Emit the marker that precedes a child envelope's inlined events.

    Field name is `envelope`, per the event table in this same document
    ("| `sub_request` | `envelope` | Child HAPPI envelope dispatched |").
    Axiom 4 makes this document the contract, so the wire field follows the
    table rather than any implementation's local choice.
    """
    _emit({"v": HAPPI_VERSION, "id": req_id, "type": "sub_request",
           "ts": _ts_ms(), "envelope": child})


def emit_idr(req_id, envelope_raw):
    """Emit an idr event with content hash of (envelope + buffered events).

    SHA-256 chains envelope_raw bytes followed by each emitted event line.
    Deterministic given identical (envelope, event sequence). Used as the
    audit terminator when flags.audit=true on the input envelope.
    """
    h = hashlib.sha256()
    h.update(envelope_raw.encode("utf-8"))
    for line in _AUDIT_BUFFER:
        h.update(line.encode("utf-8"))
    idr_ref = {
        "sha256": h.hexdigest(),
        "cid": None,
        "model_versions": list(_AUDIT_MODEL_VERSIONS),
        "block_anchor": None,
    }
    _emit({"v": HAPPI_VERSION, "id": req_id, "type": "idr",
           "ts": _ts_ms(), "idr_ref": idr_ref})


def _content_addr(body):
    """Content address of a decision body — "sha256:<hex>" over canonical JSON
    (sorted keys) EXCLUDING the volatile id/ts/audit fields. Byte-identical to
    GRIP lib/precog/idr.py::content_addr for the same body (the cross-runtime
    invariant the context chain depends on for dedup/merkle linkage)."""
    addressed = {k: v for k, v in body.items() if k not in _CONTENT_ADDR_EXCLUDE}
    canonical = json.dumps(addressed, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- Decision anatomy (v1.3) ------------------------------------------------
# The agreed Decision shape carried in a context body's "decision" slot. Inlined
# stdlib-only validator, BEHAVIOURALLY byte-identical to the lib twin
# (lib/hal/happi/decision.py) and GRIP lib/decision_record.py — the ONE shared
# schema (hypothesis H-ONE-SCHEMA). Opt-in via flags.validate_decision on
# context.append; the address algorithm is unchanged (gate before addressing).
_DECISION_ADMISSIBLE = ("what", "why", "confidence", "uncertainties",
                        "assumptions", "falsification_criteria", "falsifier_ref")
_DECISION_SURFACE_GRAFTS = ("tone", "persona", "personality", "mood", "style",
                            "voice", "vibe", "confidence_threshold",
                            "confidence_gate", "ratify", "merge_decision",
                            "approve", "auto_merge", "should_merge")
_DECISION_PREDICATE_MARKERS = re.compile(
    r"(<=|>=|==|!=|<|>|"
    r"\b(?:exceeds?|below|above|equals?|differs?|matches?|mismatch|greater|less|"
    r"within|rejects?|fails?\s+if|falsified\s+if|wrong\s+if|breaks?\s+if|"
    r"diverges?|returns?|when\b|unless\b|over\b|under\b)|"
    r"test_|\.py\b|/|\bCI\b|AGORA|hypothesis)",
    re.IGNORECASE)


def _decision_is_predicate_shaped(text):
    """True iff `text` names something an EXTERNAL process can check. Mirrors
    decision_record.is_predicate_shaped."""
    t = (text or "").strip()
    if not t:
        return False
    return bool(_DECISION_PREDICATE_MARKERS.search(t))


def _decision_smt_gate(slot):
    """SMT gate (gentner): reject surface grafts + non-admissible keys. Returns an
    error-message str on rejection, None on pass. Mirrors decision_record
    reject_surface_fields."""
    keys = set(slot)
    grafts = keys & set(_DECISION_SURFACE_GRAFTS)
    if grafts:
        return ("context.append: surface-graft decision fields rejected: "
                + ",".join(sorted(grafts)))
    unknown = keys - set(_DECISION_ADMISSIBLE)
    if unknown:
        return "context.append: non-admissible decision fields: " + ",".join(sorted(unknown))
    return None


def _decision_honesty_gate(slot):
    """Tiered honesty validate: what non-empty, confidence in [0,1], deterministic
    minimal, delegated externally-checkable. Returns error-message str or None.
    Mirrors decision_record.validate."""
    what = slot.get("what")
    if not isinstance(what, str) or not what.strip():
        return "context.append: decision.what must be a non-empty string"
    conf = slot.get("confidence", 1.0)
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) \
            or conf != conf or conf < 0.0 or conf > 1.0:
        return "context.append: decision.confidence must be a real number in [0,1]"
    if conf >= 1.0:
        if slot.get("uncertainties"):
            return ("context.append: deterministic decision (confidence=1.0) must "
                    "not list uncertainties")
        if slot.get("assumptions"):
            return ("context.append: deterministic decision (confidence=1.0) must "
                    "not list assumptions")
        return None
    fc = (slot.get("falsification_criteria") or "").strip()
    if not fc:
        return ("context.append: delegated decision (confidence<1.0) must state "
                "falsification_criteria")
    if not _decision_is_predicate_shaped(fc):
        return ("context.append: falsification_criteria must be externally "
                "checkable (a predicate/threshold/test/'fails if ...')")
    return None


def _validate_decision_body(body):
    """Validate the "decision" slot of a context body, if present (v1.3).

    Returns None on success (slot absent or valid). Returns an error-message str
    when malformed. Mirrors decision_record SMT-gate then tiered-honesty validate
    (split into two helpers to stay below the CC threshold). The ONE shared schema.
    """
    slot = body.get("decision")
    if slot is None:
        return None
    if not isinstance(slot, dict):
        return ("context.append: decision must be a JSON object (the Decision "
                "shape), got " + type(slot).__name__)
    return _decision_smt_gate(slot) or _decision_honesty_gate(slot)


def emit_context(req_id, body, meta):
    """Emit a context event content-addressing `body` (excludes id/ts/audit).

    `meta` (a flags-shaped dict) supplies the chain metadata: predecessor_context,
    snapshot_ref, kind (default "context-delta"), model_versions. The context
    event is a non-streaming terminator — the memory-chain twin of the idr
    audit receipt.
    """
    context_ref = {
        "sha256": _content_addr(body),
        "predecessor_context": meta.get("predecessor_context"),
        "snapshot_ref": meta.get("snapshot_ref"),
        "kind": meta.get("kind") or "context-delta",
        "model_versions": list(meta.get("model_versions") or []),
    }
    _emit({"v": HAPPI_VERSION, "id": req_id, "type": "context",
           "ts": _ts_ms(), "context_ref": context_ref})


_CONTINUITY_KINDS = ("continuity-commit", "continuity-verdict")


def emit_continuity(req_id, body, meta):
    """Emit a continuity event content-addressing `body` (excludes id/ts/audit).

    `meta` (a flags-shaped dict) supplies: kind (default "continuity-commit"),
    predecessor_continuity, settles (the target commit address of a verdict),
    model_versions.

    Unlike `idr` and `context`, this is NOT a terminator. A commitment is
    emitted BEFORE the dispatch whose premise it registers, because the whole
    guarantee is ordering: a reader verifies from the stream alone that the
    stated premise and its falsifier predate the outcome, so the reasoning
    cannot have been fitted to the result after the fact.
    """
    continuity_ref = {
        "sha256": _content_addr(body),
        "kind": meta.get("kind") or "continuity-commit",
        "predecessor_continuity": meta.get("predecessor_continuity"),
        "settles": meta.get("settles"),
        "model_versions": list(meta.get("model_versions") or []),
    }
    _emit({"v": HAPPI_VERSION, "id": req_id, "type": "continuity",
           "ts": _ts_ms(), "continuity_ref": continuity_ref})


def _load_body_arg(env, cmd, noun):
    """Decode args[0] as a JSON object body, or emit parse_error and return None.

    The body-carrying cmds (context.append, continuity.append,
    reasoning.append) validate args[0] identically, because the contract is
    identical: exactly one positional JSON object. `cmd` and `noun` reproduce
    the original wording of each caller verbatim — the messages are part of the
    wire contract a consumer may match on, so they must not drift.

    NOTE, and it is not a style preference: this layer is embedded inside
    `HAPPI_PY="$(cat <<'PYTHON_EOF'`, so bash pairs quotes across the whole
    polyglot while scanning for the closing paren. An ODD number of prose
    apostrophes in this file leaves one unpaired, swallows that paren, and the
    assembled happi.md stops being executable — with the error reported far
    away, in the bash dispatcher. Avoid prose apostrophes here entirely.

    Returns the decoded dict, or None having ALREADY emitted the error; a caller
    receiving None must `return 1` and emit nothing further.
    """
    args = env.get("args", [])
    if not args or not isinstance(args[0], str):
        emit_error(env["id"], "parse_error",
                   cmd + ": requires args[0] = " + noun + " JSON string")
        return None
    try:
        body = json.loads(args[0])
    except json.JSONDecodeError as e:
        emit_error(env["id"], "parse_error",
                   cmd + ": args[0] is not valid JSON — " + str(e))
        return None
    if not isinstance(body, dict):
        emit_error(env["id"], "parse_error",
                   cmd + ": " + noun + " must be a JSON object")
        return None
    return body


# A reasoning record has no "kind": there is exactly one thing it may be, and
# the wire states it rather than offering a choice. `role` is a fixed stamp, not
# a caller-supplied field — a vocabulary the caller could pick from would be a
# vocabulary a caller could claim authority in.
_REASONING_ROLE = "diagnostic"


def emit_reasoning(req_id, body, meta):
    """Emit a reasoning event content-addressing `body` (excludes id/ts/audit).

    `meta` (a flags-shaped dict) supplies: precedes (the address of the record
    this trace temporally precedes), model_versions, and an optional provider
    thinking-signature.

    A reasoning trace is admissible as a DIAGNOSTIC SPECIMEN to be examined and
    inadmissible as EVIDENCE OF CAUSE. That distinction is the entire reason the
    event type is safe to add, so the record carries it rather than leaving it to
    the specification prose: `authoritative` is stamped literally False and
    `role` literally "diagnostic", neither caller-settable. A consumer reading a
    reasoning record as the reason a decision was taken is reading it wrong, and
    the bytes say so without the spec to hand.
    """
    reasoning_ref = {
        "sha256": _content_addr(body),
        "authoritative": False,
        "role": _REASONING_ROLE,
        "precedes": meta.get("precedes"),
        "model_versions": list(meta.get("model_versions") or []),
        "signature": meta.get("signature"),
    }
    _emit({"v": HAPPI_VERSION, "id": req_id, "type": "reasoning",
           "ts": _ts_ms(), "reasoning_ref": reasoning_ref})


def parse_envelope(raw):
    """Return (envelope_dict, error_message). On success, error_message is None."""
    if not raw.strip():
        return None, "empty stdin"
    try:
        env = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, "invalid JSON: " + str(e)
    if not isinstance(env, dict):
        return None, "envelope must be a JSON object"
    if env.get("v") not in _ACCEPTED_VERSIONS:
        return None, "unsupported version: " + repr(env.get("v"))
    if not isinstance(env.get("id"), str) or not env["id"]:
        return None, "missing or invalid 'id' (must be non-empty string)"
    if not isinstance(env.get("cmd"), str) or not env["cmd"]:
        return None, "missing or invalid 'cmd' (must be non-empty string)"
    return env, None


def cmd_version(env):
    emit_delta(env["id"], HAPPI_VERSION)
    emit_completed(env["id"])
    return 0


def cmd_echo(env):
    args = env.get("args", [])
    if not isinstance(args, list):
        emit_error(env["id"], "parse_error", "echo: 'args' must be an array")
        return 1
    for a in args:
        emit_delta(env["id"], str(a))
    emit_completed(env["id"])
    return 0


_SPEC_LINES = [
    "HAPPI/1.5 - Harnessed-AI Polyglot Protocol Interface",
    "One JSON envelope in (stdin), one NDJSON event stream out (stdout).",
    "Envelope: {v, id, cmd, args?, flags?, auth?}",
    "Events: started -> (delta|tool_call|tool_result|sub_request)* -> completed|error -> [idr]",
    "v1.3: cmd compose dispatches child envelopes through this same runtime (axiom 3), emitting sub_request per child",
    "v1.1: flags.audit=true emits idr terminator with sha256 of envelope+events",
    "v1.1: cmd context.append emits a context terminator content-addressing a decision body (excl id/ts/audit)",
    "v1.3: flags.validate_decision=true on context.append validates the body decision slot against the shared Decision schema (tiered: deterministic minimal, delegated externally-checkable; SMT surface-graft rejection)",
    "v1.3: cmd cite.verify deterministically checks each citation quote is verbatim in its source (un-fakeable provenance floor)",
    "v1.4: cmd continuity.append emits a continuity event pre-registering a falsifiable commitment (claim+falsifier) BEFORE the action it governs, or settling one with a verdict",
    "v1.5: cmd reasoning.append emits a reasoning event carrying a trace as a DIAGNOSTIC SPECIMEN — authoritative is stamped false and role diagnostic, and a record claiming authority or naming settles is refused before it reaches the stream",
    "Cmds (this runtime): version, cite.verify, compose, echo, spec.describe, envelope.validate, idr.emit, context.append, continuity.append, reasoning.append, pr.reference, hypothesis.register, quine.spawn",
    "Reference: happi.md  Watermark: V>>--<<V",
]


def cmd_spec_describe(env):
    for line in _SPEC_LINES:
        emit_delta(env["id"], line)
    emit_completed(env["id"])
    return 0


def cmd_envelope_validate(env):
    """Trivial pass: parse_envelope already validated to reach this dispatch."""
    emit_delta(env["id"], "envelope OK: passed schema validation")
    emit_completed(env["id"], {"validated": True})
    return 0


def cmd_idr_emit(env):
    """Compute IDR for a recorded (envelope, event-stream) pair (v1.1).

    args[0] = envelope JSON string; args[1] = NDJSON event stream string.
    Optional: flags.model_versions (list[str]), flags.block_anchor (int|null).
    """
    args = env.get("args", [])
    if len(args) < 2 or not all(isinstance(a, str) for a in args[:2]):
        emit_error(env["id"], "parse_error",
                   "idr.emit: requires args=[envelope_json, ndjson_events]")
        return 1
    envelope_str, events_str = args[0], args[1]
    h = hashlib.sha256()
    h.update(envelope_str.encode("utf-8"))
    h.update(events_str.encode("utf-8"))
    flags = env.get("flags") or {}
    idr_ref = {
        "sha256": h.hexdigest(),
        "cid": None,
        "model_versions": flags.get("model_versions") or [],
        "block_anchor": flags.get("block_anchor"),
    }
    _emit({"v": HAPPI_VERSION, "id": env["id"], "type": "idr",
           "ts": _ts_ms(), "idr_ref": idr_ref})
    emit_completed(env["id"])
    return 0


def cmd_context_append(env):
    """Emit one context event content-addressing a decision body (v1.1).

    args[0] = decision body JSON string (the memory delta/snapshot/supersede).
    Optional flags: predecessor_context (str|null), snapshot_ref (str|null),
    kind (str, default "context-delta"), model_versions (list[str]). The content
    address excludes the volatile id/ts/audit fields and is byte-identical to
    GRIP lib/precog/idr.py::content_addr for the same body.
    """
    body = _load_body_arg(env, "context.append", "decision body")
    if body is None:
        return 1
    flags = env.get("flags") or {}
    # v1.3 opt-in (default OFF = byte-identical back-compat): validate the body's
    # "decision" slot against the ONE shared Decision schema before addressing.
    if flags.get("validate_decision"):
        err = _validate_decision_body(body)
        if err is not None:
            emit_error(env["id"], "parse_error", err)
            return 1
    # Outcome first: context is a terminator emitted AFTER completed/error --
    # the reverse is falsification clause (b) above. Mirrors _emit_terminators.
    emit_completed(env["id"], {"content_addr": _content_addr(body)})
    emit_context(env["id"], body, flags)
    return 0


def cmd_continuity_append(env):
    """Emit one continuity event content-addressing a commitment body (v1.4).

    args[0] = commitment body JSON string. For a commit that is the premise
    being pre-registered ({claim, falsifier, deps, severity, ...}); for a
    verdict, the settlement body ({verdict, axis, evidence, verifier, ...}).
    Optional flags: kind (default "continuity-commit"), predecessor_continuity,
    settles, model_versions. The content address excludes the volatile
    id/ts/audit fields and is byte-identical to GRIP
    lib/precog/idr.py::content_addr for the same body.

    Two fail-closed checks, because a malformed record is worse than none: an
    unrecognised `kind` is rejected rather than written through, and a verdict
    MUST name the commitment it settles — a settlement that settles nothing
    cannot be joined back to its premise and is therefore unverifiable.
    """
    body = _load_body_arg(env, "continuity.append", "commitment body")
    if body is None:
        return 1
    flags = env.get("flags") or {}
    kind = flags.get("kind") or "continuity-commit"
    if kind not in _CONTINUITY_KINDS:
        emit_error(env["id"], "parse_error",
                   "continuity.append: kind must be one of " +
                   ", ".join(_CONTINUITY_KINDS) + ", got " + repr(kind))
        return 1
    if kind == "continuity-verdict" and not flags.get("settles"):
        emit_error(env["id"], "parse_error",
                   "continuity.append: a continuity-verdict must set flags.settles "
                   "to the content address of the commitment it settles")
        return 1
    # Commitment FIRST, outcome second — the inverse of the idr/context
    # terminators, and the ordering IS the pre-registration guarantee.
    emit_continuity(env["id"], body, flags)
    emit_completed(env["id"], {"content_addr": _content_addr(body), "kind": kind})
    return 0


def cmd_reasoning_append(env):
    """Emit one reasoning event content-addressing a trace body (v1.5).

    args[0] = trace body JSON string. Optional flags: precedes, model_versions,
    signature.

    THE FENCE — both guards reject BEFORE any emit, mirroring
    cmd_continuity_append, because a record refused after reaching the stream
    has not been refused. A caller claiming `authoritative` is rejected: the
    field exists on the wire so a reader can SEE it is false, and is not a
    setting — a trace able to assert its own authority is precisely the failure
    this design fences against. `settles` is rejected too: settling is what a
    continuity-verdict does to a pre-registered commitment, a trace settles
    nothing, and allowing the field would let a specimen pose as a settlement.

    Deliberately NOT enforced here: refusing a continuity-verdict whose
    `settles` names a REASONING address needs an address resolved back to the
    record bearing it. This runtime holds no record store and reads one
    envelope, so it cannot decide that without inventing state. That check
    belongs to a stateful reader holding the chain; asserting it here would be a
    guard that cannot fire, which reads as protection that is not there.
    """
    body = _load_body_arg(env, "reasoning.append", "trace body")
    if body is None:
        return 1
    flags = env.get("flags") or {}
    if flags.get("authoritative"):
        emit_error(env["id"], "parse_error",
                   "reasoning.append: a reasoning trace is a diagnostic specimen, "
                   "never evidence of cause — flags.authoritative must be false")
        return 1
    if flags.get("settles"):
        emit_error(env["id"], "parse_error",
                   "reasoning.append: a reasoning record cannot settle anything — "
                   "flags.settles belongs to a continuity-verdict")
        return 1
    emit_reasoning(env["id"], body, flags)
    emit_completed(env["id"], {"content_addr": _content_addr(body),
                               "role": _REASONING_ROLE})
    return 0


def cmd_pr_reference(env):
    """Record a pull request as part of a HAPPI seed.

    Informational dispatch — no side effects. Useful for #233-style fractal
    seed envelopes whose queue lists contributing PRs.

    Required: flags.pr (int), flags.repo ("org/repo" str)
    Optional: args[0] (description str), flags.status, flags.contributes_axiom
    """
    args = env.get("args", [])
    flags = env.get("flags") or {}
    pr = flags.get("pr")
    repo = flags.get("repo")
    if not isinstance(pr, int):
        emit_error(env["id"], "parse_error",
                   "pr.reference: requires flags.pr (int)")
        return 1
    if not isinstance(repo, str) or not repo:
        emit_error(env["id"], "parse_error",
                   "pr.reference: requires flags.repo (str)")
        return 1
    description = args[0] if args and isinstance(args[0], str) else ""
    parts = ["PR #" + str(pr) + " (" + repo + ")"]
    if description:
        parts.append(description)
    if flags.get("status"):
        parts.append("status=" + str(flags["status"]))
    if flags.get("contributes_axiom"):
        parts.append("axiom=" + str(flags["contributes_axiom"]))
    emit_delta(env["id"], " | ".join(parts))
    emit_completed(env["id"], {"pr": pr, "repo": repo})
    return 0


def _hyp_validate(env):
    """Validate hypothesis envelope; return (hyp_id, flags) on success, None on error."""
    args = env.get("args", [])
    flags = env.get("flags") or {}
    if not args or not isinstance(args[0], str) or not args[0]:
        emit_error(env["id"], "parse_error",
                   "hypothesis.register: requires args[0] = hypothesis ID")
        return None
    required = ("claim", "metric", "prediction", "deadline")
    missing = [k for k in required
               if not isinstance(flags.get(k), str) or not flags[k]]
    if missing:
        emit_error(env["id"], "parse_error",
                   "hypothesis.register: missing flags: " + ",".join(missing))
        return None
    return args[0], flags


def _hyp_write(record):
    """Append one NDJSON record to the hypotheses log; return resolved out_path."""
    default = os.path.expanduser("~/.hal/data/hypotheses.jsonl")
    out_path = os.environ.get("HAL_HYPOTHESES_PATH", default)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
    return out_path


def cmd_hypothesis_register(env):
    """Register a falsifiable hypothesis to data/hypotheses.jsonl (NDJSON, append-only).

    Stdlib-only — preserves HAPPI self-bootstrap. Override path via HAL_HYPOTHESES_PATH.
    Required: args[0] (hypothesis ID), flags.claim, flags.metric, flags.prediction, flags.deadline.
    """
    validated = _hyp_validate(env)
    if validated is None:
        return 1
    hyp_id, flags = validated
    record = {
        "v": HAPPI_VERSION, "id": hyp_id,
        "claim": flags["claim"], "metric": flags["metric"],
        "prediction": flags["prediction"], "deadline": flags["deadline"],
        "registered_at_unix": int(time.time()), "status": "pending",
    }
    try:
        out_path = _hyp_write(record)
    except OSError as e:
        emit_error(env["id"], "runtime_error",
                   "hypothesis.register: cannot write — " + str(e))
        return 1
    emit_delta(env["id"], "registered " + hyp_id + " | claim: " + flags["claim"][:80])
    emit_delta(env["id"], "deadline " + flags["deadline"] + " | logged to " + out_path)
    emit_completed(env["id"], {"hypothesis_id": hyp_id, "path": out_path})
    return 0


class _QuineError(Exception):
    """Internal signal for quine.spawn helpers to surface a runtime_error event."""


def _fetch_parent_issue(repo, issue_num):
    """Fetch parent title and body via gh CLI. Raises _QuineError on failure."""
    cmd = ["gh", "issue", "view", str(issue_num), "--repo", repo,
           "--json", "title,body"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise _QuineError("gh CLI unavailable: " + type(e).__name__)
    if result.returncode != 0:
        raise _QuineError("gh issue view failed: " + result.stderr.strip()[:200])
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise _QuineError("gh returned non-JSON: " + str(e))
    return data["title"], data["body"]


def _check_and_bump(title, body, depth_limit, parent_issue):
    """Validate parent is a HAPPI seed and bump generation counter in title."""
    if "quine.spawn" not in body:
        raise _QuineError("parent #" + str(parent_issue) + " is not a HAPPI seed (no quine.spawn envelope)")
    m = re.search(r"generation\s+(\d+)", title, re.IGNORECASE)
    if not m:
        raise _QuineError("parent title lacks 'generation N' counter")
    current = int(m.group(1))
    if current >= depth_limit:
        raise _QuineError("generation " + str(current)
                          + " >= depth_limit " + str(depth_limit))
    new_title = re.sub(r"(generation\s+)(\d+)",
                       lambda x: x.group(1) + str(current + 1),
                       title, count=1, flags=re.IGNORECASE)
    return new_title, current + 1


def _spawn_dry_run(env, parent, new_title, body, next_gen, depth_limit):
    """Emit DRY-RUN events without creating any GitHub issue."""
    emit_delta(env["id"], "DRY-RUN: would spawn from #" + str(parent))
    emit_delta(env["id"], "proposed_title: " + new_title)
    emit_delta(env["id"], "body_preview: " + body[:200].replace("\n", " "))
    emit_delta(env["id"], "generation: " + str(next_gen))
    emit_delta(env["id"], "depth_limit: " + str(depth_limit)
                          + " (set flags.live=true to actually create)")
    emit_completed(env["id"], {"dry_run": True, "next_generation": next_gen})
    return 0


def _create_child_issue(repo, title, body):
    """Create child issue via gh CLI. Returns child issue number, raises on failure."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md",
                                      delete=False, encoding="utf-8") as f:
        f.write(body)
        body_file = f.name
    try:
        cmd = ["gh", "issue", "create", "--repo", repo,
               "--title", title, "--body-file", body_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise _QuineError("gh issue create failed: "
                              + result.stderr.strip()[:200])
        m = re.search(r"/issues/(\d+)", result.stdout)
        if not m:
            raise _QuineError("could not parse issue # from: "
                              + result.stdout[:200])
        return int(m.group(1))
    finally:
        os.unlink(body_file)


def _append_quine_audit(parent, child, generation, repo):
    """Append a shared audit record to data/quine-spawn-audit.jsonl. Returns path."""
    audit_path = os.path.expanduser("~/.hal/data/quine-spawn-audit.jsonl")
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    record = {
        "v": HAPPI_VERSION,
        "parent_issue": parent, "child_issue": child,
        "generation": generation, "repo": repo,
        "ts_unix": int(time.time()),
    }
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
    return audit_path


def _spawn_live(env, parent, repo, new_title, body, next_gen):
    """LIVE branch: create child issue + append audit. Emits live events."""
    try:
        child = _create_child_issue(repo, new_title, body)
    except _QuineError as e:
        emit_error(env["id"], "runtime_error", "quine.spawn LIVE: " + str(e))
        return 1
    audit_path = _append_quine_audit(parent, child, next_gen, repo)
    emit_delta(env["id"], "LIVE: spawned child #" + str(child)
                          + " from #" + str(parent))
    emit_delta(env["id"], "title: " + new_title)
    emit_delta(env["id"], "generation: " + str(next_gen))
    emit_delta(env["id"], "audit: " + audit_path)
    emit_completed(env["id"], {"parent": parent, "child": child,
                               "generation": next_gen, "audit": audit_path})
    return 0


def cmd_quine_spawn(env):
    """Spawn a child HAPPI seed issue from a parent (#233 fractal pattern).

    Required: flags.parent_issue (int), flags.repo ("org/repo")
    Optional: flags.live (default false — DRY-RUN unless explicit),
              flags.depth_limit (default 16; HAPPI_QUINE_DEPTH_MAX env override)

    DRY-RUN emits proposed transform without creating any GitHub artefact.
    LIVE creates child issue via gh CLI and appends to shared audit log.

    Dependency: requires `gh` CLI installed and authenticated for LIVE mode.
    """
    flags = env.get("flags") or {}
    parent_issue = flags.get("parent_issue")
    repo = flags.get("repo")
    if not isinstance(parent_issue, int):
        emit_error(env["id"], "parse_error",
                   "quine.spawn: requires flags.parent_issue (int)")
        return 1
    if not isinstance(repo, str) or not repo:
        emit_error(env["id"], "parse_error",
                   "quine.spawn: requires flags.repo (str)")
        return 1
    default_depth = int(os.environ.get("HAPPI_QUINE_DEPTH_MAX", "16"))
    depth_limit = flags.get("depth_limit", default_depth)
    try:
        title, body = _fetch_parent_issue(repo, parent_issue)
        new_title, next_gen = _check_and_bump(title, body, depth_limit, parent_issue)
    except _QuineError as e:
        emit_error(env["id"], "runtime_error", "quine.spawn: " + str(e))
        return 1
    if not bool(flags.get("live", False)):
        return _spawn_dry_run(env, parent_issue, new_title, body, next_gen, depth_limit)
    return _spawn_live(env, parent_issue, repo, new_title, body, next_gen)


# ---------------------------------------------------------------------------
# cite.verify (v1.3) — deterministic citation-provenance floor, any harness.
#
# Re-implements the GRIP prove_it verify ladder INLINE (stdlib only) so happi.md
# stays self-bootstrapping: a cited quote is verbatim-present in its source, or it
# is not. The same exact -> whitespace+typographic-flexible -> not_found ladder as
# GRIP lib/prove_it.py::verify_quote, so the two runtimes agree on the same
# (quote, source). A fabricated citation can never verify — that is the guarantee.
# ---------------------------------------------------------------------------

_CV_WS = re.compile(r"\s+")
# Length-preserving typographic normalisation (single-char -> single-char), so a
# match found in normalised text still indexes the ORIGINAL source verbatim.
_CV_TYPO = {
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-", 0x2015: "-",
    0x2018: "'", 0x2019: "'", 0x201B: "'",
    0x201C: '"', 0x201D: '"', 0x201F: '"',
    0x00A0: " ", 0x2007: " ", 0x2009: " ", 0x202F: " ",
}


def _cv_typo(s):
    return s.translate(_CV_TYPO)


def _cv_verify(quote, source_text):
    """(status, start, end) for `quote` in `source_text`. Deterministic ladder:
    exact substring -> whitespace+typographic-flexible -> not_found. Offsets index
    the original source verbatim (normalisation is length-preserving)."""
    q = (quote or "").strip()
    if not q:
        return "not_found", -1, -1
    idx = source_text.find(q)
    if idx != -1:
        return "verified", idx, idx + len(q)
    toks = [re.escape(t) for t in _CV_WS.split(_cv_typo(q)) if t]
    if not toks:
        return "not_found", -1, -1
    m = re.compile(r"\s+".join(toks)).search(_cv_typo(source_text))
    if m:
        return "fuzzy", m.start(), m.end()
    return "not_found", -1, -1


def _cv_index_sources(sources):
    """Map source id -> text. Returns (by_id, None) or (None, error_message)."""
    by_id = {}
    for s in sources:
        if (not isinstance(s, dict) or not isinstance(s.get("id"), str)
                or not isinstance(s.get("text"), str)):
            return None, "cite.verify: each source needs string id and text"
        by_id[s["id"]] = s["text"]
    return by_id, None


def _cv_citation_ok(c):
    """True iff citation carries the required string fields id, source_id, quote."""
    return (isinstance(c, dict) and isinstance(c.get("id"), str)
            and isinstance(c.get("source_id"), str) and isinstance(c.get("quote"), str))


def _cv_process(env, citations, by_id):
    """Verify each citation against its source, streaming a delta per citation.
    Returns (results, tally), or None after emitting a parse_error for a malformed
    citation (caller then exits non-zero)."""
    tally = {"verified": 0, "fuzzy": 0, "not_found": 0}
    results = []
    for c in citations:
        if not _cv_citation_ok(c):
            emit_error(env["id"], "parse_error",
                       "cite.verify: each citation needs string id, source_id, quote")
            return None
        src = by_id.get(c["source_id"])
        status, start, end = _cv_verify(c["quote"], src) if src is not None \
            else ("not_found", -1, -1)
        tally[status] += 1
        results.append({"id": c["id"], "source_id": c["source_id"],
                        "status": status, "start": start, "end": end})
        emit_delta(env["id"], c["id"] + " " + status)
    return results, tally


def cmd_cite_verify(env):
    """Deterministically verify that each citation's quote is verbatim-present in
    its cited source (v1.3). The un-fakeable provenance floor at protocol level,
    so ANY AI on ANY harness that speaks HAPPI gets the same guarantee.

    flags.sources   = [{"id": str, "text": str}, ...]                    (required)
    flags.citations = [{"id": str, "source_id": str, "quote": str}, ...] (required)
    flags.strict    = bool (default false). When true, ANY not_found citation makes
                      the runtime emit `error` (exit non-zero) — a gate any harness
                      can fail a build on. Default emits `completed` with the
                      provenance record; the caller reads grounding_rate and decides.

    completed.usage carries the provenance record: per-source sha256+chars,
    per-citation status+offsets, tally, grounding_rate. Same shape as GRIP
    lib/prove_it.py::provenance, so the two engines interoperate.
    """
    flags = env.get("flags") or {}
    sources = flags.get("sources")
    citations = flags.get("citations")
    if not isinstance(sources, list) or not isinstance(citations, list):
        emit_error(env["id"], "parse_error",
                   "cite.verify: requires flags.sources[] and flags.citations[]")
        return 1
    by_id, err = _cv_index_sources(sources)
    if err:
        emit_error(env["id"], "parse_error", err)
        return 1
    processed = _cv_process(env, citations, by_id)
    if processed is None:
        return 1
    results, tally = processed
    grounded = tally["verified"] + tally["fuzzy"]
    record = {
        "sources": {sid: {"sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
                          "chars": len(txt)} for sid, txt in by_id.items()},
        "citations": results,
        "tally": tally,
        "grounding_rate": round(grounded / max(len(citations), 1), 3),
    }
    if bool(flags.get("strict", False)) and tally["not_found"] > 0:
        emit_error(env["id"], "runtime_error",
                   str(tally["not_found"]) + " citation(s) NOT verbatim in source — unproven")
        return 1
    emit_completed(env["id"], record)
    return 0


def _compose_guard(children):
    """Pre-dispatch checks for `compose`, extracted from `cmd_compose` so each
    concern is independently readable (KISS -- the coordinator below just
    reads as guard -> loop -> complete). Returns an `(code, message)` pair to
    emit as `error` when `compose` must refuse before touching any child, or
    `None` when it is safe to proceed.
    """
    if not isinstance(children, list):
        return ("parse_error",
                "compose: 'flags.envelopes' must be an array of envelopes")
    if _SUB_DEPTH >= MAX_SUB_REQUEST_DEPTH:
        return ("sub_request_depth_cap",
                "sub_request depth cap reached (" +
                str(MAX_SUB_REQUEST_DEPTH) + "); refusing to recurse further")
    return None


def _compose_dispatch_one(env, child):
    """Validate and dispatch ONE `compose` child. Returns `(rc, dispatched, fatal)`.

    `fatal=True` means the CHILD ITSELF would not parse -- a defect in the
    parent's own request payload, exactly like `flags.envelopes` not being a
    list, so the caller terminates the parent with a single `error` and no
    `completed`. `fatal=False` once the child has been dispatched, whatever
    its OWN outcome (it reports on its own id and the parent still
    completes) -- keeping these apart is what preserves the invariant that
    every envelope id in a stream terminates exactly once.
    """
    global _SUB_DEPTH
    if not isinstance(child, dict):
        emit_error(env["id"], "parse_error",
                   "compose: child must be a JSON object, got " +
                   type(child).__name__)
        return 1, 0, True
    child_env, child_err = parse_envelope(json.dumps(child))
    if child_env is None:
        emit_error(env["id"], "parse_error",
                   "compose: invalid child envelope: " + str(child_err))
        return 1, 0, True
    emit_sub_request(env["id"], child_env)
    _SUB_DEPTH += 1
    try:
        child_rc = dispatch(child_env)
    finally:
        _SUB_DEPTH -= 1
    return child_rc, 1, False


def cmd_compose(env):
    """Dispatch every child envelope in `flags.envelopes` through THIS runtime.

    Axiom 3 — "`sub_request` recurses through the same runtime. The protocol is
    fractal; no privileged inner interface."

    That axiom is a claim about the CODE PATH, so this handler is written to make
    it checkable rather than merely asserted:

      * a child is validated by `parse_envelope`, the same validator that
        accepted the envelope arriving on stdin;
      * a child is dispatched by `dispatch`, the same function `main` calls.

    There is deliberately no child-only parser, no child-only handler table and
    no relaxed validation. A child that is malformed, carries an out-of-range
    version, or names an unknown cmd fails exactly as the same bytes would have
    failed on stdin. Consequently a `compose` stream carries one outcome event
    per envelope id, not one per stream.
    """
    children = (env.get("flags") or {}).get("envelopes")
    guard = _compose_guard(children)
    if guard is not None:
        emit_error(env["id"], guard[0], guard[1])
        return 1

    rc = 0
    dispatched = 0
    for child in children:
        child_rc, child_dispatched, fatal = _compose_dispatch_one(env, child)
        if fatal:
            return child_rc
        dispatched += child_dispatched
        if child_rc != 0:
            rc = child_rc
    emit_completed(env["id"], {"children": dispatched})
    return rc


CMDS = {
    "version": cmd_version,
    "compose": cmd_compose,
    "cite.verify": cmd_cite_verify,
    "echo": cmd_echo,
    "spec.describe": cmd_spec_describe,
    "envelope.validate": cmd_envelope_validate,
    "idr.emit": cmd_idr_emit,
    "context.append": cmd_context_append,
    "continuity.append": cmd_continuity_append,
    "reasoning.append": cmd_reasoning_append,
    "pr.reference": cmd_pr_reference,
    "hypothesis.register": cmd_hypothesis_register,
    "quine.spawn": cmd_quine_spawn,
}


def _emit_terminators(env, raw):
    """Emit the opt-in idr audit terminator after completed/error (at most once,
    when flags.audit=true). The v1.1 context event is emitted via the explicit
    `context.append` cmd, NOT a flags-driven auto-terminator: a generic dispatch
    envelope has no well-defined decision body, and content-addressing the whole
    envelope would fold dispatch flags (model_versions, predecessor_context) into
    the address and break the dedup coordinate. A richer runtime that carries a
    designated decision body MAY add a flags.context auto-path (HAL #429)."""
    if _AUDIT_ENABLED:
        emit_idr(env["id"], raw)


def dispatch(env):
    """Dispatch ONE parsed envelope: started -> handler -> completed|error.

    The single entry point for every envelope this runtime serves, whichever
    direction it arrived from: `main` calls it for the envelope read from stdin,
    and `cmd_compose` calls it for each child envelope. That shared call is what
    makes axiom 3 true rather than merely stated — there is no second, inner
    dispatcher a child could be routed to, so "recurses through the same
    runtime" is a property of the code and not of the documentation.

    Terminators are deliberately NOT emitted here. The `idr` audit receipt
    hashes the raw stdin bytes, which exist only at the top level; a child has
    no stdin of its own. That is a difference in available INPUT, not a
    privileged interface — every child gets the same validation, the same
    handler table and the same error contract.

    Returns the handler's exit code (0 on success).
    """
    handler = CMDS.get(env["cmd"])
    emit_started(env["id"])
    if handler is None:
        emit_error(env["id"], "unsupported_cmd",
                   "cmd " + repr(env["cmd"]) + " not supported by this runtime")
        return 1
    try:
        return handler(env)
    except Exception as e:
        emit_error(env["id"], "runtime_error",
                   type(e).__name__ + ": " + str(e))
        return 1


def main():
    global _AUDIT_ENABLED, _AUDIT_MODEL_VERSIONS
    raw = sys.stdin.read()
    env, err = parse_envelope(raw)
    if env is None:
        emit_error("req-invalid", "parse_error", err)
        return 1

    flags = env.get("flags") or {}
    _AUDIT_ENABLED = bool(flags.get("audit", False))
    _AUDIT_MODEL_VERSIONS = list(flags.get("model_versions") or [])

    rc = dispatch(env)
    _emit_terminators(env, raw)
    return rc


if __name__ == "__main__":
    sys.exit(main())
PYTHON_EOF
)"

# ----- Python resolver (python.org on Windows may only provide python/py) ----
resolve_python() {
  local candidate
  HAPPI_PYTHON=()
  for candidate in python3 python; do
    if "$candidate" -c 'import sys; sys.exit(sys.version_info[0] != 3)' >/dev/null 2>&1; then
      HAPPI_PYTHON=("$candidate")
      break
    fi
  done
  if (( ${#HAPPI_PYTHON[@]} == 0 )); then
    if py -3 -c 'import sys; sys.exit(sys.version_info[0] != 3)' >/dev/null 2>&1; then
      HAPPI_PYTHON=(py -3)
    else
      printf 'Python 3 interpreter not found (tried python3, python, py -3)\n' >&2
      exit 1
    fi
  fi
}

# ----- cmd handlers ---------------------------------------------------------

cmd_identity() {
  local size sha
  size="$(wc -c < "$HAPPI_FILE" 2>/dev/null | tr -d ' ' || echo unknown)"
  sha="$(shasum -a 256 "$HAPPI_FILE" 2>/dev/null | cut -d' ' -f1 || echo unavailable)"
  cat <<EOF
happi.md / $HAPPI_VERSION — V>>--<<V
Harnessed-AI Polyglot Protocol Interface — canonical reference

file:   $HAPPI_FILE
size:   $size bytes
sha256: $sha

Cmds:
  bash happi.md                  identity (this banner)
  bash happi.md morning          morning-boot routine (subsumed)
  bash happi.md run              exec embedded runtime; reads envelope from stdin
  bash happi.md install          symlink $HAPPI_INSTALL_DIR/happi -> $HAPPI_FILE
  bash happi.md extract <layer>  dump markdown|bash|python|envelope|openapi
  bash happi.md spec.describe    recursive dogfood
  bash happi.md help             alias for identity (also -h, --help)

Embedded runtime cmds (stdlib-only):
  version              emit runtime's protocol version as a delta
  echo                 echo args back as deltas
  spec.describe        emit spec summary
  envelope.validate    confirm envelope passed schema validation
  idr.emit       v1.1  emit idr event for recorded (envelope, ndjson_events)
  context.append v1.1  emit context event content-addressing a decision body
  continuity.append v1.4  pre-register a falsifiable commitment, or settle one
  pr.reference   v1.1  reference a PR (informational; flags.pr, flags.repo)
  hypothesis.register  v1.1  append falsifiable hypothesis to NDJSON log
  quine.spawn    v1.1  spawn child seed issue (DRY-RUN unless flags.live=true)
  cite.verify    v1.3  verify each citation quote is verbatim in its source

Try:
  echo '{"v":"happi/1.0","id":"hello","cmd":"version"}' | bash happi.md run

For the canonical spec, see the Markdown body of this file.
EOF
}

# ----- cmd_run: extract embedded Python runtime + exec ----------------------
cmd_run() {
  resolve_python || return 1
  local runtime_py="$HAPPI_TMPDIR/happi-runtime-$$.py"
  printf '%s\n' "$HAPPI_PY" > "$runtime_py"
  exec "${HAPPI_PYTHON[@]}" "$runtime_py"
}

# ----- cmd_install: symlink to ~/.local/bin/happi ---------------------------
cmd_install() {
  mkdir -p "$HAPPI_INSTALL_DIR"
  local target="$HAPPI_INSTALL_DIR/happi"
  local source_path
  if command -v realpath >/dev/null 2>&1; then
    source_path="$(realpath "$HAPPI_FILE")"
  else
    source_path="$(cd "$(dirname "$HAPPI_FILE")" && pwd)/$(basename "$HAPPI_FILE")"
  fi
  if [[ -L "$target" || -e "$target" ]]; then
    printf 'note: %s exists; replacing\n' "$target" >&2
    rm -f "$target"
  fi
  ln -s "$source_path" "$target"
  printf 'installed: %s -> %s\n' "$target" "$source_path"
  if [[ ":$PATH:" != *":$HAPPI_INSTALL_DIR:"* ]]; then
    printf '\nnote: %s is NOT on $PATH; add to your shell rc:\n' "$HAPPI_INSTALL_DIR" >&2
    printf '  export PATH="%s:$PATH"\n' "$HAPPI_INSTALL_DIR" >&2
  fi
}

# ----- cmd_extract: dump one polyglot layer ---------------------------------
cmd_extract() {
  local layer="${1:-}"
  case "$layer" in
    markdown)
      sed -n "/^: <<'HAPPI_DOC'\$/,/^HAPPI_DOC\$/p" "$HAPPI_FILE" | sed '1d;$d'
      ;;
    python)
      sed -n "/cat <<'PYTHON_EOF'\$/,/^PYTHON_EOF\$/p" "$HAPPI_FILE" | sed '1d;$d'
      ;;
    bash)
      # Bash sections: file MINUS the HAPPI_DOC body MINUS the PYTHON_EOF body.
      awk '
        /^: <<'\''HAPPI_DOC'\''$/ { print; in_doc=1; next }
        /^HAPPI_DOC$/ && in_doc { print; in_doc=0; next }
        in_doc { next }
        /cat <<'\''PYTHON_EOF'\''$/ { print; in_py=1; next }
        /^PYTHON_EOF$/ && in_py { print; in_py=0; next }
        in_py { next }
        { print }
      ' "$HAPPI_FILE"
      ;;
    envelope)
      sed -n '/<!-- happi:label=envelope -->/,/<!-- happi:label=envelope:end -->/p' "$HAPPI_FILE" \
        | sed -n '/^```json$/,/^```$/p' \
        | sed '1d;$d'
      ;;
    openapi)
      sed -n '/<!-- happi:label=openapi -->/,/<!-- happi:label=openapi:end -->/p' "$HAPPI_FILE" \
        | sed -n '/^```yaml$/,/^```$/p' \
        | sed '1d;$d'
      ;;
    "")
      printf 'usage: bash happi.md extract <layer>\n' >&2
      printf 'layers: markdown | bash | python | envelope | openapi\n' >&2
      return 1
      ;;
    *)
      printf 'unknown layer: %s\n' "$layer" >&2
      printf 'layers: markdown | bash | python | envelope | openapi\n' >&2
      return 1
      ;;
  esac
}

# ----- cmd_spec_describe: recursive dogfood --------------------------------
cmd_spec_describe() {
  printf '{"v":"happi/1.0","id":"spec-describe-%d","cmd":"spec.describe"}\n' "$$" \
    | cmd_run
}

# ----- cmd_morning: subsumed boot routine (verbatim semantics from prior --
# ----- happi.md, lines 95-360 of the pre-canonical backup) -----------------
cmd_morning() {
  resolve_python || return 1

HAL_ROOT="${HAL_ROOT:-$HOME/.hal}"
CLAUDE_DIR="${CLAUDE_DIR:-$HOME/.claude}"
CAPSULE_DIR="$HAL_ROOT/state/witness-capsules"
MARKER_DIR="$HAL_ROOT/state/boot-markers"
LOG_DIR="$HAL_ROOT/logs/happi-boot"
mkdir -p "$MARKER_DIR" "$LOG_DIR"

TODAY="$(date +%Y-%m-%d)"
MARKER="$MARKER_DIR/$TODAY.json"
LOG_FILE="$LOG_DIR/$(date +%Y%m%dT%H%M%S).log"

FORCE=0; DRY=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --dry) DRY=1 ;;
  esac
done

# ------- colour output (no external deps) -----------------------------------
if [[ -t 1 ]]; then
  G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'
  B=$'\033[34m'; D=$'\033[2m'; Z=$'\033[0m'; BL=$'\033[1m'
else
  G=""; R=""; Y=""; B=""; D=""; Z=""; BL=""
fi

log() { printf '%s\n' "$*" | tee -a "$LOG_FILE" >&2; }

# ------- stage 0: bootstrap check (fresh machine or existing install) -------
"${HAPPI_PYTHON[@]}" - "$HAL_ROOT" "$DRY" <<'PY'
import sys, pathlib
hal_root = pathlib.Path(sys.argv[1])
dry_run  = sys.argv[2] == "1"

if hal_root.is_dir():
    sys.path.insert(0, str(hal_root))
    try:
        from lib.hal.happi_boot import bootstrap_hal
        result = bootstrap_hal(hal_root, dry_run=dry_run)
        print(result.summary())
        sys.exit(0 if result.ok else 1)
    except ImportError:
        sys.exit(0)  # mid-install, carry on
else:
    import shutil, subprocess
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 11):
        print(f"!! Python >= 3.11 required; found {major}.{minor}")
        sys.exit(1)
    if not shutil.which("git"):
        print("!! git not found — install git then re-run bash happi.md morning")
        sys.exit(1)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/grip-platform/HAL.git", str(hal_root)],
            check=True, capture_output=True, timeout=120,
        )
        print(f"cloned HAL to {hal_root}")
    except Exception as e:
        print(f"!! git clone failed: {e}")
        sys.exit(1)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "anthropic", "rich", "httpx"],
            check=True, capture_output=True, timeout=120,
        )
        print("pip deps installed")
    except Exception as e:
        print(f"!! pip install failed: {e}")
        sys.exit(1)
    sys.exit(0)
PY
if (( $? != 0 )); then
  printf '%s!! bootstrap failed — see output above%s\n' "${R}" "${Z}"
  exit 1
fi

# ------- idempotency gate ---------------------------------------------------
if [[ -f "$MARKER" ]] && (( FORCE == 0 )); then
  printf '%s' "${D}HAPPI: already ran today ($TODAY). --force to re-run.${Z}\n"
  "${HAPPI_PYTHON[@]}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d, indent=2))" "$MARKER" 2>/dev/null || true
  exit 0
fi

printf '\n%sHAPPI%s %sv1.0%s · %s' "${BL}${G}" "${Z}" "${D}" "${Z}" "$(date +%H:%M:%S)"
printf '  %sV>>--<<V%s\n' "${D}" "${Z}"
printf '%s\n' "$(printf '%.0s─' {1..62})"

# ------- stage 1: find latest witness capsule -------------------------------
LATEST_CAPSULE="$(ls -t "$CAPSULE_DIR"/*.md 2>/dev/null | head -1 || true)"
if [[ -z "$LATEST_CAPSULE" ]]; then
  printf '%s!!%s no witness capsule found — was hal-cold-shutdown ever run?\n' "${Y}" "${Z}"
  exit 2
fi
printf '%scapsule%s %s\n' "${D}" "${Z}" "$(basename "$LATEST_CAPSULE")"

DRY_CAPSULE=0
if grep -q '^\*\*Mode:\*\*.*DRY-RUN' "$LATEST_CAPSULE"; then
  DRY_CAPSULE=1
  printf '%s!!%s capsule is a DRY-RUN preview — predictions informational only\n' "${Y}" "${Z}"
fi

# ------- stage 2: verify the four predictions -------------------------------
PASS=0; FAIL=0
declare -a RESULTS=()

record() {
  local status="$1" label="$2" got="$3" expected="$4"
  if [[ "$status" = "pass" ]]; then
    RESULTS+=("${G}✓${Z} $label ${D}(got=$got, expected=$expected)${Z}")
    PASS=$((PASS + 1))
  else
    RESULTS+=("${R}✗${Z} $label ${D}(got=$got, expected=$expected)${Z}")
    FAIL=$((FAIL + 1))
  fi
}

# Prediction 1: TALAppsToRelaunchAtLogin = 0
got="$(defaults -currentHost read com.apple.loginwindow TALAppsToRelaunchAtLogin 2>/dev/null \
        | grep -c 'BundleID' | tr -d ' ' || echo 0)"
[[ "$got" = "0" ]] && record pass "TALApps-relaunch-list-empty" "$got" "0" \
                  || record fail "TALApps-relaunch-list-empty" "$got" "0"

# Prediction 2: .savedState bundles = 0
got="$(find "$HOME/Library/Saved Application State" -maxdepth 1 -name '*.savedState' 2>/dev/null \
        | wc -l | tr -d ' ' || echo 0)"
[[ "$got" = "0" ]] && record pass "saved-state-bundles-empty" "$got" "0" \
                  || record fail "saved-state-bundles-empty" "$got" "0"

# Prediction 3: non-whitelisted LaunchAgents loaded = 0
got="$(launchctl list 2>/dev/null \
        | awk '{print $3}' \
        | grep -cE '^(com\.codetonight|com\.grip\.|com\.grammarly|homebrew\.mxcl\.mongodb|pm2\.)' \
        | tr -d ' ' || echo 0)"
[[ "$got" = "0" ]] && record pass "launchagents-whitelist-only" "$got" "0" \
                  || record fail "launchagents-whitelist-only" "$got" "0"

# Prediction 4: active memory at boot+60s < predicted cap (extract from capsule).
predicted_mem="$(sed -n 's/.*boot + 60 s < \([0-9]\{2,\}\).*/\1/p' "$LATEST_CAPSULE" | head -1)"
if [[ -n "$predicted_mem" ]]; then
  actual_mb="$(vm_stat 2>/dev/null | awk '
    /page size of/ { for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) { bytes=$i; break } }
    /Pages active/ { gsub(/\./,""); for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) { pages=$i; break } }
    END { if (bytes=="" || pages=="") print "?"; else printf "%d\n", pages * bytes / 1048576 }
  ')"
  if [[ "$actual_mb" != "?" ]] && (( actual_mb < predicted_mem )); then
    record pass "active-memory-under-cap" "${actual_mb}MB" "<${predicted_mem}MB"
  else
    record fail "active-memory-under-cap" "${actual_mb}MB" "<${predicted_mem}MB"
  fi
fi

# ------- stage 3: register today's fresh hypotheses -------------------------
HYP_ENGINE="$CLAUDE_DIR/lib/hypothesis_engine.py"
DEADLINE="$("${HAPPI_PYTHON[@]}" -c "from datetime import date, timedelta; print((date.today() + timedelta(days=1)).isoformat())" 2>/dev/null || echo "$TODAY")"

register_hypothesis() {
  local tag="$1" claim="$2" metric="$3" prediction="$4"
  if (( DRY == 1 )) || [[ ! -f "$HYP_ENGINE" ]]; then
    return
  fi
  PYTHONPATH="$CLAUDE_DIR" "${HAPPI_PYTHON[@]}" "$HYP_ENGINE" register \
    --pr 0 --claim "$tag: $claim" \
    --metric "$metric" --prediction "$prediction" \
    --deadline "$DEADLINE" >>"$LOG_FILE" 2>&1 || true
}

warmup_ms="$("${HAPPI_PYTHON[@]}" - <<PY 2>>"$LOG_FILE" || echo -1
import sys, time, pathlib
sys.path.insert(0, "$HAL_ROOT")
t0 = time.time()
try:
    import lib.hal            # noqa: F401  — surface the package
    import lib.hal.registry   # noqa: F401  — module import only, no init
    ok = True
except Exception:
    ok = False
print(int((time.time() - t0) * 1000) if ok else -1)
PY
)"

register_hypothesis "H-BOOT-WARMUP" \
  "lib.hal + lib.hal.registry module-import path completes in <5 s at boot" \
  "warmup_ms" "le:5000"

register_hypothesis "H-BOOT-IDEMPOTENT" \
  "Re-running happi.md morning the same day without --force exits 0 with no writes" \
  "second_run_exit" "eq:0"

polyglot_sha=""
if command -v shasum >/dev/null 2>&1; then
  polyglot_sha="$(shasum -a 256 "$0" | cut -d' ' -f1)"
fi
register_hypothesis "H-BOOT-POLYGLOT" \
  "happi.md remains a valid polyglot (bash + markdown + python + envelope + openapi) after today's session" \
  "polyglot_sha256" "eq:$polyglot_sha"

# ------- stage 4: emit glanceable report ------------------------------------
printf '\n%spredictions%s\n' "${BL}" "${Z}"
for r in "${RESULTS[@]}"; do
  printf '  %b\n' "$r"
done

printf '\n%swarmup%s ' "${BL}" "${Z}"
if [[ "$warmup_ms" = "-1" ]]; then
  printf '%smodule-import failed%s — check %s\n' "${R}" "${Z}" "$LOG_FILE"
elif (( warmup_ms < 5000 )); then
  printf 'hal package + registry import %dms %s[under 5s advisory]%s\n' "$warmup_ms" "${G}" "${Z}"
else
  printf 'hal package + registry import %dms %s[exceeds 5s advisory]%s\n' "$warmup_ms" "${Y}" "${Z}"
fi

if (( DRY_CAPSULE == 1 )); then
  printf '\n%ssummary%s  %s%d pass%s · %s%d fail%s  %s(dry-run capsule — informational)%s\n' \
    "${BL}" "${Z}" "${D}" "$PASS" "${Z}" "${D}" "$FAIL" "${Z}" "${D}" "${Z}"
else
  printf '\n%ssummary%s  %s%d pass%s · %s%d fail%s\n' \
    "${BL}" "${Z}" "${G}" "$PASS" "${Z}" "${R}" "$FAIL" "${Z}"
fi

# ------- stage 5: write idempotency marker ----------------------------------
if (( DRY == 0 )); then
  "${HAPPI_PYTHON[@]}" - "$MARKER" "$TODAY" "$(basename "$LATEST_CAPSULE")" "$PASS" "$FAIL" "$warmup_ms" "$polyglot_sha" <<'PY'
import json, sys
marker, today, capsule, p, f, warmup, sha = sys.argv[1:8]
open(marker, "w").write(json.dumps({
    "date": today,
    "capsule": capsule,
    "pass": int(p),
    "fail": int(f),
    "warmup_ms": int(warmup),
    "polyglot_sha256": sha,
    "watermark": "V>>--<<V",
}, indent=2))
PY
fi

printf '\n%s%s%s  %sV>>--<<V%s\n\n' \
  "$(printf '%.0s─' {1..62})" "" "" "${D}" "${Z}"

if (( FAIL > 0 )); then
  exit 1
fi
exit 0
}

# ----- dispatch -------------------------------------------------------------
cmd="${1:-identity}"

case "$cmd" in
  identity|"")     cmd_identity ;;
  morning)         shift; cmd_morning "$@" ;;
  run)             shift; cmd_run "$@" ;;
  install)         cmd_install ;;
  extract)         shift; cmd_extract "$@" ;;
  spec.describe)   cmd_spec_describe ;;
  -h|--help|help)  cmd_identity ;;
  *)
    printf 'unknown cmd: %s\n' "$cmd" >&2
    printf 'try: bash %s\n' "$HAPPI_FILE" >&2
    exit 1
    ;;
esac
