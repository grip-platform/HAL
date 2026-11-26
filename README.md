# HAL — Harness Abstraction Layer

> **New to HAL?** See the [Getting Started Guide](docs/GETTING-STARTED.md) — zero to productive in 10 minutes.

**The LLM-agnostic substrate. Any model. Any provider. Any domain. Zero limits — sovereign by default.**

HAL is an LLM-agnostic runtime built by [CodeTonight](https://codetonight.co.za). It routes queries across cloud and local LLM providers, manages cost and rate limits transparently, runs cross-LLM councils for critical decisions, and improves itself after every conversation.

HAL works **standalone** (Python-stdlib core, no pip install) or **inside any harness** (Claude Code, Cursor, Gemini CLI, Codex, etc.). When running inside a harness, HAL detects it automatically and integrates gracefully.

## Where HAL sits — the open-core family

HAL is the **model-agnostic substrate layer** of a small free-software family that CodeTonight calls the **HAPPIverse**. The position, in one line: **the parts others adopt (the HAPPI protocol, its SDKs and the GRASP receipt verifier) are Apache-2.0, and the reference infrastructure (GRIP, HAL and GRASP's engine) is AGPL-3.0-only, with commercial licences available**.

| Component | What it is | Licence |
|-----------|------------|---------|
| **GRIP** | The reasoning + provenance engine (critical-thinking + RSI, deterministic safety gates). HAL is the substrate it runs on. | AGPL-3.0-only |
| **HAL** (this repo) | Harness Abstraction Layer — the LLM-agnostic, multi-provider substrate. Any model, any provider, cloud + local, auto-failover, cross-LLM councils. Sovereign / non-Anthropic by default. | AGPL-3.0-only |
| **HAPPI** | The open protocol the family speaks — a single polyglot spec file (`happi.md`), not a black box. | Apache-2.0 |
| **GRASP** | The open proof layer — reference implementation of cryptographic causation, with a public verifier at [grasp.codetonight.co.za](https://grasp.codetonight.co.za). | AGPL-3.0-only (engine), Apache-2.0 (verifier) |

The whole family is free software — clone it, run it, modify it, and share it under its licence. For the AGPL-3.0-only parts, commercial licences (for use outside the AGPL's terms) are available via [licensing@codetonight.co.za](mailto:licensing@codetonight.co.za). "Deeds, not words."

## The moat — cryptographic causation

GRIP + HAL's defensible identity is **cryptographic causation**: tamper-evident, replayable, externally-anchorable records of

1. **what** an AI decided (a signed IDR decision chain),
2. **what it believed when** it decided (a signed memory / context chain), and
3. **every outward claim** backed by deterministic citation-provenance (`cite.verify` / prove-it).

The records are **falsifiable by construction** — a skeptic (regulator, auditor, opposing counsel) can independently *refute or confirm* them; the verifier is the maths (Ed25519 signatures, Merkle roots, content hashes) plus an external party, never the AI itself. "Witness it, do not trust it." Real Bitcoin anchoring is available via OpenTimestamps.

HAL is where this is recorded at the provider boundary: the HAPPI `idr` event (a stream-hash receipt) and the `context` event (the replayable signed memory-chain link) are emitted per invocation when audit is enabled. See the [HAPPI Protocol](#happi-protocol--harnessed-ai-polyglot-protocol-interface) section below.

**Scope-honest:** the cryptographic-causation substrate is roughly **70% realised** — recorded on production decision paths, not on every surface. It is *provable by construction where recorded*, not "always, everywhere, 100%". The authoritative rule (loaded into every HAL session via the shared GRIP rule stack) is `rules/cryptographic-causation.md`.

## Installation

```bash
git clone https://github.com/grip-platform/HAL ~/.hal
export PATH="$HOME/.hal/bin:$PATH"   # add to your .zshrc or .bashrc, one-time
```

Then jump to Quick Start below.

## Quick Start

```bash
# Add to PATH (one-time)
export PATH="$HOME/.hal/bin:$PATH"

# Interactive chat (warms up local models automatically)
hal

# Start web UI + API server (default :3850)
hal --serve

# Chat with a specific model
hal --model ollama/mistral:7b

# Switch model mid-chat in plain English: "use grok-4.3" or "switch to ollama"
# Inside the chat REPL, the natural-language model switch is parsed and routed
# without the operator typing /model commands.

# Open the /settings TUI panel for live provider/model configuration
hal
> /settings

# Tools mode (auto-starts local OpenAI-compatible server if not running)
hal --tools

# Chat in a domain preset
hal --mode legal

# Show providers and health
hal --status
```

## Why HAL — three signals, one architecture

Three independent events in mid-May 2026 sharpen the case for harness-level portability.

**Three signals this week.** An npm supply-chain campaign reached into Claude Code configuration files (Wiz and Socket reports, hundreds of poisoned packages, the source-of-worm open-sourced 11–13 May). Anthropic launched Claude for Legal with a closed plugin and connector stack of its own — the bundle is fast and the bundle is locked. Anthropic also moved Agent SDK use off subscription and onto metered credits, effective 15 June. Three independent events, one architectural reading: lock-in is the risk, portability is the moat.

**HAL is portability by construction.** Provider routing is a single syscall shim driven by `config/providers.yaml`. Fallback chains are tier-scoped (free, cheap, mid, premium) so failures stay inside the operator's cost band before escalating. `lib/hal/sentinel.py` does pre-flight rate-limit detection from response headers so the user never sees a "you've hit your limit" wall. `lib/hal/moe.py` runs a cross-LLM council with named dissent — agreement is reported, disagreement is preserved, not flattened. Switching providers is a one-line config change. Running fully local (Ollama, vLLM, MLX-VLM, LM Studio) requires no cloud accounts at all.

**Sibling projects.** HAL underwrites GRIP (the reasoning + provenance engine) and Mike (Will Chen's OSS document drafter, with Joseph Breda's local-LLM fork). The long-form positioning lives at [about.grip-web.com](https://about.grip-web.com) — §07 HAL, §24 MIKE, §25 AGORA. Same primitives, different audiences.

After *Munir v SSHD* [2026] UKUT 81 — the UK Upper Tribunal ruling on AI services and legal privilege — self-hosted inference moved from sales pitch to practising-certificate concern. HAL was already built for that world.

## Sovereignty — non-Anthropic by default

HAL routes to **non-Anthropic providers by default**. The multi-provider failover
ladder is on out of the box — it used to be opt-in. A request with no explicit model
is served by the sovereign ladder (Groq, then Gemini, DeepSeek, OpenAI, Qwen, and
finally a local Ollama model) with **zero Anthropic egress**. Asking for a Claude
model by name still works the moment you request it (`/model anthropic ...`); it is
simply no longer the default.

This is what "sovereign" means here: the substrate does not depend on any single vendor — and specifically not on Anthropic — to function. Sovereignty is a property of the **HAL substrate**, not of the interactive REPL that may be driving it.

**One config works on every host.** The single committed `config/providers.yaml`
declares each key as `api_key_source: keychain:grip-<provider>`, and resolution adapts
to the environment:

- **macOS** — read straight from the Keychain (`security find-generic-password -s grip-<provider> -w`).
- **Linux / CI / VPS** — fall back to `GRIP_<PROVIDER>_API_KEY`, then `<PROVIDER>_API_KEY`.

So the same file that works on a Mac dev box works unchanged on a Linux server — no
per-host config files, no secrets in the repo. On a server the keys are injected through
the process manager (systemd unit or pm2 ecosystem), so the sovereign default survives a
reboot.

This was proven live on 2026-06-27 across a Mac laptop and the production VPS — traffic
served end-to-end by non-Anthropic models, with the running model self-identifying as
DeepSeek rather than Claude. The full provider ladder, credential resolution, and
Linux/VPS setup are documented in [docs/SOVEREIGN-DEFAULT.md](docs/SOVEREIGN-DEFAULT.md).

## Architecture

```
bin/hal                    # Entry point (bash wrapper; dispatches happi/osa/save/llm/voice subcommands)
lib/hal/
  cli.py                   # Interactive chat REPL
  server.py                # HTTP API + web UI (default :3850)
  gateway.py               # OpenAI-compatible /v1/* proxy (DEFAULT_GATEWAY_PORT = 4010)
  conversation.py          # Multi-turn conversation engine
  registry.py              # Provider management + model addressing
  sentinel.py              # Rate limit monitoring + auto-failover
  router.py                # Routing strategies (fixed, fallback, quality-cost)
  moe.py                   # Cross-LLM council (mixture of experts)
  grip_runtime.py          # grip_infer — the GRIP <-> HAL AI-syscall boundary
  agent_spawn.py           # Recursive agent spawning + mesh
  cost_tracker.py          # Token-to-USD calculation per provider
  warmup.py                # Background model warm-up
  discovery.py             # Auto-detect local LLM servers
  harness_detect.py        # Detect active agentic harness
  hook_dispatch.py         # Safety enforcement hooks
  skill_loader.py          # Skill discovery + mode resolution
  memory_engine.py         # Persistent memory (MEMORY.md format)
  context_assembler.py     # Full context assembly (HAL.md chain + mode + skills)
  identity_provider.py     # Self-aware system prompt builder
  session_manager.py       # Session lifecycle
  tool_dispatch.py         # Native tool execution (Bash, Read, Write, etc.)
  home.py                  # HAL_HOME resolution
  types.py                 # Core data types (frozen dataclasses)
config/
  providers.yaml           # Provider configuration
tests/                     # Unit tests (350+)
docs/                      # Architecture diagrams + guides
```

## Providers

HAL supports any OpenAI Chat Completions-compatible endpoint. Current registry covers
13 cloud providers, 6 local server types, and a custom-adapter slot for non-OpenAI APIs:

| Provider | Type | Sample models | Cost |
|----------|------|---------------|------|
| Groq | Cloud (free tier) | llama-3.3-70b, llama-4-scout, qwen3-32b, deepseek-r1-distill | Free |
| Gemini | Cloud (free tier) | gemini-2.5-flash, gemini-3.1-pro-preview, gemma-4-27b | Free |
| Qwen (DashScope) | Cloud | qwen3.5-flash, qwen3.5-plus, qwen3.6-plus-preview | Pay per token |
| DeepSeek | Cloud | deepseek-v4-flash, deepseek-v4-pro, deepseek-reasoner | Pay per token |
| Moonshot | Cloud | moonshot-v1-{8k, 32k, 128k}, vision variants | Pay per token |
| ZhipuAI | Cloud | glm-5, glm-5.1, glm-4.7, glm-4.5-air | Pay per token |
| Mistral | Cloud | mistral-small-latest | Pay per token |
| OpenRouter | Gateway | gemma-4-31b-it, llama-4-maverick + 100+ via gateway | Pay per token |
| OpenAI | Cloud | gpt-4o, gpt-4o-mini, o3-mini | Pay per token |
| xAI | Cloud | grok-4.20 (reasoning, non-reasoning, multi-agent), grok-4.6, grok-4.5, grok-4.3 | Pay per token |
| Anthropic | Cloud | Claude Opus, Sonnet, Haiku (via direct API) | Pay per token |
| Vulavula | Cloud | translate, transcribe, sentiment (custom adapter) | Pay per call |
| SoruxGPT | Cloud (gateway) | gpt-4o, claude-3-5-sonnet, deepseek, gemini-2.0-flash | Pay per token |
| Ollama | Local | Any pulled GGUF model (gemma4:26b primary, MoE 128/9 experts) | Zero cost, fully private |
| LM Studio | Local | Any model loaded in LM Studio | Zero cost |
| vLLM | Local | Any model served by vLLM | Zero cost |
| Lemonade | Local | Any model on lemonade-server (port 7860) | Zero cost |
| MLX-VLM | Local | Vision/audio/OCR/segmentation (Gemma4-VL, Qwen2/3-VL, Phi-4-Vision, SAM 3.1) | Zero cost |
| BitNet | Local | Microsoft BitNet b1.58-2B-4T (1.58-bit ternary, CPU-only, keyless localhost port 8114) | Zero cost |

Configure in `config/providers.yaml`. HAL auto-discovers local servers on known ports.

## Limit Sentinel + Tier-Scoped Failover

The user **never sees "You've hit your limit."** HAL monitors provider utilisation via response headers and auto-routes to the next available provider before limits are hit.

Default fallback chain: `groq → gemini → xai → openrouter → qwen → deepseek → ollama (local, zero limits)`.

Beyond the global chain, HAL routes through **tier-scoped fallback chains** so failures stay within budget tier before escalating:

| Tier | Chain |
|------|-------|
| free | ollama/gemma4:26b → ollama/gemma4:e4b |
| cheap | qwen → openrouter/gemma-4-31b → groq |
| mid | gemini-2.5-flash → openrouter → deepseek-v4-flash |
| premium | xai/grok-4.20-0309-reasoning → gemini-3.1-pro-preview → deepseek-v4-pro → groq/gpt-oss-120b |

This respects the operator's cost-band intent — a `cheap` request that fails on Qwen retries on OpenRouter and Groq before considering escalation to a paid mid-tier model.

A transient provider `429` ("server is temporarily limiting requests") is a *failover* signal, not a hard stop: the sentinel records a cooldown for that provider and routes the call onward — the operation proceeds rather than blocking.

## Cross-LLM Council

Query multiple LLMs simultaneously and merge results with agreement scoring:

```
you> /council Is this contract clause fair?
  Querying 4 models: groq/llama-3.3-70b, gemini/gemini-2.5-flash, ollama/mistral:7b...
  Agreement: 78% | Models: 3/4 | Cost: $0.000000
```

For higher-stakes deliberations — code review, architecture decisions, falsification of
strong claims — HAL supports the **AGORA grounded council** pattern. AGORA differs from a
plain council in two ways: every seed prompt is enriched with canonical grounding
excerpts from rules / plans / prior councils (Rule 21-bis grounding requirement), and
councils terminate on quorum or turn-cap rather than fixed depth. The council emits a
synthesised verdict plus a per-Speaker breakdown of dissent. The **judge is exogenous**
— a different, non-Anthropic model grades the speakers, so no model marks its own
homework (`rules/exogenous-anchor-law.md`).

## HAL.md Standard

HAL introduces a polymorphic instruction file standard. Any harness can have its own
instruction file (CLAUDE.md, .cursorrules, etc.). HAL reads `HAL.md` first, then falls
back to other formats. The format is identical across harnesses — plain markdown.

See `HAL.md` for the full specification.

## HAPPI Protocol — Harnessed-AI Polyglot Protocol Interface

`happi.md` is the canonical reference for **HAPPI/1.3** — a single self-contained polyglot
file that is simultaneously valid Markdown, executable Bash (`bash happi.md run`),
an embedded stdlib-only Python runtime (no pip dependencies), a HAPPI
envelope describing itself, and an OpenAPI 3.1 schema fragment. Five parsers, one
source of truth. The runtime emits `happi/1.3` and accepts `1.0`–`1.3` envelopes
(full back-compat). HAPPI is the **free (Apache-2.0)** protocol the whole family speaks; the full licence text is in `LICENSES/Apache-2.0.txt`.

### The IDR (Intent Decision Record) — a separate signed audit receipt

HAPPI/1.0 defined the envelope contract — one JSON object in (`v`, `id`, `cmd`, `flags`,
`auth`), one NDJSON event stream out, seven event types (`started`, `delta`, `tool_call`,
`tool_result`, `sub_request`, `completed`, `error`). HAPPI/1.1 added the **`idr`
event** as a *separate terminator* emitted **after** `completed`/`error`. The IDR is
**not** the `completed` payload: it carries only an `idr_ref` —
the **content-hash (`sha256`) of the envelope + the full emitted event stream**, the
**model versions** used, and an optional **CID / block-anchor**. It is a locally
verifiable, deterministically replayable audit *receipt*, opt-in via `flags.audit=true`.

The **decision anatomy** — the decision (`what`/`why`), the confidence score,
uncertainties, assumptions, and falsification criteria — lives on the v1.3 **`context`
event** (the `decision` shape, appended via the `context.append` cmd), **not** on the
IDR. The `context` event is the replayable signed memory-chain link; the IDR is the
stream-hash receipt. Together they make every HAL invocation **deterministically
replayable** (same data + different model = a falsification test of the original
verdict) and **content-addressable audit-ready** — the protocol-level expression of the
cryptographic-causation moat above.

HAPPI/1.3 adds the **deterministic citation-provenance floor**: the `cite.verify` cmd
checks, by exact string-match, that every cited quote is verbatim-present in its source
— a fabricated citation can never verify. It is the protocol-level twin of GRIP's
`prove_it` and of GRASP's public verifier — the un-fakeable floor, available to any AI
on any harness.

### Source-of-truth build (HAPPI/306 Phase 1)

Editing `happi.md` directly forces the editor to hold three or more escaping contexts
simultaneously (bash + Python + JSON + Markdown + OpenAPI). The build script avoids that:
edit single-context layer files in `happi-sources/` and regenerate `happi.md` from them.

```bash
# Workflow
python3 scripts/build-happi.py --extract     # derive sources from happi.md (one-time)
$EDITOR happi-sources/02-markdown-body.md    # edit a single layer in its native form
python3 scripts/build-happi.py --build       # regenerate happi.md
python3 scripts/build-happi.py --check       # CI-style drift verification
```

A pre-commit hook warns (does not block) when `happi.md` is staged without matching
`happi-sources/` changes. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full edit
discipline including the Goodhart-proof testing requirement.

### Public mirror

Read-only Gist: <https://gist.github.com/operator/2483d5c218c21ecc931130bcee7dee83>

The Gist is auto-synced from `happi.md` via a local post-commit hook so the public
mirror never drifts from the canonical source. Hook is not committed (lives under
`.git/hooks/`); on a fresh clone, re-arm it once with:

```bash
bash scripts/install-happi-gist-hook.sh
```

After installation, every commit that touches `happi.md` triggers
`scripts/sync-happi-gist.sh --quiet` in the background. The script PATCHes the
Gist via `gh api` and verifies byte-identity (SHA-256). On failure the hook prints
to stderr without blocking the commit; manually retry with
`bash scripts/sync-happi-gist.sh`. Check sync status at any time:

```bash
bash scripts/sync-happi-gist.sh --verify
```

## Development

```bash
# Run tests. The core needs only the standard library; the test suite runs under
# pytest, and a few tests import pyyaml, cryptography, apsw or xxhash. Keep the
# venv outside the clone so the checkout stays clean.
python3 -m venv ~/.hal-test-venv && ~/.hal-test-venv/bin/pip install pytest pyyaml cryptography apsw xxhash
cd ~/.hal && HAL_HOME=~/.hal ~/.hal-test-venv/bin/python -m pytest -q tests

# Run demo
HAL_HOME=~/.hal python3 lib/hal/demo.py --live

# Start server directly
HAL_HOME=~/.hal python3 -m lib.hal.server --port 3850
```

## Zero Dependencies

HAL's core uses **only Python stdlib**. No pip install required. No numpy, no langchain,
no pydantic. The entire runtime is self-contained in `lib/hal/`.

## Contributing

HAL ships with a hypothesis-driven contribution model: every PR registers a falsifiable
prediction via `lib/hypothesis_engine.py` before any code lands, and verification runs
automatically at the deadline. New features ship with Goodhart-proof tests — tests that
verify the change actually changes behaviour, not just that the code runs without error.

For the full contribution protocol — hypothesis registration, edit discipline for the
polyglot `happi.md` source-of-truth build, code style, and Goodhart-proof testing
requirements — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

HAL is free software, licensed under the **GNU Affero General Public License,
version 3 only (AGPL-3.0-only)**. See [LICENSE](LICENSE) for the full text.

Copyright (C) 2024-2026 Lourens Cornelius Scheepers / CodeTonight (Pty) Ltd

In plain terms: if you run a modified version of HAL as a network service, that
counts as distribution — the users of that service must be offered the modified
source code.

Commercial licences, for use outside the AGPL's terms, are available via
[licensing@codetonight.co.za](mailto:licensing@codetonight.co.za).
