# Getting Started with HAL

**Standalone Knowledge Engine. Any model. Any provider. Any domain. Zero limits.**

HAL is an LLM-agnostic runtime that routes queries across cloud and local providers, manages cost and rate limits transparently, and runs cross-LLM councils for critical decisions. It works standalone (zero external Python dependencies) or inside any harness (Claude Code, Cursor, Gemini CLI, Codex).

This guide gets you from zero to productive in under 10 minutes.

---

## What HAL Solves

| Problem | How HAL Fixes It |
|---------|-----------------|
| **Token limits** | Limit Sentinel monitors provider utilisation and auto-routes to the next provider at 85% capacity. You never see "You've hit your limit." |
| **Provider lock-in** | Any model, any provider, same interface. Switch with `/model groq/llama-3.3-70b`. |
| **Cost** | Free-tier-first routing: Groq and Gemini free tiers handle most queries at zero cost. |
| **Privacy** | Ollama fallback keeps everything local. Zero data leaves your machine. |
| **Single points of failure** | If Groq goes down, HAL routes to Gemini. If Gemini goes down, Ollama. Automatic. |
| **Harness dependence** | Works standalone or inside Claude Code, Cursor, Gemini CLI, or any agentic harness. |

---

## Prerequisites

- **Python 3.9+** (check with `python3 --version`)
- **git** (check with `git --version`)
- **macOS, Linux, or Windows** (Git Bash)
- **Optional:** [Ollama](https://ollama.com) for local models (unlimited, zero cost, fully private)

---

## Installation

HAL reaches you as the **starter kit**: a cleaned copy of the HAL framework that ships beside GRIP. You install it from the kit itself; you do not need access to HAL's source repository.

### 1. Unpack the kit and run its installer

Unpack the kit next to your GRIP install, for example to `~/hal-starter-kit`, then run the installer it carries:

```bash
cd ~/hal-starter-kit
bash scripts/install-starter.sh                   # installs into ~/.hal
HAL_HOME=/path/to/hal bash scripts/install-starter.sh   # or installs somewhere else
```

### 2. Add your provider keys to the macOS keychain

HAL reads each provider's key from a keychain item named `grip-<provider>`, such as `grip-groq` or `grip-gemini`. Add one for each provider you have a key for:

```bash
security add-generic-password -s grip-groq -a "$USER" -w
```

Ending the command with `-w` makes `security` ask for the key, so the key never lands in your shell history. On Linux there is no keychain: set the provider's key in the environment instead (`GROQ_API_KEY` or `GRIP_GROQ_API_KEY` for `grip-groq`), and HAL reads it from there.

### 3. Start the gateway by hand once, then install it as a service

Run the gateway in a terminal first, so you can see it start and answer:

```bash
cd ~/.hal
python3 -m lib.hal.gateway --port 4010
```

When it works, stop it with Ctrl-C, then install it as a per-user service with GRIP's service tool. The service uses the same port, 4010, and starts again at login:

```bash
grip-hal-services install --service gateway
```

### What ships switched off

- **Every paid source.** `config/spend_guard.json` ships fail-closed: `paid_opt_in` is empty and every daily cap is 0, so HAL refuses every paid billing source until you add its name to `paid_opt_in`. A source HAL meters itself also needs its cap in `daily_caps_usd` raised above 0. Free providers such as Groq are not paid sources and need no opt-in.
- **The W4 provider cascade.** `config/w4_cascade.json` is withheld from the kit, so W4 (switching a session to another provider when its own runs out) stays dormant until you configure your own targets.
- **Paid-first routing.** The routing default is Groq's free tier, `groq/openai/gpt-oss-120b`, so the first call works with a free Groq key.

### What the installer does

1. Copies the framework directories from the kit into `~/.hal`. A file of the same name already there is replaced without a backup, `config/` included, so run the installer once, for a first install
2. Copies the top-level files and the docs; on an existing install, it first backs up each file it would overwrite as `<name>.pre-hal-<timestamp>`
3. Creates the directory skeleton (`config`, `facts`, `data`, `state/hal`, `memory`, and so on)
4. Creates a Python virtualenv at `~/.hal/venv` and installs the KONO dependencies
5. Creates a default `facts/identity.md`, and a default `config/providers.yaml` if there is none

### Add to PATH

```bash
# Add to your .zshrc or .bashrc (one-time)
export PATH="$HOME/.hal/bin:$PATH"
```

---

## First Run

```bash
hal                    # Interactive chat (auto-warms local models)
hal --serve            # Start web UI + API server at localhost:3850
hal --status           # Show provider health and availability
hal --tools            # List built-in capabilities
hal --model ollama/mistral:7b   # Chat with a specific model
hal --mode legal       # Chat in a domain preset
```

When you type `hal` and press Enter, you get an interactive REPL. Type your query, get a response. Type `/quit` to exit.

---

## Provider Setup

HAL supports 12+ LLM providers out of the box. Configure in `~/.hal/config/providers.yaml`.

### Free Tier Providers (Zero Cost)

| Provider | How to Get API Key | Models |
|----------|--------------------|--------|
| **Groq** (default) | [console.groq.com](https://console.groq.com) | Llama 3.3 70B, Llama 4 Scout, Qwen3 32B, DeepSeek R1 |
| **Gemini** | [aistudio.google.com](https://aistudio.google.com) | Gemini 2.5 Flash, Gemini 3.1 Pro |
| **Ollama** (local) | [ollama.com](https://ollama.com) — no API key needed | Any GGUF model (Mistral, Llama, etc.) |

### Setting API Keys

```bash
# Option 1: Environment variables
export GROQ_API_KEY="gsk_..."
export GEMINI_API_KEY="AI..."

# Option 2: macOS Keychain (recommended, more secure)
security add-generic-password -s "grip-groq" -a "hal" -w "gsk_..."
security add-generic-password -s "grip-gemini" -a "hal" -w "AI..."
```

The `providers.yaml` config uses `api_key_source: keychain:grip-groq` by default, which reads from macOS Keychain. Set the environment variable as a fallback.

### All Supported Providers

| Provider | Type | Notes |
|----------|------|-------|
| Groq | Cloud (free) | Default. Fastest inference. |
| Gemini | Cloud (free) | Google. Large context windows. |
| Qwen | Cloud | Alibaba. Strong multilingual. |
| DeepSeek | Cloud | Reasoning specialist (DeepSeek R1). |
| Moonshot | Cloud | Kimi K2.5. |
| Zhipu | Cloud | GLM series. Chinese NLP strength. |
| Mistral | Cloud | European. Mistral Small/Large. |
| OpenRouter | Gateway | 100+ models from all providers. |
| OpenAI | Cloud | GPT-4o, o3-mini. |
| Arcee | Cloud | Trinity Large Thinking. |
| Vulavula | Cloud | South African NLP (translate, transcribe, sentiment). |
| Ollama | Local | Any model. Zero cost. Fully private. |

### Auto-Discovery

HAL auto-discovers local LLM servers running on known ports. If Ollama is running (`ollama serve`), HAL finds it automatically — no configuration needed.

---

## Key Features

### Limit Sentinel — Never Hit Token Limits Again

This is HAL's signature feature. The Limit Sentinel:

1. **Monitors** `X-RateLimit` headers from every provider response
2. **Tracks** utilisation percentage per provider in real time
3. **Auto-routes** to the next available provider at 85% utilisation
4. **Handles 429s** with exponential backoff (60s base, 1h max)

Default failover chain: **Groq -> Gemini -> Ollama**

You keep working. No interruption. No error messages. No manual switching.

```
you> Analyse this 50-page contract for liability clauses
  [Groq at 84% utilisation — routing to Gemini]
  Here are the key liability clauses I found...
```

### Cross-LLM Council (`/council`)

Query multiple LLMs simultaneously and merge results with agreement scoring:

```
you> /council Is this non-compete clause enforceable in Switzerland?
  Querying 4 models...
  groq/llama-3.3-70b:    Likely enforceable with conditions...
  gemini/gemini-2.5-flash: Swiss law requires reasonable scope...
  ollama/mistral:7b:      Under OR 340a, non-competes must...
  deepseek/deepseek-chat:  Enforceable if geographically limited...
  
  Agreement: 82% | Models: 4/4 | Cost: $0.000
  Synthesis: The clause is likely enforceable provided it meets
  Swiss OR 340a requirements: reasonable geographic scope,
  duration under 3 years, and adequate compensation.
```

The council uses a 14-step pipeline with Gemini 2.5-pro-002 as the permanent judge for deterministic external scoring. Family-deduped agreement prevents related models from inflating consensus.

### Domain Presets (`/mode`)

Switch domain context instantly:

```bash
hal --mode legal       # Legal operations context
hal --mode research    # Systematic research protocols
hal --mode code        # Software development
```

Presets inject domain-specific system prompts, relevant skills, and operational constraints.

### Web UI (`hal --serve`)

Start the web server for a browser-based experience:

```bash
hal --serve            # Starts at http://localhost:3850
hal --serve --port 8080  # Custom port
```

Features:
- Rich chat interface with conversation history
- `/api/tools` endpoint for capability inspection
- Session persistence across browser refreshes
- Provider health dashboard

### In-Chat Commands

| Command | Action |
|---------|--------|
| `/model <name>` | Switch model (e.g., `/model ollama/mistral:7b`) |
| `/mode <name>` | Switch domain preset (e.g., `/mode legal`) |
| `/council <query>` | Query all available LLMs and merge results |
| `/cost` | Show session cost breakdown |
| `/providers` | List providers with health status |
| `/history` | Show conversation history |
| `/clear` | Clear conversation and start fresh |
| `/quit` | Exit HAL |

### Self-Build (Recursive Self-Improvement)

HAL can analyse its own conversations and propose improvements:

1. `RSIEngine` detects improvement proposals from conversations
2. `CodeGenerator` transforms proposals into Python modules + tests via LLM
3. 7 safety gates validate: AST parse, security scan, compile check, tests, Goodhart guard, human approval, regression suite
4. `SelfBuilder` commits to a branch and creates a PR
5. `BuildLoop` iterates until convergence or budget exhaustion

Autonomy levels: `FULL_MANUAL` (approve everything) to `FULL_AUTO` (approve non-destructive changes automatically).

---

## Architecture

```
~/.hal/
  bin/hal                    Entry point (bash wrapper)
  lib/hal/
    cli.py                   Interactive chat REPL
    server.py                HTTP API + web UI
    conversation.py          Multi-turn conversation engine
    registry.py              Provider management + model addressing
    sentinel.py              Limit Sentinel (rate limit auto-failover)
    router.py                Routing strategies (fixed, fallback, quality-cost)
    moe.py                   Cross-LLM council (mixture of experts)
    agent_spawn.py           Recursive agent spawning + mesh
    cost_tracker.py          Token-to-USD calculation per provider
    warmup.py                Background model warm-up
    discovery.py             Auto-detect local LLM servers
    harness_detect.py        Detect active agentic harness
    hook_dispatch.py         Safety enforcement hooks
    skill_loader.py          Skill discovery + mode resolution
    memory_engine.py         Persistent memory (MEMORY.md format)
    context_assembler.py     Full context assembly (HAL.md chain)
    identity_provider.py     Self-aware system prompt builder
    session_manager.py       Session lifecycle
    tool_dispatch.py         Native tool execution
    tool_adapter.py          ToolDispatcher-to-ToolProtocol bridge
  config/
    providers.yaml           Provider configuration
  skills/                    235+ reusable skill definitions
  agents/                    43 specialised agent definitions
  hooks/                     48 safety gate definitions
  modes/                     30+ domain presets
  tests/                     142 test suites
```

**Zero dependencies.** HAL's core uses only Python stdlib. No pip install required for the runtime. No numpy, no langchain, no pydantic.

---

## For GRIP Users

If you already use GRIP (Claude Code's self-improving infrastructure):

- HAL detects GRIP automatically via `harness_detect.py`
- The GRIP context bridge (`grip_bridge.py`) loads your session state into HAL
- `/council` queries are grounded in actual GRIP system knowledge
- When GRIP hits Anthropic token limits, switch to `hal` — same conversation quality, different provider
- HAL reads GRIP's `MEMORY.md` format natively — your memories carry over

---

## Recent Achievements (April 2026)

### Fleet Gateway (PRs #136, #145, #149)
7 transport adapters (WhatsApp, Slack, Discord, Web, SMS, Email, Webhook) with a unified identity resolver and per-transport message counters. HAL can now receive and respond across any messaging platform.

### RAILLM Sprint (PRs #137-#141)
- **CarryOverVerifier** (W2): pluggable judge with audit log for cross-session state validation
- **CrossSessionStateFile** (W3): `fcntl` shared/exclusive file locks for safe concurrent session access
- **Fault Injection Harness** (W4): 100-message stress test with deterministic failure injection

### Tool System (PRs #150-#152)
- `tool_adapter.py` bridges `ToolDispatcher` into `ToolProtocol` — HAL can now use structured tools natively
- `hal --tools` surfaces the built-in capability list from the CLI
- `/api/tools` returns rich spec data for Web UI parity

### HAL Council — 14-Step Pipeline
The cross-LLM council matured into a production-quality pipeline:
- PR scope extraction + AST differ for automated code review
- Mermaid diagram adapter (AST delta to visual graph)
- Family-deduped agreement scoring (prevents model-family bias)
- Gemini 2.5-pro-002 as permanent external judge

### Extended Thinking (PR #128)
The CLI now renders extended thinking alongside the answer. When a model reasons through a problem, you see the reasoning process — transparent and auditable.

### Security Hardening (PR #154 + #24 epic)
- Vite CVE patches (GHSA-4w7w, GHSA-p9ff)
- `agent_traps.py`: multi-match injection detection with 30 adversarial test variants
- `sanitise.py`: CSS class hiding detection for hidden element attacks
- `a2a.py`: queue depth cap (10,000) preventing unbounded memory growth

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "No providers available" | Check API keys: `hal --status`. Set at least one key (Groq is easiest). |
| Ollama not detected | Run `ollama serve` in a separate terminal first. |
| Permission denied on `bin/hal` | `chmod +x ~/.hal/bin/hal` |
| Import errors | Reinstall the venv's dependencies: `~/.hal/venv/bin/pip3 install xxhash apsw sqlite-vec`. (Running the kit's installer again also does this, but it copies the kit's `config/` over yours, including `spend_guard.json`.) |
| Stale state | Delete `~/.hal/state/hal/limits.json` to reset Sentinel state. |

---

## Quick Reference

```bash
# Install (from the unpacked starter kit)
bash ~/hal-starter-kit/scripts/install-starter.sh

# Set up free-tier provider (pick one)
export GROQ_API_KEY="gsk_..."       # console.groq.com
export GEMINI_API_KEY="AI..."       # aistudio.google.com

# Run
hal                                  # Interactive chat
hal --serve                          # Web UI
hal --mode legal                     # Legal domain preset
hal --status                         # Provider health

# Inside HAL chat
/council Is this contract fair?      # Multi-LLM consensus
/model gemini/gemini-2.5-flash       # Switch model
/cost                                # Session cost
/quit                                # Exit
```

---

*Built by [CodeTonight](https://codetonight.co.za). Private repository.*
