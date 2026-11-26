# HAL Architecture Overview

## Module Dependency Graph

```mermaid
graph TD
    CLI[cli.py<br/>Interactive Chat] --> CONV[conversation.py<br/>Multi-turn Engine]
    CLI --> DISC[discovery.py<br/>Local Auto-detect]
    CLI --> MOE[moe.py<br/>Cross-LLM Council]
    CLI --> WARM[warmup.py<br/>Model Warmup]
    
    SRV[server.py<br/>HTTP API + Web UI] --> CONV
    SRV --> MOE
    SRV --> REG[registry.py<br/>Provider Management]
    
    CONV --> SENT[sentinel.py<br/>Limit Monitoring]
    CONV --> COST[cost_tracker.py<br/>Cost Calculation]
    CONV --> SESS[session_manager.py<br/>Session Lifecycle]
    CONV --> CTX[context_assembler.py<br/>Context Builder]
    
    SENT --> REG
    SENT --> ROUTE[router.py<br/>Routing Strategies]
    ROUTE --> REG
    
    REG --> OAI[providers/openai_compat.py<br/>OpenAI Wire Protocol]
    REG --> TYPES[types.py<br/>Core Data Types]
    
    CTX --> SKILL[skill_loader.py<br/>Skills + Modes]
    CTX --> MEM[memory_engine.py<br/>Persistent Memory]
    
    MOE --> REG
    
    SPAWN[agent_spawn.py<br/>Recursive Agents] --> REG
    SPAWN --> MOE
    
    HOOKS[hook_dispatch.py<br/>Safety Gates] --> HOME[home.py<br/>HAL_HOME Resolution]
    
    TOOL[tool_dispatch.py<br/>Native Tools] --> HOOKS
    
    HARNESS[harness_detect.py<br/>Harness Detection]
    
    DISC --> REG

    style CLI fill:#e94560,color:#fff
    style SRV fill:#e94560,color:#fff
    style CONV fill:#4fc3f7,color:#000
    style REG fill:#4fc3f7,color:#000
    style SENT fill:#00d26a,color:#000
    style MOE fill:#ff9800,color:#000
    style SPAWN fill:#ff9800,color:#000
    style TYPES fill:#666,color:#fff
    style HOME fill:#666,color:#fff
```

## Layer Architecture

```mermaid
graph LR
    subgraph Surface["Surface Layer"]
        C[CLI]
        W[Web UI]
        M[MCP]
    end
    
    subgraph Engine["Engine Layer"]
        CONV2[Conversation]
        CTX2[Context Assembly]
        HOOKS2[Hook Dispatch]
    end
    
    subgraph Intelligence["Intelligence Layer"]
        COUNCIL[Council / MoE]
        AGENTS[Agent Spawner]
        RSI[RSI Engine]
    end
    
    subgraph Provider["Provider Layer"]
        REG2[Registry]
        SENT2[Sentinel]
        ROUTE2[Router]
    end
    
    subgraph LLM["LLM Providers"]
        GROQ[Groq]
        GEM[Gemini]
        OLL[Ollama]
        OAI2[OpenAI]
        ANT[Anthropic]
    end
    
    Surface --> Engine
    Engine --> Intelligence
    Intelligence --> Provider
    Provider --> LLM
```

## Data Flow

```
User Input
    |
    v
[CLI / Server / MCP]  <-- Surface
    |
    v
[Conversation Engine]  <-- Manages history, cost, mode
    |
    v
[Context Assembler]    <-- HAL.md chain + skills + memory + git
    |
    v
[Limit Sentinel]       <-- Checks provider health + utilisation
    |
    v
[Router]               <-- Selects strategy (fixed/fallback/quality-cost)
    |
    v
[Provider Registry]    <-- Resolves provider/model address
    |
    v
[OpenAI-compat HTTP]   <-- Universal wire protocol
    |
    v
[LLM Response]
    |
    v
[Cost Tracker]         <-- Records tokens + USD
    |
    v
[Hook Dispatch]        <-- PostToolUse / PostSampling hooks
    |
    v
User Output
```

## Gateway / Egress Rail

The module graph above covers the **conversation engine** (the `hal` CLI and
HTTP server talking to providers). HAL also ships a second, equally important
surface: the **Gateway / Egress rail** — a local Anthropic/OpenAI-compatible
reverse proxy on `/v1/*` that lets *any* coding harness (Claude Code, Aider,
Cline, OpenAI Codex CLI, Goose) reach Anthropic — or any HAL-registered
provider — through HAL, transparently.

```
Coding harness (Claude Code / Codex / Aider / Cline)
    |
    v
[gateway.py]              <-- /v1/messages reverse proxy (Anthropic-compat, P1)
[gateway_openai.py]       <-- /v1/chat/completions (OpenAI-wire harnesses, P2)
[gateway_responses.py]    <-- /v1/responses (Codex wire_api="responses", P6)
    |
    v
[gateway_guard.py]        <-- default-deny: the public hal-server tunnel is
    |                          never an unauthenticated open relay
    v
[gateway_crosswire.py /   <-- translate Anthropic<->OpenAI wire when the client
 gateway_translate.py /        and the target provider speak different wires
 gateway_stream_crosswire.py]  (streaming and non-streaming)
    |
    v
[gateway_egress.py]       <-- same-wire re-pointing: forward the original body
    |                          verbatim to a non-Anthropic provider
    v
[gateway_failover.py]     <-- ordered upstreams; on 429/5xx, try the next one
    |
    v
[gateway_witness.py]      <-- WITNESS + SIGN every inference on the chat rail
    |                          (the provability-moat / "VeriSign-of-AI" floor)
    v
Provider (Anthropic / Groq / Gemini / OpenAI / local)
```

Two shipped features here are the largest the conversation graph omits:

- **`gateway_witness.py`** — witnesses and cryptographically signs every
  inference that crosses the chat rail, closing the gateway's last
  unwitnessed surface (`/v1/chat/completions`). This is the audit / provability
  foundation that the IDR chain builds on.
- **`grip_runtime` (`grip_infer`)** — GRIP's first-class default LLM runtime.
  It drives a single happi `infer` envelope through the real happi runtime over
  the **RAILLM ladder** (CCH subscription -> Kimi/Qwen subscription -> cheap
  API tiers -> local), with stateful failover that carries conversation history
  across provider switches. happi.md is the syscall surface; RAILLM is the
  failover substrate. See `docs/ADR-0001-hal-raillm-grip-runtime.md` for the
  decision record.
