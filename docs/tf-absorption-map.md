# TF>> → HAL Absorption Map

TF>> (Claude Code fork) architecture mapped to HAL plain language terminology
and HAL module equivalents. Source: `~/CodeTonight/tf>>` (~1,850 TypeScript files).

## Terminology Translation

Per `lib/plain_language.py` and GRIP naming conventions:

| TF>> Term | HAL Plain Language | HAL Module |
|-----------|-------------------|------------|
| Tool | Tool (unchanged) | `tool_dispatch.py` |
| Hook (PreToolUse) | Automated check (pre-action) | `hook_dispatch.py` |
| Hook (PostToolUse) | Automated check (post-action) | `hook_dispatch.py` |
| Permission | Guard decision | `hook_dispatch.py` |
| Skill | Playbook | `skill_loader.py` |
| Agent | Worker | `agent_runner.py` |
| Task | Task (unchanged) | `agent_runner.py` |
| Mode | Preset | config |
| REPL | Session | `session_manager.py` |
| Query | Conversation turn | `conversation.py` |
| QueryEngine | Turn executor | `conversation.py` |
| Context | Session state | `context_assembler.py` |
| MCP | Tool bridge | `registry.py` providers |
| Coordinator | Orchestrator | `agent_runner.py` |
| Classifier | Auto-approver | `hook_dispatch.py` (hard/soft) |
| PermissionDecision | Guard verdict | `HookResult` |
| OutputStyle | Response format | config |
| Ink | Terminal UI | (GRIP-GUI replaces) |
| Screen | View | (GRIP-GUI routes) |

## Subsystem Absorption Status

### Fully Absorbed (HAL has Python equivalent)

| TF>> Subsystem | Files | HAL Equivalent | Status |
|----------------|-------|----------------|--------|
| `hooks/` | 104 | `hook_dispatch.py` + `hooks/*.py` | Complete |
| `tools/` | 183 | `tool_dispatch.py` | Core tools only |
| `skills/` | 20 | `skill_loader.py` + `skills/` | Complete |
| `state/` | 6 | `session_manager.py` | Complete |
| `context/` | 9 | `context_assembler.py` | Complete |
| `cost-tracker.ts` | 1 | `cost_tracker.py` | Complete |
| `remote/` | 4 | `server.py` (HTTP API) | Partial |
| `query/` | 4 | `conversation.py` | Complete |

### Partially Absorbed (HAL has some equivalent)

| TF>> Subsystem | Files | HAL Gap | Priority |
|----------------|-------|---------|----------|
| `services/` | 130 | MCP, analytics, OAuth, voice missing | HIGH |
| `commands/` | 189 | HAL CLI has basic commands only | MEDIUM |
| `tasks/` | 12 | `agent_runner.py` covers basics | MEDIUM |
| `bridge/` | 31 | Session bridging for remote | LOW |
| `migrations/` | 11 | No migration system | LOW |

### Not Absorbed (frontend — replaced by GRIP-GUI)

| TF>> Subsystem | Files | GRIP-GUI Equivalent | Notes |
|----------------|-------|---------------------|-------|
| `components/` | 389 | `src/components/` | Swiss Nihilism styled |
| `ink/` | 96 | N/A (web replaces terminal) | GRIP-GUI is web+Electron |
| `screens/` | 3 | `src/app/` routes | Already exists |
| `vim/` | 5 | N/A | Terminal-only feature |
| `keybindings/` | 14 | GRIP-GUI keyboard shortcuts | Partial |

### Knowledge to Absorb (architecture patterns, not code)

| TF>> Pattern | Description | HAL Application |
|--------------|-------------|-----------------|
| Tool.ts interface | Unified tool type with execute/validate | Formalise ToolDispatcher protocol |
| Task.ts lifecycle | Create/run/stop/output streaming | Extend agent_runner.py |
| QueryEngine | Turn-by-turn conversation with tool calls | Enrich conversation.py |
| coordinatorMode | Multi-agent orchestration | Enrich agent_runner.py |
| directConnect | Server mode for IDE integration | Extend HAL HTTP API |
| upstreamProxy | Relay for managed deployment | New: proxy.py |
| outputStyles | Configurable response formatting | New: output_format.py |
| sessionHistory | Conversation persistence | Extend session_manager.py |

## GRIP-GUI as HAL's Native Frontend

GRIP-GUI already has:
- Swiss Nihilism CSS (globals.css with paper-grey, cyan accent, sharp corners)
- Engine component (ChatInterface, ContextPanel, ToolUseBlock, GateIndicator)
- Electron shell with PTY integration
- MCP orchestrator
- Zustand state management
- Next.js 16 App Router

What it needs:
1. HAL API client (replace CC process bridge with HTTP to port 3850)
2. Guard flow UI (permission dialogs styled in Swiss Nihilism)
3. Worker management UI (agent list, task tracking)
4. Playbook browser (skill browser already exists, rename + enhance)
5. Provider switcher (HAL's multi-LLM selection)
6. Session management (conversation history via HAL)

## TF>> ASS-OS Absorption (Gentner SMT-Validated, Gen 506)

Structural mappings from Thomas Frumkin's ASS-OS 127D v2.0 into HAL,
validated by Structure Mapping Theory enforcer. Only mappings with genuine
causal correspondence survive — surface analogies blocked.

### Absorbed (structural correspondence confirmed)

| TF>> Concept | HAL Module | Gentner Score | Status |
|---|---|---|---|
| Shield integrity model (target/actual + regen) | `provider_monitor.py` | 94/100 | ABSORBED |
| 12-ring compiler (7 inner stages) | `pipeline.py` (planned) | 68/100 | PARTIAL (7/12) |
| Reflex triggers (~40 regex auto-dispatch) | `hooks/` | 96/100 | ALREADY EXISTS |

### Independently designed (inspired by TF>> but not mapped)

| HAL Feature | Inspiration | Why Not Mapped |
|---|---|---|
| Multi-intent confidence weighting | TF>> bloom vectors | Surface analogy (56/100). HAL's IntentWeights designed independently. |

### Not absorbed (blocked by Gentner enforcer)

| TF>> Concept | Gentner Score | Reason |
|---|---|---|
| 7 agents to ring archetypes | 25/100 | No dispatch mechanism — empty naming without causal machinery |
| Mersenne 127 to air-gap boundary | 40/100 | Numerological — 127 has no causal relevance to provider filtering |
| Bloom vector naming (7 primes) | 56/100 | Surface analogy — vector field names labelled existing intents, no new behaviour |

### 5 Mersenne outer stages (backend, unmapped)

Thomas's 12-ring compiler has 5 backend stages (LIFT, TRANSFORM, PRUNE, NEXUS,
EMIT) that map to compilation optimisation. HAL has no equivalent post-processing
pipeline yet. These remain as future absorption targets when HAL gains response
post-processing (caching, deduplication, formatting).

## Wave Plan

| Wave | Deliverable | Repo |
|------|-------------|------|
| W1 | This document + GRIPHarness TF>> scan | ~/.hal |
| W2 | HAL API surface spec (OpenAPI) | ~/.hal |
| W3 | HAL API implementation (tool/task/conversation endpoints) | ~/.hal |
| W4 | GRIP-GUI HAL client + Engine bridge | GRIP-GUI |
| W5 | Guard flow + Worker management + Provider switcher UI | GRIP-GUI |
| W6 | Full integration test + Electron packaging | GRIP-GUI |
