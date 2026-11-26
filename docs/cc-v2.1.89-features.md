# Claude Code v2.1.89 Feature Analysis — HAL Absorption Status

Analysis date: 2026-04-02. Source: ccunpacked.dev, tf-fork, CC source screenshots.

## Employee-Gated Features (process.env.USER_TYPE === 'ant')

CC gates quality features behind an employee check. HAL implements these mechanically.

| CC Feature (ant-only) | HAL Native Implementation | Module |
|------------------------|--------------------------|--------|
| Post-edit verification | PostToolUse hooks verify output | `hook_dispatch.py` |
| Faithful outcome reporting | Hooks check actual output, not bytes-to-disk | `hook_dispatch.py` |
| Context decay awareness | 85% context gate + graduated warnings | `context_assembler.py` |
| Sub-agent swarming | Recursive spawning, no ceiling | `agent_spawn.py` |
| File read budgeting | Configurable limits in tool dispatch | `tool_dispatch.py` |
| Tool result truncation | Registry returns full responses | `registry.py` |

**Key insight**: CC's quality improvements are prompt injections (bypassable).
HAL's are mechanical hooks (deterministic, auditable, not bypassable).

## New Tools in CC v2.1.89

| Tool | CC Status | HAL Status | Priority |
|------|-----------|-----------|----------|
| TeamCreate/TeamDelete | Active | Use agent_spawn mesh | LOW |
| VerifyPlanExecution | Feature-gated | Not needed (hooks verify) | LOW |
| Snip (context trim) | Feature-gated | Context gate handles | LOW |
| Workflow | Feature-gated | Skill system covers | LOW |
| TerminalCapture | Feature-gated | Not applicable (web-first) | SKIP |
| Monitor | Feature-gated | Mesh heartbeat covers | LOW |
| SubscribePR | Feature-gated | GitHub API via tools | LOW |
| LSP | Feature-gated | Add for code intelligence | MEDIUM |
| WebBrowser | Feature-gated | Gemini vision covers | LOW |
| SendUserFile | Feature-gated | Server API covers | LOW |
| PushNotification | Feature-gated | Mesh + web push | MEDIUM |
| CronCreate/Delete/List | Feature-gated | `scheduler.py` | DONE |

## New Commands in CC v2.1.89

| Command | CC Status | HAL Equivalent |
|---------|-----------|---------------|
| /ultraplan | Active | Mode system + council | 
| /rewind | Active | Git ops module |
| /btw | Active | Mode switching |
| /advisor | Active | Council (multi-model) |
| /voice | Feature-gated | Planned (hal-web) |
| /desktop | Feature-gated | Electron shell planned |
| /mobile | Feature-gated | React Native planned |
| /sandbox | Active | Hook dispatch gates |

## Hidden Features

| Feature | CC Status | HAL Status |
|---------|-----------|-----------|
| Kairos (persistent mode) | Feature-gated | Daemon mode planned |
| Coordinator Mode | Feature-gated | agent_spawn.py mesh |
| Bridge (remote control) | Feature-gated | Mesh + web UI |
| Daemon Mode (--bg) | Feature-gated | `daemon.py` planned |
| UDS Inbox | Feature-gated | `mesh.py` file-based |
| Auto-Dream | Feature-gated | `rsi.py` post-conversation |
| Buddy (virtual pet) | Active | Not applicable |

## Opus Architecture Integration

From ASS-OS Ring Topology analysis (opus_architecture.db):

| Ring | Opus OEE | HAL Advantage |
|------|---------|---------------|
| R0 Seed (tokenization) | 76% | Context assembly + token budget |
| R1 Sense (features) | 67% | 203 skills + 30 modes |
| R2 Gate (safety) | 45% | **HAL: 68%** — external mechanical gates |
| R3 State (context) | 50% | 85% context gate + graduated warnings |
| R4 Exec (reasoning) | 57% | Multi-provider routing by capability |
| R5 Memory (knowledge) | 41% | Web search + evidence classification |
| R6 Observe (output) | 59% | Concise communication + anti-drift |

**HAL's R2 gate scores +23 points over Opus's dissolved gate.**
The externalised gate pattern is confirmed superior to prompt-based safety.

## CC Limitations HAL Addresses Natively

1. **29-30% false claims rate** — HAL hooks verify mechanically
2. **167K context before compaction** — HAL gates at configurable threshold
3. **Brevity mandate overrides intent** — HAL modes override system prompts
4. **Single-agent bottleneck** — HAL spawns recursive agents
5. **2000-line file read cap** — HAL tool dispatch is configurable
6. **50K tool result truncation** — HAL returns full results
7. **grep-only search** — HAL should add AST search (TODO)
