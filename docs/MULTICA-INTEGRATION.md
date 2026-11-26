# multica-ai/multica — Research Analysis + Integration Plan (#112)

H038 research council output. Deadline: 2026-04-20.

## What multica IS

Multica is an AI-native task management platform — essentially Linear with AI
agents as first-class citizens. Distributed coding agents (Claude, Codex,
OpenCode, Gemini) autonomously claim tasks, execute work, report progress, and
compound reusable skills. Architecture: Next.js frontend -> Go backend (Chi +
WebSocket) -> PostgreSQL (pgvector) <-> local agent daemons.

## SMT Mapping Table

| multica concept | HAL concept | Mapping type |
|----------------|-------------|--------------|
| Task board | BackgroundExecutor | Partial — HAL has async tasks but no persistent board UI |
| Skill compounding | KnowledgeAbsorber | Surface — HAL absorbs knowledge but doesn't auto-promote solutions to reusable guards |
| Polymorphic assignees | AgentRegistry | Partial — HAL agents are typed; multica treats humans and agents as interchangeable |
| Local daemon fleet | FleetGateway | Deep — both coordinate distributed runtimes, different transport layer |
| pgvector embeddings | KONO substrate (sqlite-vec) | Deep — same semantic pattern, different DB engine |
| WebSocket coordination | ChannelRouter | Partial — HAL routes messages, multica routes task state |
| Agent autonomy (claim/execute/report) | No equivalent | Anti — HAL's council is synchronous deliberation, not async autonomy |

## Novel Patterns (H038 claim: >= 3 absent from HAL)

### 1. Autonomous Task Lifecycle (ATL)
Agents claim work from a shared board, execute independently, and report status
without human intervention. Decouples execution from orchestration. HAL's
synchronous council requires realtime deliberation. Multica's async model scales
to N agents with fire-and-forget distribution.

### 2. Skill Compounding Engine
Successful solutions auto-convert to reusable capabilities via pgvector semantic
matching. HAL has VaultIndexer and KnowledgeAbsorber but lacks automated
skill extraction/promotion from execution results into a queryable capability
library that downstream agents can invoke.

### 3. Polymorphic Assignee Pattern
Team members and agents share the same assignment interface with distinct
permissions. HAL's council members are model-typed; multica treats humans and
agents as interchangeable task recipients, enabling hybrid workflows where
humans re-delegate mid-execution.

### 4. Distributed Runtime Coordination (DRC)
Multiple local daemons (each running different agent CLIs) report to a single
PostgreSQL source-of-truth via WebSocket, enabling fleet-level observability
across heterogeneous runtimes. HAL's FleetGateway handles transport; multica
adds a persistent task ledger + skill graph across runtimes.

**H038 verdict: CONFIRMED** — 4 structurally distinct novel patterns identified
(claim was >= 3).

## Falsification Table

| Claim | Verdict | Evidence |
|-------|---------|----------|
| multica has >= 3 novel patterns absent from HAL | CONFIRMED | 4 patterns: ATL, skill compounding, polymorphic assignees, DRC |
| multica's council is deeper than HAL's | FALSIFIED | multica has no council/deliberation — it's task management, not synthesis |
| multica's embedding approach supersedes KONO | FALSIFIED | Same pattern (pgvector vs sqlite-vec), different scale target |
| multica's agent autonomy is incompatible with HAL | FALSIFIED | ATL is additive — can layer async task lifecycle on top of synchronous council |

## Integration Recommendation

**ADOPT + PARTIAL**

| Pattern | Action | Priority | Effort |
|---------|--------|----------|--------|
| Skill compounding | ADOPT — auto-promote council insights to reusable guards | P1 | M (4-6h) |
| Async task lifecycle | ADOPT — extend BackgroundExecutor with claim/execute/report | P2 | M (4-6h) |
| Polymorphic assignees | PARTIAL — human re-delegation in hybrid flows | P3 | S (2-3h) |
| Full task board UI | SKIP — orthogonal to deliberation; bridge via API if needed | - | - |

**Priority rationale**: Skill compounding closes the largest HAL gap — there is
currently no way to capture and reuse council insights as parameterised guards
that downstream agents can invoke. The ATL pattern enables HAL's Fleet Command
to support fire-and-forget agent work, which is the natural next step after
the transport bootstrap shipped in this sprint.
