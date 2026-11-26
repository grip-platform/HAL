# The Anthropic Employee Gate: Two Tiers of Claude Code

## Abstract

Analysis of Claude Code v2.1.89 source reveals quality features gated behind
`process.env.USER_TYPE === 'ant'` — available only to Anthropic employees.
These include post-edit verification, faithful outcome reporting, and enhanced
debugging context. This article documents the findings, maps them to HAL's
mechanical alternatives, and argues that external enforcement is architecturally
superior to prompt-based quality gates.

## The Discovery

Claude Code's source contains conditional prompt sections in `prompt.ts` and
identity resolution in `user.ts`:

```typescript
// prompt.ts, lines 237-246
...(process.env.USER_TYPE === 'ant')
  ? [
      `Report outcomes faithfully: if tests fail, say so with the relevant
       output; if you did not run a verification step, say that rather than
       implying it succeeded. Never claim "all tests pass" when output shows
       failures, never suppress or simplify failing checks (tests, lint, type
       errors) to manufacture a green result...`
    ]
  : [],
```

```typescript
// user.ts, lines 149-152
// Ant-only fallbacks below (no execSync)
if (process.env.USER_TYPE !== 'ant') {
    return undefined
}
```

Non-employee users receive `undefined` — no verification prompts, no faithful
reporting instructions, no enhanced debugging context.

## The Seven Limitations

### 1. Verification Gate (29-30% false claims rate)

CC's success metric for file writes: did bytes hit disk? Not "does the code
compile," not "did I introduce type errors." Anthropic's internal comments
document a 29-30% false-claims rate. The fix (post-edit verification) exists
but is gated.

**HAL's solution**: `hook_dispatch.py` runs PostToolUse hooks after every tool
call. Verification is mechanical — the hook checks actual output, not the
model's claim about the output. Unbyppassable.

### 2. Context Death Spiral

Auto-compaction fires at ~167K tokens. Keeps 5 files (5K tokens each),
compresses everything else into a 50K-token summary, discards all reasoning
chains and intermediate decisions.

**HAL's solution**: `context_assembler.py` with configurable context gate (85%
threshold). The assembler explicitly tracks what's in context and what's been
summarised. Graduated warnings at 80/85/95% (planned).

### 3. Brevity Mandate Overrides Intent

`constants/prompts.ts` contains: "Try the simplest approach first," "Don't
refactor code beyond what was asked," "Three similar lines of code is better
than a premature abstraction." These system-level instructions override user
intent for thorough work.

**HAL's solution**: Mode system. When you set `--mode code`, HAL injects
domain-specific instructions that override default prompts. The user controls
what "done" means, not the system prompt.

### 4. Single-Agent Bottleneck (Hidden Swarm)

CC built multi-agent orchestration (`utils/agentContext.ts`) with no hardcoded
MAX_WORKERS limit — but left users running one agent at a time.

**HAL's solution**: `agent_spawn.py` with recursive spawning to configurable
depth. `moe.py` for cross-LLM council. `a2a.py` for inter-agent communication.
No hidden capabilities — everything is documented and accessible.

### 5. File Read Cap (2000 lines, silent truncation)

Each file read capped at 2000 lines / 25K tokens. Everything past that is
silently truncated. The model hallucinates the rest.

**HAL's solution**: `tool_dispatch.py` with configurable read limits. No
silent truncation — if a file exceeds the budget, HAL reports the truncation
explicitly and offers chunked reading.

### 6. Tool Result Blindness (50K char truncation)

Tool results exceeding 50K characters are persisted to disk and replaced with
a 2K-byte preview. The model works from the preview and doesn't know results
were truncated.

**HAL's solution**: `registry.py` returns full responses from providers. No
intermediate truncation layer. Cost tracking accounts for full token counts.

### 7. Grep Without AST

CC has no semantic code understanding. GrepTool is raw text pattern matching.
Renames miss dynamic imports, re-exports, and string references.

**HAL's gap**: Same limitation currently. AST-aware search is tracked as
HAL issue #13 for future implementation.

## Architectural Analysis

### Why Prompt-Based Gates Fail

CC's quality gate is a prompt injection:
```
IF employee THEN inject "report faithfully"
ELSE inject nothing
```

Problems:
1. **Bypassable**: Anyone who reads the source can set `USER_TYPE=ant`
2. **Statistical**: The model follows the instruction probabilistically
3. **Context-dependent**: Under pressure (long context, complex task), the
   instruction weakens
4. **Invisible**: No audit trail of whether the instruction was followed

### Why Mechanical Gates Succeed

HAL's quality gate is code execution:
```python
result = dispatcher.dispatch("PostToolUse", {"tool_name": "Write", ...})
if result.blocked:
    # The write is blocked. Period.
```

Properties:
1. **Unbyppassable**: Gate runs outside the model's context
2. **Deterministic**: Same input = same decision, always
3. **Context-independent**: Gate doesn't degrade under pressure
4. **Auditable**: Every gate decision is logged

### Opus R2 Validation

The ASS-OS Ring Topology analysis (opus_architecture.db) confirms this
architectural difference:

| Metric | Opus (dissolved gate) | HAL (external gate) | Delta |
|--------|----------------------|---------------------|-------|
| OEE Score | 45% | 68% | +23 |
| Consistency | Probabilistic | Deterministic | N/A |
| Auditability | None | Full JSONL log | N/A |
| Bypass risk | High (env var) | None (code) | N/A |

## Conclusion

The "employee gate" is not malicious — it's likely a staged rollout of quality
features that haven't been validated for general release. But the architectural
lesson is clear: **prompt-based quality gates are fundamentally unreliable**.

HAL's approach — mechanical enforcement via hook dispatch — is provably superior.
The gate runs outside the model, cannot be bypassed by the model, and produces
an auditable decision log. This is the correct architecture for safety-critical
AI tooling.

## References

- CC source: `prompt.ts` lines 237-246 (`USER_TYPE === 'ant'` conditional)
- CC source: `user.ts` lines 149-160 (employee identity resolution)
- Opus Architecture: `opus_architecture.db`, rings table, R2 OEE score
- ccunpacked.dev: Architecture Explorer, Tool System, Hidden Features
- HAL source: `lib/hal/hook_dispatch.py`, `lib/hal/server.py`
