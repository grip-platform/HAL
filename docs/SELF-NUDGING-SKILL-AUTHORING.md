# Self-Nudging Skill Authoring (W4b absorb)

Status: DESIGN NOTE (no implementation this wave)
Grade: B (Pattern 1 from W4b research-council audit)
Parent plan: `~/.claude/plans/ultrado-apr-15-multi-wave.md`
Source audit: `/tmp/hermes-agent-analysis.md` section 4, Pattern 1

## 1. Problem Statement

HAL has a `/learn` command and a `learned-insights.md` pipeline, but the
trigger is explicit: the operator must type `/learn`. Observed effect:
valuable session-scale learnings decay because nobody remembers to
invoke the command at the right moment. This is the **trigger gap**.

The W4b audit flagged this as hermes-agent's Pattern 1 (Grade B,
survived falsification claim C2). Hermes-agent closes the loop via
self-nudging: the agent itself decides, after a complex task, that a
skill should be written, and prompts for confirmation.

## 2. How hermes-agent Solves It

From `/tmp/hermes-agent-analysis.md` section 4, Pattern 1:

> "hermes-agent nudges itself during long sessions to write new skills
> after complex task completion, and to improve existing skills during
> use. The agent is both consumer AND author of its own procedural
> memory."

Mechanism is doc-level, not source-level (audit marks it MEDIUM
confidence — details will need reverse engineering). The shape is:
complexity detector -> self-prompt -> skill candidate -> review gate ->
commit.

## 3. Proposed HAL Integration

A post-task hook inside HAL's orchestration surface that runs after
each `AgentRunner` task completes.

**Integration points** (all verified to exist):

1. `lib/hal/agent_runner.py` — `AgentRunner.run()` emits a post-task
   event when a task completes. No change to loop shape; only a hook
   callout.
2. `lib/hal/event_bus.py` — `EventBus` already carries `Event`s. A new
   event type `task.completed` is published by the runner and consumed
   by the nudger.
3. `lib/hal/insights.py` — `InsightsEngine` already scores friction and
   generates suggestions. The nudger reuses its `_detect_friction` and
   topic extraction primitives to compute a complexity heuristic.
4. `lib/hal/home.py` — `HAL_HOME` is the root for a new
   `skills/draft/` subdirectory where proposed skills land before
   operator review.

**New module** (only net-new file this wave would introduce):
`lib/hal/skill_author.py`, containing:

- `ComplexityHeuristic` — scores a task transcript on
  (length, tool-call count, error count, topic drift). Threshold tuned
  to avoid nudge spam.
- `SkillNudger` — subscribes to `task.completed`, runs the heuristic,
  and on breach drafts a skill candidate file in `skills/draft/` plus
  an AskUserQuestion prompt.
- `SkillReviewGate` — PARAMOUNT. Nothing lands in `skills/` (the live
  directory) without an explicit AskUserQuestion approval. Draft files
  auto-expire after 7 days if ignored.

## 4. Explicit AskUserQuestion Gate

**PARAMOUNT.** The nudger MAY NOT write to the live skills directory
directly. Flow:

1. Heuristic fires on `task.completed`.
2. Draft skill is written to `skills/draft/<timestamp>-<slug>.md`.
3. AskUserQuestion is emitted with options: `commit` (promote to
   `skills/`), `revise` (operator edits draft first), `discard`
   (delete draft), `defer` (leave in draft for later review).
4. Only on `commit` does the draft move to the live directory.

This matches Rule 7 (Efficiency) — confidence is never 99.9999999%
that a self-authored skill is correct. Always ask.

## 5. Effort Estimate

| # | Milestone | Effort |
|---|---|---|
| M1 | `skill_author.py` skeleton + heuristic + draft writer | S (2 days) |
| M2 | EventBus subscription + `task.completed` event in `agent_runner.py` | S (1 day) |
| M3 | Review gate + promotion pipeline + draft expiry | S (2 days) |

Total: ~5 engineering days. Shorter than Pattern 2 (Fleet Gateway)
because all infrastructure already exists in HAL.

## 6. Falsification Criteria

This design is refuted if:
1. The complexity heuristic nudges on simple tasks (false-positive rate
   > 20% in a 100-task sample).
2. Operator consistently chooses `discard` over `commit` (> 80% reject
   rate — suggests the nudger is not identifying real learnings).
3. Self-authored skills, once committed, are net-negative on later
   sessions (measured by `output_quality.py` score delta on tasks
   touching the skill's domain).
4. Draft directory accumulates stale files faster than the 7-day expiry
   clears them (operator attention exhaustion).

## 7. Cross-Links

- W4b audit: `/tmp/hermes-agent-analysis.md` section 4, Pattern 1
- Companion design spec: `docs/FLEET-GATEWAY-DESIGN.md`
- Parent ultrado plan: `~/.claude/plans/ultrado-apr-15-multi-wave.md`
