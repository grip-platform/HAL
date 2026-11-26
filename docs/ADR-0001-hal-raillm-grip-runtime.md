# ADR-0001 — HAL RAILLM as GRIP's First-Class LLM Runtime

- **Status**: Accepted (substrate shipped; HAL#329 core defect FIXED on `fix/raillm-compress-before-failover` — H-RAILLM-5 now CONFIRMED)
- **Date**: 2026-05-16
- **Branch / PR**: `feat/hal-raillm-grip-runtime` / CodeTonight-SA/HAL#328
- **Plan**: `plans/hal-raillm-grip-runtime.md`
- **Supersedes**: nothing (first ADR)

## Context

GRIP today calls LLMs through `~/.claude/lib/hal_llm_adapter.py` — a
stateless scorer chain (HAL-HTTP → Groq → Gemini → xAI → sentinel). It has
no Claude-Code-subscription (CCH) rung, no Kimi/Qwen subscription failover,
no conversation-history carry, and does not pass through the happi.md
envelope. GRIP is therefore bound to one provider's context window and one
auth path.

The operator goal: make GRIP run *through HAL* by default — happi.md as the
shim, an LLM ladder CCH → Kimi/Qwen → cheap API → local, cascading through
the RAILLM InfiniteMachine with history preserved, so GRIP is no longer
bound to the Claude Code Anthropic context-window constraint.

Grounded reality (verified, not assumed): ~80% of the machinery already
existed on `main` (Anthropic native provider, CCH/Kimi/Qwen subscription
transports, `RAILLMSentinel.dispatch_with_history`, the `lib/hal/happi/`
package). This work is integration + wiring + falsification, not a
greenfield build.

## Decision

Four composition layers, each leaving the core (`raillm.py`, `registry.py`,
`sentinel.py`, `happi/runtime.py`, `subscription.py`) **byte-unchanged**:

| Wave | Module | What it adds |
|------|--------|--------------|
| W1 | `lib/hal/raillm_subscription_bridge.py` | Adapts the `SubscriptionAuth` transports (CCH/Kimi/Qwen) to the registry `.get()/.chat()->ChatResponse` contract so RAILLM can cascade through them. |
| W2 | `lib/hal/raillm_ladder.py` | Encodes the canonical ladder: CCH-Anthropic primary → Kimi/Qwen subscription → cheap API → local (ollama/mlx). |
| W3 | `lib/hal/happi/cmds_infer.py` | A happi `infer` cmd: envelope → `dispatch_with_history` over the ladder → streamed events; `flags.audit=true` → deterministic `idr` terminator. |
| W4 | `lib/hal/grip_runtime.py` | `grip_infer(prompt, *, model=None, audit=False) -> GripInferResult` — the single, stable, documented entrypoint. happi.md *is* the syscall surface. |

GRIP's adapter rewire is a **separate cross-repo follow-up**
(CodeTonight-SA/GRIP#2722) — never written from the HAL session
(placement boundary).

## Falsification & findings (the science, not a press release)

Hypotheses registered up front; resolved with evidence:

- **H-RAILLM-1/2/3/4 — CONFIRMED.** Core diff empty; 22 Goodhart-proof
  hermetic tests drive the REAL runtime. The facade is pure composition.
- **H-RAILLM-5 — FALSIFIED as stated.** "Decoupled from any single
  provider's context window" is **false** when the primary rung has a
  window smaller than the live thread *and* it fails:
  `dispatch_with_history` runs `compress_if_needed(budget)` on the
  **shared** thread for *every attempted rung before the chat call*
  (`raillm.py` L377-378), and the base `LimitSentinel` does not back a
  failing primary off at realistic volumes — so the tiny-window failing
  primary destructively truncates the shared thread oldest-first *before*
  the larger fallback rung is tried. Turn-1 content is lost across the
  CCH→Kimi failover. Core fix tracked: **CodeTonight-SA/HAL#329**.
  - **True sub-claim that survives** (tests assert the precise boundary):
    failover *transparency* is intact (the caller never sees a limit), and
    history continuity *does* hold when every attempted rung's window ≥ the
    live thread size. Those are the boundaries where the goal is met today.

This is a successful wave: a falsification engine that only confirms is
not doing science. The defect was found *before* GRIP depends on the path.

## Consequences

**Positive**: provider-agnostic envelope boundary; audit-ready
(deterministic `idr`); CCH/Kimi/Qwen/local cascade with history carry;
zero core rewrite (reversible, low-risk).

**Negative / open risk**: GRIP#2722 is required before `grip_infer` is
actually GRIP's default (W4 makes it *available*, not yet *wired*).

## Resolution — HAL#329 (fixed)

The compress-before-failover defect W5 falsified is **fixed** on branch
`fix/raillm-compress-before-failover` (W1 `ebf45a61`, W2 `2e5d80bd`):

- `ConversationThread._fit_to_budget` — pure drop-oldest fitter, single
  source of truth (DRY).
- `compress_if_needed` — unchanged mutating contract (delegates to the
  fitter; 44 regression tests prove preservation).
- `compressed_prompt` — NEW non-mutating `(prompt, dropped)`.
  `dispatch_with_history`'s wire path now uses it: a failed small-window
  rung gets a per-rung compressed COPY while the **shared thread stays
  the canonical full history**. Only the serving rung's reply is appended.

The W5 regression sentinel inverted and passes —
`test_small_window_failing_primary_preserves_shared_thread` now asserts
continuity HOLDS (10 turns retained, turn-1 fact reaches the fallback
rung). **H-RAILLM-5 → CONFIRMED; H-RAILLM-7 → CONFIRMED.** 94/94 full
regression green.

Devil's-advocate scope (R0 — NOT regressions of this fix): unbounded
shared-thread growth and per-message provider-fidelity replay are noted
future enhancements, tracked separately if pursued. The #329 fix is a
strict improvement — no permanent destruction; a larger future rung sees
*more* context, never less.

## Plain-language summary

GRIP can now ask any of several AI providers a question through one front
door (`grip_infer`). If the first provider is rate-limited or down, it
automatically tries the next — subscription, then cheap, then a local
model — without the caller noticing, and it carries the conversation along
so context is not lost on the switch. We tested the strong claim "the
conversation can never be limited by one provider's memory size". W5
found a real bug — if the *first* provider had a small memory limit and
failed, the shared conversation was trimmed before the bigger provider
was tried. **That bug (HAL#329) is now fixed**: each provider gets its
own trimmed *copy* for the wire, while the full conversation is kept
intact and handed to whichever provider answers. The strong claim now
holds, and a regression test guards it.

## Durable backlog (for the /sprint · /rsi · /auto loop)

All `sprint-loop`-labelled and traceable:

- ~~**HAL#329** — core fix: don't compress the shared thread to a failing
  rung's window before failover~~ **FIXED** (`fix/raillm-compress-before-failover`,
  H-RAILLM-5/7 CONFIRMED) — closing with this branch's PR.
- ~~**GRIP#2722** — rewire `hal_llm_adapter.py` through `grip_infer`~~
  **RESOLVED 2026-05-16** — transport B LIVE: `lib/hal_llm_adapter.py`
  primary path is the HTTP POST → hal-server `/api/infer` → `grip_infer`;
  the interim subprocess bridge (transport C, #2726) is purged;
  `rules/syscall-doctrine.md` reference-set updated (GRIP session, #2747
  hardened on top). Verified in code this session, not claimed.
- **HAL#322–#327** — the W1–W6 wave tickets; W1–W5 closed on completion,
  W6 closed by this ADR.

## References

- PR CodeTonight-SA/HAL#328 — commits `3f4fb787` W1, `cbe82e25` W2,
  `6bb2e833`/`841d1c88` W3, `84d5bd12` W4, `dba78b98` W5.
- Tests: `tests/test_raillm_subscription_bridge.py`,
  `tests/test_raillm_ladder.py`, `tests/test_happi_infer.py`,
  `tests/test_grip_runtime.py`,
  `tests/test_e2e_infinite_machine.py::TestGripLadderContextWindowDecoupling`.
- Hypotheses: `data/hypotheses.jsonl` (H-RAILLM-1..6).
- Related: `docs/RAILLM.md` (mechanism), `plans/hal-raillm-grip-runtime.md`.
