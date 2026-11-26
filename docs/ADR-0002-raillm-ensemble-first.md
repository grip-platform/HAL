# ADR-0002 — RAILLM ensemble-first: DFI-FIB orchestrates, the ladder demotes to intra-call failover

- **Status**: Proposed
- **Date**: 2026-09-15
- **Supersedes (partly)**: ADR-0001 — its substrate stands; its Anthropic-first ladder
  *order*, and the ladder's standing as the primary mechanism, do not.
- **Branch**: `feat/raillm-ensemble-first`
- **Related**: `docs/ADR-0001-hal-raillm-grip-runtime.md`, `docs/RAILLM.md`,
  `lib/hal/raillm.py`, `lib/hal/raillm_ladder.py`, `lib/hal/grip_runtime.py`,
  `lib/improve_kernel.py` (`InnerspeakerGo`), `docs/DFI-FIB.md`.

## Context

ADR-0001 is Accepted and its substrate shipped: `grip_infer`, the happi `infer`
command, `RAILLMSentinel.dispatch_with_history`, and GRIP's adapter routing its
primary path to HAL `/api/infer`. GRIP already runs *through* HAL. That is not in
question here.

Two things have changed since.

**1. The dependency is being retired.** On 2026-09-14 Anthropic ended Claude
Code's temporary 50% weekly boost and replaced it with +25% over each plan's
baseline — a net ~17% cut for current users — and the Opus sub-limit was removed.
The binding constraint is limits, not price. The operator has decided to stop
depending on Anthropic's harness.

**2. A ladder is not a reasoning strategy.** `raillm_ladder.py` encodes a
*sequential, Anthropic-first* order:

    "anthropic/claude-opus-4-7",   # 1. CCH subscription (primary)
    "moonshot/kimi-k3",            # 2. ...
    "groq/..." "deepseek/..." "ollama/..." "mlx_vlm/..."

A ladder answers *"this provider is down, rate-limited or too expensive — what
next?"* It is about **availability and cost**, and it is inherently sequential.
It does not answer *"is this answer right?"*, which is a **quality** question and
is inherently parallel. The current shape asks a quality question of a failover
mechanism, and its first rung is the very dependency being retired.

The doctrine already exists in mechanical form. about.grip-web.com §03 states the
principle — *"Width without depth is gossip. Depth without width is obsession."* —
and GRIP already ships the kernel that enforces both: **DFI-FIB**'s
`InnerspeakerGo`, one improvement step plus Fibonacci depth control and a
persistent fixed-point registry.

## Decision

1. **DFI-FIB orchestrates.** Depth is the improvement loop; width is the fan-out
   *inside a single step*; convergence is the kernel's stop (threshold passed,
   fixed point, or the depth cap). No new orchestrator is built — the kernel at
   `lib/improve_kernel.py` is the one correct loop.
2. **Roles are split.** *Reasoners* produce; *judges* adjudicate and are **never
   reasoners in the same round** (`rules/exogenous-anchor-law.md`: the verifier is
   never one of the producers). A model that both reasons and judges grades its
   own homework, and disagreement stops being a signal.
3. **Reasoner core = three**, spanning three vendors and one open weight:
   `openai/gpt-6-astra`, `deepseek/deepseek-v4-pro`, `anthropic/claude-fable-5`.
4. **Grok is a conditional fourth.** `xai/grok-4.20-0309-reasoning` joins only on
   escalation — extra power on the turns that need it, not every turn.
5. **The judge is a rotating panel of two exogenous judges**, drawn per state from
   `google/gemini-3.1-pro-preview`, `qwen/qwen3.8-max`, `moonshot/kimi-k3`,
   `zhipu/glm-5.1` — all vendors absent from the reasoner set.
6. **The ladder is demoted**, not deleted: it survives *only* as intra-call
   failover inside a rung. It stops pretending to be orchestration.
7. **A difficulty router** decides breadth: routine turns take the cheap tier;
   hard turns take the reasoner core; contested turns escalate.

## The step (contract)

```python
def ensemble_step(state, depth) -> ImproveResult:
    tier     = tier_for(depth)                      # cheap → core → escalate
    reasoners = CORE if tier < ESCALATE else CORE + [GROK]
    answers   = raillm.fan_out(state, role="reasoner", models=reasoners, parallel=True)
    agree     = agreement_of(answers)               # deterministic; no model call to decide
    if agree >= 0.9:
        return ImproveResult(consensus(answers), {"agreement": agree}, depth, fp(consensus))
    judges    = judge_combo(fp(state), seed)        # rotating panel, exogenous to `reasoners`
    ruling    = raillm.judge(state, answers, models=judges, parallel=True)
    return ImproveResult(ruling.text, {"agreement": agree, "judge_conf": ruling.conf}, depth, fp(ruling.text))

InnerspeakerGo(converge=ensemble_step,
               threshold={"agreement": ("ge", 0.9)},
               max_depth=13).__call__(task)        # F(7), matches the RSI wave cap
```

**Purity is a precondition.** `docs/DFI-FIB.md` ("When NOT to use the kernel")
warns that a non-deterministic step breaks fixed-point detection: a model call is
non-deterministic, so as written the step would either never reach a fixed point
or short-circuit wrongly on a repeated state. The step is therefore made pure **by
construction** — model outputs are memoised by `(prompt, model, params)`, so
`ensemble_step` is a deterministic function of its input. This is a hard
dependency, not an optimisation.

## Rotation, and why it still replays

The judge panel is drawn as `random.Random(f"{fingerprint(state)}:{seed}")`
sampled without replacement from the exogenous pool, with the invariant
`judges ∩ reasoners = ∅` enforced per round. The `seed` is sealed in the IDR.

Consequence: rotation is **reproducible within a state** (the same problem always
draws the same panel, so a ruling replays exactly) while **decorrelated across
states** (different problems draw different panels, so no single model's blind
spots become a permanent anchor). Rotation makes rulings harder to game *without*
making them harder to replay — which is the only reason it is admissible in a
substrate whose product is a replayable record.

## Escalation trigger

Deterministic, evaluated on the round's own metrics — never on vibes:

    escalate  iff  agreement < 0.9  and  judge_conf < CONF_FLOOR
             or    the turn carries a high-stakes flag

So Grok buys in exactly when the trio and the judges could not settle.

## Cost and latency

The judge runs **only on disagreement**, so a settled turn costs three reasoner
calls and no adjudication. The persistent fixed-point registry means a state that
has already been settled is never re-paid for. Astra (`$10/$50` per 1M) is the
expensive specialist, not the default path; Fable is flat-fee subscription at zero
marginal cost; DeepSeek is a fraction of Astra's price. Frontline cost is a
hard-turn cost, not a per-turn cost.

## Falsification pre-registration

Registered before implementation, per the converge mandate:

- **H1** — ensemble-first beats a single frontier model on a held-out reasoning set
  at equal or lower cost. *Falsified if* the quality gain does not exceed the
  cost delta.
- **H2** — a rotating exogenous judge catches errors a fixed judge misses.
  *Falsified if* A/B shows no difference on a seeded error set.
- **H3** — fingerprint-keyed rotation replays exactly. *Falsified if* the same
  state + seed yields a different panel.
- **H4** — the purity memo makes fixed-point detection correct. *Falsified if*
  removing the memo does not break the fixed-point test.

## Defect found and fixed alongside: xAI catalogue ↔ provider drift

Probed live against `api.x.ai/v1/models` on 2026-09-15 — and **both** config files
were wrong, in opposite directions:

- `providers.yaml` listed six ids, **five of which the endpoint no longer serves**:
  `grok-3`, `grok-3-mini`, `grok-4-0709`, `grok-4-fast-non-reasoning`,
  `grok-code-fast-1`. Any rung naming one of these 404s at runtime.
- `model_catalog.yaml` advertised `grok-4.5` and `grok-4.20-0309-reasoning`, which
  *are* served, while omitting the also-served `grok-4.6`.

Served text models, from the live probe: `grok-4.20-0309-reasoning`,
`grok-4.20-0309-non-reasoning`, `grok-4.20-multi-agent-0309`, `grok-4.6`,
`grok-4.5`, `grok-4.3`, `grok-build-0.1`.

The escalation rung is therefore **`xai/grok-4.20-0309-reasoning`** (the
1M-context reasoning variant), not `grok-4.6`. `grok-4.20-multi-agent-0309` is
recorded as a candidate for a future multi-agent fan-out tier — noted, not adopted.
Both config files are corrected in this branch.

## Consequences

**Positive**: reasoning quality bought with architectural independence rather than
a bigger single model; availability preserved by a demoted ladder; replay intact;
the Anthropic first rung retires from the critical path.

**Negative / open**: frontier fan-out is slower and dearer than one call on hard
turns; the rotating panel adds a configuration surface; the memo layer is now a
correctness dependency. The single-node scope is unchanged — this is one machine's
runtime, not a distributed fleet.

**Placement**: code lands in HAL. GRIP-side wiring changes are a separate GRIP
ticket, never written from a HAL session (ADR-0001's placement boundary).

## Plain-language summary

Today HAL tries one model, and if it fails, tries the next in a fixed order. That
answers "who is awake?", not "is the answer right?". This ADR keeps a small
failover list but stops using it as the strategy. Instead, three strong models
from three different companies answer in parallel; if they agree, that is the
answer and nothing else is spent. If they disagree, a rotating panel of two other
models — never the same companies — rules on the disagreement, and if they still
cannot settle, a fourth model (Grok) is brought in. Every round is recorded and
can be replayed exactly, because the panel is chosen from the content of the
question rather than at random. A ladder tells you who is awake; this tells you
whether to believe the answer.
