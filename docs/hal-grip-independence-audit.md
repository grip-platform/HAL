# HAL → GRIP Independence Audit

**Status:** evidence-backed inventory of the remaining coupling points
between HAL (`~/.hal`) and GRIP (`~/.claude`), the two personas those
couplings serve, and a pragmatic roadmap for full runtime independence.

**Date:** 2026-04-19 · **Sprint:** ultrado research wave after β-4b + γ-1b shipped.

---

## TL;DR

HAL is already *architecturally* close to GRIP-substitutable — there is
an explicit one-way absorption machinery (`lib/hal/grip_harness.py`)
that copies GRIP's skills, agents, and hooks into HAL's tree on demand.
The runtime import graph is already almost disjoint. What keeps HAL
practically dependent falls into **seven concrete buckets**, and only
two of them affect HAL's user-visible CLI. The rest affect the
*session persona* — the meta-orchestration layer (slash commands, RSI,
hypothesis tracking) that currently only works inside a GRIP-aware
Claude Code session.

Three-sentence version:

1. **HAL-the-CLI** (`hal chat`, `hal --serve`, `hal osa run`, `hal happi speak`) is near-substitutable for a GRIP-less host today — remaining leaks are soft reads for chat-context injection and a cosmetic env-var default.
2. **HAL-the-session** (slash commands, autonomous loops, `/save`, `/ultrado`, `/broly`) is *runtime*-coupled to GRIP via PYTHONPATH tricks and inherits the entire slash-command surface from GRIP's `commands/` directory — substitutability is not the same as behavioural independence.
3. **The absorption pipeline** is the crucial lever — once absorbed skills, hooks, and command definitions get *path-rewritten during copy* (currently they do not), HAL can run both personas without reaching into `~/.claude` at runtime.

> **Substitutability vs behavioural independence.** A HAL that starts
> without GRIP installed but then *behaves indistinguishably from
> GRIP* (same commands, same rules, same agent catalogue) is
> substitutable, not independent. This audit treats the two as
> distinct — and flags the places where HAL's current strategy
> conflates them.

---

## Personas of HAL

| Persona | Entry point | Who runs it | GRIP-dependence today |
|---------|-------------|-------------|------------------------|
| CLI | `hal`, `hal --serve`, `hal osa run`, `hal happi speak` | End-user at a terminal, or another tool via subprocess (this is how GRIP's `lib/hal_osa.py` adapter talks to HAL) | Near-zero runtime imports; soft reads for LLM context injection |
| Session | `/ultrado`, `/converge`, `/broly`, `/save-exit`, `/models` | Operator inside Claude Code with `cwd=~/.hal` | Borrows GRIP libs by `PYTHONPATH=$HOME/.claude` tricks |
| Background | Hooks checked into `~/.hal/hooks/` | GRIP's hook dispatcher (not HAL's runtime) | These *are* GRIP hooks, checked into HAL for mirror-absorption purposes |

Separating personas is the single biggest clarification this audit
produces. "Is HAL GRIP-independent?" has two answers, not one.

---

## Seven coupling buckets (evidence-backed)

### Bucket 1 — Soft context injection (FAIL-SAFE, low severity)

**What**: `lib/hal/grip_bridge.py` reads GRIP state (CLAUDE.md, session-context.md,
`cache/genome.json`, guild graphs) and injects a ~500-token summary into
the chat model's context window, so the model "knows what GRIP is".

**Where it's called from**: `lib/hal/context_assembler.py:145` —
`grip_ctx = load_grip_context(); sections.append(f"## GRIP System Context\n{grip_ctx}")`

**Failure mode today**: `if not CLAUDE_DIR.exists(): return ""`.
HAL runs fine on hosts without GRIP — the GRIP section is simply
absent from the model's context. This is already *correct behaviour*
for a standalone runtime.

**Severity**: cosmetic. This bucket is *already* GRIP-independent in
the sense that matters — it fails closed. Worth keeping for HAL-in-
GRIP sessions; no action needed for independence.

### Bucket 2 — The GRIP harness absorption machinery (INTENTIONAL, keep)

**What**: `lib/hal/grip_harness.py` is an explicit pipeline:

```
scan(~/.claude/skills) → classify portability (regex on CC markers)
                       → copy portable → ~/.hal/skills/
scan(~/.claude/agents) → same pipeline → ~/.hal/agents/
scan(~/.claude/hooks)  → same pipeline → ~/.hal/hooks/
```

HAL today has **~50 absorbed skills, 41 agents, 48 hooks** in its tree
(counted 2026-04-19). This is the prescribed route for HAL to inherit
GRIP's innovations without importing them at runtime.

**The load-bearing gap**: absorption copies files verbatim.
`shutil.copytree`/`copy2` does NOT rewrite `~/.claude` → `~/.hal` in the
copied content. Run a grep on `~/.hal/hooks/`: you'll find dozens of
hooks that HAL now ships but which still hardcode `~/.claude`. HAL
thinks it absorbed them; at runtime, those hooks still reach into
GRIP's state directory.

**Evidence**:
- `hooks/auto-memory-hook.py:31` → `CLAUDE_DIR = Path(os.environ.get("CLAUDE_DIR") or Path.home() / ".claude")`
- `hooks/api_audit.py:47` → `AUDIT_LOG = Path.home() / ".claude" / "logs" / "api-audit.jsonl"` (no env override)
- `hooks/recall-worker.py:24` → `sys.path.insert(0, str(Path.home() / ".claude" / "lib"))`

**Severity**: high. This is the single biggest gap between "HAL has
absorbed it" and "HAL runs it standalone".

**Fix**: `grip_harness._absorb_one` should path-rewrite during copy:
`content.replace(".claude/", ".hal/")` with an allow-list of literal
path tokens. The auto-absorb README already says "portable components"
— portable *means* path-rewritten.

### Bucket 3 — Runtime lib imports via PYTHONPATH (SESSION ONLY)

**What**: HAL's slash commands reach into GRIP's lib by env-var:

| GRIP lib | Invoked by | Purpose |
|----------|-----------|---------|
| `~/.claude/lib/hypothesis_engine.py` | `/ultrado`, every HAL PR in this session | H-code registry, falsification tracking |
| `~/.claude/lib/gates/yolo_mode.py` | `/ultrado` phase 0.1 | Halting-gate bypass for autonomous loops |
| `~/.claude/lib/gates/session_isolation.py` | `/ultrado` phase 0.0 | Rule 13 worktree enforcement |
| `~/.claude/lib/ultrado_checkpoint.py` | `/ultrado` phase 6 | Wave-level crash recovery |
| `~/.claude/lib/ultrado_precision.py` | `/ultrado` phase 0.5 | Context-adaptive wave planning |
| `~/.claude/lib/grip-worktree.sh` | `/ultrado`, β-4b sprint | Git-crypt-aware worktree spawn |
| `~/.claude/lib/grip_save.py` | `/save-exit` | Cross-project state persistence |

These never reach HAL-the-CLI. They are only invoked inside a GRIP
session when the operator types `/ultrado` etc.

**Severity**: high for the session persona, zero for the CLI persona.

**Fix for the session persona**: port each lib into `~/.hal/lib/hal/`
or (kinder) accept that the session persona lives on top of GRIP's
skill/rule substrate and document that explicitly. The RSI engine is
conceptually GRIP-shaped; forcing it to relocate to HAL would fork the
innovation surface.

### Bucket 4 — Memory sync's legacy naming (COSMETIC)

**What**: `lib/hal/memory_sync.py:67` still accepts a `grip_dir=` kwarg
for back-compat:

```python
def __init__(self, peer_dir: str = "", hal_dir: str = "", *, grip_dir: str = ""):
    self._peer = peer_dir or grip_dir or _PEER_MEMORY
```

**Severity**: cosmetic. The concept has already been renamed
`peer_dir`; `grip_dir` is a deprecated alias. The coupling is a *name*,
not a mechanism — HAL's memory-sync can peer with any GRIP-shaped
project directory.

**Fix**: remove `grip_dir` kwarg at the next major bump; update any
remaining caller sites. Grep confirms the live call sites have already
moved to `peer_dir`.

### Bucket 5 — CLAUDE_DIR env var in satellite apps (PARTIAL)

**What**: `apps/copilot-api/main.py:27`:

```python
CLAUDE_DIR = Path(os.environ.get("CLAUDE_DIR", Path.home() / ".claude"))
```

The copilot-api satellite reads `CLAUDE_DIR` and writes into it. Useful
as an integration endpoint for GRIP's copilot stack; problematic if a
standalone HAL user doesn't have GRIP installed and the default path
doesn't exist.

**Severity**: medium — satellite apps should support `HAL_HOME` as a
first-class alternative. The env-var override already exists; making
`HAL_HOME` the default when `CLAUDE_DIR` is unset would fix the
standalone case without breaking the GRIP-integrated case.

### Bucket 6 — Slash-command definition surface (UPSTREAM-OWNED)

**What**: Slash commands themselves — `/ultrado`, `/converge`,
`/save-exit`, `/broly-auto`, `/models`, `/compact`, `/exit` — are
authored as files under `~/.claude/skills/<name>/SKILL.md` and
`~/.claude/commands/`. Session-HAL inherits them by being invoked
inside a Claude Code session whose `CLAUDE_DIR` points at GRIP. Even
after bucket 2 rewrites absorbed command *content*, the *authoring
convention* (where new commands appear, who reviews them, which repo
they live in) remains GRIP's.

**Severity**: architectural. A HAL that absorbs commands but never
authors them is a perpetual downstream — it will always lag GRIP by
one absorption cycle, and any HAL-specific command must either be
pushed upstream to GRIP or cause a schism in the command surface.

**Fix**: decide whether HAL owns a parallel command-authoring
convention (`~/.hal/skills/<name>/SKILL.md`, absorbed back into GRIP
if portable) or whether the session persona explicitly is GRIP's
concern. The answer is bound to the Phase C strategic choice below.

### Bucket 7 — Missing standalone equivalents (FOUNDATIONAL)

Features that exist in GRIP and are used by HAL's session persona but
have no HAL-native implementation:

| Feature | GRIP lives at | HAL-native equivalent |
|---------|---------------|------------------------|
| Hypothesis engine (H-codes, falsification registry) | `~/.claude/lib/hypothesis_engine.py` | None — session persona reaches up |
| Save orchestrator | `~/.claude/lib/grip_save.py` | `plans/hal-save-orchestrator.md` backlog (not shipped) |
| KONO memory substrate | `~/.claude/lib/kono_*` | **Partially decoupled** — see `docs/HAL-KONO-STANDALONE.md` and `plans/kono-standalone-vendor.md`; work in progress |
| Worktree-aware git helper | `~/.claude/lib/grip-worktree.sh` | HAL uses it via `--repo ~/.hal` flag — no HAL copy |
| Genome / DNA | `~/.claude/cache/genome.json` + `lib/converge_dna.py` | None; HAL reads GRIP's genome via `grip_bridge._read_genome` |
| Broly meta-agent | `~/.claude/agents/broly.md` + skill scaffolding | Absorbed but depends on GRIP's agent dispatcher |
| Yolo mode marker | `~/.claude/state/session/yolo-mode.json` | None |

**Severity**: foundational. A HAL session persona that could run without
GRIP would need its own hypothesis engine, its own save orchestrator,
its own checkpoint system. That's real engineering, not a rename.

---

## Severity-weighted roadmap

In order of highest-impact-per-effort:

### Phase A (high impact, bounded-but-real effort)

1. **Path-aware absorber for `grip_harness._absorb_one`** — bucket 2.
   **Not a one-function edit**. The broly council flagged four
   breakages that turn this into a bounded subsystem with a test
   matrix:

   - **Idempotency**: content-hash change-detection trips on every
     absorb because rewritten bytes differ from source. Solution:
     hash the *rewritten* form, not the source, when determining
     "changed since last absorb".
   - **`chflags uappnd`-locked files** (audit chains): writes are
     rejected by the OS. The rewriter must detect the append-only
     flag and skip such files — silent failure corrupts the audit
     chain.
   - **Symlinks**: copying a symlink's target string and rewriting
     its path produces a broken link when `~/.hal` mirror doesn't
     exist. The rewriter must detect `os.path.islink` and either
     copy the link verbatim or relocate the target.
   - **Binary assets and git-crypt-encrypted files**: text
     substitution on binary content is corruption. The rewriter
     must whitelist text MIME types (driven by `.gitattributes` in
     git-crypt repos).

   The fix is a ~150-line function with a ~15-case unit-test matrix,
   not a two-edit patch. Budget accordingly.

2. **Default `CLAUDE_DIR` to `HAL_HOME` in satellite apps** — bucket 5.
   Two-line change in `apps/copilot-api/main.py`: if `CLAUDE_DIR`
   unset AND `HAL_HOME` set, use `HAL_HOME`.

3. **Retire `grip_dir=` kwarg** — bucket 4. One deprecation cycle and
   it's gone.

### Phase B (medium impact, medium effort)

4. **Ship `bin/hal save`** — bucket 7 + known backlog
   (`plans/hal-save-orchestrator.md`). Port the subsystem chain
   (insights flush, genome sync, session context write) into
   `~/.hal/lib/hal/save_orchestrator.py`. Keep `grip_save.py`
   subprocess-callable for users who have both installed.

5. **Port hypothesis engine** — bucket 3+7. Extract `hypothesis_engine.py`
   into `~/.hal/lib/hal/hypothesis/`; keep H-code continuity so
   existing registrations don't break.

6. **Port yolo gate + session isolation** — bucket 3. These are
   small, self-contained Python modules. Relocate as
   `~/.hal/lib/hal/gates/`.

7. **Decide command-authoring home** — bucket 6. Either (a) HAL
   publishes commands under `~/.hal/skills/*/SKILL.md` and GRIP
   absorbs from HAL (reverse the flow for commands that originate
   in HAL), or (b) HAL formally states that commands are authored
   upstream only. Either decision closes the loop; leaving it open
   drifts.

### Phase C (high impact, high effort — sequencing question)

7. **Decide on the session persona's home.**

   - **C.a HAL-as-CLI-only** *(staging checkpoint, not a terminal
     choice)*: the session persona explicitly belongs to GRIP. Close
     out buckets 1, 2, 4, 5 so HAL-the-CLI is fully standalone;
     accept that `/ultrado` is a GRIP command operating on HAL's
     project tree. Useful as an **interim**: publishes a clean HAL
     CLI to users without GRIP, gives time to absorb the session
     substrate carefully.
   - **C.b HAL-as-full-harness** *(terminal state required by Fleet
     Command)*: finish absorbing the session layer (hypothesis
     engine, yolo, checkpoint, save orchestrator, command authoring
     convention) so HAL ships as a standalone competitor to Managed
     Agents et al.

   **The broly MESH council's integrator role observed that C.a is
   not viable as a terminal state if "HAL Fleet Command" (the
   provider-agnostic harness competitor memorandum) is the product
   vision.** Fleet Command competes at the harness layer; shipping
   HAL without the harness would cede exactly the ground it claims
   to win. The strategic question is therefore *sequencing*, not
   *optionality*: ship C.a first as a checkpoint, commit to C.b as
   the terminal state. Treating C.a as final would orphan the
   session persona inside GRIP forever.

### Phase C — decision (r3, 2026-04-19)

**Decision: commit to Phase C.a as the 30-day terminal state; defer
C.b (Fleet Command) to an explicit triphase plan gated by operator
approval after the C.a evidence window closes (2026-05-19).**

This is a *staging* commitment, not a *strategic* one: C.b remains
the long-term terminal state per the Fleet Command memorandum, but
the intermediate 30 days prioritise closing the substitutability
gap (HAL-the-CLI runs without GRIP installed) before taking on the
harness-authorship cost. The decision is tracked as **H164**
(deadline 2026-04-26 for the decision doc itself; separate 30-day
evidence window for the C.a→C.b pivot criteria below).

**Evidence supporting C.a-first**:

1. Phase B landed standalone `lib/hal/hypothesis`, `bin/hal save`,
   and `lib/hal/gates/yolo_mode.py` under 1 054 lines with 36 tests
   green — proof that the most load-bearing substrate ports are
   tractable one at a time without a full harness rewrite.
2. Phase A's path-aware absorber (15-case matrix + 4 helpers, 19
   tests) confirms H145: the one-way GRIP→HAL sync can be made
   idempotent + chflags-safe without rewriting the absorption
   direction.
3. γ-2a + γ-2b + γ-2c shipped tier-3 consent, rate limits, and
   allowlists (54 tests across 3 PRs, zero cross-wave breakage)
   — each fits the HAL-as-CLI scope without needing a new command
   authoring home.
4. The substitutability test `HAL_HOME=/tmp/fresh hal save "msg"`
   works today (verified in Phase B smoke); H-indep-5 can be closed
   with 2–3 more bucket-5 PRs rather than an architectural pivot.

**The next 5 PRs that realise C.a** (in priority order):

| PR | Surface | Closes |
|----|---------|--------|
| 1 | Path-aware absorber invoked against a clean `HAL_HOME=/tmp/fresh` | Bucket 2 end-to-end |
| 2 | `copilot-api` + satellite apps resolve `HAL_HOME` before `CLAUDE_DIR` | Bucket 5 |
| 3 | `hal chat "hello"` with `CLAUDE_DIR` unset / unreadable | H-indep-2 live |
| 4 | Remove `grip_dir=` deprecated kwarg from HAL call sites | Bucket 4 |
| 5 | Audit report: `grep -rn '\.claude' ~/.hal/hooks ~/.hal/skills ~/.hal/agents` returns zero matches post-absorb | H-indep-1 closure |

γ-2d (shortcut backend) and γ-2e (seed registry entries) continue in
parallel; they fit the osa sub-plan, not the C.a bucket.

**Falsification criteria that would flip the decision to C.b now**
(not at the end of the 30-day window):

- **F1 — CLI substitutability blocked**: more than one of the 5 PRs
  above requires introducing a new harness-authoring convention (new
  slash-command loader, new agent registry, new skill resolution
  path). If the surface area of C.a grows past existing abstractions,
  the pivot is C.b or nothing. Falsification check: after each of
  the 5 PRs lands, verify that the GRIP surface remains single-file
  and path-portable.

- **F2 — Fleet Command date anchor slips**: if the Fleet Command
  memorandum attracts an external customer commit or a public date
  anchor during the 30-day window, C.a becomes dead weight and the
  pivot to C.b is immediate — shipping without the harness surrenders
  the competitive position Fleet Command claims.

- **F3 — Substitutability proven but unused**: if, after PRs 1–5
  land, nobody actually runs `HAL_HOME=/tmp/fresh hal` without GRIP
  in the wild (V>> does not, no external user does, no CI smoke
  exercises it), then C.a was a correct port but useless delivery
  — the 30-day window closes without evidence and C.b becomes the
  only forward path. Falsification check: at day 30, does at least
  one entity that is not V>>'s dev laptop run HAL standalone?

**On all three F-criteria**: the flip is explicit (operator runs
`/pivot-to-Cb`-style intent signal); no ambient ratchet. Keeping
the decision reversible is the point.

**Hypotheses registered for the decision window**:

- **H164** — this decision doc is committed by 2026-04-26 with all
  three bullets above (next 5 PRs, 3 falsification criteria,
  operator pivot signal contract). CONFIRMED by this PR.
- **H-indep-5** — remains active; deadline aligns with the 30-day
  window (close by 2026-05-19).

**On sequencing**: the triphase plan for C.b is *NOT* authored in
this PR. Authoring it now would anchor the pivot prematurely and
waste operator attention on a route we may not take. The plan is
drafted only if F2 fires (date anchor) or the 30-day window closes
without refutation and V>> explicitly signals the C.b commit.

---

## Falsifiable claims

- **H-indep-1**: After Phase A lands, `grep -rn '\.claude' ~/.hal/hooks ~/.hal/skills ~/.hal/agents` returns zero matches in absorbed components. *Falsified if* any absorbed hook/skill/agent still contains a literal `.claude` path after the next `GRIPHarness.scan().absorb()` run.

- **H-indep-2**: After Phase A lands, `HOME=/tmp/fresh-user hal chat "hello"` runs to first model response without raising an AttributeError or reading anything under `~/.claude`. *Falsified if* any file under `~/.claude` is touched by the HAL-CLI happy path.

- **H-indep-3**: After Phase B lands, `bin/hal save "achievement X"` persists HAL project memory *without* invoking `~/.claude/lib/grip_save.py`. *Falsified if* the HAL save still subprocess-calls GRIP.

- **H-indep-4**: The session persona (`/ultrado`, `/save-exit`) continues to work unchanged after Phase A + Phase B — operators see no regression. *Falsified if* any slash command that worked before Phase A starts failing after.

- **H-indep-5**: HAL's copilot-api satellite starts correctly with `HAL_HOME` set and `CLAUDE_DIR` unset. *Falsified if* it raises FileNotFoundError for `~/.claude` paths after bucket 5 is closed.

---

## What this audit is NOT

- **It is not a prescription**: which stance to take (Phase C.a vs C.b)
  is V>>'s decision, driven by the Fleet Command vision. This
  document gives the data; the choice is strategic.
- **It is not exhaustive on the hooks**: `~/.hal/hooks/` has 48 files;
  this audit sampled ~10. A full pass will likely surface additional
  `~/.claude` literals that a path-rewriter would fix mechanically.
- **It is not a critique of the absorption model**: one-way absorption
  from GRIP is a sensible architecture. The only flaw is that the
  current absorber does raw copy instead of path-aware copy.

---

## How this audit was produced

1. **Scoping greps** of `~/.hal/lib/**`, `~/.hal/hooks/**`,
   `~/.hal/bin/**`, `~/.hal/apps/**` for `.claude`, `CLAUDE_DIR`,
   `from grip|import grip`, `~/.claude/lib`.
2. **Close read** of `lib/hal/grip_bridge.py`, `lib/hal/grip_harness.py`
   (both entry points into the coupling).
3. **Enumeration** of the 40+ `~/.claude` literals found across HAL's
   hook tree to confirm the absorption-drift pattern.
4. **Cross-reference** against existing HAL plans
   (`plans/kono-standalone-vendor.md`, `docs/HAL-KONO-STANDALONE.md`,
   `plans/hal-save-orchestrator.md`) — this is not the first time
   independence has been investigated; prior work is narrower in
   scope.
5. **Structural-mapping test**: each bucket has (a) concrete file
   evidence, (b) a named failure mode, (c) a proposed fix. No
   hand-waving; every claim cites a path.

---

## Appendix — one-liner audit commands for re-running this

```bash
# Bucket 2: absorbed components still pointing at GRIP
grep -rn "\.claude" ~/.hal/hooks ~/.hal/skills ~/.hal/agents | wc -l

# Bucket 3: live PYTHONPATH reaches into GRIP from HAL skills
grep -rn "PYTHONPATH=\$HOME/.claude\|PYTHONPATH=~/.claude" ~/.hal/skills

# Bucket 5: satellite apps defaulting to ~/.claude
grep -rn "CLAUDE_DIR" ~/.hal/apps ~/.hal/bin

# Verify HAL-CLI standalone-ness (should not touch ~/.claude)
HOME=/tmp/fresh HAL_HOME=/tmp/fresh-hal strace -e openat -o /tmp/hal-calls.log \
    python3 -m lib.hal.cli --help 2>&1
grep "\.claude" /tmp/hal-calls.log
```

If those commands return non-empty at the end of Phase A, the phase
isn't done. If they return empty, H-indep-1 and H-indep-2 hold.

---

## Revision history

- **2026-04-19 r1**: initial draft, six buckets.
- **2026-04-19 r2**: incorporated broly MESH critique
  (APPROVE-WITH-CHANGES). Added **bucket 6** (slash-command
  definition surface) — prior draft elided it. Reclassified Phase A
  absorber fix from "one-function edit" to "bounded subsystem with
  test matrix" (idempotency, chflags, symlinks, binaries, git-crypt).
  Reframed Phase C.a from terminal choice to staging checkpoint —
  Fleet Command vision commits HAL to C.b. Added
  substitutability-vs-independence distinction to TL;DR to tighten
  the "percent independent" framing. Analyst/Devil's Advocate/
  Integrator roles in the broly council all contributed load-bearing
  changes; none was cosmetic.

- **2026-04-19 r3**: committed to Phase C.a as 30-day terminal state
  with explicit flip criteria (F1 CLI substitutability blocked, F2
  Fleet Command date anchor slips, F3 substitutability proven but
  unused). Listed the next 5 PRs that realise C.a in priority order.
  Registered **H164** (this decision doc committed by 2026-04-26).
  Evidence supporting C.a-first: Phase A (19 tests, 15-case matrix,
  PR #247) + Phase B (36 tests across 3 surfaces, PR #251) +
  γ-2a/γ-2b/γ-2c (58 tests across 3 PRs — #244, #250, #259) landed
  without needing a new command-authoring home. H143 offline argv
  shape pinned by 9 synthetic tests (PR #260); live smoke runbook
  shipped. Triphase C.b plan *deliberately NOT authored* in r3 to
  avoid anchoring a pivot we may not take.
