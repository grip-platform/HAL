# RAILLM -- Redundant Array of Independent LLMs

> CompSci 101 -- redundancy. The chain of operations cannot break under any
> circumstances. RAILLM is RAID-1 for LLMs: full replication of capability
> across an equivalence class, not striping of a single query across providers.

## What it does

RAILLM is a rate-limit-aware LLM routing system. When a primary provider
hits its quota, goes down, or returns an error, the `RAILLMSentinel`
transparently routes the **full conversation history** to an equivalent
model on another provider. The user never sees "You have hit your limit."

Three properties distinguish RAILLM from plain fallback chains:

1. **Stateful carry-over** -- the conversation history IS the state. It
   travels with the failover, compressed to fit the target's context window.
2. **Tier-aware cascade** -- models are grouped by capability tier. Failover
   tries peers in the same tier before dropping to the next tier down.
3. **Verification** -- the `CarryOverVerifier` can detect information loss
   at every provider switch boundary.

## Architecture

```
HAL caller
    |
    v
RAILLMSentinel.dispatch_with_history(model, user_message)
    |
    +-- append user_message to ConversationThread (stateful)
    +-- classify_model(target) -> primary ModelTier
    +-- _build_tier_chain(primary -> peers -> cascade)
    |
    v
For each attempt in cascade chain:
    |
    +-- LimitSentinel._is_near_limit(provider)? -> skip
    +-- thread.compress_if_needed(target_context_window)
    +-- provider.chat(prompt=thread.to_prompt(), model=...)
    |
    +-- on success:
    |     +-- append assistant response to thread
    |     +-- record response headers (limit bookkeeping)
    |     +-- return ChatResponse
    |
    +-- on error:
          +-- record error (backoff bookkeeping)
          +-- continue to next attempt
    |
    v
All exhausted -> raise RAILLMExhausted(errors)
```

## Model tiers

RAILLM groups models into four capability tiers. The cascade traverses
them in order: FRONTIER -> FAST -> FREE_TIER -> LOCAL.

| Tier | Purpose | Default members |
|------|---------|-----------------|
| FRONTIER | Best reasoning, largest context | Gemini 3.1 Pro, GPT-4o, DeepSeek Reasoner |
| FAST | Low-latency, good-enough quality | Groq Llama 3.3, GPT-4o-mini, Qwen 3.5 Flash, DeepSeek Chat |
| FREE_TIER | Zero-cost cloud inference | Groq Llama 3.3, Gemini 2.5 Flash, Gemma 4 31B (OpenRouter) |
| LOCAL | On-device, air-gapped, zero egress | Ollama Gemma4, Lemonade, mlx_vlm |

Within a tier, models are **functionally interchangeable** -- that is the
equivalence class. A frontier model failing over to another frontier model
preserves quality; dropping to FAST is a graceful degradation.

## Quick start (5 minutes)

### 1. Ensure at least two providers are configured

Check `config/providers.yaml`. You need at least one cloud provider and
one local provider, or two cloud providers with different failure domains.

```yaml
providers:
  - name: groq
    type: cloud
    api_key_source: keychain:grip-groq
    models: [llama-3.3-70b-versatile]

  - name: ollama
    type: local
    base_url: http://localhost:11434/v1
    detect: true
    models: auto
```

### 2. Start a local model (optional but recommended)

```bash
ollama pull gemma4:26b
ollama serve  # default port 11434
```

### 3. Use RAILLM in Python

```python
from lib.hal.registry import ProviderRegistry
from lib.hal.raillm import RAILLMSentinel, ConversationThread

# Load providers from config
registry = ProviderRegistry.from_config("config/providers.yaml")

# Create a sentinel with a fresh thread
sentinel = RAILLMSentinel(registry=registry)

# Dispatch with history -- failover is automatic
response = sentinel.dispatch_with_history(
    model_address="groq/llama-3.3-70b-versatile",
    user_message="What is RAID in computer science?",
)
print(response.text)

# Second message carries the full conversation history
response = sentinel.dispatch_with_history(
    model_address="groq/llama-3.3-70b-versatile",
    user_message="How does RAILLM apply that concept to LLMs?",
)
print(response.text)

# Inspect the thread at any time
print(f"Thread: {len(sentinel.thread.messages)} messages, "
      f"~{sentinel.thread.token_estimate()} tokens")
```

### 4. Force a failover (testing)

```python
# Specify an explicit fallback chain
response = sentinel.dispatch_with_history(
    model_address="groq/llama-3.3-70b-versatile",
    user_message="Continue our discussion about redundancy.",
    tier_fallback=["ollama/gemma4:26b"],
)

# The response includes which provider/model actually served it
print(f"Served by: {response.provider}/{response.model}")
```

## ConversationThread API

The thread is the stateful core. It holds messages and supports
budget-aware compression.

```python
from lib.hal.raillm import ConversationThread, Message

thread = ConversationThread(system_prompt="You are a helpful assistant.")

# Append messages
thread.append(Message(role="user", content="Hello"))
thread.append(Message(role="assistant", content="Hi there!",
                      provider_hint="groq", model_hint="llama-3.3-70b"))

# Token estimation (~4 chars per token heuristic)
print(thread.token_estimate())  # e.g. 15

# Compression -- drop oldest messages to fit a budget
dropped = thread.compress_if_needed(budget_tokens=10000)
print(f"Dropped {dropped} oldest messages")

# Export formats
prompt_str = thread.to_prompt()      # XML-tagged string
messages = thread.to_messages()       # OpenAI-style [{"role":..., "content":...}]
```

## CarryOverVerifier

Detects information loss when providers switch. Pluggable judge function:
production uses a broly council, tests use substring matching.

```python
from lib.hal.carryover_verifier import (
    CarryOverVerifier, record_switch, CarryOverAuditLog,
)

verifier = CarryOverVerifier()  # default: substring matcher
log = CarryOverAuditLog()

# After a provider switch, verify fact retention
report = record_switch(
    log=log,
    verifier=verifier,
    thread_before=sentinel.thread,
    response_after=response,
    from_provider="groq",
    to_provider="ollama",
)

print(f"Loss detected: {report.has_loss}")
print(f"Confidence: {report.confidence_score:.2f}")
print(f"Retained: {report.retained_facts}")
print(f"Lost: {report.lost_facts}")

# Audit trail
print(f"Total switches: {len(log.entries)}")
print(f"Loss rate: {log.observed_rate():.1%}")
```

## CrossSessionStateFile

Safe concurrent access to shared state (e.g. `state/hal/limits.json`)
across multiple Claude Code sessions. Uses `fcntl.LOCK_EX` for atomicity.

```python
from lib.hal.cross_session_state import CrossSessionStateFile

sf = CrossSessionStateFile("state/hal/limits.json")

# Read (shared lock -- multiple readers OK)
data = sf.read()

# Write (exclusive lock -- blocks other writers)
sf.write({"groq": {"remaining": 500}})

# Atomic read-modify-write (exclusive lock held for full round trip)
new = sf.update(lambda d: {**d, "groq": {"remaining": 499}})
```

## Custom equivalence maps

Override the default tier assignments for custom deployments:

```python
from lib.hal.raillm import (
    RAILLMSentinel, ModelTier, ModelEquivalenceClass,
)

custom_map = {
    ModelTier.FRONTIER: ModelEquivalenceClass(
        tier=ModelTier.FRONTIER,
        members=["openai/gpt-4o", "anthropic/claude-opus-4-7"],
        context_window=200_000,
    ),
    ModelTier.LOCAL: ModelEquivalenceClass(
        tier=ModelTier.LOCAL,
        members=["ollama/gemma4:26b", "vllm/mistral-7b"],
        context_window=32_000,
    ),
    # ... add FAST and FREE_TIER as needed
}

sentinel = RAILLMSentinel(
    registry=registry,
    equivalence_map=custom_map,
)
```

## Cascade chain construction

When `dispatch_with_history` is called with model address `M` in tier `T`:

1. `M` (primary -- try first)
2. Other members of tier `T` (peer failover)
3. Explicit `tier_fallback` list (operator-supplied)
4. FRONTIER members (skip if `T` is FRONTIER)
5. FAST members (skip if `T` is FAST)
6. FREE_TIER members (skip if `T` is FREE_TIER)
7. LOCAL members (skip if `T` is LOCAL)

Duplicates are removed while preserving first-occurrence order. The chain
is O(n) with `dict.fromkeys` deduplication.

## Configuration reference

RAILLM reads provider configuration from `config/providers.yaml`. The
`routing` section controls default behaviour:

```yaml
routing:
  default_model: groq/llama-3.3-70b-versatile
  fallback_chain: [groq/..., gemini/..., ollama/...]
  tier_chains:
    free:    [ollama/gemma4:26b]
    cheap:   [qwen/qwen3.5-flash, groq/llama-3.3-70b-versatile]
    mid:     [gemini/gemini-2.5-flash, deepseek/deepseek-chat]
    premium: [gemini/gemini-3.1-pro-preview, deepseek/deepseek-reasoner]
  council: [groq/..., gemini/..., qwen/..., deepseek/...]
```

## Modules

| Module | Purpose |
|--------|---------|
| `lib/hal/raillm.py` | RAILLMSentinel, ConversationThread, ModelTier, cascade builder |
| `lib/hal/carryover_verifier.py` | CarryOverVerifier, fact extraction, audit log |
| `lib/hal/cross_session_state.py` | CrossSessionStateFile, fcntl locking |
| `lib/hal/sentinel.py` | LimitSentinel base class (rate-limit detection, backoff) |

## Hypotheses

| ID | Claim | Status |
|----|-------|--------|
| H047 | RAILLMSentinel subclass enables failover without base class changes | CONFIRMED (PR #117) |
| H048 | CarryOverVerifier detection recall >= 0.95 on synthetic suite | CONFIRMED (PR #137) |
| H049 | fcntl.LOCK_EX concurrent state safety with >= 3 writers | CONFIRMED (PR #139) |
| H050 | 100-message cascade with 30% fault injection, zero halts | CONFIRMED (PR #141) |
| H051 | /council preserves 9/10 facts across 4-provider cascade | PENDING (#142) |
| H052 | OTel traces reconstruct 50-hop chain in < 5 min | PENDING (#143) |
| H053 | New operator sets up RAILLM in < 15 min with this doc | PENDING (#144) |
| H054 | Broly council fails to kill zero-loss claim in 5+ scenarios | PENDING (#146) |
| H-RAILLM-1 | SubscriptionAuth transports made RAILLM-resolvable, core unchanged | CONFIRMED (PR #328) |
| H-RAILLM-2 | providers ladder == CCH->Kimi->Qwen->cheap->local | CONFIRMED (PR #328) |
| H-RAILLM-3 | happi `infer` envelope -> RAILLM, deterministic audit `idr` | CONFIRMED (PR #328) |
| H-RAILLM-4 | `grip_runtime.grip_infer` facade, zero core change | CONFIRMED (PR #328) |
| H-RAILLM-5 | conversation decoupled from any single provider window | CONFIRMED (W5 falsified-as-stated; HAL#329 fix resolved it) |
| H-RAILLM-6 | remaining hardening decomposed into durable sprint-loop backlog | CONFIRMED (PR #328) |
| H-RAILLM-7 | non-mutating per-rung wire copy; shared thread preserved across failing primary | CONFIRMED (HAL#329) |

## GRIP runtime entrypoint (W1-W6, ADR-0001)

`lib/hal/grip_runtime.py::grip_infer(prompt, *, model=None, audit=False)`
is the single, stable front door that makes "GRIP runs through HAL" real:
one happi `infer` envelope -> `RAILLMSentinel.dispatch_with_history` over
the W2 ladder (CCH-Anthropic -> Kimi/Qwen -> cheap -> local), via the W1
subscription bridge. `audit=True` appends the deterministic `idr`
terminator. Returns a normalised `GripInferResult` (`ok`, `text`,
`provider`, `failover`, `idr_ref`, ...). Pure composition -- the RAILLM
core is byte-unchanged.

**HAL#329 — RESOLVED (H-RAILLM-5 -> CONFIRMED):**
`dispatch_with_history` previously ran the *mutating* `compress_if_needed`
on the **shared** thread for every attempted rung, so a small-window
failing primary destructively truncated shared history before the larger
fallback rung was tried. **Fixed** on `fix/raillm-compress-before-failover`:
the shared `ConversationThread` is the canonical full history; each rung
gets a per-rung **non-mutating** `compressed_prompt` wire copy
(`_fit_to_budget` is the DRY shared fitter; `compress_if_needed` keeps its
mutating contract for direct callers). A failed rung no longer destroys
history; only the serving rung's reply is appended. The W5 sentinel
inverted (`test_small_window_failing_primary_preserves_shared_thread`) and
passes; 94/94 regression green. Failover transparency was always
unaffected. See ADR-0001 §Resolution and
`tests/test_e2e_infinite_machine.py::TestGripLadderContextWindowDecoupling`.
