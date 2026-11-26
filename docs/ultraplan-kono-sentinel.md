# Ultraplan: The Sentinel — KONO Everpresent Memory Across All LLMs

## Goal

Wire KONO semantic memory into HAL's conversation layer so every LLM query,
regardless of provider, gets query-relevant memory context. The Sentinel:
a session that never forgets.

## Current State (Gap Analysis)

| Component | Status | Gap |
|-----------|--------|-----|
| KONO substrate | Production (86MB, 768-dim, 4D spatial) | Only fires in CC hooks |
| HAL MemoryEngine | Production (flat MEMORY.md) | No semantic search |
| GRIP bridge | Production (static ~500 tokens) | No query-time recall |
| Guild pipeline | Production (39 channels, 261 insights) | Batch, not continuous to KONO |
| HALCouncil | Production (parallel 3-phase debate) | No KONO context in phases |

## Architecture: O(1) Coordinator Pattern

```
User Query -> Conversation.send(message)
  |
  +-> kono_bridge.recall(message)  [NEW - O(1) lookup, 1.5s timeout]
  |     |
  |     +-> MemorySubstrate.recall(query, limit=3)
  |     |     +-> EmbeddingEngine.search_hybrid() [RRF-fused]
  |     |     +-> Spatial expansion [4D neighbors]
  |     |     +-> Fitness filtering [decay model]
  |     +-> Format as context section (~500 tokens max)
  |
  +-> system = base_system + kono_context  [DYNAMIC per turn]
  |
  +-> Provider.chat(system=enriched_system)
        |
        +-- Groq, Gemini, Ollama, DeepSeek... ALL get memory
```

## Fibonacci Wave Execution

### Wave 1 (1 file + tests) — The Bridge
- `lib/hal/kono_bridge.py` — Single entry point for KONO recall from HAL
- `tests/test_kono_bridge.py` — Unit tests with mocked substrate

### Wave 2 (2 files) — The Wiring
- `lib/hal/conversation.py` — Per-turn KONO recall in send()
- `lib/hal/moe.py` — KONO context injected into council system prompt

### Wave 3 (2 files + tests) — The Upgrade
- `lib/hal/grip_bridge.py` — Dynamic recall + static state
- `tests/test_kono_integration.py` — Integration tests

### Wave 5 — Verification
- Full test suite (1167+ tests), zero regressions

## Key Design Decisions

1. **KONO recall lives in conversation.py, not context_assembler.py** — because
   context_assembler builds the base prompt once, while KONO recall is per-query.

2. **kono_bridge.py is a thin adapter** — it handles import, timeout, formatting.
   The heavy lifting stays in KONO substrate (SRP).

3. **Same memory for all providers** — no per-provider filtering. Knowledge is
   knowledge regardless of which LLM processes it. KISS.

4. **Graceful degradation** — if KONO is unavailable (no sqlite-vec, no DB),
   HAL continues without memory. Never crashes.

5. **O(1) from caller** — one function call, one string back. The coordinator
   pattern: dispatch to the right subsystem, return formatted result.

## Success Criterion

converge(
    task="KONO everpresent across all HAL LLMs",
    criterion="Any HAL query to any provider includes relevant KONO memories AND all tests pass",
    max_depth=3
)

## 2026-04-15 Addendum — Standalone KONO Port

The original Sentinel wave loaded GRIP's shared substrate at
`~/.claude/embeddings.db` via a `sys.path` hack. HAL now owns its copy.

- Vendored package: `lib/hal/kono/{constants,core,counters,schema,substrate}.py`
- HAL-owned DB: `~/.hal/data/embeddings.db` (seeded once from GRIP, then diverges)
- Model file: `~/.hal/models/nomic-embed-text-v1.5.Q8_0.gguf` symlinked to GRIP's copy
- Runtime dependency on `~/.claude/lib` for KONO: eliminated
- Hybrid search: FTS5-only (GRIP's `EmbeddingEngine.search_hybrid` not vendored — YAGNI)

See `plans/kono-standalone-vendor.md` and `docs/HAL-KONO-STANDALONE.md`
for the port rationale, gap analysis, and rollback procedure.
