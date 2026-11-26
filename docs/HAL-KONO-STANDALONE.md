# HAL KONO Standalone

HAL owns its own KONO memory substrate. Conversation recall reads from
`~/.hal/data/embeddings.db` via the vendored `lib/hal/kono/` package. HAL
no longer imports anything from `~/.claude/lib` at runtime for memory.

## What shipped (2026-04-15)

| Artefact | Location | Origin |
|----------|----------|--------|
| Constants module | `lib/hal/kono/constants.py` | New — sole source for `HAL_DIR`, `DB_PATH`, `MODEL_PATH`, `LEMBED_CACHE_PATH`, `SESSION_COUNTER_FILE` |
| Core dataclasses | `lib/hal/kono/core.py` | Vendored from `~/.claude/lib/kono.py` |
| Session counters | `lib/hal/kono/counters.py` | Vendored from `~/.claude/lib/kono_counters.py`, paths rewired to `HAL_DIR` |
| Schema/migrations | `lib/hal/kono/schema.py` | Vendored from `~/.claude/lib/kono_schema.py` |
| Memory substrate | `lib/hal/kono/substrate.py` | Vendored from `~/.claude/lib/kono_substrate.py` (~1900 lines) |
| HAL bridge | `lib/hal/kono_bridge.py` | Rewritten — drops `sys.path.insert(~/.claude/lib)`; imports from `lib.hal.kono.substrate` |
| Seed DB | `~/.hal/data/embeddings.db` | One-time copy of GRIP's store (~76 MB, gitignored) |
| Model | `~/.hal/models/nomic-embed-text-v1.5.Q8_0.gguf` | Symlink to GRIP's copy |

## Sync strategy: one-time seed, then diverge

There is no ongoing sync between GRIP and HAL memory stores. At port time HAL
copied GRIP's `embeddings.db` verbatim. After that, each substrate evolves
independently. This is deliberate (chosen by V>> via AskUserQuestion during
the port): it avoids a sync daemon, conflict resolution, and cross-process
locking, at the cost of accepting that HAL's store will drift away from
GRIP's over time.

## Model file caveat

The nomic-embed-text-v1.5.Q8_0.gguf model is ~150 MB. Rather than copy it,
HAL symlinks to GRIP's copy at `~/.claude/models/`. Consequences:

- If GRIP is uninstalled, the symlink goes dangling. `lembed` will fail to
  load the model, which the substrate handles gracefully by marking lembed
  unhealthy and degrading to FTS5-only keyword search.
- Recall quality drops without the model (no vector similarity), but HAL
  still returns relevant hits via FTS5. The `_execute_hybrid_search` method
  in `lib/hal/kono/substrate.py` is intentionally FTS-only in HAL's vendored
  copy, so the degradation path is the default path until HAL vendors
  `EmbeddingEngine.search_hybrid()`.
- To make HAL fully standalone on this axis, run:
  `cp ~/.claude/models/nomic-embed-text-v1.5.Q8_0.gguf ~/.hal/models/`
  then remove the symlink. That pattern is captured as a follow-up, not
  this wave's work.

## External Python dependencies

HAL continues to rely on whichever Python environment is active. The
substrate imports `apsw`, `sqlite_vec`, and (optionally) `sqlite_lembed`
at module top. Required packages:

```
pip install xxhash apsw sqlite-vec sqlite-lembed
```

The `xxhash` dependency is shared with `lib/hal/utils/cch.py` (the Claude
Code billing-header signer). `hooks/startup/init.sh` installs it when
creating the bootstrap venv.

When `apsw` is unavailable (e.g. the system Python that a terminal shell
defaults to), the bridge's timeout wrapper swallows the `ImportError` and
`recall_for_query()` returns an empty string. HAL never crashes on a
missing KONO install — conversation continues without memory context.

## Rollback

If the port needs to be reverted (KONO corruption, schema drift, or a
decision to re-share with GRIP):

```bash
git checkout lib/hal/kono_bridge.py tests/test_kono_bridge.py
rm -rf lib/hal/kono/
rm -f ~/.hal/data/embeddings.db
rm -f ~/.hal/models/nomic-embed-text-v1.5.Q8_0.gguf
```

The bridge file in `main` predating this port read from GRIP's store via a
`sys.path` hack. Reverting restores that behaviour with zero user impact.

## Open follow-ups (out of scope for this wave)

1. **HAL write path** — the bridge is still read-only. Writing KONO memories
   from HAL conversations is a separate plan.
2. **Schema drift detection** — GRIP's schema may move ahead of HAL's
   vendored copy. A `kono_meta` version check on substrate connect would
   detect drift early.
3. **Vector recall** — re-vendor `EmbeddingEngine.search_hybrid()` if HAL
   needs RRF-fused vector + keyword recall rather than FTS5 alone.
4. **Model file standalone copy** — replace the symlink with a binary copy
   once HAL is deployed on machines without GRIP.
