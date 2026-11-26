# Obsidian Vault MCP Integration — Scope Document (#131)

## Context

Split from HAL #30 (3-Layer Memory Stack). KONO standalone (PR #126) provides
the semantic-search engine. This issue tracks the remaining Layer 2 work:
bridging external Obsidian vaults into HAL's knowledge graph.

## What exists

| Component | Location | Status |
|-----------|----------|--------|
| KONO core | `lib/hal/kono/core.py` | Done — InstanceDNA, GRIPInstance, MemoryBlock |
| KONO substrate | `lib/hal/kono/substrate.py` | Done — SQLite+apsw+sqlite-vec |
| KONO bridge | `lib/hal/kono_bridge.py` | Done — O(1) recall API, cache per turn |
| MCP server | `lib/hal/mcp_server.py` | Done — chat, council, status, agents |
| Knowledge absorber | `lib/hal/knowledge_absorber.py` | Done — PDF, web, audio, GitHub, wikilinks |

## What is needed

### 1. Vault indexer (`lib/hal/obsidian_vault.py`)

Indexes an Obsidian vault directory into KONO:
- Walk `.md` files, extract frontmatter + body
- Parse `[[wikilinks]]` into graph edges
- Parse `#tags` into categories
- Incremental index: only re-index files changed since last scan (mtime check)
- Store as KONO MemoryBlock entries with `source: "obsidian:{vault_path}"`

### 2. MCP vault tools (extend `lib/hal/mcp_server.py`)

New MCP tools for vault interaction:
- `vault_search(query)` — semantic search across vault notes
- `vault_browse(path)` — list notes in a directory
- `vault_read(note)` — read a specific note
- `vault_graph(note)` — show backlinks and forward links for a note

### 3. Vault browser panel (HAL Commander UI)

Electron/web component in HAL Commander:
- Tree view of vault directory
- Search bar with semantic + keyword modes
- Note preview with rendered markdown
- Backlink graph visualisation (D3.js force-directed)

### 4. "Top of Mind" integration

Surface high-relevance KONO entries at conversation start:
- Query recent + high-relevance + high-recency MemoryBlocks
- Inject as system context (already done via `kono_bridge.recall_for_query`)
- Add vault-sourced entries to the candidate pool

## Dependencies

- Obsidian vault path configured in `config/hal.yaml` or env var `HAL_OBSIDIAN_VAULT`
- KONO substrate operational (sqlite-vec, sqlite-lembed)
- For Commander UI: Electron app already scaffolded

## Estimated effort

| Phase | Effort | Priority |
|-------|--------|----------|
| Vault indexer | M (4-6h) | P1 |
| MCP vault tools | S (2-3h) | P1 |
| Commander UI | L (8-12h) | P2 |
| Graph visualisation | L (6-8h) | P3 |

## Non-goals (this issue)

- Custom Obsidian plugin (out of scope — MCP bridge is the interface)
- Real-time sync (incremental mtime-based index is sufficient)
- Non-markdown file indexing (Obsidian is markdown-native)
