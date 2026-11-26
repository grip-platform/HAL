# HAL 100+ Module Reference

Complete reference for all HAL modules, organised by architectural layer.
Generated from source docstrings. Current count: 202 top-level modules
(284 incl. subpackages).

---

## Layer 1: Provider Infrastructure (15 modules)

| Module | Purpose |
|--------|---------|
| `registry` | Config-driven provider management. Loads from `config/providers.yaml`, manages health state, and provides model addressing via `provider/model-name` convention. |
| `sentinel` | Zero-limits through proactive detection and auto-failover. Monitors response headers, tracks limit state per provider, routes away before limits hit. The user never sees "You've hit your limit." |
| `discovery` | Auto-detect local LLM servers. Probes Ollama (11434), vLLM (8000), LM Studio (1234), and port 8080, disambiguating mlx_vlm from llama.cpp via `/v1/models` response inspection. The full local-provider set declared in `config/providers.yaml` is lemonade, ollama, vllm, lmstudio, mlx_vlm, and bitnet (keyless localhost). |
| `cost_tracker` | Token-to-USD calculation for any provider. Pricing sourced from CC's `modelCost.ts` and extended with Groq, Gemini, OpenRouter, and local model costs. |
| `cost_router` | Cheapest-provider selection that meets quality needs. Simple queries go to Groq free tier; medium to Gemini Flash / DeepSeek; complex to full-capability models. |
| `provider_analytics` | Per-provider latency, error rate, and cost tracking across sessions. Enables data-driven provider selection and trend surfacing. |
| `rate_limiter` | Proactive RPM and TPM rate limiting. Enforces limits before sending requests rather than reacting to 429s. Complements the Sentinel. |
| `async_provider` | Non-blocking LLM calls via asyncio. Enables parallel council queries, background warmup, and non-blocking chat in server contexts. |
| `smart_retry` | Retry with provider rotation. On failure, rotates to the next available provider instead of retrying the same one. Combines exponential backoff with failover. |
| `response_cache` | SHA256-keyed LRU cache with TTL. Caches LLM responses to avoid redundant calls in workflows, agent loops, and retry scenarios. Thread-safe. |
| `credential_store` | Secure API key management. Uses system keychain (macOS Keychain, Linux Secret Service) with env-var fallback. No plaintext keys in config files. |
| `plugin` | Provider plugin auto-discovery. Users add new LLM providers by placing plugin files in `~/.hal/plugins/providers/` — no core changes needed. |
| `provider_failover` | Explicit failover chain configuration. User-controlled order: if provider A fails, try B then C. More predictable than smart_retry's automatic rotation. |
| `provider_monitor` | Continuous background health checking. Periodically pings all registered providers, tracks availability over time, alerts on outages via the notification system. |
| `retry_policy` | Composable retry strategy objects: exponential, linear, and fixed delay. Used by smart_retry and other retry-aware modules. |

---

## Layer 2: Conversation Engine (13 modules)

| Module | Purpose |
|--------|---------|
| `conversation` | Multi-turn chat without any external harness. Core loop: user message → context assembly → LLM query → response → memory update → next turn. Works with any HAL provider. |
| `context_assembler` | Builds the full context for any LLM runtime. Assembles: HAL.md chain → mode prompt injection → skill prompts → git status → memory → ContextBundle. Runtime-agnostic. |
| `context_warnings` | Graduated alerts at 80/85/95% context usage. Monitors token accumulation, recommends actions (compact, snapshot, new session) — never auto-acts. |
| `compaction` | Smart summarisation preserving code and decisions. When the context window fills, summarises older turns while retaining key decisions, code snippets, and action items. |
| `session_manager` | Session lifecycle abstraction. Manages session ID, state persistence, and cross-session continuity without depending on CC's specific environment variables. |
| `conversation_branch` | Fork conversations for "what if" exploration. Creates branches at any conversation point, explores alternatives, and optionally merges insights back. |
| `conversation_merger` | Combine parallel conversation branches or separate sessions into a unified chronological history. |
| `conversation_search` | TF-weighted search across saved session replays. Finds past conversations by content or topic using keyword matching with relevance scoring. |
| `conversation_stats` | Rich conversation analytics: word frequency, response times, topic distribution, and engagement metrics. |
| `context_snapshot` | Portable state save/restore. Captures full conversation state (history, model, mode, cost, context usage) for session handoff, backup, or resumption. |
| `session_handoff` | Cross-device conversation transfer. Packages conversation state and metadata into a portable format for transfer to another HAL instance. |
| `session_replay` | Timeline recording of all conversation turns, tool calls, and decisions. Enables post-session review, annotation, and export. Adapted from Recordly's timeline model. |
| `response_streamer` | Token-by-token response delivery via SSE, WebSocket, or callback. Handles buffering and flush control. |

---

## Layer 3: Intelligence (11 modules)

| Module | Purpose |
|--------|---------|
| `moe` | Cross-LLM council with agreement scoring. Queries multiple providers in parallel, synthesises responses, and runs cross-context debate rounds. Every registry provider is a potential council member. |
| `rsi` | Recursive self-improvement. After each conversation, a different model analyses the interaction to detect patterns and propose improvements. Applies the YSH principle: when a pattern is seen 3+ times, abstract it. |
| `autodream` | Memory consolidation engine. Runs between sessions to merge, deduplicate, and strengthen memories. HAL-native implementation of the CC autoDream pattern using mechanical consolidation rules. |
| `insights` | Post-conversation analytics. Extracts effectiveness metrics, topic patterns, friction points, and improvement suggestions. Feeds the RSI engine. |
| `attribution` | Input section importance scoring. Determines which parts of a prompt most influenced the LLM output. Two methods: LLM-based rating (cheap) and ablation testing (accurate). |
| `quality_scorer` | 4-dimension response evaluation: relevance, completeness, structure, and conciseness. Each dimension scores 0–25, total 0–100. |
| `intent_detector` | 7-category intent classification for smart routing. Categories: code generation, explanation, debugging, research, conversation, command, and other. |
| `suggestions` | Proactive next-action recommendations. Analyses conversation context and suggests useful next steps. Non-intrusive — suggestions appear only when relevant and are never auto-executed. |
| `model_compare` | Side-by-side model evaluation. Sends the same query to multiple models, compares responses on quality, cost, and latency. |
| `model_selector` | Automatic model selection. Combines intent detection, cost routing, and model catalog to pick the optimal model for any query. |
| `knowledge_absorber` | Universal knowledge extraction from any source: papers, articles, APIs, datasets, URLs, or local files. |

---

## Layer 4: Agent and Skill System (9 modules)

| Module | Purpose |
|--------|---------|
| `agent_runner` | Parse GRIP agent `.md` files, extract prompt/tools/skills, and spawn agents. Under CC, delegates to the Task tool. Standalone, runs HAL's own query with the agent prompt. |
| `agent_spawn` | Recursive agent spawning to infinite depth. Each spawned agent can use a different LLM provider. Coordinator-convergence duality: spawning is breadth, each agent converges at depth. |
| `skill_loader` | Runtime-agnostic skill resolution. Discovers skills from the `skills/` directory, resolves mode-aware skill sets (foundation + mode-specific), and injects skill content into context. |
| `prompt_library` | Reusable prompt templates. Ships with built-in templates; supports user-defined custom templates via `{variable}` substitution. |
| `tool_registry` | Dynamic tool discovery and registration. Manages built-in tools, MCP-discovered tools, and user-registered custom tools. |
| `workflow` | Multi-step automated task execution. Steps can be LLM prompts, tool calls, or conditional branches. Steps reference previous outputs via `{step_name.output}`. |
| `council_agent` | Multi-LLM deliberation per response. Queries 3–5 models from different providers simultaneously. Unlike single-vendor platforms, HAL councils span Anthropic, Google, Groq, and local models. |
| `self_improving_agent` | Agents that evolve their own definitions. Analyses own performance history and proposes optimised versions of itself — static agent configs become dynamic. |
| `agent_registry` | Registry of active and registered agents with lifecycle tracking. |

---

## Layer 5: Memory and Search (11 modules)

| Module | Purpose |
|--------|---------|
| `memory_engine` | CC-compatible MEMORY.md format interface. Reads/writes with 200-line / 25 KB limits, integrates with GRIP's KONO semantic search, runtime-agnostic. |
| `memory_manager` | VRAM-resident model lifecycle manager. Keeps local Ollama models warm in VRAM by pinging before the 5-minute Ollama eviction timeout. |
| `memory_sync` | Bidirectional sync between GRIP and HAL memory. Syncs MEMORY.md index and individual files between `~/.claude/memory/` and `~/.hal/memory/`. Conflict resolution: newest-wins. |
| `user_adapt` | Few-shot preference learning. Learns user preferences from minimal interaction signals (feedback, style patterns, topic focus) and adapts future responses without fine-tuning. |
| `knowledge_base` | Karpathy LLM wiki architecture. `raw/` → LLM-compiled wiki of `.md` articles → Q&A → outputs filed back. The LLM maintains all wiki content. |
| `kono_bridge` | Semantic memory recall bridge. Connects HAL's conversation layer to GRIP's KONO 4D memory substrate so every LLM query gets query-relevant context, regardless of provider. |
| `code_search` | AST-aware Python code search. Searches by structure (functions, classes, imports) using Python's `ast` module. Falls back to regex for non-Python files. |
| `tag_manager` | Categorisation and labelling for conversations, memory entries, and resources. Enables filtered retrieval and organisation. |
| `prompt_optimizer` | Token-saving prompt compression. Removes redundancy, compresses whitespace, shortens verbose instructions. Saves cost on every query. |
| `summary_generator` | Structured conversation summaries for session handoff, stakeholder reporting, and documentation. Supports 3 styles. |
| `diff_tracker` | File change tracking via SHA256. Records state snapshots and computes diffs. Audits what an agent session changed. |

---

## Layer 6: Safety and Security (10 modules)

| Module | Purpose |
|--------|---------|
| `hook_dispatch` | In-process Python hook runner. Runs GRIP's enforcement hooks without CC's subprocess invocation. Hard hooks run synchronously (deny blocks action); soft hooks fail-open. 38 event types. |
| `sanitise` | Web content sanitisation defending against AI Agent Trap Class 1 (Content Injection): strips hidden instructions from HTML comments, CSS, image metadata, and accessibility tags. |
| `verify` | Post-edit verification. CC gates this behind `USER_TYPE=ant`. HAL implements it as a standalone module — after any file write/edit, runs compilation checks before reporting success. |
| `harness_detect` | Auto-detect which agentic tool is running. Detects CC, Cursor, standalone, or other harnesses so HAL can adapt behaviour (delegate tool dispatch or run tools natively). |
| `agent_traps` | 6-class adversarial trap defence based on the AI Agent Traps research paper. Mechanical defences for Content Injection, Prompt Hijacking, Tool Poisoning, and 3 additional classes. |
| `content_filter` | PII and credential redaction before LLM submission. Strips API keys, passwords, and email addresses from prompts before they reach external providers. |
| `safety_scanner` | Python security static analysis. Detects SQL injection, command injection, hardcoded secrets, insecure deserialisation, and path traversal. |
| `guard` | Three-way tool access control (allow/deny/ask). Ported from TF>> `useCanUseTool` + permissions system. Evaluates rules per tool invocation; integrates with hook dispatch. |
| `tracing` | OTel-compatible span model without requiring the full OpenTelemetry SDK. Traces LLM calls, tool executions, and workflow steps. Can export to JSON or feed OTel collectors. |
| `audit_log` | Compliance-ready operation tracking. Records every significant operation with timestamps, actors, and outcomes. Designed for regulated environments (legal, healthcare, finance). |

---

## Layer 7: Surfaces and Infrastructure (15 modules)

| Module | Purpose |
|--------|---------|
| `cli` | Unified `hal` command. Entry point for interactive chat with model warmup, plus subcommands for council, status, export, and config. |
| `server` | HTTP API + web UI + streaming. Routes all queries through the registry and conversation engine. Supports multiple concurrent sessions and JSONL streaming for GRIP-GUI. |
| `daemon` | Always-on background process. Runs HTTP server, mesh heartbeat, model warmup, and scheduler as a persistent background daemon. |
| `mcp_server` | MCP tool provider (stdio JSON-RPC). Exposes HAL as a tool provider to any MCP-compatible client: Claude Desktop, Cursor, VS Code, etc. |
| `health` | Unified health dashboard. Aggregates health signals from all HAL subsystems: providers, memory, disk, and module integrity. Available as CLI display and API response. |
| `export` | Multi-format conversation export. Exports as markdown, JSON, HTML, or plain text for sharing, archiving, or feeding into other tools. |
| `response_formatter` | LLM response formatting for different targets: terminal (ANSI), plain text, markdown, HTML, and JSON. Handles code block styling and structural formatting. |
| `output_validator` | LLM output format validation. Checks that output matches expected schema (JSON, code, markdown, list, etc.). Used by workflows and automated pipelines. |
| `benchmark` | Systematic provider evaluation. Runs standardised test prompts against multiple providers; measures quality, latency, and cost. Results persist for trend analysis. |
| `usage_reporter` | Aggregated usage reports for billing and tracking. Supports daily, weekly, and monthly rollups across sessions. |
| `config_manager` | Unified configuration from multiple sources: built-in defaults → `config/hal.json` user overrides → environment variables, in priority order. |
| `notification` | Event-driven pub/sub alerts. Notifies when background tasks complete, errors occur, or important events happen. Channels: in-process callbacks, file-based signals, optional webhooks. |
| `progress_tracker` | Multi-step task completion tracking. Provides percentage completion, ETA, and status reporting for workflows, sprints, and batch processing. |
| `command_router` | Slash command registry. Routes `/status`, `/council`, `/help`, etc. with argument parsing and help text generation. |
| `web_demo` | Lightweight single-file HTTP server and interactive UI. No external dependencies. Serves the demo UI and proxies LLM requests. |

---

## Layer 8: RSI and Self-Build Pipeline (8 modules)

| Module | Purpose |
|--------|---------|
| `build_loop` | Continuous self-build with convergence detection. Ties RSI analysis to self-build. Runs iterations until convergence (no new proposals), budget exhaustion, or max iterations. |
| `code_gen` | RSI-proposal-to-Python code generation. Takes an RSI Proposal or structured CodeSpec and produces valid Python source with tests. Uses LLM for generation, mechanical checks for validation. |
| `code_reviewer` | Mechanical + semantic code review for the RSI pipeline. Safety scan + syntax check always runs. LLM analysis runs only when an `llm_fn` is supplied. Both gates fail-safe. |
| `self_build` | RSI proposal → code gen → safety gates → branch → PR pipeline. The full automated self-improvement workflow. |
| `autoresearch` | Karpathy loop applied to routing config. Autonomous hill-climbing: propose config delta → benchmark → keep/discard. Runs unattended; human directs via `config/rsi-direction.md`. |
| `background_executor` | Async long-running agent tasks. Sentinel auto-failover mid-task with multi-provider routing. If a provider hits its limit during a long task, switches mid-execution. |
| `trace_capture` | Step-by-step reasoning trace capture for the meta-agent loop. Captures WHY something failed (not just WHETHER it failed) for RSI analysis. |
| `pipeline` | Composable staged message handling. Decomposes `Conversation.send()` into 7 ordered stages, each doing one thing. Inspired by compiler pipeline architecture. |

---

## Layer 9: Cross-Cutting Infrastructure (20 modules)

| Module | Purpose |
|--------|---------|
| `types` | Shared frozen dataclasses for the universal LLM abstraction. No external dependencies — stdlib only. |
| `home` | HAL_HOME resolution. Priority: `HAL_HOME` env var → `~/.hal`. Used by every module that needs file-system paths. |
| `warmup` | Background model pre-ping. Ollama models take 20–30s cold start but respond in <100ms when warm. Pings local models in a background thread at startup. |
| `scheduler` | Lightweight task scheduling without system cron. Uses `threading.Timer` for periodic execution. Persists schedules to `config/schedules.json`. |
| `a2a` | Agent-to-agent message bus. In-process, thread-safe. Agents at any depth can send messages to any other agent without direct imports. |
| `mesh` | File-based inter-instance peer discovery and messaging. Multiple HAL instances discover each other via heartbeat files and exchange messages through the filesystem. No HTTP, no mDNS — just files. |
| `features` | Feature flag management. Toggles features without code changes. Persists to `config/features.json`. |
| `cc_env_bridge` | CC hidden feature activation. When HAL detects it's running inside Claude Code, sets environment variables to unlock gated features (employee-only or beta features). |
| `auto_harness` | Universal repo scanning and feature scoring. Given any GitHub repository, scans structure and README, scores features for portability, and proposes absorption into HAL. |
| `employee_gate` | CC employee-gated features made native. Implements features gated behind `process.env.USER_TYPE === 'ant'` (Anthropic employees only) as HAL-native modules available to all users. |
| `budget` | Token spending limits. Prevents runaway costs by enforcing configurable USD or token budgets per session or conversation. |
| `diagnostic` | Automated issue detection. When something goes wrong (provider errors, import failures, config issues), runs targeted checks and suggests fixes. |
| `model_catalog` | Model metadata registry. Tracks capabilities, context windows, pricing, and recommended use cases. Used by cost_router and model_compare. |
| `system_info` | Single source of truth for HAL's version, build info, environment details, and runtime capabilities. |
| `event_bus` | Cross-module publish/subscribe communication. Enables loose coupling between HAL modules — any module can emit events and any other can subscribe without direct imports. |
| `request_throttle` | Fair queuing for concurrent requests. When multiple conversations or agents send requests simultaneously, ensures fair distribution across providers. |
| `fleet` | Fleet-level aggregation and statistics. Multi-provider cost analytics across multiple HAL instances — zero-cost tracking, provider diversity. |
| `router` | Multi-provider LLM dispatch strategies. Three modes: Fixed (explicit provider/model), Cost-aware (via cost_router), and Warm-first (via warm_router). |
| `warm_router` | Warm-model-first routing. Probes Ollama `/api/ps` before firing the full API chain. Directs queries to already-loaded models to minimise cold-start latency. |
| `output_quality` | Dismissive pattern detection in LLM responses. Catches language that deflects responsibility for found issues (e.g. "pre-existing issue", "not related to this change"). Implements HAL's No Issue Hierarchy principle. |

---

## Layer 10: Integration and Support (15 modules)

| Module | Purpose |
|--------|---------|
| `environment` | Lightweight execution profiles. Config-driven profiles enforced through the hook dispatch system. No Docker required — alternative to heavyweight container-based environments. |
| `event_protocol` | Typed, structured event streaming. CC-compatible event types plus HAL-exclusive events: council votes, sentinel failovers, mesh results. |
| `git_ops` | Git and GitHub abstraction for standalone operation. Wraps git CLI and gh CLI calls in a clean interface for MCP apps, context assembler, and session management. |
| `grip_bridge` | GRIP context injection into HAL. Two modes: static snapshot (~500 tokens) and live bridge for active sessions. |
| `grip_harness` | GRIP-to-HAL absorption harness. The recursive loop: GRIP innovates via RSI sprints → HAL detects new/changed components → scores portability → absorbs portable ones → HAL standalone grows. |
| `hal_init` | Bootstrap the complete HAL runtime. Single entry point initialising all HAL subsystems in the correct dependency order. Used by `cli.py` and `daemon.py`. |
| `identity_provider` | User/org identity without any external harness. Reads from `facts/identity.md` (GRIP-native) with zero external dependency. No CC environment variables required. |
| `io_utils` | Shared file operations. Extracted to eliminate DRY violation across `agent_registry`, `background_executor`, and `environment` modules. |
| `kono_bridge` | Semantic memory recall bridge to GRIP's KONO 4D memory substrate. Every LLM query gets query-relevant context regardless of provider. |
| `media` | Multimodal pipeline. Detects local file paths in user messages, reads and base64-encodes them, and builds OpenAI-compatible `ContentPart` arrays for multimodal LLM APIs. |
| `raillm` | Redundant Array of Independent LLMs. RAID-inspired failover with stateful conversation history carried across provider switches. Primary hits limit → secondary continues with full context intact. |
| `session_registry` | Multi-session conversation management. Manages multiple concurrent conversations, each with its own session ID, history, model, and cost tracking. Disk-backed for persistence. |
| `tool_dispatch` | Native tool execution without external harness. HAL runs Bash, Read, Write, Edit, Glob, and Grep natively with sandbox enforcement via hook dispatch. |
| `tool_executor` | Concurrent tool execution. Read-only tools (`is_concurrent_safe=True`) run in parallel via `ThreadPoolExecutor`. Write tools run serially. Ported from TF>> `StreamingToolExecutor`. |
| `tool_protocol` | Formal interface for all tool implementations. Ported from TF>> `Tool.ts`: unified type with `execute`/`validate`, input schema, and concurrency safety flag. |

---

## Layer 11: Gateway / Egress Rail and GRIP Runtime (14 modules)

The local Anthropic/OpenAI-compatible reverse-proxy rail (`/v1/*`) plus the
RAILLM ladder and the `grip_infer` runtime that makes "GRIP runs through HAL"
real (see `docs/ADR-0001-hal-raillm-grip-runtime.md`).

| Module | Purpose |
|--------|---------|
| `gateway` | Local Anthropic-compat `/v1/messages` reverse proxy (P1). Verbatim HTTP + byte-identical retry; repairs the harness `thinking`/`redacted_thinking` signature-drop 400. |
| `gateway_openai` | OpenAI-compatible `/v1/chat/completions` endpoint (P2). Any OpenAI-wire harness (Aider, Cline, Codex CLI, Goose) points `OPENAI_API_BASE` at the gateway and reaches Anthropic through it. |
| `gateway_responses` | OpenAI Responses API `/v1/responses` endpoint (P6). Lets OpenAI Codex CLI (`wire_api="responses"`) reach Anthropic; translates Responses <-> Anthropic Messages, streaming and non-streaming. |
| `gateway_crosswire` | Anthropic Messages -> OpenAI Chat translation. The inverse of `gateway_openai`: an Anthropic-wire client (Claude Code) targeting an OpenAI-family model. |
| `gateway_stream_crosswire` | Token-incremental cross-wire: consumes a provider's OpenAI Chat SSE stream delta-by-delta and translates each chunk into Anthropic Messages SSE events as they arrive. |
| `gateway_translate` | Shared OpenAI->Anthropic translation helpers (image-block + cross-cutting concerns) reused by the `/v1/chat/completions` and `/v1/responses` handlers. |
| `gateway_egress` | Same-wire multi-provider re-pointing. When a request routes to a non-Anthropic provider, forwards the original OpenAI-wire body verbatim to that provider's endpoint with its own Bearer key — no Anthropic translation. |
| `gateway_failover` | Failover + cost/audit layer (P3). Ordered upstream list; on transport-retry exhaustion or a failover-class status (429/5xx) the next upstream is tried, transparently to the caller. |
| `gateway_witness` | Witnesses and signs every inference on the chat rail — the unwitnessed `/v1/chat/completions` surface now records the failover-layer audit. The provability-moat foundation. |
| `gateway_guard` | Default-deny access control for the `/v1/*` proxy. Prevents the publicly-tunnelled `hal-server` from becoming an unauthenticated open relay to Anthropic. |
| `gateway_threads` | Cross-transport thread store. Per-canonical-user thread state so the same human reaching HAL from Telegram and Slack sees one coherent conversation. |
| `raillm_ladder` | The operator's canonical RAILLM fall-over order: Anthropic via CCH subscription -> Kimi/Qwen subscription -> cheap/zero-cost API tiers -> local. |
| `raillm_subscription_bridge` | Exposes `SubscriptionAuth` transports (CCH, Kimi) as RAILLM-resolvable providers so the sentinel can fail over across subscription and API auth uniformly. |
| `grip_runtime` | GRIP's first-class default LLM runtime — `grip_infer` over the RAILLM ladder, driving a single happi `infer` envelope through the real happi runtime. happi.md is the syscall surface; RAILLM is the failover substrate. |

---

## Module Count by Layer

| Layer | Count |
|-------|-------|
| Provider Infrastructure | 15 |
| Conversation Engine | 13 |
| Intelligence | 11 |
| Agent and Skill System | 9 |
| Memory and Search | 11 |
| Safety and Security | 10 |
| Surfaces and Infrastructure | 15 |
| RSI and Self-Build Pipeline | 8 |
| Cross-Cutting Infrastructure | 20 |
| Integration and Support | 15 |
| Gateway / Egress Rail and GRIP Runtime | 14 |

> The per-layer groupings above are a curated tour, not an exhaustive
> filesystem listing. The canonical count is **202 top-level modules in
> `lib/hal/*.py`** (**284** incl. subpackages such as `lib/hal/happi/`).
> Regenerate with `ls lib/hal/*.py | wc -l` and
> `find lib/hal -name '*.py' | wc -l`.

---

## Test Coverage

- 353 test files in `tests/`
- Every module has at least one corresponding test file
- Tests use pytest with no external test framework dependencies

---

## Key Design Constraints

- **Zero external dependencies for core** — stdlib only in `types`, `home`, and all cross-cutting modules
- **Runtime-agnostic** — all modules work inside Claude Code, Cursor, or HAL standalone
- **Provider-agnostic** — every intelligence feature works across Anthropic, Google, Groq, OpenRouter, and local Ollama models
- **Fail-safe** — hard hooks block; soft hooks fail-open; all external calls have timeout and retry
