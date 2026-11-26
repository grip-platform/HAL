"""HAL MCP Server — expose HAL as a tool provider via MCP protocol.

Any MCP-compatible client (Claude Desktop, Cursor, VS Code, etc.) can
connect to HAL and use its tools: chat, council, status, providers.

Implements the MCP tool listing and tool execution protocol over stdio.

Usage:
    # As MCP server (stdio transport):
    python3 -m lib.hal.mcp_server

    # In .mcp.json:
    {
      "hal": {
        "command": "python3",
        "args": ["-m", "lib.hal.mcp_server"],
        "cwd": "~/.hal"
      }
    }
"""
import json
import os
import sys

_hal_home = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
sys.path.insert(0, _hal_home)


# MCP tool definitions
_TOOLS = [
    {
        "name": "hal_chat",
        "description": "Send a message to HAL and get a response. Supports multi-turn conversation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The message to send"},
                "model": {"type": "string", "description": "Optional model address (e.g. groq/llama-3.3-70b)"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "hal_council",
        "description": "Query multiple LLMs simultaneously and merge results with agreement scoring.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question to pose to all council members"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "hal_status",
        "description": "Get HAL system status: providers, skills, modes, agents, health.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "hal_providers",
        "description": "List all configured LLM providers with health status and available models.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "hal_cost",
        "description": "Get current session cost breakdown by provider and model.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # Fleet Command tools
    {
        "name": "hal_agent_create",
        "description": "Create a new managed agent with model, routing strategy, and optional council models.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Agent name"},
                "model": {"type": "string", "description": "Model address (e.g. groq/llama-3.3-70b)"},
                "system_prompt": {"type": "string", "description": "Agent system prompt"},
                "routing_strategy": {"type": "string", "enum": ["fixed", "fallback", "quality_cost", "council"], "description": "A RoutingStrategy enum member (lib/hal/types.py). 'sentinel' is NOT a strategy: the Limit-Sentinel is an always-on failover layer, not a selectable routing strategy."},
                "fallback_chain": {"type": "array", "items": {"type": "string"}, "description": "Provider failover chain"},
                "council_models": {"type": "array", "items": {"type": "string"}, "description": "Multi-LLM council models"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "hal_agent_list",
        "description": "List all managed agent definitions with versions.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "hal_fleet",
        "description": "Get fleet-wide statistics: agents, sessions, cost by provider, zero-cost ratio, council usage.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "hal_task_submit",
        "description": "Submit a background task to an agent for async execution.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID to run the task"},
                "description": {"type": "string", "description": "Task description/prompt"},
            },
            "required": ["agent_id", "description"],
        },
    },
    # --- Obsidian Vault tools ---
    {
        "name": "hal_vault_search",
        "description": "Search notes in an Obsidian vault by keyword. Returns matching notes sorted by relevance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "hal_vault_browse",
        "description": "List all indexed notes in the Obsidian vault with titles and tags.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "hal_vault_read",
        "description": "Read a specific note from the Obsidian vault by relative path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within the vault (e.g. Projects/HAL.md)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "hal_vault_graph",
        "description": "Show forward links and backlinks for a note — knowledge graph fragment.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path of the note"},
            },
            "required": ["path"],
        },
    },
    # --- A2A peer delegation (three-harness mesh, downstream plane) ---
    {
        "name": "a2a_ask_peer",
        "description": (
            "Delegate a prompt to a PEER AI harness — claude_code, codex, or "
            "antigravity (Gemini) — running in its NATIVE read-only mode, and get "
            "its reply. Use this for a second opinion or another giant's capability. "
            "Each peer is capability-locked read-only (no writes, no shell) by "
            "construction; an unknown harness fails closed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "harness": {"type": "string", "description": "Peer: claude_code (claude/cc), codex, antigravity (gemini/agy)"},
                "prompt": {"type": "string", "description": "The prompt to delegate to the peer"},
                "timeout": {"type": "number", "minimum": 0, "description": "Optional timeout in seconds (default 120)"},
            },
            "required": ["harness", "prompt"],
        },
    },
    {
        "name": "a2a_list_peers",
        "description": "List the peer AI harnesses this gateway can delegate to (read-only, capability-locked).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class HALMCPServer:
    """MCP server exposing HAL tools over stdio."""

    def __init__(self):
        self._conversation = None
        self._council = None
        self._registry = None
        self._agent_registry = None
        self._fleet = None
        self._executor = None
        self._vault = None

    def _ensure_init(self):
        """Lazy init to avoid import overhead for tool listing."""
        if self._registry is not None:
            return
        from lib.hal.agent_registry import get_agent_registry
        from lib.hal.background_executor import BackgroundExecutor
        from lib.hal.conversation import Conversation
        from lib.hal.cost_tracker import CostTracker
        from lib.hal.discovery import discover_local_providers
        from lib.hal.fleet import FleetManager
        from lib.hal.moe import HALCouncil
        from lib.hal.registry import get_registry
        from lib.hal.session_registry import SessionRegistry

        self._registry = get_registry()
        discover_local_providers(self._registry)
        self._conversation = Conversation(registry=self._registry)
        self._council = HALCouncil(self._registry)
        self._agent_registry = get_agent_registry()
        sessions = SessionRegistry(registry=self._registry)
        self._fleet = FleetManager(self._agent_registry, sessions, CostTracker())
        self._executor = BackgroundExecutor(self._agent_registry, provider_registry=self._registry)
        vault_path = os.environ.get("HAL_OBSIDIAN_VAULT", "")
        if vault_path:
            from lib.hal.obsidian_vault import VaultIndexer
            self._vault = VaultIndexer(vault_path)

    def handle_request(self, request: dict) -> dict:
        """Handle a single MCP JSON-RPC request."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
        }

        handler = handlers.get(method)
        if handler is None:
            return _error_response(req_id, -32601, f"Method not found: {method}")

        try:
            result = handler(params)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as e:
            return _error_response(req_id, -32000, str(e)[:500])

    def _handle_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "hal", "version": "1.0.0"},
            "capabilities": {"tools": {}},
        }

    def _handle_tools_list(self, params: dict) -> dict:
        return {"tools": _TOOLS}

    def _handle_tools_call(self, params: dict) -> dict:
        self._ensure_init()
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        tool_handlers = {
            "hal_chat": self._tool_chat,
            "hal_council": self._tool_council,
            "hal_status": self._tool_status,
            "hal_providers": self._tool_providers,
            "hal_cost": self._tool_cost,
            "hal_agent_create": self._tool_agent_create,
            "hal_agent_list": self._tool_agent_list,
            "hal_fleet": self._tool_fleet,
            "hal_task_submit": self._tool_task_submit,
            "hal_vault_search": self._tool_vault_search,
            "hal_vault_browse": self._tool_vault_browse,
            "hal_vault_read": self._tool_vault_read,
            "hal_vault_graph": self._tool_vault_graph,
            "a2a_ask_peer": self._tool_a2a_ask_peer,
            "a2a_list_peers": self._tool_a2a_list_peers,
        }

        handler = tool_handlers.get(tool_name)
        if handler is None:
            return {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True}

        return handler(arguments)

    def _tool_chat(self, args: dict) -> dict:
        message = args.get("message", "")
        model = args.get("model")
        if model:
            self._conversation.configure(model=model)
        response = self._conversation.send(message)
        text = (
            f"{response.text}\n\n"
            f"---\n"
            f"Model: {response.provider}/{response.model} | "
            f"Latency: {response.latency_ms:.0f}ms | "
            f"Cost: ${response.cost_usd or 0:.6f}"
        )
        return {"content": [{"type": "text", "text": text}]}

    def _tool_council(self, args: dict) -> dict:
        query = args.get("query", "")
        models = _get_council_models(self._registry)
        result = self._council.deliberate(query, models)
        lines = [
            result.merged_text,
            "",
            f"---",
            f"Agreement: {result.agreement_score:.0%} | Models: {len(result.votes)}",
        ]
        for vote in result.votes:
            lines.append(f"  {vote.model}: {vote.response.text[:150]}...")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    def _tool_status(self, args: dict) -> dict:
        from lib.hal.agent_runner import AgentRunner
        from lib.hal.memory_engine import MemoryEngine
        from lib.hal.skill_loader import SkillLoader

        loader = SkillLoader()
        skills = loader.discover()
        modes = loader.list_modes()
        agents = AgentRunner().list_agents()
        memory = MemoryEngine()

        text = (
            f"HAL Status\n"
            f"Skills: {len(skills)} | Modes: {len(modes)} | Agents: {len(agents)}\n"
            f"Memory entries: {len(memory.list_entries())}\n"
            f"Providers: {len(self._registry.list_providers())}"
        )
        return {"content": [{"type": "text", "text": text}]}

    def _tool_providers(self, args: dict) -> dict:
        from lib.hal.warmup import _get_ollama_models
        lines = []
        for p in self._registry.list_providers():
            models = list(p.models)
            if p.name == "ollama" and not models:
                models = _get_ollama_models()
            health = "healthy" if p.healthy else "down"
            lines.append(f"{p.name} ({p.type.value}) [{health}]: {', '.join(models[:3])}")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    def _tool_cost(self, args: dict) -> dict:
        status = self._conversation.status()
        text = (
            f"Session Cost: ${status['total_cost_usd']:.6f}\n"
            f"Turns: {status['turns']} | Tokens: {status['total_tokens']}\n"
        )
        for provider, cost in status["cost_by_provider"].items():
            text += f"  {provider}: ${cost:.6f}\n"
        return {"content": [{"type": "text", "text": text}]}

    # --- Fleet Command tools ---

    def _tool_agent_create(self, args: dict) -> dict:
        defn = self._agent_registry.create(
            name=args.get("name", "unnamed"),
            model=args.get("model", ""),
            system_prompt=args.get("system_prompt", ""),
            routing_strategy=args.get("routing_strategy", "fallback"),
            fallback_chain=args.get("fallback_chain", []),
            council_models=args.get("council_models", []),
        )
        text = f"Agent created: {defn.name} (id={defn.agent_id}, v{defn.version}, strategy={defn.routing_strategy})"
        return {"content": [{"type": "text", "text": text}]}

    def _tool_agent_list(self, args: dict) -> dict:
        agents = self._agent_registry.list_agents()
        if not agents:
            return {"content": [{"type": "text", "text": "No agents defined."}]}
        lines = [f"Agents ({len(agents)}):"]
        for a in agents:
            lines.append(f"  {a.agent_id}: {a.name} v{a.version} [{a.routing_strategy}] model={a.model}")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    def _tool_fleet(self, args: dict) -> dict:
        snap = self._fleet.snapshot()
        text = (
            f"Fleet Status\n"
            f"Agents: {snap.total_agents} | Sessions: {snap.total_sessions}\n"
            f"Cost: ${snap.total_cost_usd:.4f} | Zero-cost: {snap.zero_cost_ratio:.0%}\n"
            f"Council: {snap.council_usage} | Sentinel: {snap.sentinel_interventions}\n"
            f"Providers: {snap.provider_diversity}"
        )
        return {"content": [{"type": "text", "text": text}]}

    def _tool_task_submit(self, args: dict) -> dict:
        agent_id = args.get("agent_id", "")
        description = args.get("description", "")
        if not agent_id or not description:
            return {"content": [{"type": "text", "text": "Error: agent_id and description required"}], "isError": True}
        task_id = self._executor.submit(agent_id, description)
        return {"content": [{"type": "text", "text": f"Task submitted: {task_id} (status: queued)"}]}

    # --- Obsidian Vault tools ---

    def _tool_vault_search(self, args: dict) -> dict:
        if not self._vault:
            return {"content": [{"type": "text", "text": "Vault not configured. Set HAL_OBSIDIAN_VAULT env var."}], "isError": True}
        query = args.get("query", "")
        limit = args.get("limit", 10)
        results = self._vault.search(query, limit=limit)
        if not results:
            return {"content": [{"type": "text", "text": f"No results for '{query}'"}]}
        lines = [f"Vault search: '{query}' ({len(results)} results)"]
        for note in results:
            tags = f" [{', '.join(note.tags[:3])}]" if note.tags else ""
            lines.append(f"  {note.path} — {note.title}{tags}")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    def _tool_vault_browse(self, args: dict) -> dict:
        if not self._vault:
            return {"content": [{"type": "text", "text": "Vault not configured. Set HAL_OBSIDIAN_VAULT env var."}], "isError": True}
        report = self._vault.index()
        lines = [f"Vault: {report.vault_path} ({len(report.notes)} notes)"]
        for note in sorted(report.notes, key=lambda n: n.path):
            tags = f" [{', '.join(note.tags[:3])}]" if note.tags else ""
            links = f" -> {len(note.wikilinks)} links" if note.wikilinks else ""
            lines.append(f"  {note.path} — {note.title}{tags}{links}")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    def _tool_vault_read(self, args: dict) -> dict:
        if not self._vault:
            return {"content": [{"type": "text", "text": "Vault not configured. Set HAL_OBSIDIAN_VAULT env var."}], "isError": True}
        path = args.get("path", "")
        content = self._vault.read_note(path)
        if content is None:
            return {"content": [{"type": "text", "text": f"Note not found: {path}"}], "isError": True}
        return {"content": [{"type": "text", "text": content}]}

    def _tool_vault_graph(self, args: dict) -> dict:
        if not self._vault:
            return {"content": [{"type": "text", "text": "Vault not configured. Set HAL_OBSIDIAN_VAULT env var."}], "isError": True}
        path = args.get("path", "")
        g = self._vault.graph(path)
        lines = [f"Graph: {g['note']}"]
        lines.append(f"  Forward links: {', '.join(g['forward_links']) or 'none'}")
        lines.append(f"  Backlinks: {', '.join(g['backlinks']) or 'none'}")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    # --- A2A peer delegation ---

    def _tool_a2a_ask_peer(self, args: dict) -> dict:
        from lib.hal.a2a_peer import UnknownPeerError, ask_peer
        # Validate at this boundary (untrusted MCP-client input): harness and
        # prompt must be NON-EMPTY STRINGS. A non-string (e.g. 123 or a list) or
        # a whitespace-only value is the documented "required" error here, never
        # passed to ask_peer (which would TypeError at spawn or raise a confusing
        # UnknownPeerError). The timeout cast is guarded so a non-numeric value is
        # a clean isError, not an uncaught ValueError.
        harness = args.get("harness")
        prompt = args.get("prompt")
        if (not isinstance(harness, str) or not harness.strip()
                or not isinstance(prompt, str) or not prompt.strip()):
            return {"content": [{"type": "text",
                                 "text": "Error: harness and prompt required (non-empty strings)"}],
                    "isError": True}
        harness = harness.strip()
        try:
            timeout = float(args.get("timeout", 120.0))
        except (TypeError, ValueError):
            return {"content": [{"type": "text", "text": "Error: timeout must be a number"}],
                    "isError": True}
        if timeout < 0:  # enforce the schema minimum at the boundary (not just declared)
            return {"content": [{"type": "text", "text": "Error: timeout must be >= 0"}],
                    "isError": True}
        try:
            reply = ask_peer(harness, prompt, timeout=timeout)
        except UnknownPeerError as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        verdict = "ok" if reply.ok else f"FAILED (exit {reply.exit_code})"
        text = f"{reply.output}\n\n---\nPeer: {reply.harness} | {verdict}"
        return {"content": [{"type": "text", "text": text}], "isError": not reply.ok}

    def _tool_a2a_list_peers(self, args: dict) -> dict:
        from lib.hal.a2a_peer import list_peers
        return {"content": [{"type": "text",
                             "text": "Peer harnesses: " + ", ".join(list_peers())}]}

    def run_stdio(self) -> None:
        """Run as stdio MCP server (blocking)."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            except json.JSONDecodeError:
                err = _error_response(None, -32700, "Parse error")
                sys.stdout.write(json.dumps(err) + "\n")
                sys.stdout.flush()


def _get_council_models(registry) -> list:
    """Pick available models for council."""
    from lib.hal.warmup import _get_ollama_models
    models = []
    ollama_fetched = None
    for p in registry.list_providers():
        if p.models:
            models.append(f"{p.name}/{p.models[0]}")
        elif p.name == "ollama":
            if ollama_fetched is None:
                ollama_fetched = _get_ollama_models()
            if ollama_fetched:
                models.append(f"ollama/{ollama_fetched[0]}")
    return models[:5]


def _error_response(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


if __name__ == "__main__":
    server = HALMCPServer()
    server.run_stdio()
