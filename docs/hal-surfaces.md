# HAL Multi-Surface Architecture

## Surface Layer

```mermaid
graph TD
    subgraph Surfaces["User Surfaces"]
        CLI[CLI REPL<br/>bin/hal]
        WEB[Web App<br/>hal-web / React]
        PWA[PWA<br/>Installable]
        ELECTRON[Electron<br/>Desktop App]
        MOBILE[Mobile<br/>React Native]
        MCP_C[MCP Client<br/>Claude Desktop, Cursor]
        VOICE[Voice<br/>STT + TTS]
    end
    
    subgraph API["HAL API Layer"]
        HTTP[HTTP REST<br/>/api/chat, /api/council]
        WS[WebSocket<br/>/ws/chat]
        SSE[SSE Stream<br/>/api/stream]
        MCP_S[MCP Server<br/>Tool provider]
    end
    
    subgraph Core["HAL Core"]
        CONV[Conversation Engine]
        COUNCIL[Council / MoE]
        AGENTS[Agent Spawner]
        TOOLS[Tool Dispatch]
    end
    
    subgraph Providers["LLM Providers"]
        CLOUD[Cloud<br/>Groq, Gemini, OpenAI]
        LOCAL[Local<br/>Ollama, vLLM, LM Studio]
    end
    
    CLI --> HTTP
    WEB --> HTTP
    WEB --> SSE
    PWA --> HTTP
    ELECTRON --> HTTP
    MOBILE --> HTTP
    MCP_C --> MCP_S
    VOICE --> HTTP
    
    HTTP --> CONV
    WS --> CONV
    SSE --> CONV
    MCP_S --> CONV
    
    CONV --> COUNCIL
    CONV --> AGENTS
    CONV --> TOOLS
    
    COUNCIL --> Providers
    AGENTS --> Providers
    CONV --> Providers
    
    style CLI fill:#e94560,color:#fff
    style WEB fill:#4fc3f7,color:#000
    style MCP_S fill:#ff9800,color:#000
    style CONV fill:#00d26a,color:#000
```

## Current Status

| Surface | Technology | Status |
|---------|-----------|--------|
| CLI (REPL) | Python (`bin/hal`) | Done |
| Web (embedded) | Inline HTML in `server.py` | Done |
| Web (React app) | React + Vite + Tailwind | Planned (hal-web) |
| PWA | Web + manifest + SW | Planned |
| Electron | Electron shell around hal-web | Planned |
| Mobile | React Native (from grip-mobile) | Planned |
| MCP Server | Python MCP protocol | Planned |
| Voice | STT + TTS integration | Planned |

## API Endpoints

```
GET  /                    # Web UI (embedded HTML or hal-web)
GET  /api/status          # System status (providers, skills, modes, agents)
GET  /api/providers       # Provider list with health
POST /api/query           # Single-turn query (provider + prompt)
POST /api/chat            # Multi-turn conversation (message + optional model)
POST /api/council         # Cross-LLM council query
GET  /api/stream          # SSE token streaming (planned)
WS   /ws/chat             # WebSocket chat (planned)
```

## Daemon Mode (Planned)

```
hal --daemon start        # Background HTTP server + mesh + warmup
hal --daemon stop         # Graceful shutdown
hal --daemon status       # Show PID, uptime, connections

Always on. No sessions. Cross-device handoff.
Mesh heartbeat broadcasts to ~/.hal/state/mesh/{pid}.json
```
