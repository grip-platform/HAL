# HAL Conversation Flow

## Multi-Turn Conversation

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as CLI/Server
    participant Conv as Conversation
    participant Ctx as Context Assembler
    participant Skill as Skill Loader
    participant Mem as Memory Engine
    participant Sent as Sentinel
    participant Cost as Cost Tracker
    
    U->>CLI: "What is a Knowledge Engine?"
    CLI->>Conv: send(message)
    
    Note over Conv: First turn — build context
    Conv->>Ctx: assemble(mode="legal")
    Ctx->>Ctx: Load HAL.md chain
    Ctx->>Skill: resolve_for_mode("legal")
    Skill-->>Ctx: [skill prompts]
    Ctx->>Mem: read_memory()
    Mem-->>Ctx: MEMORY.md entries
    Ctx-->>Conv: ContextBundle
    
    Conv->>Conv: Append user message to history
    Conv->>Sent: dispatch(model, message, system, history)
    Sent-->>Conv: ChatResponse
    
    Conv->>Cost: record_response(model, provider, usage)
    Conv->>Conv: Append assistant message to history
    Conv-->>CLI: ChatResponse
    CLI-->>U: Display response + metadata
    
    Note over U,CLI: Next turn — history preserved
    U->>CLI: "How does it help law firms?"
    CLI->>Conv: send(message)
    Note over Conv: History includes previous exchange
    Conv->>Sent: dispatch(model, message, system, history=[prev...])
    Sent-->>Conv: ChatResponse
    Conv-->>CLI: ChatResponse
```

## Context Assembly

```mermaid
graph TD
    HAL_MD[HAL.md Chain] --> SP[System Prompt]
    MODE[Mode Prompt] --> SP
    SKILLS[Skill Prompts] --> SP
    GIT[Git Status + Branch] --> SP
    MEM[MEMORY.md Entries] --> SP
    ID[Identity / Self-Awareness] --> SP
    
    SP --> LLM[LLM Query]
    HISTORY[Conversation History] --> LLM
    USER[User Message] --> LLM
    
    style SP fill:#4fc3f7,color:#000
    style LLM fill:#e94560,color:#fff
```

## HAL.md Resolution Chain

```
Priority order (first found wins per level):

1. <project>/.hal/HAL.md        ← Project-specific HAL instructions
2. <project>/.claude/CLAUDE.md  ← Project-specific CC instructions  
3. <project>/CLAUDE.md          ← Legacy location
4. ~/.hal/HAL.md                ← Global HAL instructions
5. ~/.hal/CLAUDE.md             ← Global fallback

All found files are concatenated into the system prompt.
HAL.md takes priority over CLAUDE.md at the same directory level.
```
