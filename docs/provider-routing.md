# HAL Provider Routing

## Routing Flow

```mermaid
sequenceDiagram
    participant U as User/Agent
    participant C as Conversation
    participant S as Sentinel
    participant R as Router
    participant P as Registry
    participant L as LLM Provider
    
    U->>C: send("query")
    C->>S: dispatch(model, query)
    
    Note over S: Check provider health
    alt Provider healthy + under limit
        S->>P: chat(model, query)
        P->>L: POST /v1/chat/completions
        L-->>P: Response + headers
        P-->>S: ChatResponse
        Note over S: Parse utilisation headers
        S-->>C: ChatResponse
    else Provider at limit (>85%)
        Note over S: Auto-failover
        S->>S: next in fallback chain
        S->>P: chat(fallback_model, query)
        P->>L: POST to fallback provider
        L-->>P: Response
        P-->>S: ChatResponse
        S-->>C: ChatResponse
    end
    
    C->>C: Record cost + update history
    C-->>U: Response
```

## Fallback Chain

```mermaid
graph LR
    A[groq/llama-3.3-70b] -->|rate limited| B[gemini/gemini-2.5-flash]
    B -->|rate limited| C[ollama/local model]
    C -->|always available| D((Zero Limits))
    
    style A fill:#4fc3f7,color:#000
    style B fill:#00d26a,color:#000
    style C fill:#ff9800,color:#000
    style D fill:#e94560,color:#fff
```

## Routing Strategies

| Strategy | Selection Logic | Use Case |
|----------|----------------|----------|
| **Fixed** | User specifies `provider/model` directly | Explicit model selection |
| **Fallback** | Try chain in order, skip on failure/rate-limit | Default: resilient routing |
| **Quality-Cost** | Triage by query complexity (simple→cheap, complex→premium) | Cost optimisation |
| **Council** | Query all available models, merge by weighted vote | Critical decisions |

## Provider Health States

```mermaid
stateDiagram-v2
    [*] --> Healthy
    Healthy --> Warning: utilisation > 70%
    Warning --> Limited: utilisation > 85%
    Limited --> Cooldown: 429/402 received
    Cooldown --> Healthy: cooldown expires
    
    Warning --> Healthy: utilisation drops
    Limited --> Warning: utilisation drops
```
