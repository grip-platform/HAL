# HAL Agent Spawning and Council Mesh

## Recursive Agent Spawning

```mermaid
graph TD
    ROOT[Root Agent<br/>groq/llama-3.3-70b<br/>depth=0] --> A1[Agent 1<br/>groq/qwen3-32b<br/>depth=1]
    ROOT --> A2[Agent 2<br/>ollama/mistral:7b<br/>depth=1]
    ROOT --> A3[Agent 3<br/>gemini/gemini-2.5-flash<br/>depth=1]
    
    A1 --> A1A[Sub-agent<br/>depth=2]
    A1 --> A1B[Sub-agent<br/>depth=2]
    
    A2 --> A2A[Sub-agent<br/>depth=2]
    
    A1A --> LEAF1[...<br/>depth=N]
    
    style ROOT fill:#e94560,color:#fff
    style A1 fill:#4fc3f7,color:#000
    style A2 fill:#ff9800,color:#000
    style A3 fill:#00d26a,color:#000
```

**Key properties:**
- Any agent can spawn sub-agents to configurable depth (default max_depth=21)
- Each agent can use a **different LLM provider** (cross-LLM mesh)
- Results propagate up: sub-agents report to their parent
- Cost tracked at every level

## Council (Mixture of Experts)

```mermaid
sequenceDiagram
    participant Q as Query
    participant C as HALCouncil
    participant M1 as Groq/Llama-70B
    participant M2 as Gemini/Flash
    participant M3 as Ollama/Mistral
    
    Q->>C: deliberate("Is this fair?")
    
    par Query all models
        C->>M1: chat(query, system)
        C->>M2: chat(query, system)
        C->>M3: chat(query, system)
    end
    
    M1-->>C: Vote 1 (weight=1.0)
    M2-->>C: Vote 2 (weight=1.0)
    M3-->>C: Vote 3 (weight=1.0)
    
    C->>C: Merge by weighted quality
    C->>C: Compute agreement (Jaccard)
    
    C-->>Q: CouncilResult{merged_text, agreement, votes}
```

## Agreement Scoring

The council uses Jaccard word-overlap similarity against a reference vote:

```
agreement = average(
    |words_ref AND words_vote| / |words_ref OR words_vote|
    for each non-reference vote
)
```

- **High agreement (>60%)**: Models converge on answer — high confidence
- **Medium agreement (30-60%)**: Partial convergence — review individual votes
- **Low agreement (<30%)**: Models diverge — indicates complex/ambiguous query

## Spawn Mesh Pattern

```mermaid
graph LR
    subgraph Orchestrator
        O[Root Agent]
    end
    
    subgraph Workers
        W1[Review for bugs<br/>groq/llama-70b]
        W2[Check security<br/>ollama/mistral]
        W3[Suggest improvements<br/>groq/qwen3-32b]
    end
    
    subgraph Results
        R[Merged Result]
    end
    
    O -->|spawn| W1
    O -->|spawn| W2
    O -->|spawn| W3
    
    W1 -->|result| R
    W2 -->|result| R
    W3 -->|result| R
    R -->|merged| O
```
