# HAL Sovereign Default

*Council-authored (DeepSeek + Gemini Pro judge; Anthropic calls: 0).
IDR: precog-1782593088-a2bbb9c2. Verify: grip-sovereign-council --verify 439b2860a784656b.*

## TL;DR

HAL routes all requests through Groq's Llama-3.3-70B-Versatile by default, with zero Anthropic egress. A six-provider failover ladder ensures resilience across environments without per-host configuration.

## Why Non-Anthropic Default

- **Cost efficiency**: Groq's Llama-3.3-70B-Versatile delivers competitive quality at a fraction of Anthropic's per-token pricing, reducing operational overhead.
- **Provider independence**: Relying on a single proprietary provider creates vendor lock-in risk. The multi-provider ladder distributes dependency across Groq, Gemini, DeepSeek, OpenAI, Qwen, and local Ollama models.

## Provider Ladder (Failover Order)

| Priority | Provider | Models | Key Source |
|----------|----------|--------|------------|
| 1 (default) | Groq | `llama-3.3-70b-versatile`, `llama-4-scout-17b-16e-instruct` | `keychain:grip-groq` |
| 2 | Gemini | `gemini-2.5-flash`, `gemini-3.1-pro-preview` | `keychain:grip-gemini` |
| 3 | DeepSeek | `deepseek-chat`, `deepseek-v4-pro`, `deepseek-reasoner` | `keychain:hal-deepseek` |
| 4 | OpenAI | `gpt-4o-mini`, `gpt-4o` | `keychain:hal-openai` |
| 5 | Qwen | `qwen3.5-flash`, `qwen3.5-plus` | `keychain:grip-qwen` |
| 6 (offline fallback) | Ollama | Any local model | N/A (local) |

## Credential Resolution

The single committed `config/providers.yaml` uses `api_key_source: keychain:grip-<provider>` entries. Resolution follows this fallback chain:

### macOS (Development)

Reads directly from the macOS Keychain:

```bash
security find-generic-password -s grip-<provider> -w
```

Store credentials once with:

```bash
security add-generic-password -s grip-groq -a "$USER" -w "gsk_..."
security add-generic-password -s grip-gemini -a "$USER" -w "AIza..."
security add-generic-password -s hal-deepseek -a "$USER" -w "sk-..."
```

### Linux / CI / VPS

Falls back to environment variables in this order:

1. `GRIP_<PROVIDER>_API_KEY` (e.g. `GRIP_GROQ_API_KEY`)
2. `<PROVIDER>_API_KEY` (e.g. `GROQ_API_KEY`)

This means the same committed `providers.yaml` works everywhere:

- **Mac dev** — keychain lookup
- **Linux VPS** — env vars set in pm2 ecosystem or systemd unit
- **CI** — env vars set in repository secrets

No per-host configuration files are required.

## Setting Up on Linux / VPS

Set these environment variables before starting HAL:

```bash
export GRIP_GROQ_API_KEY=gsk_...
export GRIP_GEMINI_API_KEY=AIza...
export GRIP_DEEPSEEK_API_KEY=sk-...
```

For production deployments, inject via your process manager:

```bash
# pm2 ecosystem.cjs
env: {
  GRIP_GROQ_API_KEY: process.env.GRIP_GROQ_API_KEY,
  GRIP_GEMINI_API_KEY: process.env.GRIP_GEMINI_API_KEY,
}

# systemd
Environment=GRIP_GROQ_API_KEY=gsk_...
```

## Switching Models

Use the `/model` command to select a specific provider and model at runtime:

```
/model groq llama-4-scout-17b-16e-instruct
/model gemini gemini-3.1-pro-preview
/model deepseek deepseek-reasoner
/model openai gpt-4o
/model qwen qwen3.5-plus
/model ollama llama3.2
```

The command accepts either the provider name alone (uses its default model) or `provider model-name`. If the specified provider is unavailable, HAL falls through the ladder to the next available provider.
