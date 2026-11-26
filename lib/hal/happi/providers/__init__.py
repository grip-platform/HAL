"""HAPPI/1.1 provider adapters (v1.0 envelopes accepted).

Each adapter translates the native streaming format of one LLM provider
into HAPPI NDJSON events (delta | thinking | tool_call | tool_result |
completed | error).
Adapters are protocol-version-agnostic: they emit via the shared
`lib.hal.happi.events` encoder, whose stamped version is resolved from the
P2 feature flag (`HAPPI_LIB_PROTOCOL_VERSION`). v1.0 callers continue to
dispatch byte-equivalently; v1.1 callers gain idr/audit semantics handled
in the runtime layer.

α-3 scope: Anthropic only, via `CLAUDE_CODE_OAUTH_TOKEN` subscription or
explicit API key.

α-3.5 scope: OpenAI-compat family (openai, groq, mistral, deepseek,
openrouter, moonshot, ollama) via a single parameterised adapter.

α-3.7 scope: Extended OpenAI-compat coverage (HAL#302) — qwen, zhipu,
xai, soruxgpt, vulavula (cloud) plus lemonade, vllm, lmstudio,
mlx_vlm (local). Same adapter, additional PROVIDER_CONFIGS entries.
The HAPPI router whitelist now matches `config/providers.yaml`; a sync
test prevents future drift.

α-3.6 scope: Google Gemini via the streamGenerateContent endpoint
(contents+parts body, candidates[].content.parts[].text response, no
[DONE] sentinel — separate from OpenAI-compat by API shape).

Interface contract:
    stream_<provider>(envelope: HappiEnvelope) -> Iterator[str]

Each function yields NDJSON event lines (without trailing newline).
Runtime layer owns `started`; the provider owns `delta`, `thinking`,
`tool_call`, `tool_result`, `completed`, and `error`. No provider executes a
tool: `tool_call` reports what the model asked for, the caller runs it and
returns the output on the next envelope (see openai_compat.py, W1b).
"""
from lib.hal.happi.providers.anthropic import stream_anthropic
from lib.hal.happi.providers.gemini import stream_gemini
from lib.hal.happi.providers.openai_compat import (
    PROVIDER_CONFIGS as OPENAI_COMPAT_PROVIDER_CONFIGS,
    SUPPORTED_PROVIDERS as OPENAI_COMPAT_SUPPORTED_PROVIDERS,
    stream_openai_compat,
)

__all__ = [
    "OPENAI_COMPAT_PROVIDER_CONFIGS",
    "OPENAI_COMPAT_SUPPORTED_PROVIDERS",
    "stream_anthropic",
    "stream_gemini",
    "stream_openai_compat",
]
