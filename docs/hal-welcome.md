# `hal welcome` — spoken context-aware greeting

> "Tomorrow morning after a reboot I want HAL to welcome me back, by talking to me — in real time."
> — V>>, 2026-04-18

## What it does

Assembles a snapshot of your repo state (current branch, HEAD commit, the last six merged PR subjects, the tail of `state/session-context.md`, your `active-identity` signature) into a system prompt, dispatches a HAPPI `chat` envelope, and streams the LLM response through macOS `say` via stdin.

Because each LLM `delta` event writes directly to `say`'s stdin and flushes, **the first spoken word starts playing while the LLM is still generating the rest of the reply.** First audible word is typically under 1.5s on a warm connection — the `time_to_first_spoken_word_ms` hypothesis (H160) measures this.

## Usage

```bash
# Default: empty provider → HAPPI's Anthropic default, Jamie Premium voice.
hal welcome

# Choose a different provider (any HAPPI family).
hal welcome --provider gemini
hal welcome --provider groq/llama-3.1-70b-versatile
hal welcome --provider openrouter/anthropic/claude-sonnet-4-5

# Different macOS voice.
hal welcome --voice Samantha
hal welcome --voice Daniel --rate 200

# Speak a fixed string (bypass the LLM entirely — always-works fallback).
hal welcome --text "Welcome back."

# Inspect the assembled envelope + context without speaking.
hal welcome --dry-run
```

## Auto-run on reboot

Three mechanisms, pick one. The command itself is the same; only the trigger differs.

### Option 1 — manual (zero config)

Type `hal welcome` yourself when you open a terminal in the morning. Zero cognitive load once it's in muscle memory. This is the default and the most reliable — no hooks, no launchd, no surprises.

### Option 2 — shell alias on first interactive login

Add this to `~/.zshrc` (or `~/.bashrc`). It runs only for the FIRST interactive shell opened after login (via `$HAL_WELCOMED_SESSION`):

```bash
# ~/.zshrc — greet once per login session
if [[ -o interactive ]] && [[ -z "$HAL_WELCOMED_SESSION" ]]; then
    export HAL_WELCOMED_SESSION=1
    hal welcome 2>/dev/null || true
fi
```

The `|| true` + `2>/dev/null` guard means a provider outage never blocks your shell.

### Option 3 — launchd LaunchAgent on login

For an always-fires-on-login experience even before a terminal opens, install the LaunchAgent template at `docs/com.codetonight.hal.welcome.plist.example`:

```bash
# one-time install
cp ~/.hal/docs/com.codetonight.hal.welcome.plist.example \
   ~/Library/LaunchAgents/com.codetonight.hal.welcome.plist
# update the path to hal in the plist if ~/.hal/bin isn't in your launchd PATH
launchctl load ~/Library/LaunchAgents/com.codetonight.hal.welcome.plist
```

The plist sets `RunAtLoad=true` so it fires once on each login. `KeepAlive=false` so it doesn't restart if the greeting ends. Logs land in `~/Library/Logs/hal-welcome.log`.

To unregister:

```bash
launchctl unload ~/Library/LaunchAgents/com.codetonight.hal.welcome.plist
rm ~/Library/LaunchAgents/com.codetonight.hal.welcome.plist
```

## Architecture

```
hal welcome
    │
    ▼
lib.hal.welcome.main()
    │
    ├── gather_context() ──► signature, branch, HEAD, 6 recent PRs, session notes
    │       └── git log + filesystem reads (5s timeouts, no network)
    │
    ├── render_system_prompt(context) ──► ~1200 char prompt with facts
    │
    ├── build_welcome_envelope(context, provider) ──► HappiEnvelope cmd=chat
    │
    └── stream_say(envelope) ──► lib.hal.voice.stream_speak.stream_say
            │
            ├── runtime._dispatch(envelope) yields NDJSON event lines
            │       (same dispatch path as hal --happi-api, HTTP, socket)
            │
            └── for each delta event: Popen("say").stdin.write(delta.text) + flush
                    │
                    └── macOS say speaks as it reads — first word out < 1.5s
```

Every layer delegates to an existing primitive. No new transport, no new HTTP, no new subprocess pattern. `stream_say` is ~130 LOC on top of the runtime + a stdlib Popen call.

## Stdlib-only

The welcome command uses **zero** external dependencies:

- Context assembly: `subprocess` (for git) + `pathlib` (filesystem)
- Envelope construction: `lib.hal.happi.envelope`
- LLM dispatch: `lib.hal.happi.runtime._dispatch` (urllib under the hood)
- TTS: macOS `say` via `subprocess.Popen`

No `requests`, no `anthropic`, no `google-cloud-texttospeech`. On a fresh macOS install with `python3` on path, `hal welcome --text "hi"` works immediately. Provider-backed greetings need a key somewhere HAPPI's runtime can find it (env var or Keychain) — see individual provider docs.

## Platform support

- **macOS (Darwin)**: full support, Jamie Premium default voice. Requires `say` on PATH (every macOS has it).
- **Linux / Windows**: `stream_say` raises `PlatformUnsupported` at the default spawn path. A cross-platform streaming TTS backend (Piper streaming, espeak-ng, etc.) is a future atom — the interface is already abstracted via the `spawn` parameter of `stream_say`, so the swap is one function per backend.
- **Non-platform override**: `--text` with a subprocess-swap-capable consumer (or injecting `spawn=` at the library layer) works everywhere tests run.

## Provider selection

Any HAPPI family works as a `--provider` argument:

| Family | Example provider arg | Notes |
|--------|----------------------|-------|
| Anthropic | `""` (default) or `"claude-opus-4-7"` or `"anthropic/claude-sonnet-4-5"` | Uses `CLAUDE_CODE_OAUTH_TOKEN` or `grip-claude-oauth` Keychain |
| Gemini | `"gemini"` or `"gemini/gemini-2.0-flash"` | `GEMINI_API_KEY` env / `grip-gemini` Keychain |
| OpenAI | `"openai/gpt-4o-mini"` | `OPENAI_API_KEY` / `hal-openai` Keychain |
| Groq | `"groq/llama-3.1-70b-versatile"` | fastest first-token in my testing |
| Mistral / DeepSeek / OpenRouter / Moonshot | `"<family>/<model>"` | |
| Ollama | `"ollama/llama3.2"` | local; no auth |

If no provider keys are available (fresh machine, pre-setup), `hal welcome --text "Welcome back."` always works because it bypasses the LLM entirely.

## Falsification (H160)

The welcome command claims: **first audible word plays within 1500ms of envelope dispatch on a warm connection.**

Measured via the `time_to_first_spoken_word_ms` metric. Confirmed locally during shipping — actual numbers depend on:

- LLM time-to-first-token (Groq ~300ms, Anthropic ~500-800ms, Gemini ~400-600ms)
- `say`'s own buffering threshold (empirically ~100-200ms of text before first phoneme)
- Audio-subsystem initialisation (one-shot per process)

Falsified by:

- First spoken word consistently beyond 1.5s across providers
- `say` buffers the entire stream before speaking (would suggest stdin isn't flushed properly — possible on future macOS versions)
- `subprocess.Popen` overhead dominates — needs investigation, could switch to `os.execvp` + pipe

If falsification holds, the fix is to measure specifically where the latency lives and either (a) pre-warm say via a no-op first delta, or (b) pipe audio bytes directly from a faster TTS backend like Piper.

## Tests

- `tests/test_voice_stream_speak.py` — 20 unit tests covering the streaming primitive (argv building, stdin write/flush order, null-byte stripping, error-event short-circuit, platform guard, envelope vs iterator source normalisation).
- `tests/test_hal_welcome.py` — 23 unit tests covering context assembly (mocked git + filesystem), system-prompt rendering, envelope construction, fixed-text path, argparse + `--dry-run` + `--text` + `--provider` forwarding.

Both suites run in under 200ms. No real LLM, no real `say` — every test injects a mock `spawn` factory.
