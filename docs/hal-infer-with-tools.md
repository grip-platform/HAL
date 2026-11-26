# HAL `/api/infer-with-tools` — capability-locked Anthropic tool-loop

## What this does (plain English)

HAL ships a new HTTP route that lets a Claude model **inspect files** —
`Read`, `Grep`, `Glob` — to answer a question, while never being able to
*change* anything. The model can read your code; it cannot run shell
commands, write files, edit text, or browse the web from this surface.

The boundary is bolted on in two independent places so a single bug in
either side cannot widen it:

1. **HAL's `SandboxedDispatcher`** hardcodes the allowed tool names as a
   Python `frozenset({"Read", "Grep", "Glob"})`. Anything else the model
   tries — `Bash`, `Write`, `Edit`, `Task`, `Agent`, `WebFetch`,
   `WebSearch`, etc. — gets a blocked `tool_result` and the model has to
   continue without it.
2. **HAL's `file_sandbox`** primitives refuse any path that resolves
   (via `os.path.realpath`) outside the sandbox root. Symlinks pointing
   at `/etc/passwd`, `../../../etc/sudoers`, `/etc/*` glob patterns —
   all refused with an audit-logged escape error.

The route is at `POST /api/infer-with-tools` on `hal-server` (port 3850
by default). It uses the same Claude OAuth subscription seat the
existing `/api/infer` route uses (CCH signer), so it's free per the
operator's Claude Code subscription — no per-token billing, no separate
API key, no env-propagation pitfalls.

## When to use it

| Goal | Route |
|---|---|
| Answer a question about a file in a sandbox | `/api/infer-with-tools` |
| Generate text, no file access needed | `/api/infer` (existing) |
| Anything that needs `Bash` / `Write` / `Edit` | Neither — this route refuses by design |

If you need write access, you are in the wrong place. The whole point of
this route is that an untrusted prompt can't reach those tools through
it.

## Request

```http
POST /api/infer-with-tools HTTP/1.1
Content-Type: application/json

{
  "prompt": "What is the import contract for lib/hal/conversation.py?",
  "sandbox_root": "~/.claude",
  "tools": ["Read", "Grep", "Glob"],
  "model": "claude-opus-4-7",
  "max_iterations": 8,
  "max_tokens": 4096,
  "system": null
}
```

| Field | Required | Default | Notes |
|---|---|---|---|
| `prompt` | yes | — | User-level question. Untrusted content goes here. |
| `sandbox_root` | yes | — | Absolute path; must exist; must not be `/`, `/etc`, `/bin`, `/sbin`, `/root`, `/System`, or a symlink resolving into any of those. Real path is canonicalised before any tool fires. |
| `tools` | no | `["Read","Grep","Glob"]` | Subset of the allowlist. Caller-supplied list is intersected with the dispatcher's hardcoded frozenset BEFORE tool schemas reach the model. |
| `model` | no | `claude-opus-4-7` | Any Claude model supported by the CCH subscription. |
| `max_iterations` | no | `8` | Cap on chat ↔ tool round-trips. The loop terminates with `hit_iteration_cap=true` if the model keeps emitting `tool_use` past this number. |
| `max_tokens` | no | `4096` | Anthropic `max_tokens` per call. |
| `system` | no | `null` | Optional system prompt. |

## Response (HTTP 200)

```json
{
  "ok": true,
  "text": "lib/hal/conversation.py exposes Conversation(model, system) ...",
  "tool_calls": [
    {
      "tool": "Read",
      "input": {"file_path": "lib/hal/conversation.py"},
      "is_error": false,
      "iteration": 0
    }
  ],
  "iterations": 2,
  "hit_iteration_cap": false,
  "error_code": null,
  "error_message": null
}
```

The route is **fail-safe like `/api/infer`** — any internal failure
returns HTTP 200 with `ok=false` so the GRIP adapter can degrade
cleanly. The only HTTP 400 responses are for genuinely missing inputs
(no `prompt`, no `sandbox_root`).

### Fail-safe `error_code` values

| `error_code` | Meaning | Operator action |
|---|---|---|
| `import_failed` | A W1/W2 module didn't load | Deploy issue — check `lib/hal/file_sandbox.py` / `lib/hal/sandboxed_tool_dispatch.py` exist on the host |
| `sandbox_root_refused` | Bad root (`/etc`, non-existent, symlink to system dir) | Fix the request — never the server |
| `no_allowed_tools` | Caller's `tools` list intersected with the allowlist is empty | Send `tools=null` to use the default |
| `cch_subscription_unavailable` | `CLAUDE_CODE_OAUTH_TOKEN` missing from env + keychain | Set the OAuth token; see CCH subscription docs |
| `tool_loop_raised` | The loop itself crashed (rare) | File a bug; the response carries the exception message |

## Capability-lock contract (the boundary)

Per `rules/untrusted-content-capability-lock.md` (GRIP), the surface a
spawned action can touch must be restricted, not the sender or the
permission prompt. This route is the canonical instance for HAL.

| Layer | Where | What it does |
|---|---|---|
| 1 — Request filter | `lib/hal/server.py::_iwt_build_dispatcher_and_schemas` | Intersects caller's `tools` with `_ALLOWED_TOOLS` BEFORE schemas reach the model. Drops `Bash`/`Write`/`Edit`/etc. silently. |
| 2 — Dispatcher allowlist | `lib/hal/sandboxed_tool_dispatch.py::SandboxedDispatcher.execute` | Per-`tool_use` check against the hardcoded `frozenset`. A model that emits `tool_use:Bash` despite the schemas gets `blocked=true`, never reaches a handler. |
| 3 — Filesystem boundary | `lib/hal/file_sandbox.py::_resolve_safe` | Every Read/Grep/Glob path runs through `os.path.realpath` + descendant check. Symlinks, `..`, absolute paths outside the root all refused with `SandboxEscapeError`. |
| 4 — Audit log | `~/.hal/logs/infer-with-tools.jsonl` | Every refusal AND every successful tool call appends one JSONL line, best-effort. |

The frozenset is the source of truth. Renaming an allowlist member
without updating it breaks the layer-1 filter AND the layer-2
dispatcher in the same place. There is no second mutable list to drift
out of sync.

## Client (GRIP)

Use `lib/hal_llm_adapter.infer_with_tools` — see
`tests/test_hal_llm_adapter_infer_with_tools.py` for the exact shape.
Module-level function, not a `HALLLMAdapter` method (the tool-loop is
stateless on the GRIP side; the entire fail-safe lives in HAL).

```python
from lib.hal_llm_adapter import infer_with_tools

result = infer_with_tools(
    "What does conversation.py expose?",
    sandbox_root=os.path.expanduser("~/.claude"),
    tools=["Read", "Grep", "Glob"],
)
if result.must_refuse():
    log.error("HAL refused: %s %s",
              result.error_code, result.error_message)
else:
    print(result.text)
    for call in result.tool_calls:
        print(f"  {call.tool}({call.input}) "
              f"→ {'err' if call.is_error else 'ok'}")
```

`result.must_refuse()` is True iff HAL is down (`hal_down=True`) OR the
server returned `ok=false`. IDR-aware callers should branch on it
before emitting any output, the same way `HALResult.must_refuse()`
guards the non-tools path.

## Audit log

`~/.hal/logs/infer-with-tools.jsonl` — one line per tool call, including
refusals:

```json
{"ts":"2026-05-29T05:30:11Z","tool":"Read","root":"~/.claude",
 "path":"lib/foo.py","allowed":true,"bytes":2341,"truncated":false}
{"ts":"2026-05-29T05:30:12Z","tool":"Read","root":"~/.claude",
 "path":"../../../etc/passwd","allowed":false,"reason":"escape"}
```

Best-effort: an I/O failure on the log never blocks the request.

## Tests (Goodhart-resistant — Rule 14)

| Suite | Count | Surface |
|---|---|---|
| `tests/test_file_sandbox.py` | 14 + 4 sub | W1 primitives — read/grep/glob + escape, root validation, audit |
| `tests/test_sandboxed_tool_dispatch.py` | 28 | W2 dispatcher — capability lock, sandbox escape, frozenset immutable, result shape |
| `tests/test_server_infer_with_tools.py` | 25 | W3 server route — 400 on missing input, fail-safe on bad root/no CCH, tools filter, end-to-end round-trip, response-shape lock |
| `tests/test_tool_loop.py` | 20 | Pre-existing loop driver |

Every test docstring names the mutation it survives. Suite total: 90+
across HAL.

## Falsifier

This route is wrong if any of these are observable in the wild:

- A tool name outside `{"Read", "Grep", "Glob"}` reaches a handler.
- A filesystem operation escapes `sandbox_root` via symlink, `..`, or
  an absolute path outside the root.
- A 5xx response on any non-input-validation failure (the contract is
  fail-safe HTTP 200 + `ok=false`).
- The dispatcher's `_ALLOWED_TOOLS` is observed as a non-frozenset (i.e.
  mutable from outside the module).

If you see any of these, file an issue + treat the route as untrusted
until the regression test is added.
