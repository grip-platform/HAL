# Adding a New Provider to happi-lib

> **Audience.** Engineers wiring a new LLM family (cloud or local) into
> HAL's HAPPI/1.1 runtime so it routes from any HAPPI-aware caller —
> CLI, a chat handler, voice gateway, future shims — without touching the
> caller-side code.

> **Why this exists.** A HAPPI caller's own provider-parity smoke test
> verifies that the three currently-shipped families (anthropic, gemini,
> openai_compat) emit a structurally identical event sequence
> `[started, delta+, completed]`. A fourth provider that diverges from
> that contract breaks every HAPPI consumer silently. This doc is the
> falsifiable recipe to keep the contract intact.

> **Scope.** Adding a *new family* (different wire dialect, different
> body shape, different SSE format). Adding *another model under an
> existing family* — say a new OpenAI-compat endpoint — only requires
> appending one entry to `PROVIDER_CONFIGS` in
> `lib/hal/happi/providers/openai_compat.py`. See the [Worked Example
> for an OpenAI-compat addition](#shortcut-adding-a-new-openai-compat-endpoint)
> at the bottom.

---

## The 5-Step Recipe

### 1. Read the contract

The provider adapter is a generator that translates one provider's
native streaming format into HAPPI NDJSON event lines.

```python
def stream_<name>(envelope: HappiEnvelope) -> Iterator[str]:
    """Yield NDJSON event lines (no trailing newline)."""
```

Mandatory event sequence on success:

```
delta+ → completed
```

The runtime layer (`lib/hal/happi/runtime.py`) emits the leading
`started` event before delegating to your adapter. Your adapter owns
`delta`, `completed`, and any error-path `error` event.

Failure paths each end with `[error, completed]` (in that order). The
existing three adapters share the same five failure paths:

| Failure | Code |
|---|---|
| Token unavailable from any source | `missing_credentials` |
| HTTP non-2xx response | `provider_http_<status>` |
| Network / DNS / TLS error | `provider_network` |
| Provider-emitted error event | `provider_error:<type>` |
| Unknown provider in family | `unsupported_provider` |

Reuse those codes — they are part of the consumer-facing contract.

### 2. Pick the closest existing adapter as a template

| New provider's wire format | Template to copy |
|---|---|
| OpenAI-style `data: {...}\n\n` + `[DONE]` sentinel | `providers/openai_compat.py` |
| Google-style `data: {...}\n\n` (no `[DONE]`) | `providers/gemini.py` |
| Named-event SSE (`event: foo\ndata: {...}`) | `providers/anthropic.py` |

If the new wire format matches none of these, you are introducing a
genuinely new dialect — copy the closest match and document the
divergence in your module docstring.

### 3. Implement `lib/hal/happi/providers/<name>.py`

The minimum surface every adapter exposes:

```python
from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.events import emit_completed, emit_delta, emit_error


# Module-level constants — endpoint, defaults, env-var name, keychain
# service, redaction pattern. Mirror the structure used in anthropic.py.

DEFAULT_MODEL = "<model-id>"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_S = 120
KEYCHAIN_SERVICE = "grip-<name>"
API_KEY_ENV = "<NAME>_API_KEY"


def _redact(text: str) -> str:
    ...  # strip provider-style tokens before logging error detail


def _resolve_token(envelope: HappiEnvelope) -> str | None:
    ...  # apikey-scheme literal -> env var -> Keychain (in that order)


def _build_request_body(envelope: HappiEnvelope) -> dict:
    ...  # envelope.content || envelope.flags.message || " ".join(args)


def _open_stream(body, token, timeout):
    ...  # urllib.request.urlopen(...) — patched in tests


def _translate_and_terminate(envelope, response):
    ...  # iterate provider SSE; yield emit_delta(..) per token
    ...  # yield emit_completed(envelope.id, usage={...}) at end


def stream_<name>(envelope: HappiEnvelope) -> Iterator[str]:
    """Main entry: token resolve -> body build -> open -> translate."""
    token = _resolve_token(envelope)
    if not token:
        yield emit_error(envelope.id, ..., code="missing_credentials")
        yield emit_completed(envelope.id, usage={})
        return

    body = _build_request_body(envelope)
    timeout = int(envelope.flags.get("timeout_s") or DEFAULT_TIMEOUT_S)
    try:
        response = _open_stream(body, token, timeout)
    except urllib.error.HTTPError as exc:
        # emit provider_http_<code> + completed
        ...
    except urllib.error.URLError as exc:
        # emit provider_network + completed
        ...

    try:
        yield from _translate_and_terminate(envelope, response)
    finally:
        try:
            response.close()
        except Exception:
            pass
```

### 4. Register the adapter in two places

#### 4a. Public re-export

`lib/hal/happi/providers/__init__.py`:

```python
from lib.hal.happi.providers.<name> import stream_<name>

__all__ = [
    "stream_anthropic",
    "stream_gemini",
    "stream_openai_compat",
    "stream_<name>",   # ← new
]
```

#### 4b. Runtime dispatch

`lib/hal/happi/runtime.py` `_dispatch_chat`:

```python
def _dispatch_chat(envelope: HappiEnvelope) -> Iterator[str]:
    provider_spec = str(envelope.flags.get("provider") or "")
    family = _select_family(provider_spec)

    if family == "" or family.startswith(ANTHROPIC_FAMILY_PREFIXES):
        from lib.hal.happi.providers.anthropic import stream_anthropic
        yield from stream_anthropic(envelope)
        return

    if family == "gemini":
        from lib.hal.happi.providers.gemini import stream_gemini
        yield from stream_gemini(envelope)
        return

    # ← new branch (place BEFORE the openai_compat fallback)
    if family == "<name>":
        from lib.hal.happi.providers.<name> import stream_<name>
        yield from stream_<name>(envelope)
        return

    # openai_compat catch-all
    ...
```

### 5. Test it

Copy the closest template under `tests/test_happi_providers_*.py`
and adapt it. The reusable mocking machinery is
`_FakeSSEResponse` — same shape across all three existing test files:

```python
class _FakeSSEResponse:
    def __init__(self, lines: list[str]):
        self._lines = [line.encode("utf-8") for line in lines]
        self.closed = False
    def __iter__(self):
        return iter(self._lines)
    def close(self):
        self.closed = True
```

Patch `_open_stream` on the adapter module to short-circuit the
`urllib.request.urlopen` call:

```python
with patch.object(provider, "_open_stream", return_value=fake_response):
    events = _collect(provider.stream_<name>(env))
```

Mandatory test coverage (mirrors the matrix in
`test_happi_providers_anthropic.py` and `_gemini.py`):

| Test | Asserts |
|---|---|
| `test_redact_strips_<name>_style_tokens` | secrets never escape into logs |
| `test_resolve_token_apikey_literal` | apikey scheme returns the literal token |
| `test_resolve_token_env_var` | env var is used when scheme is subscription |
| `test_resolve_token_keychain_fallback` | Keychain is consulted when env unset |
| `test_build_request_body_uses_args` | args fall through when no `flags.message` |
| `test_build_request_body_uses_message_flag` | `flags.message` wins over args |
| `test_stream_successful` | delta + completed events emitted in order |
| `test_stream_handles_upstream_error` | upstream error → error + completed |
| `test_stream_http_403` | 403 → `provider_http_403` + completed |
| `test_stream_network_error` | URLError → `provider_network` + completed |
| `test_stream_missing_credentials` | no token → `missing_credentials` + completed |
| `test_runtime_routes_<name>_spec_to_<name>_adapter` | end-to-end runtime dispatch |

#### Update your caller's own parity matrix

If your HAPPI-aware caller keeps its own provider-parity suite (mirroring
this doc's own 12-test matrix above, one level up), open it. Append a
`pytest.param(...)` row to `PROVIDER_MATRIX` and a sentinel-text row
to `sentinel_text_by_id`:

```python
PROVIDER_MATRIX = [
    pytest.param("claude-opus-4-7", anthropic_provider, ..., id="anthropic"),
    pytest.param("gemini/gemini-2.0-flash", gemini_provider, ..., id="gemini"),
    pytest.param("openai/gpt-4o-mini", openai_compat_provider, ..., id="openai_compat"),
    pytest.param("<name>/<model>", <name>_provider, _<name>_sse_lines(),
                 "<NAME>_API_KEY", "<fake-token>", id="<name>"),  # ← new
]
```

That single edit re-enrolls the new provider in all four parity tests
(structural, count parity, type-sequence parity, routing-key
exclusivity). **Caller-side handlers DO NOT change** — provider selection
remains `flags.provider="<name>/<model>"`.

---

## Worked Example: Hypothetical `vertex` Provider

Adding Google Vertex AI (different from Gemini's Generative Language
API — uses regional endpoints, OAuth2 not API key, Anthropic-style
request body):

1. **Module:** `lib/hal/happi/providers/vertex.py`
   ```python
   ENDPOINT_TEMPLATE = (
       "https://{region}-aiplatform.googleapis.com/v1/projects/"
       "{project}/locations/{region}/publishers/anthropic/models/"
       "{model}:streamRawPredict"
   )
   DEFAULT_MODEL = "claude-opus-4-7@20251015"
   DEFAULT_REGION = "us-east5"
   KEYCHAIN_SERVICE = "grip-vertex"
   API_KEY_ENV = "VERTEX_ACCESS_TOKEN"

   def stream_vertex(envelope: HappiEnvelope) -> Iterator[str]:
       ...
   ```

2. **Re-export:** add `stream_vertex` to `providers/__init__.py`.

3. **Runtime dispatch:** add a `family == "vertex"` branch in
   `_dispatch_chat` before the openai_compat fallback.

4. **Tests:** `tests/test_happi_providers_vertex.py` — copy the
   gemini test file (closest wire template), swap the URL prefix and
   the body assertions.

5. **Parity matrix:** if your caller keeps one (see "Update your
   caller's own parity matrix" above), append a row to it with
   `id="vertex"`.

A caller's own chat handler now accepts `provider="vertex"` with
**zero changes** to the caller's own handler code. The shim discovers the
adapter through the runtime registry, not through any caller-specific code.

---

## Shortcut: Adding a New OpenAI-Compat Endpoint

If the provider speaks the OpenAI `/v1/chat/completions` dialect with
SSE + `[DONE]` sentinel, **do not** create a new module. Add an entry
to `PROVIDER_CONFIGS` in
`lib/hal/happi/providers/openai_compat.py`:

```python
PROVIDER_CONFIGS: dict[str, dict[str, Any]] = {
    ...,
    "<name>": {
        "endpoint": "https://api.<name>.example/v1/chat/completions",
        "keychain": "grip-<name>",
        "env": "<NAME>_API_KEY",
        "auth": "bearer",
        "default_model": "<default-model-id>",
    },
}
```

Then mirror the same entry in `config/providers.yaml`. The sync test
`test_happi_router_covers_yaml_registry_no_drift` will fail if the
two registries diverge — that test exists specifically to catch the
HAL#302 class of bug.

No caller-side change. No new test file (the existing
`test_happi_providers_openai_compat.py` parameterises across the
whole `PROVIDER_CONFIGS` family, so your new entry inherits the test
suite for free).

---

## Falsification

This doc is wrong if any of the following hold:

- Following the recipe verbatim still leaves a HAPPI consumer broken
  (a caller's handler can't reach the new provider via
  `flags.provider="<name>"`). Fix: update the runtime dispatch
  documentation in step 4b above.
- The 5-step recipe omits a registry update (e.g. config/providers.yaml,
  the runtime sync test, the LiteLLM equivalency table). Fix: append
  the missing step here and add a regression test catching the
  omission.
- A real new provider needs a sixth surface that isn't covered (e.g.
  custom auth refresh loop, persistent connection pooling). Fix: split
  the recipe into "stateless" and "stateful" halves and document the
  stateful path here.

If any of those happen, file an issue against this doc and update
both the recipe and the regression-test surface so the next person
following it does not hit the same gap.
