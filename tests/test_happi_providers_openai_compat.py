"""Golden tests for the OpenAI-compat provider adapter (α-3.5).

Covers the 7-provider family (openai, groq, mistral, deepseek, openrouter,
moonshot, ollama) via one parameterised adapter. Exercises spec parsing,
model defaulting, token resolution (apikey / env / keychain / none), body
translation, SSE streaming, error paths (HTTP / network / provider-level /
unsupported family), and runtime dispatch of `cmd: chat` envelopes.

All HTTP traffic is mocked — no network dependency.
"""
from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.providers import openai_compat as provider
from lib.hal.happi.runtime import run


class _FakeSSEResponse:
    """Minimal OpenAI-style SSE response matching what urlopen returns."""

    def __init__(self, lines: list[str]):
        self._lines = [line.encode("utf-8") for line in lines]
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


def _envelope(**overrides) -> HappiEnvelope:
    payload = {
        "v": "happi/1.0",
        "id": "req-test",
        "cmd": "chat",
        "args": ["Hello, HAPPI."],
        "flags": {"provider": "groq/llama-3.1-70b-versatile", "max_tokens": 100},
        "auth": {"scheme": "apikey", "token": "gsk_faketokenvalue0123456789abcdef"},
    }
    payload.update(overrides)
    return HappiEnvelope.parse(payload)


def _collect(events_iter) -> list[dict]:
    return [json.loads(line) for line in events_iter]


def _successful_sse_lines() -> list[str]:
    return [
        'data: {"id":"c1","choices":[{"delta":{"content":"Hel"},"index":0}]}',
        "",
        'data: {"id":"c1","choices":[{"delta":{"content":"lo"},"index":0}]}',
        "",
        'data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":5,"completion_tokens":2}}',
        "",
        "data: [DONE]",
        "",
    ]


# --- Token redaction --------------------------------------------------------

def test_redact_strips_openai_style_tokens():
    text = "curl failed with sk-proj-abcdefghijklmnopqrstuvwxyz123456 attached"
    out = provider._redact(text)
    assert "sk-proj" not in out
    assert "[REDACTED]" in out


def test_redact_strips_groq_gsk_tokens():
    text = "bad token gsk_realsecretvalue0123456789abcdef sent"
    out = provider._redact(text)
    assert "gsk_" not in out


# --- Provider spec parsing --------------------------------------------------

@pytest.mark.parametrize(
    "spec,expected",
    [
        ("", ("", "")),
        ("groq", ("groq", "")),
        ("GROQ", ("groq", "")),
        ("groq/llama-3.1-70b-versatile", ("groq", "llama-3.1-70b-versatile")),
        ("openrouter/anthropic/claude-sonnet-4-5", ("openrouter", "anthropic/claude-sonnet-4-5")),
        ("UNKNOWN", ("unknown", "")),
    ],
)
def test_parse_provider_spec_forms(spec, expected):
    assert provider._parse_provider_spec(spec) == expected


# --- Model resolution -------------------------------------------------------

def test_resolve_model_uses_default_for_family_when_absent():
    assert provider._resolve_model("groq", "") == "llama-3.1-70b-versatile"
    assert provider._resolve_model("openai", "") == "gpt-4o-mini"


def test_resolve_model_passthrough_when_provided():
    assert provider._resolve_model("groq", "custom-model") == "custom-model"


# --- Token resolution -------------------------------------------------------

def test_resolve_token_apikey_literal():
    env = _envelope(auth={"scheme": "apikey", "token": "literal-key"})
    assert provider._resolve_token("groq", env) == "literal-key"


def test_resolve_token_env_var_takes_precedence(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-groq-value")
    env = _envelope(auth={"scheme": "subscription"})
    assert provider._resolve_token("groq", env) == "env-groq-value"


def test_resolve_token_falls_back_to_keychain(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    env = _envelope(auth={"scheme": "subscription"})
    with patch.object(provider, "_fetch_keychain_token", return_value="keychain-value"):
        assert provider._resolve_token("groq", env) == "keychain-value"


def test_resolve_token_ollama_returns_empty_sentinel():
    env = _envelope(auth={})
    assert provider._resolve_token("ollama", env) == ""


def test_resolve_token_returns_none_when_all_sources_fail(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    env = _envelope(auth={"scheme": "subscription"})
    with patch.object(provider, "_fetch_keychain_token", return_value=None):
        assert provider._resolve_token("groq", env) is None


# --- Request body ----------------------------------------------------------

def test_build_request_body_uses_args_as_user_message():
    env = _envelope(args=["Tell", "me", "a", "joke"])
    body = provider._build_request_body("groq-m", env)
    assert body["model"] == "groq-m"
    assert body["messages"] == [{"role": "user", "content": "Tell me a joke"}]
    assert body["stream"] is True
    assert body["max_tokens"] == 100


# --- Egress headers ----------------------------------------------------------

def test_open_stream_sends_hal_user_agent(monkeypatch):
    """Egress must carry an explicit UA: urllib's default "Python-urllib/3.x"
    is bot-flagged by some provider CDNs (Groq 403s with an HTML challenge
    page, so a speaker leg parses as empty). Mirrors the non-happi adapter's
    "HAL/1.0" (lib/hal/providers/openai_compat.py)."""
    captured = {}

    class _Stop(Exception):
        pass

    def fake_urlopen(req, timeout=0):
        captured["req"] = req
        raise _Stop()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    try:
        provider._open_stream(
            "groq", {"model": "llama-3.3-70b-versatile"}, "gsk-test", 5)
    except _Stop:
        pass
    assert captured["req"].get_header("User-agent") == "HAL/1.0"


def test_build_request_body_remaps_gpt5_max_tokens():
    env = _envelope(flags={"provider": "openai/gpt-5.5", "max_tokens": 100})
    body = provider._build_request_body("gpt-5.5", env)
    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == 100


def test_build_request_body_keeps_gpt4_max_tokens():
    env = _envelope(flags={"provider": "openai/gpt-4o", "max_tokens": 100})
    body = provider._build_request_body("gpt-4o", env)
    assert body["max_tokens"] == 100
    assert "max_completion_tokens" not in body


def test_build_request_body_prepends_system_message_when_provided():
    env = _envelope(flags={"provider": "openai/gpt-4o-mini", "system": "Be terse."})
    body = provider._build_request_body("gpt-4o-mini", env)
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user"]
    assert body["messages"][0]["content"] == "Be terse."


# --- Sampling flags → request body -----------------------------------------

def test_build_request_body_threads_temperature_zero():
    # AGORA's determinism case: temperature 0.0 must reach the top-level body.
    env = _envelope(flags={"provider": "openai/gpt-4o-mini", "temperature": 0.0})
    body = provider._build_request_body("gpt-4o-mini", env)
    assert body["temperature"] == 0.0


def test_build_request_body_threads_top_p_and_seed():
    env = _envelope(flags={"provider": "openai/gpt-4o-mini", "top_p": 0.9, "seed": 42})
    body = provider._build_request_body("gpt-4o-mini", env)
    assert body["top_p"] == 0.9
    assert body["seed"] == 42


def test_build_request_body_omits_sampling_by_default():
    # Backwards-compat: no sampling keys present unless the caller sets them.
    body = provider._build_request_body("groq-m", _envelope())
    assert "temperature" not in body
    assert "top_p" not in body
    assert "seed" not in body


# --- Successful streaming ---------------------------------------------------

def test_successful_stream_emits_deltas_and_usage():
    env = _envelope()
    fake = _FakeSSEResponse(_successful_sse_lines())
    with patch.object(provider, "_open_stream", return_value=fake):
        events = _collect(provider.stream_openai_compat(env))
    types = [e["type"] for e in events]
    assert types == ["delta", "delta", "completed"]
    assert events[0]["text"] == "Hel"
    assert events[1]["text"] == "lo"
    assert events[2]["usage"] == {"in": 5, "out": 2}
    assert fake.closed is True


def test_stream_handles_upstream_error_event():
    env = _envelope()
    err_lines = [
        'data: {"error":{"message":"bad request","type":"invalid_request_error"}}',
        "",
        "data: [DONE]",
    ]
    fake = _FakeSSEResponse(err_lines)
    with patch.object(provider, "_open_stream", return_value=fake):
        events = _collect(provider.stream_openai_compat(env))
    assert [e["type"] for e in events] == ["error", "completed"]
    assert events[0]["code"] == "provider_error:invalid_request_error"


def test_http_401_emits_http_failure():
    env = _envelope()
    http_err = urllib.error.HTTPError(
        url="https://api.groq.com/openai/v1/chat/completions",
        code=401, msg="Unauthorized", hdrs={}, fp=io.BytesIO(b'{"error":"bad key"}'),
    )
    with patch.object(provider, "_open_stream", side_effect=http_err):
        events = _collect(provider.stream_openai_compat(env))
    assert [e["type"] for e in events] == ["error", "completed"]
    assert events[0]["code"] == "provider_http_401"


def test_network_error_emits_network_failure():
    env = _envelope()
    url_err = urllib.error.URLError("nodename nor servname provided")
    with patch.object(provider, "_open_stream", side_effect=url_err):
        events = _collect(provider.stream_openai_compat(env))
    assert [e["type"] for e in events] == ["error", "completed"]
    assert events[0]["code"] == "provider_network"


def test_missing_credentials_emits_auth_failure(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = _envelope(
        flags={"provider": "openai/gpt-4o-mini"},
        auth={"scheme": "subscription"},
    )
    with patch.object(provider, "_fetch_keychain_token", return_value=None):
        events = _collect(provider.stream_openai_compat(env))
    assert [e["type"] for e in events] == ["error", "completed"]
    assert events[0]["code"] == "missing_credentials"


def test_unsupported_family_direct_call_yields_error():
    env = _envelope(flags={"provider": "cohere/command-r"})
    events = _collect(provider.stream_openai_compat(env))
    assert [e["type"] for e in events] == ["error", "completed"]
    assert events[0]["code"] == "unsupported_provider"


# --- Runtime dispatch integration ------------------------------------------

def test_runtime_routes_openai_spec_to_openai_compat(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-fake-runtime-key-value")
    payload = {
        "v": "happi/1.0",
        "id": "req-runtime-openai",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "openai/gpt-4o-mini"},
    }
    fake = _FakeSSEResponse(_successful_sse_lines())
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    with patch.object(provider, "_open_stream", return_value=fake):
        rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 0
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert [e["type"] for e in events] == ["started", "delta", "delta", "completed"]


def test_runtime_routes_bare_family_spec_to_openai_compat(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_bare_family_test_key_value_0123")
    payload = {
        "v": "happi/1.0",
        "id": "req-runtime-groq-bare",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "groq"},
    }
    fake = _FakeSSEResponse(_successful_sse_lines())
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    with patch.object(provider, "_open_stream", return_value=fake):
        rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 0
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert [e["type"] for e in events] == ["started", "delta", "delta", "completed"]


def test_runtime_routes_ollama_without_auth(monkeypatch):
    payload = {
        "v": "happi/1.0",
        "id": "req-runtime-ollama",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "ollama/llama3.2"},
    }
    fake = _FakeSSEResponse(_successful_sse_lines())
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    with patch.object(provider, "_open_stream", return_value=fake):
        rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 0
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert [e["type"] for e in events] == ["started", "delta", "delta", "completed"]


# --- HAL#302 extended-coverage (α-3.7) -------------------------------------
# Origin: hal --status reported 18 providers but hal --happi-api only routed
# 9 families. The 10 missing families (qwen/zhipu/xai/arcee/soruxgpt/vulavula
# /lemonade/vllm/lmstudio/mlx_vlm) returned unsupported_provider from HAPPI
# even though they were registered in config/providers.yaml. The fix expanded
# PROVIDER_CONFIGS by 10 entries; these tests pin that fix and prevent regression.

EXTENDED_CLOUD_PROVIDERS = ("qwen", "zhipu", "xai", "soruxgpt", "vulavula")
EXTENDED_LOCAL_PROVIDERS = ("lemonade", "vllm", "lmstudio", "mlx_vlm")
ALL_EXTENDED_PROVIDERS = EXTENDED_CLOUD_PROVIDERS + EXTENDED_LOCAL_PROVIDERS


@pytest.mark.parametrize("family", ALL_EXTENDED_PROVIDERS)
def test_extended_provider_config_has_required_keys(family):
    cfg = provider.PROVIDER_CONFIGS[family]
    for key in ("endpoint", "keychain", "env", "auth", "default_model"):
        assert key in cfg, f"{family} missing {key!r}"
    assert cfg["endpoint"].startswith(("http://", "https://"))
    assert cfg["endpoint"].endswith("/chat/completions")
    assert cfg["auth"] in ("bearer", "none")
    assert cfg["default_model"], f"{family} default_model is empty"


@pytest.mark.parametrize("family", EXTENDED_CLOUD_PROVIDERS)
def test_extended_cloud_providers_use_bearer_auth(family):
    cfg = provider.PROVIDER_CONFIGS[family]
    assert cfg["auth"] == "bearer"
    assert cfg["keychain"], f"{family} cloud provider missing keychain"
    assert cfg["env"], f"{family} cloud provider missing env var name"


@pytest.mark.parametrize("family", EXTENDED_LOCAL_PROVIDERS)
def test_extended_local_providers_use_no_auth(family):
    cfg = provider.PROVIDER_CONFIGS[family]
    assert cfg["auth"] == "none"
    assert cfg["keychain"] is None
    assert cfg["env"] is None
    assert "localhost" in cfg["endpoint"] or "127.0.0.1" in cfg["endpoint"]


@pytest.mark.parametrize("family", ALL_EXTENDED_PROVIDERS)
def test_extended_provider_resolves_in_supported_set(family):
    assert family in provider.SUPPORTED_PROVIDERS


def test_hal_302_qwen_repro_no_longer_unsupported_provider(monkeypatch):
    """The exact envelope from HAL#302's reproduction snippet must not return unsupported_provider.

    Before fix: returned unsupported_provider because qwen was not in PROVIDER_CONFIGS.
    After fix: routes through openai_compat; the bearer auth attempt fails (no real
    keychain entry in test env) but the error code is missing_credentials or
    provider_http_*, NOT unsupported_provider.
    """
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(provider, "_fetch_keychain_token", lambda _: None)
    payload = {
        "v": "happi/1.0",
        "id": "test-1",
        "cmd": "chat",
        "args": [],
        "flags": {"provider": "qwen"},
        "auth": {},
        "ctx": {},
        "content": "ping",
    }
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    rc = run(stream_in=stdin, stream_out=stdout)
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    error_events = [e for e in events if e["type"] == "error"]
    assert error_events, "expected at least one error event for unauthenticated call"
    for ev in error_events:
        assert ev["code"] != "unsupported_provider", (
            f"HAL#302 regression: qwen still returns unsupported_provider: {ev}"
        )
    assert rc == 0


@pytest.mark.parametrize("family", ALL_EXTENDED_PROVIDERS)
def test_extended_family_routes_via_openai_compat_not_unsupported(family, monkeypatch):
    """Each new family must route through openai_compat (not error at the router layer)."""
    for env_var in ("DASHSCOPE_API_KEY", "ZHIPU_API_KEY", "XAI_API_KEY",
                    "SORUXGPT_API_KEY", "VULAVULA_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(provider, "_fetch_keychain_token", lambda _: None)
    fake = _FakeSSEResponse(_successful_sse_lines())
    payload = {
        "v": "happi/1.0",
        "id": f"req-{family}",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": family},
    }
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    with patch.object(provider, "_open_stream", return_value=fake):
        rc = run(stream_in=stdin, stream_out=stdout)
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    error_events = [e for e in events if e["type"] == "error"]
    for ev in error_events:
        assert ev["code"] != "unsupported_provider", (
            f"{family} routes to unsupported_provider — should reach openai_compat layer"
        )
    assert rc == 0


# --- Sync invariant: HAPPI router whitelist matches providers.yaml ----------


def _yaml_provider_names() -> set[str]:
    """Read provider names from config/providers.yaml without depending on registry.py.

    Stdlib-only line scan keyed on `  - name:` — robust to YAML structure changes
    elsewhere in the file. Excludes anthropic + gemini which use bespoke adapters.
    """
    import os
    yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config", "providers.yaml",
    )
    if not os.path.exists(yaml_path):
        pytest.skip(f"providers.yaml not found at {yaml_path}")
    names: set[str] = set()
    with open(yaml_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("- name:"):
                _, _, val = stripped.partition(":")
                names.add(val.strip().strip("'\""))
    return names


def test_happi_router_covers_yaml_registry_no_drift():
    """HAPPI PROVIDER_CONFIGS must cover every OpenAI-compat family in providers.yaml.

    This is the falsifiable safeguard from HAL#302: if someone adds a provider to
    config/providers.yaml without a matching PROVIDER_CONFIGS entry, this test
    fails in CI. The 'happi router whitelist diverges from status registry' bug
    is now mechanically prevented.

    Excluded: anthropic and gemini families — they have bespoke adapters
    (lib/hal/happi/providers/anthropic.py and gemini.py) so they don't need
    a PROVIDER_CONFIGS entry.
    """
    yaml_names = _yaml_provider_names()
    bespoke_adapters = {"anthropic", "gemini"}
    yaml_for_compat = yaml_names - bespoke_adapters
    happi_compat = set(provider.SUPPORTED_PROVIDERS)
    missing_in_happi = yaml_for_compat - happi_compat
    assert not missing_in_happi, (
        f"HAPPI router missing {len(missing_in_happi)} families that are in "
        f"providers.yaml: {sorted(missing_in_happi)}. "
        f"Add them to PROVIDER_CONFIGS in lib/hal/happi/providers/openai_compat.py."
    )


def test_happi_router_does_not_overshoot_yaml_registry():
    """No phantom families: every PROVIDER_CONFIGS entry must appear in providers.yaml.

    Falsifies a different drift direction: PROVIDER_CONFIGS having entries that
    were silently removed from providers.yaml. Keeps the two surfaces aligned
    in both directions.
    """
    yaml_names = _yaml_provider_names()
    happi_compat = set(provider.SUPPORTED_PROVIDERS)
    phantom_in_happi = happi_compat - yaml_names
    assert not phantom_in_happi, (
        f"PROVIDER_CONFIGS has {len(phantom_in_happi)} phantom families not in "
        f"providers.yaml: {sorted(phantom_in_happi)}. "
        f"Either remove from openai_compat.py or restore in config/providers.yaml."
    )


# --- P5: v1.1 acceptance + v1.0 regression --------------------------------

def test_openai_compat_accepts_v11_envelope(monkeypatch):
    """A v1.1 envelope dispatches successfully through the openai_compat adapter."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-v11-test-key")
    payload = {
        "v": "happi/1.1",
        "id": "req-v11-openai-compat",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "openai/gpt-4o-mini"},
    }
    fake = _FakeSSEResponse(_successful_sse_lines())
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    with patch.object(provider, "_open_stream", return_value=fake):
        rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 0
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert types == ["started", "delta", "delta", "completed"]


def test_openai_compat_v10_envelope_unchanged(monkeypatch):
    """Regression guard: v1.0 envelopes dispatch byte-equivalently post-P5."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-v10-regress")
    payload = {
        "v": "happi/1.0",
        "id": "req-v10-openai-compat-regress",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "openai/gpt-4o-mini"},
    }
    fake = _FakeSSEResponse(_successful_sse_lines())
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    with patch.object(provider, "_open_stream", return_value=fake):
        rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 0
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert types == ["started", "delta", "delta", "completed"]
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert deltas == ["Hel", "lo"]


def test_moonshot_default_model_is_a_live_id():
    """PROVIDER_CONFIGS["moonshot"] defaulted to moonshot-v1-auto, which the
    vendor retired on 2026-08-31 (HTTP 404 since), so every bare ``moonshot``
    envelope that relied on the family default failed. Red before 2026-09-07."""
    cfg = provider.PROVIDER_CONFIGS["moonshot"]
    assert cfg["default_model"] == "kimi-k2.6"
    assert not cfg["default_model"].startswith(("moonshot-v1", "kimi-k2.5"))


# --- W1: Kimi request-shape parity (lib/hal/model_quirks.py consumer) -------
#
# Live defect (2026-09-07): AGORA pins ``temperature: 0.0`` via
# happi/sampling.py, the body carried it, and Moonshot answered HTTP 400
# "invalid temperature: only 1 is allowed for this model" -- every kernel call
# to a Kimi id failed. _fix_request_body is the model-aware hook that already
# runs after resolve_sampling, so the quirks are applied there and nowhere else.

def _kimi_body(model: str, **flags) -> dict:
    env = _envelope(flags={"provider": f"moonshot/{model}", "max_tokens": 100, **flags})
    return provider._build_request_body(model, env)


def test_kimi_k3_body_drops_every_locked_sampling_param():
    body = _kimi_body("kimi-k3", temperature=0.0, top_p=0.9, seed=7)
    for locked in ("temperature", "top_p", "n", "presence_penalty", "frequency_penalty"):
        assert locked not in body, locked
    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == 100
    # seed is not on the vendor's locked list; it rides through untouched so a
    # provider that rejects it answers its own clear 400 (sampling.py policy).
    assert body["seed"] == 7
    assert body["model"] == "kimi-k3"


def test_kimi_k3_body_pins_vendor_default_reasoning_effort():
    # kimi-k3: top-level reasoning_effort, default max. Sent explicitly and
    # held constant so a session never flips levels (prefix-cache invalidation).
    body = _kimi_body("kimi-k3")
    assert body["reasoning_effort"] == "max"
    assert "thinking" not in body


@pytest.mark.parametrize("effort, wire", [("low", "low"), ("HIGH", "high"), ("max", "max")])
def test_kimi_k3_flags_effort_reaches_reasoning_effort(effort, wire):
    assert _kimi_body("kimi-k3", effort=effort)["reasoning_effort"] == wire


def test_kimi_k3_unknown_effort_passes_through_for_the_vendor_to_reject():
    # Not second-guessed here (sampling.py policy): Moonshot's own 400 names it.
    assert _kimi_body("kimi-k3", effort="xhigh")["reasoning_effort"] == "xhigh"


def test_kimi_k3_thinking_flag_is_not_a_k3_knob():
    # kimi-k3 reports supports_thinking_type "only"; its knob is reasoning_effort.
    body = _kimi_body("kimi-k3", thinking=False)
    assert "thinking" not in body
    assert body["reasoning_effort"] == "max"


def test_kimi_k26_body_default_path_has_no_reasoning_fields():
    body = _kimi_body("kimi-k2.6", temperature=0.0)
    assert "temperature" not in body
    assert "thinking" not in body
    assert "reasoning_effort" not in body
    assert body["max_completion_tokens"] == 100


@pytest.mark.parametrize("flag, wire", [
    (False, {"type": "disabled"}),
    (True, {"type": "enabled"}),
    ({"type": "disabled"}, {"type": "disabled"}),
])
def test_kimi_k26_thinking_only_when_the_caller_asked(flag, wire):
    assert _kimi_body("kimi-k2.6", thinking=flag)["thinking"] == wire


def test_kimi_k27_code_never_sends_thinking_even_when_asked():
    # "Forced on, cannot be turned off" -> a thinking block can only 400.
    body = _kimi_body("kimi-k2.7-code", thinking=False)
    assert "thinking" not in body
    assert "reasoning_effort" not in body
    assert body["max_completion_tokens"] == 100


def test_kimi_k27_code_highspeed_shares_the_k27_rules():
    body = _kimi_body("kimi-k2.7-code-highspeed", thinking=False, temperature=0.0)
    assert "thinking" not in body and "temperature" not in body
    assert body["max_completion_tokens"] == 100


def test_non_kimi_bodies_are_byte_identical_to_pre_w1():
    # Leak guard: the Kimi rules must not touch any other provider's body, even
    # when the envelope carries the new effort/thinking flags.
    flags = {"temperature": 0.0, "top_p": 0.9, "seed": 42, "max_tokens": 100,
             "system": "Be terse.", "effort": "low", "thinking": False}
    for model in ("deepseek/deepseek-chat", "openai/gpt-4o-mini"):
        env = _envelope(flags={"provider": model, **flags})
        bare = model.split("/", 1)[1]
        assert provider._build_request_body(bare, env) == {
            "model": bare,
            "messages": [{"role": "system", "content": "Be terse."},
                         {"role": "user", "content": "Hello, HAPPI."}],
            "max_tokens": 100,
            "stream": True,
            "temperature": 0.0,
            "top_p": 0.9,
            "seed": 42,
        }


def test_gpt5_keeps_its_rename_and_its_temperature():
    # gpt-5.x renames max_tokens but is NOT sampling-locked.
    env = _envelope(flags={"provider": "openai/gpt-5.5", "max_tokens": 100,
                           "temperature": 0.0, "effort": "low"})
    body = provider._build_request_body("gpt-5.5", env)
    assert body["max_completion_tokens"] == 100 and "max_tokens" not in body
    assert body["temperature"] == 0.0
    assert "reasoning_effort" not in body and "thinking" not in body


# --- W1: Moonshot nests `usage` inside choices[0] -----------------------------
#
# Seeded from a REAL final chunk captured from kimi-k3 on 2026-09-07 (never an
# invented shape). OpenAI puts `usage` at the top level of the last chunk;
# Moonshot puts it inside the finishing choice, so _extract_usage returned {}
# and every Kimi call reached the cost tracker and the IDR witness as zero.

_MOONSHOT_FINAL_CHUNK = (
    'data: {"id":"chatcmpl-6a9f0dfd653a9f26c52347bc","object":"chat.completion.chunk",'
    '"created":1788808702,"model":"kimi-k3","choices":[{"index":0,"delta":{},'
    '"finish_reason":"stop","usage":{"prompt_tokens":93,"completion_tokens":40,'
    '"total_tokens":133,"completion_tokens_details":{"reasoning_tokens":24}}}],'
    '"system_fingerprint":"fpv0_48a36b83"}'
)


def test_usage_is_read_from_moonshot_choice_level_final_chunk():
    lines = [
        'data: {"id":"c1","model":"kimi-k3","choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}',
        "",
        _MOONSHOT_FINAL_CHUNK,
        "",
        "data: [DONE]",
    ]
    env = _envelope(flags={"provider": "moonshot/kimi-k3"})
    fake = _FakeSSEResponse(lines)
    with patch.object(provider, "_open_stream", return_value=fake):
        events = _collect(provider.stream_openai_compat(env))
    assert [e["type"] for e in events] == ["delta", "completed"]
    assert events[-1]["usage"] == {"in": 93, "out": 40}


def test_extract_usage_top_level_still_wins_over_choice_level():
    # OpenAI shape unchanged; when both are present the top-level block is
    # authoritative and the nested one is ignored.
    data = {"usage": {"prompt_tokens": 5, "completion_tokens": 2},
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop",
                         "usage": {"prompt_tokens": 99, "completion_tokens": 99}}]}
    assert provider._extract_usage(data, {}) == {"in": 5, "out": 2}


def test_extract_usage_ignores_chunks_without_any_usage():
    data = {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}
    assert provider._extract_usage(data, {"in": 1, "out": 1}) == {"in": 1, "out": 1}


# --- Mistral content blocks (HAL#898) ---------------------------------------


def _mistral_block_sse_lines() -> list[str]:
    """Mistral Large 4's shape: thinking blocks first, then text blocks, then a plain string."""
    thinking = {"type": "thinking", "thinking": [{"type": "text", "text": "Let me think."}]}
    chunks = [
        {"id": "m1", "choices": [{"delta": {"content": [thinking]}, "index": 0}]},
        {"id": "m1", "choices": [{"delta": {"content": [{"type": "text", "text": "OK"}]}, "index": 0}]},
        {"id": "m1", "choices": [{"delta": {"content": " done"}, "index": 0}]},
        {"id": "m1", "choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
    ]
    lines = []
    for chunk in chunks:
        lines += [f"data: {json.dumps(chunk)}", ""]
    return lines + ["data: [DONE]", ""]


def test_mistral_content_blocks_become_thinking_and_string_deltas():
    env = _envelope(flags={"provider": "mistral/mistral-large-4", "max_tokens": 100})
    fake = _FakeSSEResponse(_mistral_block_sse_lines())
    with patch.object(provider, "_open_stream", return_value=fake):
        events = _collect(provider.stream_openai_compat(env))
    assert [e["type"] for e in events] == ["thinking", "delta", "delta", "completed"]
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert deltas == ["OK", " done"] and all(isinstance(d, str) for d in deltas)
    assert "message" not in events[-1], "Mistral thinking must not trigger a reasoning_content echo"


@pytest.mark.parametrize("content, expected", [
    ("plain", ("plain", "")),
    (None, ("", "")),
    ([{"type": "thinking", "thinking": "hmm"}], ("", "hmm")),
    ([{"type": "thinking", "thinking": [{"type": "text", "text": "a"}, "b"]}, {"type": "text", "text": "c"}], ("c", "ab")),
    (["x", {"type": "image_url", "image_url": "u"}, {"type": "text", "text": 3}], ("x", "")),
])
def test_content_parts_splits_answer_from_thinking(content, expected):
    assert provider._content_parts(content) == expected
