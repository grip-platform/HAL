"""OpenAI billing is checked at resolved send destinations, with fake credentials only.

The mutation runs for these assertions are recorded in the PR test plan.
"""
import ast
import json
import os
import re
import urllib.request
from pathlib import Path

import pytest

from lib.hal import gateway_egress, spend_guard, tool_cli_integration, tool_runtime
from lib.hal.gateway import is_transport_error
from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.providers import openai_compat as happi
from lib.hal.providers.openai_compat import OpenAICompatProvider
from lib.hal.voice.providers import openai_realtime as voice
from lib.hal.voice.realtime_base import RealtimeConfig

REPO = Path(__file__).resolve().parents[1]
OPENAI = "https://api.openai.com/v1/chat/completions"
FAKE = "fake-account-A-key"


@pytest.fixture(autouse=True)
def strict(monkeypatch):
    monkeypatch.delenv("GRIP_CODEX_ALLOW_PAID", raising=False)


@pytest.mark.parametrize("value", [None, "", "true", "01", "1"])
@pytest.mark.parametrize("opt_in", [[], ["openai-api-key"], ["*"]])
def test_literal_switch_is_the_only_opt_in(tmp_path, monkeypatch, value, opt_in):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"paid_opt_in": opt_in}))
    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(path))
    if value is not None:
        monkeypatch.setenv("GRIP_CODEX_ALLOW_PAID", value)
    assert spend_guard.decide_paid("openai-api-key").allowed is (value == "1")


@pytest.mark.parametrize("defect", ["missing", "malformed", "bug"])
def test_policy_defects_refuse_even_with_literal_opt_in(tmp_path, monkeypatch, defect):
    monkeypatch.setenv("GRIP_CODEX_ALLOW_PAID", "1")
    path = tmp_path / "policy.json"
    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(path))
    if defect == "malformed":
        path.write_text('{"paid_opt_in": "*"}')
    if defect == "bug":
        def broken():
            raise RuntimeError("fake policy bug")
        monkeypatch.setattr(spend_guard, "load_policy", broken)
    with pytest.raises(spend_guard.SpendRefused):
        spend_guard.require_openai_destination(OPENAI)


class Sent(Exception):
    """The fake transport received a request; never opens a socket."""


PATHS = ["gateway", "provider", "happi", "tools", "cli", "voice"]


def send(path, monkeypatch, calls, third_party=False):
    url = "https://api.groq.com/openai/v1/chat/completions" if third_party else OPENAI
    family = "groq" if third_party else "openai"
    model = "openai/gpt-oss-120b" if third_party else "gpt-6-astra"

    def transport(*args, **kwargs):
        calls.append(True)
        raise Sent()

    monkeypatch.setattr(urllib.request, "urlopen", transport)
    if path == "gateway":
        return gateway_egress.forward_same_wire(
            family, "/v1/chat/completions", json.dumps({"model": model}).encode(),
            opener=transport, key_resolver=lambda _: FAKE)
    if path == "provider":
        # The name is deliberately unrelated: billing follows the configured URL.
        provider = OpenAICompatProvider("alias", url.rsplit("/chat", 1)[0], api_key=FAKE)
        return provider._request("POST", "/chat/completions", {"model": model})
    if path == "happi":
        return happi._open_stream(family, {"model": model}, FAKE, 1)
    if path == "tools":
        return tool_runtime._post_openai_chat(url=url, api_key=FAKE, body={"model": model}, timeout_s=1)
    if path == "cli":
        monkeypatch.setattr(tool_cli_integration, "_resolve_bearer", lambda _: FAKE)
        fn, _ = tool_cli_integration.detect_tool_chat_fn(f"{family}/{model}", max_tokens=16)
        return fn(messages=[{"role": "user", "content": "hi"}], tools=[])
    backend = voice.OpenAiRealtimeBackend(config=RealtimeConfig(), api_key=FAKE, ws_factory=transport)
    return backend.start()


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("allowed", [False, True])
def test_every_key_send_checks_before_transport(monkeypatch, path, allowed):
    if allowed:
        monkeypatch.setenv("GRIP_CODEX_ALLOW_PAID", "1")
    calls = []
    with pytest.raises(Sent if allowed else spend_guard.SpendRefused) as caught:
        send(path, monkeypatch, calls)
    assert len(calls) == int(allowed)
    if not allowed:
        assert caught.value.decision.source == "openai-api-key"
        assert not isinstance(caught.value, OSError)
        assert not is_transport_error(caught.value)


@pytest.mark.parametrize("path", PATHS[:-1])
def test_an_openai_model_billed_by_groq_passes(monkeypatch, path):
    calls = []
    with pytest.raises(Sent):
        send(path, monkeypatch, calls, third_party=True)
    assert calls == [True]


def test_happi_refusal_is_one_distinct_terminal_event(monkeypatch):
    monkeypatch.setattr(happi, "_resolve_token", lambda *_: FAKE)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: pytest.fail("network reached"))
    envelope = HappiEnvelope.parse({"v": "happi/1.0", "id": "test", "cmd": "chat",
                                   "content": "hi", "flags": {"provider": "openai/gpt-6-astra"}})
    events = [json.loads(line) for line in happi.stream_openai_compat(envelope)]
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "spend_refused"


# New key/API references require an explicit audit; existing sends require a
# guard call in their function. Dynamic tests above catch a removed/no-op guard.
REFERENCES = {
    "lib/hal/spend_guard.py", "lib/hal/gateway_egress.py",
    "lib/hal/tool_cli_integration.py", "lib/hal/happi/providers/openai_compat.py",
    "lib/hal/voice/providers/openai_realtime.py",
    "lib/hal/credential_store.py",  # generic environment credential lookup, no send
    "lib/hal/happi_boot.py",  # environment allowlist, no send
    "lib/hal/gateway_cascade.py",  # names the OpenAI source only to ask decide_paid; sends via gateway_egress
    "lib/hal/raillm_ladder.py",  # asks decide_paid whether to splice the Astra rung; sends via openai_compat
    "lib/hal/cascade_policy.py",  # names the OpenAI source only to ask decide_paid; no send of its own
}
SENDS = {
    "lib/hal/gateway_egress.py": "forward_same_wire",
    "lib/hal/providers/openai_compat.py": "_do_request",
    "lib/hal/happi/providers/openai_compat.py": "_open_stream",
    "lib/hal/tool_runtime.py": "_post_openai_chat",
    "lib/hal/tool_cli_integration.py": "_dispatch_openai_compat",
    "lib/hal/voice/providers/openai_realtime.py": "start",
}


def test_new_openai_key_paths_require_a_guard_audit():
    tokens = re.compile(r"hal-openai|OPENAI_API_KEY|api\.openai\.com")
    found = set()
    excluded = {"node_modules", "venv", "dist", "build", "__pycache__", "target", "vendor"}
    for root, dirs, files in os.walk(REPO / "lib"):
        dirs[:] = [name for name in dirs if name not in excluded]
        for name in files:
            path = Path(root) / name
            try:
                if tokens.search(path.read_text()):
                    found.add(str(path.relative_to(REPO)))
            except UnicodeDecodeError:
                pass
    assert found <= REFERENCES, f"unaudited OpenAI path: {found - REFERENCES}"
    for path, name in SENDS.items():
        tree = ast.parse((REPO / path).read_text())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "require_openai_destination" for n in ast.walk(fn)), path


def test_council_fallback_keeps_billing_refusal_distinct(monkeypatch):
    from lib.hal import gateway_openai
    calls = []
    decision = spend_guard.Decision(False, "openai-api-key", "fake billing refusal")
    monkeypatch.setenv("GRIP_COUNCIL_FAILOPEN_MODEL", "openai/gpt-6-astra")
    monkeypatch.setattr(gateway_egress, "same_wire_provider", lambda _: "openai")
    def refuse(*_args):
        raise spend_guard.SpendRefused(decision)
    monkeypatch.setattr(gateway_openai, "_relay_same_wire", refuse)
    monkeypatch.setattr(gateway_openai, "send_openai_spend_refusal",
                        lambda handler, d: calls.append(d))
    monkeypatch.setattr(gateway_openai, "_send_openai_error",
                        lambda *_a: pytest.fail("billing refusal became a transport error"))
    gateway_openai._relay_council_failopen(object(), {}, "council", 0, "fake outage")
    assert calls == [decision]


@pytest.mark.parametrize("url", [OPENAI, "https://API.OPENAI.COM:443/v1/responses",
                                 "wss://api.openai.com./v1/realtime"])
def test_resolved_openai_host_forms_cannot_bypass_the_guard(url):
    with pytest.raises(spend_guard.SpendRefused):
        spend_guard.require_openai_destination(url)
