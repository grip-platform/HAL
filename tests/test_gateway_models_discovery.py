"""``GET /v1/models`` discovery (Kimi plan W1 gateway bullet, shipped in W2).

Before: a verbatim Anthropic GET plus the Codex ``models`` injection, so an
OpenAI-wire harness with no Anthropic credential got Anthropic's 401 and
could not see a single HAL-routed model (``:4010/v1/models`` answered
``x-api-key header is required`` live on 2026-09-07). After: the ids
declared in ``config/providers.yaml`` are served offline from the provider
registry in OpenAI list shape (``<provider>/<model>``, which the gateway
routes by explicit prefix), and the Anthropic passthrough is merged in only
when the client sent a credential. Hermetic except the one registry smoke
test, which reads the checkout's providers.yaml (no network).
"""
import io
import json
import unittest
from unittest import mock

from lib.hal.gateway import (
    GatewayUpstreamError,
    UpstreamResponse,
    declared_model_entries,
    handle_models_request,
)
from lib.hal.types import ProviderInfo, ProviderType


class _FakeHandler:
    def __init__(self, headers=None):
        self.command = "GET"
        self.headers = headers if headers is not None else {}
        self.rfile = io.BytesIO(b"")
        self.wfile = io.BytesIO()
        self.status = None
        self.sent_headers = []

    def send_response(self, code):
        self.status = code

    def send_header(self, k, v):
        self.sent_headers.append((k, v))

    def end_headers(self):
        pass

    def body(self):
        return json.loads(self.wfile.getvalue())


class _Registry:
    def __init__(self, infos):
        self._infos = infos

    def list_providers(self):
        return self._infos


_DECLARED = [
    {"id": "moonshot/kimi-k3", "object": "model", "owned_by": "moonshot"},
    {"id": "moonshot/kimi-k2.6", "object": "model", "owned_by": "moonshot"},
]
_ANTHROPIC_OK = json.dumps({
    "data": [{"type": "model", "id": "claude-opus-5", "display_name": "Claude Opus 5"}],
    "has_more": False, "first_id": "claude-opus-5", "last_id": "claude-opus-5",
}).encode()
_ANTHROPIC_401 = json.dumps({"type": "error", "error": {
    "type": "authentication_error", "message": "x-api-key header is required"}}).encode()


def _upstream(status, raw):
    return UpstreamResponse(status, [("Content-Type", "application/json")], io.BytesIO(raw))


class TestDeclaredModelEntries(unittest.TestCase):

    def test_every_declared_id_in_canonical_address_form(self):
        reg = _Registry([
            ProviderInfo(name="moonshot", type=ProviderType.CLOUD,
                         base_url="https://api.moonshot.ai/v1", models=("kimi-k3", "kimi-k2.6")),
            ProviderInfo(name="ollama", type=ProviderType.LOCAL,
                         base_url="http://127.0.0.1:11434/v1", models=()),
        ])
        self.assertEqual(declared_model_entries(reg), _DECLARED)

    def test_registry_failure_is_an_empty_list_never_a_crash(self):
        class _Boom:
            def list_providers(self):
                raise RuntimeError("yaml missing")
        self.assertEqual(declared_model_entries(_Boom()), [])

    def test_checkout_providers_yaml_declares_kimi_k3(self):
        # Offline smoke over the real registry (HAL_HOME is this checkout).
        ids = {e["id"] for e in declared_model_entries()}
        self.assertIn("moonshot/kimi-k3", ids)
        self.assertNotIn("moonshot/kimi-k2.5", ids)  # retired 2026-08-31


class TestModelsRequest(unittest.TestCase):

    def test_no_credential_serves_declared_list_without_touching_anthropic(self):
        handler = _FakeHandler(headers={"User-Agent": "codex"})
        proxy = mock.Mock()
        handle_models_request(handler, proxy, "/v1/models", declared=_DECLARED)
        proxy.forward.assert_not_called()
        self.assertEqual(handler.status, 200)
        body = handler.body()
        self.assertEqual(body["object"], "list")
        self.assertEqual([m["id"] for m in body["data"]], ["moonshot/kimi-k3", "moonshot/kimi-k2.6"])
        self.assertEqual(body["models"], [{"id": "moonshot/kimi-k3"}, {"id": "moonshot/kimi-k2.6"}])
        self.assertIn(("Content-Type", "application/json"), handler.sent_headers)

    def test_credential_merges_anthropic_passthrough_ahead_of_declared(self):
        handler = _FakeHandler(headers={"x-api-key": "sk-ant-api03-x"})
        proxy = mock.Mock()
        proxy.forward.return_value = _upstream(200, _ANTHROPIC_OK)
        handle_models_request(handler, proxy, "/v1/models", declared=_DECLARED)
        proxy.forward.assert_called_once()
        self.assertEqual(handler.status, 200)
        body = handler.body()
        self.assertEqual([m["id"] for m in body["data"]],
                         ["claude-opus-5", "moonshot/kimi-k3", "moonshot/kimi-k2.6"])
        self.assertIs(body["has_more"], False)  # Anthropic fields preserved
        self.assertEqual(body["models"][0], {"id": "claude-opus-5"})

    def test_bearer_counts_as_a_credential(self):
        handler = _FakeHandler(headers={"Authorization": "Bearer sk-ant-oat01-x"})
        proxy = mock.Mock()
        proxy.forward.return_value = _upstream(200, _ANTHROPIC_OK)
        handle_models_request(handler, proxy, "/v1/models", declared=_DECLARED)
        proxy.forward.assert_called_once()

    def test_upstream_401_still_serves_declared_as_200(self):
        handler = _FakeHandler(headers={"Authorization": "Bearer doorstop"})
        proxy = mock.Mock()
        proxy.forward.return_value = _upstream(401, _ANTHROPIC_401)
        handle_models_request(handler, proxy, "/v1/models", declared=_DECLARED)
        self.assertEqual(handler.status, 200)
        body = handler.body()
        self.assertEqual([m["id"] for m in body["data"]], ["moonshot/kimi-k3", "moonshot/kimi-k2.6"])
        self.assertNotIn("error", body)

    def test_upstream_unreachable_still_serves_declared_as_200(self):
        handler = _FakeHandler(headers={"x-api-key": "sk-ant-api03-x"})
        proxy = mock.Mock()
        proxy.forward.side_effect = GatewayUpstreamError("upstream unreachable")
        handle_models_request(handler, proxy, "/v1/models", declared=_DECLARED)
        self.assertEqual(handler.status, 200)
        self.assertEqual(handler.body()["data"], _DECLARED)

    def test_nothing_declared_keeps_the_verbatim_passthrough(self):
        # Degenerate case (no providers.yaml): exactly today's behaviour --
        # the upstream answer, status included, with the Codex injection.
        handler = _FakeHandler(headers={})
        proxy = mock.Mock()
        proxy.forward.return_value = _upstream(401, _ANTHROPIC_401)
        handle_models_request(handler, proxy, "/v1/models", declared=[])
        proxy.forward.assert_called_once()
        self.assertEqual(handler.status, 401)
        self.assertEqual(handler.body()["type"], "error")

    def test_real_registry_lists_kimi_without_a_credential(self):
        # The plan's acceptance test: /v1/models lists moonshot/kimi-k3 with
        # no Anthropic credential, and Anthropic is never asked.
        handler = _FakeHandler(headers={})
        proxy = mock.Mock()
        handle_models_request(handler, proxy, "/v1/models")
        proxy.forward.assert_not_called()
        self.assertEqual(handler.status, 200)
        self.assertIn("moonshot/kimi-k3", [m["id"] for m in handler.body()["data"]])


if __name__ == "__main__":
    unittest.main()
