"""Tests: CostRouter wired into Conversation.send().

All three tests are falsifiable — they fail if the routing logic is broken.
Uses importlib.util to load modules directly from the worktree, bypassing
the sys.modules cache entirely.
"""
import importlib.util
import pathlib
import unittest
from unittest.mock import MagicMock, patch

_REPO = pathlib.Path(__file__).parent.parent


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cost = _load("lib/hal/cost_router.py", "_wt_cost_router")
_conv = _load("lib/hal/conversation.py", "_wt_conversation")

RouteDecision = _cost.RouteDecision
Conversation = _conv.Conversation


def _make_registry(model_addr: str = "groq/llama-3.3-70b-versatile") -> MagicMock:
    provider = MagicMock()
    provider.name = model_addr.split("/")[0]
    provider.models = [model_addr.split("/", 1)[1]]
    provider.type.value = "cloud"

    routing = MagicMock()
    routing.default_model = model_addr

    registry = MagicMock()
    registry.list_providers.return_value = [provider]
    registry.routing = routing
    return registry


def _make_dispatch_response(model: str = "groq/llama-3.3-70b-versatile") -> MagicMock:
    response = MagicMock()
    response.text = "ok"
    response.model = model
    response.provider = model.split("/")[0]
    response.usage = MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)
    return response


class TestCostRouterUsedWhenNoModelSet(unittest.TestCase):
    """CostRouter.select() is called when user has not explicitly set a model."""

    def test_cost_router_called_on_send(self):
        registry = _make_registry()
        routed_model = "qwen/qwen3.5-flash"
        route = RouteDecision(
            model=routed_model, tier="cheap",
            reason="simple query -> cheap tier", estimated_cost_usd=0.0,
        )

        with patch.object(_conv, "get_registry", return_value=registry), \
             patch.object(_conv, "LimitSentinel") as mock_sentinel_cls, \
             patch.object(_conv, "SessionManager"), \
             patch.object(_conv, "SessionRecorder"), \
             patch.object(_conv, "CostRouter") as mock_router_cls, \
             patch.object(_conv, "IntentDetector", return_value=MagicMock()):

            mock_sentinel_cls.return_value.dispatch.return_value = _make_dispatch_response(routed_model)
            mock_router_cls.return_value.select.return_value = route

            conv = Conversation(registry=registry)
            conv.send("Hello")

            mock_router_cls.assert_called_once_with(registry)
            mock_router_cls.return_value.select.assert_called_once_with("Hello")
            call_args = mock_sentinel_cls.return_value.dispatch.call_args
            self.assertEqual(call_args.args[0], routed_model)


class TestStaticModelHonouredWhenUserSet(unittest.TestCase):
    """CostRouter is NOT called when user explicitly set a model via configure()."""

    def test_no_cost_router_when_model_explicit(self):
        registry = _make_registry()
        explicit_model = "anthropic/claude-sonnet-4-6"

        with patch.object(_conv, "get_registry", return_value=registry), \
             patch.object(_conv, "LimitSentinel") as mock_sentinel_cls, \
             patch.object(_conv, "SessionManager"), \
             patch.object(_conv, "SessionRecorder"), \
             patch.object(_conv, "CostRouter") as mock_router_cls, \
             patch.object(_conv, "IntentDetector", return_value=MagicMock()):

            mock_sentinel_cls.return_value.dispatch.return_value = _make_dispatch_response(explicit_model)

            conv = Conversation(registry=registry)
            conv.configure(model=explicit_model)
            conv.send("Hello")

            mock_router_cls.return_value.select.assert_not_called()
            call_args = mock_sentinel_cls.return_value.dispatch.call_args
            self.assertEqual(call_args.args[0], explicit_model)


class TestCostRouterFallbackOnEmptyResult(unittest.TestCase):
    """Falls back to self._model when CostRouter returns an empty model string."""

    def test_empty_route_uses_default(self):
        default_model = "groq/llama-3.3-70b-versatile"
        registry = _make_registry(default_model)
        empty_route = RouteDecision(
            model="", tier="none",
            reason="No providers available", estimated_cost_usd=0.0,
        )

        with patch.object(_conv, "get_registry", return_value=registry), \
             patch.object(_conv, "LimitSentinel") as mock_sentinel_cls, \
             patch.object(_conv, "SessionManager"), \
             patch.object(_conv, "SessionRecorder"), \
             patch.object(_conv, "CostRouter") as mock_router_cls, \
             patch.object(_conv, "IntentDetector", return_value=MagicMock()):

            mock_sentinel_cls.return_value.dispatch.return_value = _make_dispatch_response(default_model)
            mock_router_cls.return_value.select.return_value = empty_route

            conv = Conversation(registry=registry)
            conv.send("Hello")

            call_args = mock_sentinel_cls.return_value.dispatch.call_args
            self.assertEqual(call_args.args[0], default_model)


if __name__ == "__main__":
    unittest.main()
