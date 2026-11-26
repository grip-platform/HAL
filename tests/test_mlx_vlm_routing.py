"""Wave 5: VLM vision routing tests.

Verifies that CostRouter short-circuits to mlx_vlm for vision queries
before the tier matrix is consulted, and falls back to tier logic for
text-only queries.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock

from lib.hal.cost_router import CostRouter, _get_vlm_model
from lib.hal.types import ProviderInfo, ProviderType


def _make_registry(with_vlm: bool = True) -> MagicMock:
    registry = MagicMock()
    providers = []
    if with_vlm:
        vlm = ProviderInfo(
            name="mlx_vlm",
            type=ProviderType.LOCAL,
            base_url="http://localhost:8080",
            healthy=True,
            models=("mlx-community/Qwen2-VL-7B-Instruct-4bit",),
        )
        providers.append(vlm)
    # Always add a text provider for fallback assertions
    text = ProviderInfo(
        name="groq",
        type=ProviderType.CLOUD,
        base_url="https://api.groq.com",
        healthy=True,
        models=("llama-3.3-70b-versatile",),
    )
    providers.append(text)
    registry.list_providers.return_value = providers
    registry.routing_config = None
    return registry


class TestVlmRouting(unittest.TestCase):
    """Vision queries should route to mlx_vlm when it is available."""

    def test_image_query_routes_to_vlm(self):
        router = CostRouter(_make_registry(with_vlm=True))
        decision = router.select("Describe this image")
        self.assertTrue(decision.model.startswith("mlx_vlm/"))
        self.assertEqual(decision.tier, "free")
        self.assertEqual(decision.estimated_cost_usd, 0.0)

    def test_ocr_query_routes_to_vlm(self):
        router = CostRouter(_make_registry(with_vlm=True))
        decision = router.select("OCR the receipt and extract total")
        self.assertTrue(decision.model.startswith("mlx_vlm/"))
        self.assertEqual(decision.tier, "free")

    def test_screenshot_query_routes_to_vlm(self):
        router = CostRouter(_make_registry(with_vlm=True))
        decision = router.select("What does this screenshot show?")
        self.assertTrue(decision.model.startswith("mlx_vlm/"))

    def test_png_file_routes_to_vlm(self):
        router = CostRouter(_make_registry(with_vlm=True))
        decision = router.select("analyse diagram.png and list the components")
        self.assertTrue(decision.model.startswith("mlx_vlm/"))

    def test_vision_query_without_vlm_falls_through_to_text(self):
        # mlx_vlm not registered — vision query falls through to tier logic
        router = CostRouter(_make_registry(with_vlm=False))
        decision = router.select("Describe this image")
        self.assertFalse(decision.model.startswith("mlx_vlm/"))

    def test_text_query_does_not_route_to_vlm(self):
        router = CostRouter(_make_registry(with_vlm=True))
        decision = router.select("Fix the null pointer in auth.py")
        self.assertFalse(decision.model.startswith("mlx_vlm/"))

    def test_reason_string_identifies_vision_path(self):
        router = CostRouter(_make_registry(with_vlm=True))
        decision = router.select("segment the foreground from this photo")
        self.assertIn("mlx_vlm", decision.reason)

    def test_privacy_mode_still_routes_local_vlm(self):
        # mlx_vlm is local — privacy=True must still allow it
        router = CostRouter(_make_registry(with_vlm=True))
        decision = router.select("OCR this document.png", privacy=True)
        self.assertTrue(decision.model.startswith("mlx_vlm/"))


class TestVlmPrivacy(unittest.TestCase):
    """Privacy flag must block cloud VLM providers."""

    def test_cloud_vlm_blocked_under_privacy(self):
        registry = MagicMock()
        cloud_vlm = ProviderInfo(
            name="mlx_vlm",
            type=ProviderType.CLOUD,  # hypothetical future cloud VLM
            base_url="https://api.vlm.example.com",
            healthy=True,
            models=("cloud-vlm-model",),
        )
        registry.list_providers.return_value = [cloud_vlm]
        registry.routing_config = None
        result = _get_vlm_model(registry, local_only=True)
        self.assertEqual(result, "")


class TestGetVlmModel(unittest.TestCase):
    """Unit tests for _get_vlm_model helper."""

    def test_returns_model_address_when_registered(self):
        registry = _make_registry(with_vlm=True)
        result = _get_vlm_model(registry)
        self.assertEqual(result, "mlx_vlm/mlx-community/Qwen2-VL-7B-Instruct-4bit")

    def test_returns_empty_when_not_registered(self):
        registry = _make_registry(with_vlm=False)
        result = _get_vlm_model(registry)
        self.assertEqual(result, "")

    def test_returns_empty_when_unhealthy(self):
        registry = MagicMock()
        vlm = ProviderInfo(
            name="mlx_vlm", type=ProviderType.LOCAL,
            base_url="http://localhost:8080",
            healthy=False, models=("model",),
        )
        registry.list_providers.return_value = [vlm]
        self.assertEqual(_get_vlm_model(registry), "")

    def test_returns_empty_when_no_models(self):
        registry = MagicMock()
        vlm = ProviderInfo(
            name="mlx_vlm", type=ProviderType.LOCAL,
            base_url="http://localhost:8080",
            healthy=True, models=(),
        )
        registry.list_providers.return_value = [vlm]
        self.assertEqual(_get_vlm_model(registry), "")

    def test_returns_empty_when_registry_is_none(self):
        self.assertEqual(_get_vlm_model(None), "")


if __name__ == "__main__":
    unittest.main()
