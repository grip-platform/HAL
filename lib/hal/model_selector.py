"""HAL Model Selector — automatic model selection based on task.

Combines intent detection, cost routing, and model catalog to
automatically pick the optimal model for any query.

Usage:
    from lib.hal.model_selector import ModelSelector
    selector = ModelSelector(registry)
    model = selector.select("Fix the auth bug in server.py")
"""
from dataclasses import dataclass
from lib.hal.intent_detector import IntentDetector
from lib.hal.cost_router import CostRouter


@dataclass(frozen=True)
class Selection:
    model: str
    reason: str
    intent: str
    tier: str


class ModelSelector:
    """Automatic model selection combining intent + cost routing."""

    def __init__(self, registry=None):
        self._detector = IntentDetector()
        self._router = CostRouter(registry)

    def select(self, query: str, privacy: bool = False) -> Selection:
        intent = self._detector.classify(query)
        route = self._router.select(query, privacy=privacy)
        return Selection(
            model=route.model,
            reason=f"{intent.category} query -> {route.tier} tier",
            intent=intent.category,
            tier=route.tier,
        )
