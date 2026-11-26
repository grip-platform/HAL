"""HAL Input Attribution — explain which input sections influenced the response.

Determines which parts of a user's prompt most influenced the LLM's output.
Two methods: LLM-based importance rating (cheap, single call) and ablation
(accurate, multiple calls — one per section).

Adapted from NForge's modality attribution concept: instead of measuring
text/audio/video contribution to brain vertices, HAL measures instruction/
context/code contribution to response content.

Usage:
    from lib.hal.attribution import InputAttributor

    attributor = InputAttributor(registry)
    scores = attributor.attribute_sections(
        sections={"instructions": "...", "context": "...", "code": "..."},
        response="The LLM's response text",
    )
    # scores = {"instructions": 0.15, "context": 0.60, "code": 0.25}
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AttributionScore:
    """Attribution score for a single input section."""
    section: str
    importance: float  # 0.0 to 1.0
    rationale: str = ""


@dataclass(frozen=True)
class AttributionReport:
    """Full attribution report for a prompt-response pair."""
    scores: tuple[AttributionScore, ...]
    method: str  # "rating" or "ablation"
    total_sections: int

    @property
    def dominant_section(self) -> str:
        """The section with highest attribution."""
        if not self.scores:
            return ""
        return max(self.scores, key=lambda s: s.importance).section

    @property
    def summary(self) -> str:
        parts = [f"{s.section}: {s.importance:.0%}" for s in sorted(self.scores, key=lambda x: -x.importance)]
        return f"Attribution ({self.method}): {', '.join(parts)}"


class InputAttributor:
    """Determines which input sections most influenced the LLM response."""

    def __init__(self, registry=None):
        self._registry = registry

    def attribute_sections(
        self,
        sections: dict[str, str],
        response: str,
        method: str = "rating",
        model: str = "",
    ) -> AttributionReport:
        """Attribute response to input sections.

        Args:
            sections: Named input sections {name: content}
            response: The LLM's response text
            method: "rating" (single LLM call) or "ablation" (N+1 calls)
            model: Model to use for analysis (defaults to first available)
        """
        if not sections:
            return AttributionReport(scores=(), method=method, total_sections=0)

        dispatch = {"rating": self._rate, "ablation": self._ablate}
        handler = dispatch.get(method, self._rate)
        scores = handler(sections, response, model)
        return AttributionReport(scores=tuple(scores), method=method, total_sections=len(sections))

    def _rate(self, sections: dict[str, str], response: str, model: str) -> list[AttributionScore]:
        """Single-call LLM rating of section importance."""
        if not self._registry:
            return _equal_attribution(sections)

        target = model or _first_model(self._registry)
        if not target:
            return _equal_attribution(sections)

        section_desc = "\n".join(
            f"[{name}] ({len(content)} chars): {content[:200]}..."
            for name, content in sections.items()
        )

        prompt = (
            f"Rate how much each input section influenced this response.\n\n"
            f"INPUT SECTIONS:\n{section_desc}\n\n"
            f"RESPONSE:\n{response[:500]}\n\n"
            f"Output ONLY a JSON object mapping section names to importance (0.0-1.0), "
            f"values must sum to 1.0. Example: {{\"instructions\": 0.3, \"code\": 0.7}}"
        )

        try:
            resp = self._registry.chat(target, prompt, max_tokens=200)
            return _parse_scores(resp.text, sections)
        except Exception:
            return _equal_attribution(sections)

    def _ablate(self, sections: dict[str, str], response: str, model: str) -> list[AttributionScore]:
        """Ablation: remove each section, measure response divergence."""
        if not self._registry:
            return _equal_attribution(sections)

        target = model or _first_model(self._registry)
        if not target:
            return _equal_attribution(sections)

        full_prompt = "\n\n".join(f"[{k}]\n{v}" for k, v in sections.items())

        # Baseline: response with all sections
        baseline_words = set(response.lower().split())
        divergences: dict[str, float] = {}

        for removed_name in sections:
            # Build prompt without this section
            ablated = "\n\n".join(f"[{k}]\n{v}" for k, v in sections.items() if k != removed_name)
            try:
                ablated_resp = self._registry.chat(target, ablated, max_tokens=500)
                ablated_words = set(ablated_resp.text.lower().split())
                # Jaccard distance: how different is the response without this section
                union = baseline_words | ablated_words
                intersection = baseline_words & ablated_words
                divergences[removed_name] = 1.0 - (len(intersection) / len(union)) if union else 0.0
            except Exception:
                divergences[removed_name] = 0.0

        # Normalise divergences to sum to 1.0
        total = sum(divergences.values())
        return [
            AttributionScore(section=name, importance=div / total if total else 1.0 / len(sections))
            for name, div in divergences.items()
        ]


def _equal_attribution(sections: dict[str, str]) -> list[AttributionScore]:
    """Fallback: equal attribution across all sections."""
    equal = 1.0 / len(sections) if sections else 0.0
    return [AttributionScore(section=name, importance=equal) for name in sections]


def _first_model(registry) -> str:
    """Get first available model from registry."""
    for p in registry.list_providers():
        if p.models:
            return f"{p.name}/{p.models[0]}"
    return ""


def _parse_scores(text: str, sections: dict[str, str]) -> list[AttributionScore]:
    """Parse LLM JSON response into attribution scores."""
    import json
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return _equal_attribution(sections)
    try:
        data = json.loads(text[start:end + 1])
        if not isinstance(data, dict):
            return _equal_attribution(sections)
        total = sum(float(v) for v in data.values())
        return [
            AttributionScore(
                section=name,
                importance=float(data.get(name, 0)) / total if total else 1.0 / len(sections),
            )
            for name in sections
        ]
    except (json.JSONDecodeError, ValueError, TypeError):
        return _equal_attribution(sections)
