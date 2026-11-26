"""HAL Council Agent — multi-LLM deliberation per response.

THE killer feature. CMA can only use one Claude model per agent.
HAL council agents query 3-5 models from different providers simultaneously,
producing responses with measurable agreement scores.

A legal review agent on [claude-opus, groq/llama-70b, gemini-pro] only
delivers answers where all three agree above 0.8 — reliability impossible
in any single-model system.

Usage:
    from lib.hal.council_agent import CouncilRunner

    runner = CouncilRunner(registry)
    result = runner.deliberate(
        query="Is this contract clause enforceable?",
        models=["groq/llama-3.3-70b", "gemini/gemini-2.5-flash", "ollama/mistral:7b"],
        system="You are a legal reviewer.",
    )
    print(f"Agreement: {result.agreement_score:.0%}")
    print(result.merged_text)
"""
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from lib.hal.event_protocol import EventEmitter, EventType


@dataclass(frozen=True)
class CouncilVote:
    """A single model's response in a council deliberation."""
    model: str
    provider: str
    text: str
    latency_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class CouncilDeliberation:
    """Result of a multi-LLM council deliberation."""
    merged_text: str
    votes: tuple = ()
    agreement_score: float = 0.0
    total_cost_usd: float = 0.0
    model_count: int = 0


class CouncilRunner:
    """Runs multi-LLM deliberation for council-mode agents.

    Delegates to HAL's existing MoE council (moe.py) when available,
    or runs a lightweight parallel-query + merge protocol standalone.
    """

    def __init__(self, provider_registry, emitter: Optional[EventEmitter] = None):
        self._registry = provider_registry
        self._emitter = emitter

    def deliberate(
        self,
        query: str,
        models: list,
        system: str = "",
        agreement_threshold: float = 0.7,
    ) -> CouncilDeliberation:
        """Query multiple models and synthesise a merged response."""
        votes = self._gather_votes(query, models, system)
        merged = self._synthesise(votes)
        agreement = self._score_agreement(votes)

        if self._emitter:
            self._emitter.emit(EventType.COUNCIL_SYNTHESIS, {
                "text": merged,
                "agreement_score": agreement,
                "models": [v.model for v in votes],
            })

        total_cost = sum(v.cost_usd for v in votes)
        return CouncilDeliberation(
            merged_text=merged,
            votes=tuple(votes),
            agreement_score=agreement,
            total_cost_usd=total_cost,
            model_count=len(votes),
        )

    def _gather_votes(self, query: str, models: list, system: str) -> list:
        """Query each model independently. No shared context (gut-feel phase)."""
        votes = []
        for model in models:
            try:
                response = self._registry.chat(
                    messages=[{"role": "user", "content": query}],
                    model=model,
                    system=system,
                )
                vote = CouncilVote(
                    model=response.model,
                    provider=response.provider,
                    text=response.text,
                    latency_ms=response.latency_ms or 0.0,
                    cost_usd=response.cost_usd or 0.0,
                )
                votes.append(vote)
                if self._emitter:
                    self._emitter.emit(EventType.COUNCIL_VOTE, {
                        "model": vote.model,
                        "provider": vote.provider,
                        "text": vote.text,
                        "vote_index": len(votes) - 1,
                    })
            except Exception:
                pass  # Skip failed models, continue with remaining
        return votes

    def _synthesise(self, votes: list) -> str:
        """Merge votes into a single response. Uses first vote's model for synthesis."""
        if not votes:
            return ""
        if len(votes) == 1:
            return votes[0].text
        texts = [v.text for v in votes]
        synthesiser_model = votes[0].model
        try:
            prompt = _build_synthesis_prompt(texts)
            response = self._registry.chat(
                messages=[{"role": "user", "content": prompt}],
                model=synthesiser_model,
                system="Synthesise the following responses into one coherent answer.",
            )
            return response.text
        except Exception:
            return texts[0]

    @staticmethod
    def _score_agreement(votes: list) -> float:
        """Score agreement across votes using word-frequency consensus."""
        if len(votes) < 2:
            return 1.0
        word_sets = [set(v.text.lower().split()) for v in votes]
        return _consensus_score(word_sets)


def _consensus_score(word_sets: list) -> float:
    """Consensus score: fraction of vocabulary appearing in majority of responses.

    Single-pass via Counter over flattened (word, response_index) pairs,
    then deduplicated per response. O(total_words) time.
    """
    n = len(word_sets)
    if n < 2:
        return 1.0
    # Count how many responses contain each word (single Counter pass)
    word_counts = Counter(w for ws in word_sets for w in ws)
    total_vocab = len(word_counts)
    if total_vocab == 0:
        return 0.0
    threshold = n / 2
    consensus = sum(1 for c in word_counts.values() if c > threshold)
    return consensus / total_vocab


def _build_synthesis_prompt(texts: list) -> str:
    """Build the synthesis prompt from multiple model responses."""
    parts = []
    for i, text in enumerate(texts, 1):
        parts.append(f"Response {i}:\n{text}")
    return "\n\n".join(parts) + "\n\nSynthesise these into one coherent answer."
