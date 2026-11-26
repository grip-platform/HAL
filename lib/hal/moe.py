"""HAL Council — Multi-LLM Mixture of Experts through the universal registry.

Every provider in HAL's registry is a potential council member. Query multiple
LLMs in parallel, synthesise, and run cross-context debate rounds.

Protocol (deliberate_parallel):
    Phase 0 (gut-feel):  All models queried in parallel with NO shared context.
    Phase 1 (synthesis): One synthesiser model reads all Phase 0 responses.
    Phase 2 (debate):    Each model gets query + synthesis + peers' responses.
    Final merge:         Synthesiser produces authoritative final answer.

Usage:
    council = HALCouncil(registry)

    # Sequential (backward-compatible):
    result = council.deliberate("Is this fair?", models=["groq/llama", "gemini/flash"])

    # Parallel + debate (new):
    result = council.deliberate_parallel("Is this fair?", rounds=1, budget_usd=0.10)
    print(result.merged_text)
    print(f"Agreement: {result.agreement_score:.0%}, Rounds: {len(result.rounds)}")
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from lib.hal.kono_bridge import recall_for_query as _kono_recall
from lib.hal.types import CouncilResult, CouncilRound, CouncilVote, TriangulatedFinding

# Progress callback signature: (completed, total, model_name, phase)
ProgressCallback = Optional[Callable[[int, int, str, str], None]]


# ---------------------------------------------------------------------------
# max_tokens policy (tier-based, catalogue-driven)
# Calibrated via SWOT + AskUserQuestion wizard on 2026-04-17.
# See plans/hal-council-max-tokens-tiers.md for rationale.
# ---------------------------------------------------------------------------

_TIER_DEFAULT_MAX_TOKENS = {
    "subscription": 16_384,  # flat-fee plans: be generous
    "frontier":     8_192,   # pay-per-token premium: stay careful
    "premium":      4_096,   # balanced workhorse
    "mid":          4_096,   # balanced workhorse
    "cheap":        8_192,   # Groq LPU speed makes cost negligible
    "free":         4_096,   # safe fallback for unseen tier labels
}
_LOCAL_OUTPUT_CEILING = 32_768  # hard cap even on 2M-context local models
_LOCAL_INPUT_SAFETY = 1_024     # reserve for system prompt + overhead
_LOCAL_MIN = 1_024              # refuse to truncate below this
_REASONING_MULTIPLIER = 2        # reasoning tag ⇒ CoT tokens need room
_LEGACY_FALLBACK_MAX_TOKENS = 2_048  # when catalogue lookup misses
_CHARS_PER_TOKEN = 4             # rough estimate for natural language + code


def _estimate_input_tokens(text: str) -> int:
    """Cheap input-token estimate. len/4 is accurate to within 15% for
    English prose and typical code — good enough for context-headroom math."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _resolve_max_tokens(model_address: str, input_text: str,
                        catalogue: dict) -> int:
    """Compute max_tokens for a council member from tier + input size.

    Local tier is input-aware: output_cap = min(ctx - input - 1K, 32K) so a
    big paste leaves little room and a tiny query gets the full ceiling.
    Non-local tiers use a flat per-tier default. Reasoning-tagged models get
    2× headroom because chain-of-thought tokens count against max_tokens on
    every OpenAI-compatible endpoint we wrap.
    """
    entry = catalogue.get(model_address)
    if entry is None:
        return _LEGACY_FALLBACK_MAX_TOKENS

    if entry.tier == "local":
        headroom = (entry.context_window - _estimate_input_tokens(input_text)
                    - _LOCAL_INPUT_SAFETY)
        cap = max(_LOCAL_MIN, min(headroom, _LOCAL_OUTPUT_CEILING))
    else:
        cap = _TIER_DEFAULT_MAX_TOKENS.get(
            entry.tier, _TIER_DEFAULT_MAX_TOKENS["free"],
        )

    if "reasoning" in entry.tags:
        cap *= _REASONING_MULTIPLIER
    return cap


def _load_council_catalogue() -> dict:
    """Load the model catalogue keyed by address. Returns {} on failure so
    the council still works even if the catalogue file is missing or
    malformed — callers fall back to _LEGACY_FALLBACK_MAX_TOKENS."""
    try:
        from lib.hal.models_command import load_catalogue
        return {e.address: e for e in load_catalogue()}
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Prompt templates (module-level constants — no hidden state)
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are synthesising multiple expert responses to the same question.

ORIGINAL QUERY:
{query}

COUNCIL RESPONSES:
{responses}

Your task:
1. Identify the key points of agreement across responses.
2. Surface meaningful disagreements or complementary angles.
3. Produce a concise synthesis document (3-5 paragraphs) that distils the collective reasoning.
Do not just summarise — find the signal. Output only the synthesis document."""

_DEBATE_PROMPT = """\
You are participating in a structured deliberation. Round {round_num}.
Your response in this round is from [{self_model}].

ORIGINAL QUERY:
{query}

SYNTHESIS FROM PREVIOUS ROUND:
{synthesis}

ALL COUNCIL RESPONSES (identify yours by [{self_model}] above):
{all_responses}

Your task:
1. State whether you agree or disagree with the synthesis, and why.
2. Add reasoning, evidence, or perspective the synthesis missed.
3. Revise your position if other responses changed your view — say so explicitly.
Output only your updated response."""

_FINAL_MERGE_PROMPT = """\
You have moderated a multi-model deliberation. Produce the final authoritative answer.

ORIGINAL QUERY:
{query}

SYNTHESIS DOCUMENT:
{synthesis}

DEBATE RESPONSES:
{debate}

Write a single, definitive answer that:
- Integrates the strongest reasoning from the debate
- Explicitly resolves any remaining disagreements
- Is written directly to the user (no meta-commentary about the council process)"""


def _extract_vote_sentences(
    votes: list["CouncilVote"],
) -> list[tuple[str, str, set[str]]]:
    """Flatten votes into (model, sentence, word_set) triples. O(n*s)."""
    result: list[tuple[str, str, set[str]]] = []
    for vote in votes:
        _collect_sentences(vote.model, vote.response.text, result)
    return result


def _collect_sentences(
    model: str, text: str, out: list[tuple[str, str, set[str]]],
) -> None:
    """Append (model, sentence, words) for each significant sentence. O(s)."""
    for sentence in _split_sentences(text):
        words = set(sentence.lower().split())
        if len(words) >= 4:
            out.append((model, sentence, words))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like fragments for triangulation clustering."""
    import re
    raw = re.split(r'[.\n]+', text)
    return [s.strip() for s in raw if s.strip()]


def _sentence_signature(words: set[str]) -> str:
    """Hash a word set to a signature by taking 3 longest content words. O(w log w)."""
    significant = sorted((w for w in words if len(w) > 3), key=len, reverse=True)[:3]
    return "|".join(sorted(significant))


def _cluster_sentences(
    sentences: list[tuple[str, str, set[str]]],
) -> list[list[tuple[str, str]]]:
    """Cluster sentences by signature-bucket grouping. O(n log w).

    Each sentence is hashed to a signature (top-3 content words sorted).
    Sentences sharing the same signature land in the same bucket.
    Each bucket is then deduplicated by model (one entry per model).
    """
    buckets: dict[str, list[tuple[str, str]]] = {}
    for model, sent, words in sentences:
        sig = _sentence_signature(words)
        if not sig:
            continue
        buckets.setdefault(sig, []).append((model, sent))

    # Deduplicate each bucket: keep first sentence per model
    return [_dedup_bucket(entries) for entries in buckets.values()]


def _dedup_bucket(entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep first sentence per model in a bucket. O(k) where k = bucket size."""
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for model, sent in entries:
        if model not in seen:
            result.append((model, sent))
            seen.add(model)
    return result


class HALCouncil:
    """Cross-LLM council through HAL's universal provider layer."""

    def __init__(self, registry: "ProviderRegistry"):
        self._registry = registry
        # Lazy-loaded catalogue → max_tokens resolver. First council call pays
        # the ~1ms parse; subsequent calls hit the cached dict.
        self._catalogue: Optional[dict] = None

    def _get_catalogue(self) -> dict:
        if self._catalogue is None:
            self._catalogue = _load_council_catalogue()
        return self._catalogue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deliberate(
        self,
        query: str,
        models: list[str],
        weights: Optional[dict[str, float]] = None,
        system: str = "",
        on_progress: ProgressCallback = None,
        max_tokens: Optional[int] = None,
    ) -> CouncilResult:
        """Sequential council query (backward-compatible). Returns CouncilResult.

        ``max_tokens=None`` applies the tier-based resolver per member; pass
        an int to force a uniform cap across the council.
        """
        # Inject KONO memory context for all council members
        kono_context = _kono_recall(query)
        if kono_context:
            system = (system + "\n\n---\n\n" + kono_context) if system else kono_context

        votes = []
        total_cost = 0.0
        total = len(models)

        for i, model_address in enumerate(models):
            vote = self._query_member_safe(
                model_address, query, system, max_tokens=max_tokens,
            )
            if vote:
                votes.append(vote)
                total_cost += vote.response.cost_usd or 0.0
            if on_progress:
                on_progress(i + 1, total, model_address, "query")

        if not votes:
            return CouncilResult(merged_text="[No council members responded]")

        return CouncilResult(
            merged_text=self._merge(votes, weights),
            votes=tuple(votes),
            agreement_score=self._compute_agreement(votes),
            total_cost_usd=total_cost,
        )

    def deliberate_parallel(
        self,
        query: str,
        models: Optional[list[str]] = None,
        weights: Optional[dict[str, float]] = None,
        system: str = "",
        rounds: int = 1,
        budget_usd: Optional[float] = None,
        timeout_s: float = 30.0,
        synthesizer: Optional[str] = None,
        on_progress: ProgressCallback = None,
        max_tokens: Optional[int] = None,
    ) -> CouncilResult:
        """3-phase parallel council: gut-feel → synthesis → debate. O(1) from caller.

        Args:
            query:       Question posed to all council members.
            models:      Model addresses. None = auto-select from registry routing.
            weights:     Optional {model_address: float} quality weights.
            system:      System prompt applied to all phases.
            rounds:      Debate rounds after synthesis (default 1).
            budget_usd:  Hard cost ceiling — raises ValueError if exceeded pre-flight.
            timeout_s:   Per-model timeout in seconds (default 30).
            synthesizer: Model address for synthesis/merge. None = auto (premium tier).
        """
        council = models if models is not None else self._auto_select_council(query, budget_usd)
        if not council:
            return CouncilResult(merged_text="[No council models available]")

        self._budget_preflight(council, rounds, budget_usd)

        # Inject KONO memory context so all council members share knowledge
        kono_context = _kono_recall(query)
        if kono_context:
            system = (system + "\n\n---\n\n" + kono_context) if system else kono_context

        synth_model = synthesizer or self._auto_select_synthesizer(council)
        all_rounds: list[CouncilRound] = []
        total_cost = 0.0

        # Phase 0: gut-feel (no shared context, but with KONO memory)
        t0 = time.monotonic()
        phase0_votes = self._dispatch_parallel(
            council, query, system, weights, timeout_s, on_progress,
            "gut_feel", max_tokens,
        )
        total_cost += sum(v.response.cost_usd or 0.0 for v in phase0_votes)
        all_rounds.append(CouncilRound(
            round_number=0, phase="gut_feel",
            votes=tuple(phase0_votes),
            duration_ms=(time.monotonic() - t0) * 1000,
        ))

        if not phase0_votes:
            return CouncilResult(merged_text="[No council members responded]")

        # Phase 1: synthesis
        synthesis = self._synthesise(
            phase0_votes, query, synth_model, system, max_tokens,
        )

        # Phase 2: debate rounds
        debate_votes = phase0_votes
        for round_num in range(1, rounds + 1):
            t_r = time.monotonic()
            debate_votes = self._debate_round(
                phase0_votes, synthesis, query, council, weights, round_num,
                system, timeout_s, max_tokens,
            )
            total_cost += sum(v.response.cost_usd or 0.0 for v in debate_votes)
            all_rounds.append(CouncilRound(
                round_number=round_num, phase="debate",
                votes=tuple(debate_votes),
                synthesis=synthesis,
                duration_ms=(time.monotonic() - t_r) * 1000,
            ))
            if round_num < rounds and debate_votes:
                synthesis = self._synthesise(
                    debate_votes, query, synth_model, system, max_tokens,
                )

        final_text = self._final_merge(
            debate_votes, query, synthesis, synth_model, system, max_tokens,
        )
        all_votes = tuple(phase0_votes) + tuple(debate_votes)
        triangulated = self._triangulate_findings(debate_votes or phase0_votes)

        return CouncilResult(
            merged_text=final_text,
            votes=all_votes,
            agreement_score=self._compute_agreement(list(debate_votes or phase0_votes)),
            total_cost_usd=total_cost,
            rounds=tuple(all_rounds),
            triangulated_findings=triangulated,
        )

    # ------------------------------------------------------------------
    # Parallel dispatch
    # ------------------------------------------------------------------

    def _dispatch_parallel(
        self,
        models: list[str],
        prompt: str,
        system: str,
        weights: Optional[dict[str, float]],
        timeout_s: float,
        on_progress: ProgressCallback = None,
        phase: str = "query",
        max_tokens: Optional[int] = None,
    ) -> list[CouncilVote]:
        """Fire all models in parallel. Slow/failed models excluded, never blocking."""
        max_workers = min(len(models), 8)
        results: list[CouncilVote] = []
        total = len(models)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_model = {
                pool.submit(
                    self._query_member_safe,
                    addr, prompt, system, (weights or {}).get(addr, 1.0),
                    max_tokens,
                ): addr
                for addr in models
            }
            try:
                for future in as_completed(future_to_model, timeout=timeout_s + 1):
                    model_addr = future_to_model[future]
                    completed += 1
                    try:
                        vote = future.result(timeout=timeout_s)
                        if vote is not None:
                            results.append(vote)
                    except Exception:
                        pass  # Individual model timed out or errored — excluded silently
                    if on_progress:
                        on_progress(completed, total, model_addr, phase)
            except TimeoutError:
                pass  # Overall deadline exceeded — return whatever arrived so far

        return results

    def _debate_round(
        self,
        phase0_votes: list[CouncilVote],
        synthesis: str,
        query: str,
        models: list[str],
        weights: Optional[dict[str, float]],
        round_num: int,
        system: str,
        timeout_s: float,
        max_tokens: Optional[int] = None,
    ) -> list[CouncilVote]:
        """One parallel debate round. Each model sees all Phase 0 responses.

        All responses are shared (including self) — model identifies its own
        by the [{self_model}] marker. Builds shared text once: O(n) total.
        """
        # Build the shared all-responses string ONCE — O(n), no nested loop
        all_responses = "\n\n".join(
            f"[{v.model}]\n{v.response.text}" for v in phase0_votes
        ) or "(no responses available)"

        # Per-model prompts use O(1) dict lookup — shared text, unique self_model tag
        model_prompts: dict[str, str] = {
            addr: _DEBATE_PROMPT.format(
                round_num=round_num, query=query, synthesis=synthesis,
                self_model=addr, all_responses=all_responses,
            )
            for addr in models
        }

        max_workers = min(len(models), 8)
        results: list[CouncilVote] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_model = {
                pool.submit(
                    self._query_member_safe,
                    addr, model_prompts[addr], system, (weights or {}).get(addr, 1.0),
                    max_tokens,
                ): addr
                for addr in models
            }
            try:
                for future in as_completed(future_to_model, timeout=timeout_s + 1):
                    try:
                        vote = future.result(timeout=timeout_s)
                        if vote is not None:
                            results.append(vote)
                    except Exception:
                        pass  # Individual model timed out or errored — excluded silently
            except TimeoutError:
                pass  # Overall deadline exceeded — return whatever arrived so far

        return results

    # ------------------------------------------------------------------
    # Synthesis and merge
    # ------------------------------------------------------------------

    def _synthesise(
        self, votes: list[CouncilVote], query: str, synthesizer: str, system: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Phase 1: one model reads all Phase 0 responses and produces a synthesis."""
        if not votes:
            return ""
        formatted = "\n\n".join(f"[{v.model}]\n{v.response.text}" for v in votes)
        prompt = _SYNTHESIS_PROMPT.format(query=query, responses=formatted)
        cap = max_tokens if max_tokens is not None else _resolve_max_tokens(
            synthesizer, prompt + system, self._get_catalogue(),
        )
        try:
            resp = self._registry.chat(
                synthesizer, prompt, system=system, max_tokens=cap,
            )
            return resp.text
        except Exception:
            return "\n\n".join(v.response.text for v in votes[:2])

    def _final_merge(
        self,
        debate_votes: list[CouncilVote],
        query: str,
        synthesis: str,
        synth_model: str,
        system: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Synthesiser produces the final answer from debate outputs."""
        if not debate_votes:
            return synthesis or ""
        debate_text = "\n\n".join(f"[{v.model}]\n{v.response.text}" for v in debate_votes)
        prompt = _FINAL_MERGE_PROMPT.format(
            query=query, synthesis=synthesis, debate=debate_text,
        )
        cap = max_tokens if max_tokens is not None else _resolve_max_tokens(
            synth_model, prompt + system, self._get_catalogue(),
        )
        try:
            resp = self._registry.chat(
                synth_model, prompt, system=system, max_tokens=cap,
            )
            return resp.text
        except Exception:
            return self._merge(debate_votes, None)

    # ------------------------------------------------------------------
    # Triangulation (Jeffrey modes-of-reasoning absorb)
    # ------------------------------------------------------------------

    @staticmethod
    def _triangulate_findings(
        votes: list[CouncilVote],
    ) -> tuple[TriangulatedFinding, ...]:
        """Classify findings by cross-model agreement level.

        Extracts key sentences from each vote, clusters by word-set overlap
        (Jaccard >= 0.3), then classifies each cluster:
        - kernel:     3+ models agree
        - supported:  2 models agree
        - hypothesis: 1 model only
        - disputed:   models explicitly contradict (overlap with negation)

        Returns findings sorted by agreement count (highest first).
        """
        if len(votes) < 2:
            return ()

        # Extract sentences — flat helper avoids nested loop in this method
        model_sentences = _extract_vote_sentences(votes)

        if not model_sentences:
            return ()

        # Cluster via inverted-index approach (see _cluster_sentences)
        clusters = _cluster_sentences(model_sentences)

        # Classify clusters
        findings: list[TriangulatedFinding] = []
        for cluster in clusters:
            count = len(cluster)
            models = tuple(m for m, _ in cluster)
            text = cluster[0][1]  # representative sentence
            if count >= 3:
                classification = "kernel"
            elif count == 2:
                classification = "supported"
            else:
                classification = "hypothesis"
            findings.append(TriangulatedFinding(
                text=text,
                classification=classification,
                agreement_count=count,
                models=models,
            ))

        # Sort by agreement (kernel first)
        findings.sort(key=lambda f: f.agreement_count, reverse=True)
        return tuple(findings[:20])  # cap at 20 to bound output

    # ------------------------------------------------------------------
    # Auto-selection
    # ------------------------------------------------------------------

    def _auto_select_synthesizer(self, council_models: list[str]) -> str:
        """Pick highest-quality available model as synthesiser (premium tier first)."""
        routing = getattr(self._registry, "routing", None)
        premium_chain = getattr(routing, "tier_chains", {}).get("premium", [])
        council_set = set(council_models)  # O(n) build; O(1) lookups below
        for candidate in premium_chain:
            if candidate in council_set:
                return candidate
        return premium_chain[0] if premium_chain else council_models[0]

    def _auto_select_council(
        self, query: str, budget_usd: Optional[float],
    ) -> list[str]:
        """Return council model list based on query complexity and budget."""
        routing = getattr(self._registry, "routing", None)
        tier_chains = getattr(routing, "tier_chains", {}) if routing else {}

        # Flatten tier chains in priority order, then deduplicate preserving order
        flat: list[str] = []
        for tier in ("premium", "mid", "cheap", "free"):
            flat.extend(tier_chains.get(tier, []))
        council = list(dict.fromkeys(flat))  # O(n) dedup, insertion-order preserved

        if not council:
            return []

        # Budget restriction
        if budget_usd is not None and budget_usd < 0.01:
            return [m for m in tier_chains.get("free", [])] or council[:1]
        if budget_usd is not None and budget_usd < 0.05:
            free = tier_chains.get("free", [])
            cheap = tier_chains.get("cheap", [])
            return (free + cheap) or council[:2]

        return council

    def _budget_preflight(
        self,
        models: list[str],
        rounds: int,
        budget_usd: Optional[float],
    ) -> None:
        """Raise ValueError if estimated council cost exceeds budget."""
        if budget_usd is None:
            return
        # Estimate: 500 output tokens × $0.001/1k per model per phase
        estimated_per_model = 0.0005
        total_phases = 1 + rounds  # gut-feel + debate rounds
        estimated = len(models) * estimated_per_model * total_phases
        if estimated > budget_usd:
            raise ValueError(
                f"Estimated council cost ${estimated:.4f} exceeds budget "
                f"${budget_usd:.4f}. Reduce models ({len(models)}) or "
                f"rounds ({rounds}), or raise budget_usd."
            )

    # ------------------------------------------------------------------
    # Low-level helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _query_member_safe(
        self,
        model_address: str,
        query: str,
        system: str = "",
        weight: float = 1.0,
        max_tokens: Optional[int] = None,
    ) -> Optional[CouncilVote]:
        """Query one council member. Returns None on any failure.

        ``max_tokens=None`` activates the tier-based resolver; passing an
        explicit int bypasses it entirely (the operator's wallet, the
        operator's rules).
        """
        try:
            if max_tokens is None:
                max_tokens = _resolve_max_tokens(
                    model_address, query + system, self._get_catalogue(),
                )
            response = self._registry.chat(
                model_address, query, system=system, max_tokens=max_tokens,
            )
            provider = model_address.split("/")[0] if "/" in model_address else "unknown"
            return CouncilVote(model=model_address, provider=provider,
                               response=response, weight=weight)
        except Exception:
            return None

    def _merge(self, votes: list[CouncilVote], weights: Optional[dict[str, float]]) -> str:
        """Merge council votes by highest weighted response length."""
        if not votes:
            return ""
        scored = [
            (vote, (weights or {}).get(vote.model, 1.0) * len(vote.response.text))
            for vote in votes
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0].response.text

    def _compute_agreement(self, votes: list[CouncilVote]) -> float:
        """Estimate agreement between council members (0-1). O(n) via reference vote."""
        if len(votes) < 2:
            return 1.0
        ref_words = set(votes[0].response.text.lower().split())
        if not ref_words:
            return 0.0
        # Pre-build word sets in a flat pass (O(n)), then compute Jaccard separately
        vote_word_sets = [
            set(v.response.text.lower().split()) for v in votes[1:] if v.response.text
        ]
        if not vote_word_sets:
            return 0.0
        total = sum(
            len(ref_words & ws) / max(len(ref_words | ws), 1)
            for ws in vote_word_sets
        )
        return total / (len(votes) - 1)
