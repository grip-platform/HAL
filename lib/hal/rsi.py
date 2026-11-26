"""HAL RSI Engine — Recursive Self-Improvement through cross-model analysis.

After each conversation, a DIFFERENT model analyses the interaction to detect
patterns, propose improvements, and feed the learning loop. Uses the YSH
principle: when a pattern is seen 3+ times, abstract it.

Usage:
    from lib.hal.rsi import RSIEngine

    engine = RSIEngine(registry)
    proposals = engine.analyse(conversation_history)
    for p in proposals:
        print(f"{p.type}: {p.description} (confidence={p.confidence:.0%})")
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass

from lib.hal.home import HAL_HOME as _HAL_HOME
from lib.hal.types import ChatMessage

_PATTERNS_PATH = os.path.join(_HAL_HOME, "state", "hal", "rsi-patterns.json")
_YSH_THRESHOLD = 3  # Abstract after 3 occurrences


@dataclass(frozen=True)
class Pattern:
    """A detected behavioural pattern from conversation analysis."""
    id: str
    description: str
    category: str  # "error_recovery", "tool_usage", "query_style", "domain_knowledge"
    occurrences: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class Proposal:
    """A proposed improvement based on detected patterns."""
    type: str  # "skill", "mode", "hook", "config", "memory"
    description: str
    pattern_id: str
    confidence: float = 0.0
    action: str = ""  # Concrete action to take
    auto_apply: bool = False  # Safe to apply without human review


class RSIEngine:
    """Post-conversation analysis and self-improvement engine."""

    def __init__(self, registry: "ProviderRegistry"):
        self._registry = registry
        self._patterns: dict[str, dict] = {}
        self._load_patterns()

    def analyse(
        self,
        history: list[ChatMessage],
        analysis_model: str = "",
    ) -> list[Proposal]:
        """Analyse a conversation and propose improvements.

        Uses a DIFFERENT model than the conversation to avoid self-bias.
        """
        if len(history) < 4:  # Need at least 2 turns
            return []

        # Pick analysis model (different from conversation model)
        model = analysis_model or self._pick_analysis_model()
        if not model:
            return []

        # Extract patterns from the conversation
        new_patterns = self._detect_patterns(history, model)

        # Update pattern store
        for pattern in new_patterns:
            self._record_pattern(pattern)

        # Generate proposals for patterns at YSH threshold
        proposals = self._generate_proposals()

        # Persist
        self._save_patterns()

        return proposals

    def analyse_for_build(
        self,
        history: list[ChatMessage],
        analysis_model: str = "",
        min_confidence: float = 0.7,
    ) -> list[Proposal]:
        """Analyse and return only proposals suitable for code generation.

        Filters to types that can be auto-built (skill, hook) and above
        the confidence threshold.
        """
        proposals = self.analyse(history, analysis_model)
        buildable_types = {"skill", "hook"}
        return [
            p for p in proposals
            if p.type in buildable_types and p.confidence >= min_confidence
        ]

    def get_patterns(self) -> list[Pattern]:
        """Get all recorded patterns."""
        return [
            Pattern(
                id=pid,
                description=p["description"],
                category=p["category"],
                occurrences=p["occurrences"],
                first_seen=p["first_seen"],
                last_seen=p["last_seen"],
                examples=tuple(p.get("examples", [])[-3:]),
            )
            for pid, p in self._patterns.items()
        ]

    def _pick_analysis_model(self) -> str:
        """Pick a model for analysis — prefer fast/cheap cloud models."""
        # Use a different model than the default for cross-model analysis
        candidates = [
            "groq/llama-3.3-70b-versatile",
            "gemini/gemini-3.5-flash-lite",
            "groq/openai/gpt-oss-120b",
        ]
        providers_by_name = {
            p.name: p for p in self._registry.list_providers()
        }
        for candidate in candidates:
            provider_name = candidate.split("/")[0]
            p = providers_by_name.get(provider_name)
            if p and p.healthy:
                return candidate
        return ""

    def _detect_patterns(
        self, history: list[ChatMessage], model: str,
    ) -> list[dict]:
        """Use an LLM to detect patterns in conversation history."""
        # Build analysis prompt
        turns = []
        for msg in history[-20:]:  # Last 10 turns max
            turns.append(f"{msg.role}: {msg.content[:500]}")

        prompt = (
            "Analyse this conversation and identify recurring patterns. "
            "For each pattern, output a JSON object with: "
            "description (what the pattern is), "
            "category (one of: error_recovery, tool_usage, query_style, "
            "domain_knowledge, workflow, preference), "
            "example (a brief example from the conversation). "
            "Output ONLY a JSON array of pattern objects. No other text.\n\n"
            "Conversation:\n" + "\n".join(turns)
        )

        try:
            response = self._registry.chat(model, prompt, max_tokens=1000)
            return _parse_patterns(response.text)
        except Exception:
            return []

    def _record_pattern(self, pattern: dict) -> None:
        """Record or update a pattern in the store."""
        pid = _pattern_id(pattern["description"])
        now = time.time()

        if pid in self._patterns:
            self._patterns[pid]["occurrences"] += 1
            self._patterns[pid]["last_seen"] = now
            examples = self._patterns[pid].get("examples", [])
            if pattern.get("example"):
                examples.append(pattern["example"])
                self._patterns[pid]["examples"] = examples[-5:]  # Keep last 5
        else:
            self._patterns[pid] = {
                "description": pattern["description"],
                "category": pattern.get("category", "unknown"),
                "occurrences": 1,
                "first_seen": now,
                "last_seen": now,
                "examples": [pattern["example"]] if pattern.get("example") else [],
            }

    def _generate_proposals(self) -> list[Proposal]:
        """Generate improvement proposals for patterns at YSH threshold."""
        proposals: list[Proposal] = []
        for pid, p in self._patterns.items():
            if p["occurrences"] < _YSH_THRESHOLD:
                continue

            proposal_type, action = _propose_action(p)
            proposals.append(Proposal(
                type=proposal_type,
                description=f"Pattern seen {p['occurrences']}x: {p['description']}",
                pattern_id=pid,
                confidence=min(0.95, 0.5 + (p["occurrences"] - _YSH_THRESHOLD) * 0.1),
                action=action,
                auto_apply=p["occurrences"] >= _YSH_THRESHOLD * 2,
            ))
        return proposals

    def _load_patterns(self) -> None:
        """Load pattern store from disk."""
        try:
            with open(_PATTERNS_PATH, "r", encoding="utf-8") as f:
                self._patterns = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._patterns = {}

    def _save_patterns(self) -> None:
        """Persist pattern store to disk."""
        os.makedirs(os.path.dirname(_PATTERNS_PATH), exist_ok=True)
        with open(_PATTERNS_PATH, "w", encoding="utf-8") as f:
            json.dump(self._patterns, f, indent=2)


def _pattern_id(description: str) -> str:
    """Generate a stable ID for a pattern description."""
    normalised = description.lower().strip()
    return hashlib.sha256(normalised.encode()).hexdigest()[:12]


def _parse_patterns(text: str) -> list[dict]:
    """Parse LLM response into pattern dicts. Tolerant of malformed JSON."""
    text = text.strip()
    # Try to find JSON array in the response
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        patterns = json.loads(text[start:end + 1])
        if not isinstance(patterns, list):
            return []
        return [
            p for p in patterns
            if isinstance(p, dict) and "description" in p
        ]
    except json.JSONDecodeError:
        return []


def _propose_action(pattern: dict) -> tuple[str, str]:
    """Map a pattern category to a proposal type and action."""
    category = pattern.get("category", "")
    description = pattern.get("description", "")

    category_actions = {
        "error_recovery": ("hook", f"Create a hook to detect and handle: {description}"),
        "tool_usage": ("skill", f"Create a skill codifying: {description}"),
        "query_style": ("config", f"Add config option for: {description}"),
        "domain_knowledge": ("memory", f"Persist knowledge: {description}"),
        "workflow": ("skill", f"Create workflow skill: {description}"),
        "preference": ("config", f"Add user preference: {description}"),
    }

    return category_actions.get(category, ("memory", f"Record pattern: {description}"))
