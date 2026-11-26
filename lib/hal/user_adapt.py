"""HAL User Adaptation — few-shot preference learning from interactions.

Learns user preferences from minimal interaction signals (response feedback,
style patterns, topic focus) and adapts future responses without fine-tuning.

Adapted from NForge's cross-subject adaptation: instead of fitting ridge
regression on fMRI calibration scans, HAL fits preference weights on
interaction signals.

Usage:
    from lib.hal.user_adapt import UserAdapter

    adapter = UserAdapter()
    adapter.observe("verbosity", "concise")   # User prefers short answers
    adapter.observe("depth", "expert")         # User is technical
    adapter.observe("style", "direct")         # No filler words

    profile = adapter.get_profile()
    prompt_modifier = adapter.to_system_prompt()
"""
import json
import os
import time
from dataclasses import dataclass, field

from lib.hal.home import HAL_HOME as _HAL_HOME

_PROFILES_DIR = os.path.join(_HAL_HOME, "state", "hal", "profiles")


# Preference dimensions with valid values and defaults
_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "verbosity": ("concise", "balanced", "detailed"),
    "depth": ("beginner", "intermediate", "expert"),
    "style": ("formal", "direct", "casual"),
    "examples": ("none", "minimal", "rich"),
    "explanations": ("skip", "brief", "thorough"),
    "code_comments": ("none", "minimal", "detailed"),
}

_DEFAULTS: dict[str, str] = {
    "verbosity": "balanced",
    "depth": "intermediate",
    "style": "direct",
    "examples": "minimal",
    "explanations": "brief",
    "code_comments": "minimal",
}

# System prompt fragments for each preference value
_PROMPT_MAP: dict[tuple[str, str], str] = {
    ("verbosity", "concise"): "Keep responses short and direct. No filler.",
    ("verbosity", "detailed"): "Provide thorough, detailed responses.",
    ("depth", "beginner"): "Explain concepts simply, avoid jargon.",
    ("depth", "expert"): "Assume deep technical knowledge. Skip basics.",
    ("style", "formal"): "Use formal, professional language.",
    ("style", "casual"): "Use casual, conversational tone.",
    ("style", "direct"): "Be direct and to the point.",
    ("explanations", "skip"): "Do not explain unless asked.",
    ("explanations", "thorough"): "Include explanations of reasoning.",
    ("examples", "rich"): "Include code examples when relevant.",
    ("examples", "none"): "Skip examples unless specifically requested.",
}

# Inference rules: keyword -> (dimension, value)
_INFERENCE_RULES: dict[str, tuple[str, str]] = {
    "too long": ("verbosity", "concise"),
    "shorter": ("verbosity", "concise"),
    "concise": ("verbosity", "concise"),
    "brief": ("verbosity", "concise"),
    "more detail": ("verbosity", "detailed"),
    "elaborate": ("verbosity", "detailed"),
    "expand": ("verbosity", "detailed"),
    "too basic": ("depth", "expert"),
    "obvious": ("depth", "expert"),
    "i know": ("depth", "expert"),
    "confused": ("depth", "beginner"),
    "simpler": ("depth", "beginner"),
    "don't explain": ("explanations", "skip"),
    "skip explanation": ("explanations", "skip"),
    "just code": ("explanations", "skip"),
    "explain more": ("explanations", "thorough"),
}


@dataclass
class Preference:
    """A single user preference observation."""
    dimension: str
    value: str
    confidence: float  # 0.0 to 1.0
    observed_at: float = field(default_factory=time.time)
    source: str = ""  # "explicit", "inferred", "feedback"


@dataclass
class UserProfile:
    """Aggregated user preference profile."""
    user_id: str
    preferences: dict[str, str] = field(default_factory=dict)
    observation_count: int = 0
    last_updated: float = 0.0

    @property
    def summary(self) -> str:
        prefs = ", ".join(f"{k}={v}" for k, v in sorted(self.preferences.items()))
        return f"Profile {self.user_id}: {prefs} ({self.observation_count} observations)"

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "observation_count": self.observation_count,
            "last_updated": self.last_updated,
        }


class UserAdapter:
    """Learns and applies user preferences from interaction signals."""

    def __init__(self, user_id: str = "default"):
        self._user_id = user_id
        self._observations: list[Preference] = []
        self._profile = _load_profile(user_id)

    @property
    def profile(self) -> UserProfile:
        return self._profile

    def observe(self, dimension: str, value: str, confidence: float = 0.8, source: str = "explicit") -> None:
        """Record a preference observation."""
        if dimension not in _DIMENSIONS or value not in _DIMENSIONS[dimension]:
            return
        self._observations.append(Preference(
            dimension=dimension, value=value,
            confidence=confidence, source=source,
        ))
        self._resolve_preferences()

    def observe_from_feedback(self, feedback: str) -> list[str]:
        """Infer preferences from natural language feedback. Returns updated dimensions."""
        updated = []
        feedback_lower = feedback.lower()
        seen_dims: set[str] = set()

        for keyword, (dimension, value) in _INFERENCE_RULES.items():
            if keyword in feedback_lower and dimension not in seen_dims:
                self.observe(dimension, value, confidence=0.6, source="inferred")
                updated.append(dimension)
                seen_dims.add(dimension)

        return updated

    def to_system_prompt(self) -> str:
        """Generate a system prompt modifier based on learned preferences."""
        modifiers = [
            text for (dim, val), text in _PROMPT_MAP.items()
            if self._profile.preferences.get(dim) == val
        ]
        return ("User preferences: " + " ".join(modifiers)) if modifiers else ""

    def save(self) -> None:
        """Persist profile to disk."""
        _save_profile(self._profile)

    def _resolve_preferences(self) -> None:
        """Resolve multiple observations into final preferences.

        Strategy: highest confidence wins; recency breaks ties.
        """
        resolved: dict[str, str] = dict(_DEFAULTS)
        best: dict[str, tuple[float, float]] = {}

        for obs in self._observations:
            current = best.get(obs.dimension, (0.0, 0.0))
            if obs.confidence > current[0] or (obs.confidence == current[0] and obs.observed_at > current[1]):
                resolved[obs.dimension] = obs.value
                best[obs.dimension] = (obs.confidence, obs.observed_at)

        self._profile = UserProfile(
            user_id=self._user_id,
            preferences=resolved,
            observation_count=len(self._observations),
            last_updated=time.time(),
        )


def _load_profile(user_id: str) -> UserProfile:
    """Load profile from disk, or return defaults."""
    path = os.path.join(_PROFILES_DIR, f"{user_id}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return UserProfile(
                user_id=data.get("user_id", user_id),
                preferences=data.get("preferences", dict(_DEFAULTS)),
                observation_count=data.get("observation_count", 0),
                last_updated=data.get("last_updated", 0.0),
            )
        except (json.JSONDecodeError, OSError):
            pass
    return UserProfile(user_id=user_id, preferences=dict(_DEFAULTS))


def _save_profile(profile: UserProfile) -> None:
    """Persist profile to disk."""
    os.makedirs(_PROFILES_DIR, exist_ok=True)
    path = os.path.join(_PROFILES_DIR, f"{profile.user_id}.json")
    with open(path, "w") as f:
        json.dump(profile.to_dict(), f, indent=2)
