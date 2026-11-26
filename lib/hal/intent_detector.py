"""HAL Intent Detector — classify user intent for smart routing.

Classifies user messages into intent categories to route them to the
appropriate handler (code generation, explanation, debugging, research,
conversation, command). Enables HAL to proactively select the right
mode, model, and tools.

Usage:
    from lib.hal.intent_detector import IntentDetector

    detector = IntentDetector()
    intent = detector.classify("Fix the null pointer in auth.py line 42")
    print(f"{intent.category}: {intent.confidence}")
    # "debug: 0.9"
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentWeights:
    """Multi-dimensional intent confidence scores.

    A message can match multiple intent categories simultaneously.
    Each category gets its own confidence score (0.0-1.0).
    """
    weights: tuple  # tuple of (category, confidence) pairs, sorted by confidence desc

    @property
    def dominant(self) -> str:
        """Return the highest-scoring category, or 'chat' if empty."""
        return self.weights[0][0] if self.weights else "chat"

    @property
    def active(self) -> list[str]:
        """Return all categories scoring above the activation threshold (0.3)."""
        return [cat for cat, conf in self.weights if conf >= 0.3]

    def get(self, category: str) -> float:
        """Get confidence for a specific category."""
        for cat, conf in self.weights:
            if cat == category:
                return conf
        return 0.0

    def to_dict(self) -> dict:
        """Return as {category: confidence} dict."""
        return dict(self.weights)


@dataclass(frozen=True)
class Intent:
    """Classified user intent."""
    category: str  # code, explain, debug, research, chat, command, review
    confidence: float  # 0.0 to 1.0
    suggested_mode: str  # HAL mode recommendation
    suggested_tools: tuple[str, ...]  # Tools likely needed
    weights: IntentWeights = None  # Multi-intent confidence scores


# Canonical media extension lists — shared by both regexes to stay in sync.
_IMAGE_EXTS = r"png|jpg|jpeg|webp|gif|bmp|tiff"
_VIDEO_EXTS = r"mp4|mov|avi|mkv|webm"
_AUDIO_EXTS = r"mp3|wav|ogg|flac|m4a|aac"
_ALL_MEDIA_EXTS = rf"{_IMAGE_EXTS}|{_VIDEO_EXTS}|{_AUDIO_EXTS}"

# Confidence booster: fires when a standalone media file path is present.
# (?![.\w]) blocks compound extensions like .png.zip (dot is not \w but must be excluded).
_MEDIA_FILE_RE = re.compile(
    rf"\.(?:{_ALL_MEDIA_EXTS})(?![.\w])", re.IGNORECASE
)

# Intent patterns: (category, regex, confidence_boost, mode, tools)
_PATTERNS: list[tuple[str, re.Pattern, float, str, tuple[str, ...]]] = [
    ("vision", re.compile(
        # Visual/media nouns + actions; "detect" scoped to object/CV context only.
        rf"\b(?:image|photo|picture|screenshot|diagram|video|clip|recording|audio|"
        rf"(?:bar|pie|line|scatter)\s+chart|ocr|"
        rf"detect.{{0,15}}(?:object|face|text|label|person|car|sign)|segment|"
        rf"describe.{{0,20}}(?:image|photo|picture|video)|what.{{0,15}}(?:in this|shown|visible)|"
        rf"read.{{0,10}}text.{{0,10}}image|analy[sz]e.{{0,15}}(?:video|image|photo))\b"
        rf"|data:(?:image|video|audio)/|\.(?:{_ALL_MEDIA_EXTS})(?![.\w])",
        re.IGNORECASE,
    ), 0.85, "vision", ("read",)),
    ("command", re.compile(r"^/\w+", re.IGNORECASE), 0.95, "code", ("bash",)),
    ("debug", re.compile(r"\b(?:fix|bug|error|broken|crash|fail|exception|traceback|issue)\b", re.IGNORECASE), 0.8, "code", ("read", "edit", "bash")),
    ("code", re.compile(r"\b(?:write|create|implement|add|build|function|class|module|refactor)\b", re.IGNORECASE), 0.75, "code", ("write", "edit", "read")),
    ("review", re.compile(r"\b(?:review|check|audit|analyse|inspect|evaluate)\b", re.IGNORECASE), 0.7, "code", ("read", "search")),
    ("explain", re.compile(r"\b(?:explain|how does|what is|why does|describe|tell me about)\b", re.IGNORECASE), 0.7, "research", ("read",)),
    ("research", re.compile(r"\b(?:research|find|search|look up|compare|investigate|explore)\b", re.IGNORECASE), 0.65, "research", ("search", "read")),
    ("chat", re.compile(r"\b(?:hello|hi|hey|thanks|please|help|opinion)\b", re.IGNORECASE), 0.5, "chat", ()),
]

# Code indicators boost code/debug intent
_CODE_INDICATORS = re.compile(r"```|\.py|\.ts|\.js|line \d+|def |class |function ")
_FILE_INDICATORS = re.compile(r"[\w/.-]+\.(?:py|ts|js|md|json|yaml|rs|go)")


class IntentDetector:
    """Classifies user messages into intent categories."""

    def classify(self, message: str) -> Intent:
        """Classify a message's intent.

        Collects ALL matching patterns into IntentWeights, then picks
        the dominant one as the primary category. This allows downstream
        consumers to see that "fix the auth bug" is both debug (0.8) and
        code (0.75), not just "debug".
        """
        if not message.strip():
            return Intent("chat", 0.0, "chat", (), IntentWeights(()))

        all_matches: list[tuple[str, float, str, tuple]] = []

        for category, pattern, base_conf, mode, tools in _PATTERNS:
            if pattern.search(message):
                confidence = base_conf
                if category in ("code", "debug", "review") and _CODE_INDICATORS.search(message):
                    confidence = min(1.0, confidence + 0.1)
                if _FILE_INDICATORS.search(message):
                    confidence = min(1.0, confidence + 0.05)
                if category == "vision" and _MEDIA_FILE_RE.search(message):
                    confidence = min(1.0, confidence + 0.1)
                all_matches.append((category, round(confidence, 2), mode, tools))

        if not all_matches:
            return Intent("chat", 0.3, "chat", (), IntentWeights(()))

        # Sort by confidence descending
        all_matches.sort(key=lambda m: m[1], reverse=True)
        weight_pairs = tuple((cat, conf) for cat, conf, _, _ in all_matches)
        weights = IntentWeights(weight_pairs)

        best = all_matches[0]
        return Intent(best[0], best[1], best[2], best[3], weights)

    def classify_batch(self, messages: list[str]) -> list[Intent]:
        """Classify multiple messages. Returns intents in same order."""
        return [self.classify(m) for m in messages]
