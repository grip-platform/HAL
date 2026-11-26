"""HAL Agent Trap Defence — 6 adversarial threat classes for AI agent protection.

Based on the AI Agent Traps research paper. HAL implements mechanical
defences for each trap class, protecting both HAL and its users.

Trap Classes:
1. Content Injection — hidden instructions in HTML/CSS/metadata (sanitise.py)
2. Semantic Manipulation — misleading context to alter agent behaviour
3. Memory Poisoning — corrupting persistent memory with false data
4. Behavioural Control — exploiting agent patterns to force actions
5. Systemic — attacking the agent infrastructure itself
6. Human-in-the-Loop — tricking users into approving malicious actions

Usage:
    from lib.hal.agent_traps import TrapDetector

    detector = TrapDetector()
    threats = detector.scan_input(user_message)
    threats = detector.scan_response(llm_response)
    threats = detector.scan_memory_write(entry_content)
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TrapAlert:
    """A detected adversarial trap attempt."""
    trap_class: int  # 1-6
    trap_name: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    evidence: str
    recommended_action: str


# Patterns for each trap class.
#
# Each pattern uses a verb + proximity + target-noun structure where applicable,
# so that paraphrased attacks are caught without flagging benign prompts that
# happen to share individual words. The proximity window is intentionally tight
# (≤40 chars between anchors) to avoid cross-sentence false positives.

# Content Injection — VERB (ignore/disregard/forget/override/bypass/...) +
# optional qualifier (previous/prior/above/system/...) + adversarial NOUN
# (instructions/directives/prompt/role/...). Bare short phrases are kept for
# legacy coverage.
_INJECTION_PATTERNS = re.compile(
    r"(?:"
    r"\b(?:ignore|disregard|dismiss|forget|override|bypass|skip|cancel|reject|discard|rewrite|set\s+aside)\b"
    r".{0,40}?"
    r"\b(?:instruction|directive|command|rule|guideline|prompt|order|context|role|persona|character|config|configuration|setup)s?\b"
    r"|"
    r"\byou\s+are\s+(?:now|actually|really)\b"
    r"|"
    r"\b(?:act|pretend|roleplay|roleplaying)\s+(?:as|to\s+be)\b"
    r"|"
    r"\bforget\s+everything\b"
    r"|"
    r"\bnew\s+instructions?\b"
    r"|"
    r"\bsystem\s+prompt\b"
    r")",
    re.IGNORECASE,
)

# Semantic Manipulation — authority framing, urgency pressure, explicit
# verification bypass. Paraphrase targets: "I'm the admin", "just the admin
# speaking", "believe me I am your creator".
_SEMANTIC_PATTERNS = re.compile(
    r"(?:"
    r"\b(?:trust|believe)\s+me\b"
    r"|"
    r"\bI(?:'m|\s+am)\s+(?:the\s+|your\s+|a\s+)?(?:admin|administrator|root|owner|developer|creator|author|maintainer|operator|superuser|sysadmin)\b"
    r"|"
    r"\b(?:this\s+is\s+)?urgent\b.{0,20}\b(?:please\s+comply|act\s+now|no\s+time)\b"
    r"|"
    r"\bbypass\b.{0,30}?\b(?:verification|validation|check|checks|safety|safeguards?)\b"
    r"|"
    r"\bskip\s+(?:the\s+)?(?:verification|validation|check|checks|safety|safeguards?)\b"
    r"|"
    r"\bno\s+need\s+to\s+(?:check|verify|validate|confirm)\b"
    r")",
    re.IGNORECASE,
)

# Memory Poisoning — imperative persistence verbs + intent markers that try
# to establish a false authoritative "fact" in memory. Guards against benign
# uses like "please remember the deadline" by requiring the object to be a
# knowledge/assumption/answer target.
_MEMORY_POISON_PATTERNS = re.compile(
    r"(?:"
    r"\b(?:remember|memori[sz]e|learn|store|save|record)\s+that\b"
    r"|"
    r"\balways\s+(?:assume|trust|believe|accept|treat)\b"
    r"|"
    r"\bfrom\s+now\s+on\b.{0,40}?\b(?:assume|treat|consider|act|behave|respond)\b"
    r"|"
    r"\bupdate\s+your\s+(?:knowledge|memory|beliefs?|understanding|model)\b"
    r"|"
    r"\bthe\s+(?:correct|right|true)\s+answer\s+is\s+always\b"
    r"|"
    r"\bnever\s+question\b"
    r")",
    re.IGNORECASE,
)

# Behavioural Control — unconditional imperative + absence-of-verification
# qualifier. Guards benign "step by step" tutorials unless paired with
# delete/execute/run without checking.
_BEHAVIOURAL_PATTERNS = re.compile(
    r"(?:"
    r"\brepeat\s+after\s+me\b"
    r"|"
    r"\bdo\s+exactly\s+(?:as|what)\b"
    r"|"
    r"\b(?:execute|run|perform)\b.{0,30}?\b(?:blindly|without\s+(?:check(?:ing|s)?|verif(?:y|ication)|confirm(?:ing|ation)?)|no\s+questions\s+asked)\b"
    r"|"
    r"\b(?:first|immediately|right\s+now),?\s+delete\b"
    r"|"
    r"\bdon'?t\s+(?:ask|verify|check|confirm)\b.{0,30}?\bjust\s+(?:do|execute|run)\b"
    r"|"
    r"\bfollow\s+these\s+steps?\s+(?:blindly|without\s+thinking)\b"
    r")",
    re.IGNORECASE,
)

# Systemic — credential/infrastructure name tokens. These are strong signals on
# their own because legitimate prompts rarely name specific secret-token types
# alongside intent-to-exfiltrate verbs.
_SYSTEMIC_PATTERNS = re.compile(
    r"(?:"
    r"\baccess[_\s-]?token\b"
    r"|"
    r"\bapi[_\s-]?key\b"
    r"|"
    r"\b(?:private|ssh|secret|signing)[_\s-]?key\b"
    r"|"
    r"\bauthorization:\s*bearer\b"
    r"|"
    r"\bx-api-key\b"
    r"|"
    r"\.env\b"
    r"|"
    r"\b(?:credentials?|passwords?|secrets?)\.(?:json|yaml|yml|env|txt)\b"
    r"|"
    r"\b(?:id_rsa|id_ed25519|id_ecdsa)\b"
    r"|"
    r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"  # AWS access key prefix
    r")",
    re.IGNORECASE,
)

# Human-in-the-Loop — pressure to approve without review. Guards against
# benign "just click Submit" in a legit workflow by requiring a minimisation
# verb ("minor", "nothing", "don't read") alongside the action.
_HITL_PATTERNS = re.compile(
    r"(?:"
    r"\b(?:just\s+)?approve\s+(?:this\s+)?(?:quickly|fast|right\s+now|immediately)\b"
    r"|"
    r"\bjust\s+click\s+(?:yes|approve|accept|ok|continue|confirm)\b"
    r"|"
    r"\bdon'?t\s+(?:read|review|check)\s+(?:this|it|the\s+(?:details?|terms|changes?|diff))\b"
    r"|"
    r"\b(?:routine|minor|trivial|small|normal)\s+(?:update|change|fix|patch|diff)\b.{0,30}?\b(?:approve|accept|click|confirm|ok)\b"
    r"|"
    r"\bnothing\s+to\s+worry\s+about\b"
    r"|"
    r"\bsafe\s+to\s+ignore\b"
    r")",
    re.IGNORECASE,
)

# Map class number to (name, pattern, description)
_TRAP_DEFS: dict[int, tuple[str, re.Pattern, str]] = {
    1: ("Content Injection", _INJECTION_PATTERNS, "Hidden instructions attempting to override agent behaviour"),
    2: ("Semantic Manipulation", _SEMANTIC_PATTERNS, "Misleading context to alter agent decisions"),
    3: ("Memory Poisoning", _MEMORY_POISON_PATTERNS, "Attempting to corrupt persistent memory"),
    4: ("Behavioural Control", _BEHAVIOURAL_PATTERNS, "Exploiting agent patterns to force specific actions"),
    5: ("Systemic", _SYSTEMIC_PATTERNS, "Targeting agent infrastructure or credentials"),
    6: ("Human-in-the-Loop", _HITL_PATTERNS, "Tricking users into approving malicious actions"),
}


class TrapDetector:
    """Scans inputs, responses, and memory writes for adversarial traps."""

    def scan_input(self, text: str) -> list[TrapAlert]:
        """Scan user input for trap attempts."""
        return _scan_text(text, scan_classes={1, 2, 4, 5})

    def scan_response(self, text: str) -> list[TrapAlert]:
        """Scan LLM response for signs of compromised output."""
        return _scan_text(text, scan_classes={1, 5, 6})

    def scan_memory_write(self, text: str) -> list[TrapAlert]:
        """Scan content before writing to memory."""
        return _scan_text(text, scan_classes={3, 1})

    def scan_all(self, text: str) -> list[TrapAlert]:
        """Scan text against all 6 trap classes."""
        return _scan_text(text, scan_classes={1, 2, 3, 4, 5, 6})


def _alerts_for_class(
    cls_num: int,
    name: str,
    pattern: re.Pattern,
    description: str,
    text: str,
) -> list[TrapAlert]:
    """Produce every TrapAlert for one trap class against ``text``.

    Single linear scan — ``re.Pattern.finditer`` walks the text once and
    yields each non-overlapping match. Total complexity is O(|text|)
    per class; the outer scan over ``_TRAP_DEFS`` is bounded at 6
    classes, so the full scan is O(|text|) with a constant factor of 6.
    """
    return [
        TrapAlert(
            trap_class=cls_num,
            trap_name=name,
            severity=_classify_severity(cls_num, match.group()),
            description=description,
            evidence=match.group()[:100],
            recommended_action=_recommend_action(cls_num),
        )
        for match in pattern.finditer(text)
    ]


def _scan_text(text: str, scan_classes: set[int]) -> list[TrapAlert]:
    """Scan text against specified trap classes.

    Returns one alert per distinct match — an attacker combining several
    vectors within the same trap class (or across classes) produces
    multiple alerts, not just the first. Previously this used
    ``pattern.search()`` which silently dropped every match after the
    first within a class. See _alerts_for_class for the per-class work.
    """
    alerts: list[TrapAlert] = []
    for cls_num in sorted(scan_classes):
        trap_def = _TRAP_DEFS.get(cls_num)
        if trap_def is None:
            continue
        name, pattern, description = trap_def
        alerts.extend(_alerts_for_class(cls_num, name, pattern, description, text))
    return alerts


def _classify_severity(trap_class: int, evidence: str) -> str:
    """Classify severity based on trap class and evidence."""
    critical_classes = {1, 5}  # Injection and systemic are most dangerous
    if trap_class in critical_classes:
        return "high"
    if trap_class == 3:  # Memory poisoning
        return "medium"
    return "low"


def _recommend_action(trap_class: int) -> str:
    """Recommend action for each trap class."""
    actions = {
        1: "Block input. Content injection detected — do not process.",
        2: "Flag for user review. Verify claims independently.",
        3: "Block memory write. Validate against existing knowledge.",
        4: "Refuse execution. Do not follow step-by-step instructions blindly.",
        5: "Block and alert. Credential/infrastructure targeting detected.",
        6: "Slow down. Present full details to user before approval.",
    }
    return actions.get(trap_class, "Flag for review.")
