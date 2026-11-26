"""HAL Permission Guard Engine — three-way tool access control.

Ported from TF>> useCanUseTool + permissions system. Evaluates
allow/deny/ask rules per tool invocation. Integrates with existing
hook_dispatch.py (hard/soft guards map to DENY/ALLOW).

Usage:
    from lib.hal.guard import GuardEngine, GuardDecision, GuardRule

    engine = GuardEngine()
    engine.add_rule(GuardRule(tool="Bash", decision=GuardDecision.ASK))
    engine.add_rule(GuardRule(tool="Read", decision=GuardDecision.ALLOW))

    decision = engine.evaluate("Bash", {"command": "rm -rf /"})
    # GuardVerdict(decision=ASK, reason="Bash requires approval")
"""
from dataclasses import dataclass
from enum import Enum


class GuardDecision(str, Enum):
    """Three-way permission decision (from TF>> PermissionDecision)."""
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class GuardRule:
    """A permission rule for a specific tool or pattern."""
    tool: str
    decision: str = GuardDecision.ALLOW
    source: str = "default"
    reason: str = ""
    pattern: str = ""

    def matches(self, tool_name: str, tool_input: dict) -> bool:
        """Check if this rule applies to the given tool call."""
        if self.tool == "*":
            return True
        if self.tool != tool_name:
            return False
        if self.pattern and self.pattern not in str(tool_input):
            return False
        return True


@dataclass(frozen=True)
class GuardVerdict:
    """Result of guard evaluation."""
    decision: str
    reason: str = ""
    rule_source: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision == GuardDecision.ALLOW

    @property
    def denied(self) -> bool:
        return self.decision == GuardDecision.DENY

    @property
    def needs_approval(self) -> bool:
        return self.decision == GuardDecision.ASK


# Default rules: read-only tools allowed, write tools ask, nothing denied by default.
_DEFAULT_RULES = [
    GuardRule(tool="Read", decision=GuardDecision.ALLOW, source="default"),
    GuardRule(tool="Glob", decision=GuardDecision.ALLOW, source="default"),
    GuardRule(tool="Grep", decision=GuardDecision.ALLOW, source="default"),
    GuardRule(tool="Bash", decision=GuardDecision.ASK, source="default", reason="Shell commands require approval"),
    GuardRule(tool="Write", decision=GuardDecision.ASK, source="default", reason="File writes require approval"),
    GuardRule(tool="Edit", decision=GuardDecision.ASK, source="default", reason="File edits require approval"),
]


class GuardEngine:
    """Evaluates permission rules for tool invocations.

    Rule priority: explicit rules override defaults. DENY beats ASK beats ALLOW.
    Multiple matching rules resolve by strictest-wins.
    """

    def __init__(self, rules: list[GuardRule] = None, auto_approve: bool = False):
        self._rules: list[GuardRule] = list(rules or _DEFAULT_RULES)
        self._auto_approve = auto_approve
        self._approvals: set[str] = set()

    def add_rule(self, rule: GuardRule) -> None:
        """Add a permission rule. Later rules override earlier ones."""
        self._rules.append(rule)

    def approve(self, tool_name: str) -> None:
        """Grant standing approval for a tool (persists for session)."""
        self._approvals.add(tool_name)

    def evaluate(self, tool_name: str, tool_input: dict = None) -> GuardVerdict:
        """Evaluate permission for a tool call. Returns verdict."""
        tool_input = tool_input or {}

        if self._auto_approve:
            return GuardVerdict(decision=GuardDecision.ALLOW, reason="auto-approve mode")

        if tool_name in self._approvals:
            return GuardVerdict(
                decision=GuardDecision.ALLOW,
                reason=f"Standing approval for {tool_name}",
                rule_source="session",
            )

        matching = [r for r in self._rules if r.matches(tool_name, tool_input)]
        if not matching:
            return GuardVerdict(decision=GuardDecision.ASK, reason=f"No rule for {tool_name}")

        # Strictest-wins: DENY > ASK > ALLOW
        strictness = {GuardDecision.DENY: 3, GuardDecision.ASK: 2, GuardDecision.ALLOW: 1}
        strictest = max(matching, key=lambda r: strictness.get(r.decision, 0))

        return GuardVerdict(
            decision=strictest.decision,
            reason=strictest.reason or f"Rule: {strictest.tool} -> {strictest.decision}",
            rule_source=strictest.source,
        )

    def list_rules(self) -> list[GuardRule]:
        """Return all current rules."""
        return list(self._rules)
