"""HAL Chat Templates — pre-built conversation starters for common workflows.

Usage:
    from lib.hal.chat_templates import TemplateLibrary
    template = TemplateLibrary().get("code_review")
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatTemplate:
    name: str
    description: str
    system_prompt: str
    starter_message: str
    follow_ups: tuple[str, ...]
    category: str = "general"


_TEMPLATES: dict[str, ChatTemplate] = {
    "code_review": ChatTemplate("code_review", "Structured code review",
        "You are conducting a code review. Focus on correctness, performance, security.",
        "Paste the code to review with language and context.",
        ("Critical issues?", "How to refactor?", "Security concerns?"), "development"),
    "debug": ChatTemplate("debug", "Guided debugging",
        "You are debugging. Ask questions, form hypotheses, trace execution.",
        "Describe the bug: expected vs actual, error messages.",
        ("Already tried?", "Reproducible?", "Recent changes?"), "development"),
    "architecture": ChatTemplate("architecture", "Architecture design",
        "Discussing architecture. Consider trade-offs, scalability, maintainability.",
        "What system? Key requirements and constraints?",
        ("Target scale?", "Team strengths?", "Non-negotiables?"), "development"),
    "research": ChatTemplate("research", "Research and analysis",
        "Conducting research. Be thorough, cite sources, multiple perspectives.",
        "Topic to research? Specific questions?",
        ("Priority sources?", "Competing views?", "Deadline?"), "research"),
    "onboarding": ChatTemplate("onboarding", "Codebase onboarding",
        "Helping understand a new codebase. Big picture first.",
        "Which codebase? Your role and area?",
        ("Entry point?", "Run tests?", "Contributors?"), "development"),
    "writing": ChatTemplate("writing", "Technical writing",
        "Helping write technical content. Clarity, accuracy, audience-appropriate.",
        "What are you writing? Audience? Key message?",
        ("Tone?", "Length?", "Reference examples?"), "writing"),
}


class TemplateLibrary:
    def get(self, name: str) -> ChatTemplate | None:
        return _TEMPLATES.get(name)

    def list_templates(self, category: str = "") -> list[ChatTemplate]:
        templates = list(_TEMPLATES.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return sorted(templates, key=lambda t: t.name)

    def categories(self) -> list[str]:
        return sorted({t.category for t in _TEMPLATES.values()})
