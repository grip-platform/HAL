"""HAL Agent Runner — parse agent definitions and abstract spawn mechanism.

Reads GRIP agent .md files, extracts prompt/tools/skills, and provides a
runtime-agnostic interface for spawning agents. Under CC, delegates to Task
tool. Under standalone, delegates to HAL's own query() with agent prompt.

Usage:
    from lib.hal.agent_runner import AgentRunner

    runner = AgentRunner()
    agents = runner.list_agents()
    spec = runner.parse_agent("broly")
    # spec.name, spec.prompt, spec.skills, spec.tools
"""
import os
import re
from typing import Optional

from lib.hal.home import HAL_HOME as _HAL_HOME
_AGENTS_DIR = os.path.join(_HAL_HOME, "agents")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class AgentSpec:
    """Parsed agent definition."""

    def __init__(
        self,
        name: str,
        prompt: str = "",
        description: str = "",
        skills: Optional[list[str]] = None,
        tools: Optional[list[str]] = None,
        model: str = "",
    ):
        self.name = name
        self.prompt = prompt
        self.description = description
        self.skills = skills or []
        self.tools = tools or []
        self.model = model

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "skills": self.skills,
            "tools": self.tools,
            "model": self.model,
        }


class AgentRunner:
    """Parses agent definitions and manages agent lifecycle."""

    def __init__(self, agents_dir: Optional[str] = None):
        self._dir = agents_dir or _AGENTS_DIR
        self._cache: dict[str, AgentSpec] = {}

    def list_agents(self) -> list[str]:
        """List all available agent names."""
        if not os.path.isdir(self._dir):
            return []
        return sorted(
            f.replace(".md", "")
            for f in os.listdir(self._dir)
            if f.endswith(".md") and not f.startswith("_") and f != "CLAUDE.md"
        )

    def parse_agent(self, name: str) -> Optional[AgentSpec]:
        """Parse an agent definition file into AgentSpec."""
        if name in self._cache:
            return self._cache[name]

        path = os.path.join(self._dir, f"{name}.md")
        if not os.path.isfile(path):
            return None

        spec = _parse_agent_file(name, _read_file(path))
        self._cache[name] = spec
        return spec

    def get_agent_prompt(self, name: str) -> str:
        """Get the full prompt text for an agent (body after frontmatter)."""
        spec = self.parse_agent(name)
        return spec.prompt if spec else ""


def _parse_agent_file(name: str, raw: str) -> AgentSpec:
    """Parse an agent .md file with frontmatter."""
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return AgentSpec(name=name, prompt=raw.strip())

    fm = _parse_frontmatter(match.group(1))
    prompt = raw[match.end():].strip()

    skills_raw = fm.get("skills", "")
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()] if skills_raw else []

    tools_raw = fm.get("tools", "")
    tools = [t.strip() for t in tools_raw.split(",") if t.strip()] if tools_raw else []

    return AgentSpec(
        name=name,
        prompt=prompt,
        description=fm.get("description", ""),
        skills=skills,
        tools=tools,
        model=fm.get("model", ""),
    )


def _parse_frontmatter(text: str) -> dict[str, str]:
    result = {}
    for line in text.split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return ""
