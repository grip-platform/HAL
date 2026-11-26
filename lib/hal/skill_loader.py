"""HAL Skill Loader — runtime-agnostic skill resolution.

Discovers skills from the skills/ directory, resolves mode-aware skill sets
(foundation + mode-specific), and provides skill content for context injection.

Usage:
    from lib.hal.skill_loader import SkillLoader

    loader = SkillLoader()
    skills = loader.resolve_for_mode("code")   # Foundation + code mode skills
    content = loader.get_skill_content("feature-complete")
"""
import os
import re
from typing import Optional

from lib.hal.home import HAL_HOME as _HAL_HOME
_SKILLS_DIR = os.path.join(_HAL_HOME, "skills")
_MODES_DIR = os.path.join(_HAL_HOME, "modes", "definitions")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Foundation skills that load with EVERY mode (from modes/foundation.md)
_FOUNDATION_SKILLS = [
    "ultrathink", "asking-users", "context-gate",
    "improvement-tracking", "design-smell-response",
    "self-improvement-engine", "recursive-learning",
    "autonomous-learning", "optimizing-complexity", "design-principles",
]


class SkillInfo:
    """Metadata about a discovered skill."""

    def __init__(self, name: str, path: str, description: str = "",
                 effort: str = "", version: str = ""):
        self.name = name
        self.path = path
        self.description = description
        self.effort = effort
        self.version = version


class ModeDefinition:
    """Parsed mode YAML definition."""

    def __init__(self, name: str, description: str = "",
                 always_skills: Optional[list[str]] = None,
                 optional_skills: Optional[list[str]] = None,
                 token_budget: int = 0):
        self.name = name
        self.description = description
        self.always_skills = always_skills or []
        self.optional_skills = optional_skills or []
        self.token_budget = token_budget


class SkillLoader:
    """Discovers and loads skills for context injection."""

    def __init__(self, skills_dir: Optional[str] = None,
                 modes_dir: Optional[str] = None):
        self._skills_dir = skills_dir or _SKILLS_DIR
        self._modes_dir = modes_dir or _MODES_DIR
        self._cache: dict[str, SkillInfo] = {}

    def discover(self) -> list[SkillInfo]:
        """Scan skills directory and return all discovered skills."""
        if not os.path.isdir(self._skills_dir):
            return []

        skills = []
        for entry in sorted(os.listdir(self._skills_dir)):
            skill_path = os.path.join(self._skills_dir, entry, "SKILL.md")
            if os.path.isfile(skill_path):
                info = self._parse_skill(entry, skill_path)
                skills.append(info)
                self._cache[entry] = info
        return skills

    def get_skill_content(self, name: str) -> str:
        """Read the full content of a skill's SKILL.md."""
        skill_path = os.path.join(self._skills_dir, name, "SKILL.md")
        return _read_file(skill_path) if os.path.isfile(skill_path) else ""

    def resolve_for_mode(self, mode_name: str) -> list[str]:
        """Resolve the full skill set for a mode: foundation + mode skills."""
        mode = self._load_mode(mode_name)
        skills = list(_FOUNDATION_SKILLS)
        if mode:
            for skill in mode.always_skills:
                if skill not in skills:
                    skills.append(skill)
        return skills

    def get_mode(self, mode_name: str) -> Optional[ModeDefinition]:
        """Load a mode definition by name."""
        return self._load_mode(mode_name)

    def list_modes(self) -> list[str]:
        """List all available mode names."""
        if not os.path.isdir(self._modes_dir):
            return []
        return sorted(
            f.replace("-mode.yaml", "").replace(".yaml", "")
            for f in os.listdir(self._modes_dir) if f.endswith(".yaml")
        )

    def _parse_skill(self, name: str, path: str) -> SkillInfo:
        """Parse skill metadata from SKILL.md frontmatter."""
        raw = _read_file(path)
        match = _FRONTMATTER_RE.match(raw)
        if not match:
            return SkillInfo(name=name, path=path)
        fm = _parse_frontmatter(match.group(1))
        return SkillInfo(
            name=fm.get("name", name), path=path,
            description=fm.get("description", ""),
            effort=fm.get("effort", ""), version=fm.get("version", ""),
        )

    def _load_mode(self, mode_name: str) -> Optional[ModeDefinition]:
        """Load mode definition from YAML file."""
        for suffix in (f"{mode_name}-mode.yaml", f"{mode_name}.yaml"):
            path = os.path.join(self._modes_dir, suffix)
            if os.path.isfile(path):
                return _parse_mode_yaml(_read_file(path))
        return None


def _parse_mode_yaml(raw: str) -> ModeDefinition:
    """Parse mode YAML into ModeDefinition."""
    name, description, token_budget = "", "", 0
    always_skills: list[str] = []
    optional_skills: list[str] = []
    section = ""

    for line in raw.split("\n"):
        stripped = line.strip()
        if stripped.startswith("name:"):
            name = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("description:"):
            description = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("token_budget:"):
            val = stripped.split(":", 1)[1].strip().lstrip("~")
            token_budget = int(val) if val.isdigit() else 0
        elif stripped == "always:":
            section = "always"
        elif stripped == "optional:":
            section = "optional"
        elif stripped.startswith("- ") and section == "always":
            always_skills.append(stripped[2:].strip())
        elif stripped.startswith("- ") and section == "optional":
            optional_skills.append(stripped[2:].strip())
        elif ":" in stripped and not stripped.startswith("-"):
            section = ""

    return ModeDefinition(name=name, description=description,
                          always_skills=always_skills,
                          optional_skills=optional_skills,
                          token_budget=token_budget)


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse simple key: value frontmatter."""
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
