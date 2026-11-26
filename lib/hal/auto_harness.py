"""HAL Auto-Harness — scan any repo, score features, propose absorption.

The generalised cc-auto-harness pattern. Given any GitHub repository, HAL:
1. Scans the repo structure and README
2. Identifies features relevant to HAL's architecture
3. Scores each feature on applicability, complexity, novelty, burden
4. Proposes integration points with specific HAL modules
5. Generates GitHub issues for approved absorptions

Usage:
    from lib.hal.auto_harness import AutoHarness

    harness = AutoHarness(registry)
    report = harness.scan("google-research/timesfm")
    print(report.summary)
    for feature in report.features:
        print(f"{feature.name}: {feature.score}/40 — {feature.recommendation}")
"""
import json
import os
import subprocess
import time
from dataclasses import dataclass

from lib.hal.home import HAL_HOME as _HAL_HOME

_REPORTS_DIR = os.path.join(_HAL_HOME, "state", "hal", "auto-harness")


@dataclass
class FeatureScore:
    """Scored feature from a scanned repository."""
    name: str
    description: str
    applicability: int  # 1-10: how relevant to HAL
    complexity: int     # 1-10: integration difficulty
    novelty: int        # 1-10: new capability added
    burden: int         # 1-10: ongoing maintenance cost
    recommendation: str  # "absorb", "monitor", "skip"
    hal_module: str     # Target HAL module for integration
    rationale: str

    @property
    def score(self) -> int:
        """Composite score: high applicability + novelty, low complexity + burden."""
        return self.applicability + self.novelty - self.complexity - self.burden + 20

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "applicability": self.applicability,
            "complexity": self.complexity,
            "novelty": self.novelty,
            "burden": self.burden,
            "score": self.score,
            "recommendation": self.recommendation,
            "hal_module": self.hal_module,
            "rationale": self.rationale,
        }


@dataclass
class HarnessReport:
    """Complete auto-harness scan report for a repository."""
    repo: str
    scan_time: float
    description: str
    language: str
    stars: int
    features: list  # list[FeatureScore]
    summary: str

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "scan_time": self.scan_time,
            "description": self.description,
            "language": self.language,
            "stars": self.stars,
            "features": [f.to_dict() for f in self.features],
            "summary": self.summary,
        }


class AutoHarness:
    """Scan repositories and propose feature absorption into HAL."""

    def __init__(self, registry=None):
        self._registry = registry

    def scan(self, repo: str, deep: bool = False) -> HarnessReport:
        """Scan a GitHub repo and produce an absorption report.

        Args:
            repo: GitHub repo in owner/name format (e.g. "google-research/timesfm")
            deep: If True, clone repo and scan actual source code (slower, richer)
        """
        start = time.time()

        # Fetch repo metadata
        metadata = self._fetch_metadata(repo)

        # Fetch structure
        structure = self._fetch_structure(repo)

        # Fetch README
        readme = self._fetch_readme(repo)

        # Deep scan: clone and build code mental model
        code_summary = ""
        if deep:
            code_summary = self._deep_scan_code(repo, structure)

        # Analyse with LLM council if registry available
        features = []
        if self._registry:
            features = self._analyse_with_council(
                repo, metadata, structure, readme, code_summary,
            )

        summary = self._build_summary(repo, metadata, features)

        report = HarnessReport(
            repo=repo,
            scan_time=time.time() - start,
            description=metadata.get("description", ""),
            language=metadata.get("primaryLanguage", {}).get("name", ""),
            stars=metadata.get("stargazerCount", 0),
            features=features,
            summary=summary,
        )

        # Persist report
        self._save_report(report)

        return report

    def _fetch_metadata(self, repo: str) -> dict:
        """Fetch repo metadata via gh CLI.

        The gh CLI rejects the whole ``--json`` query on any unknown field
        (``Unknown JSON field: 'X'``) with a non-zero exit, which would
        silently return ``{}`` here and leave the council prompt without
        description/language/stars context. The only fields requested are
        ones the downstream code actually reads — keep this list tight.
        """
        try:
            result = subprocess.run(
                ["gh", "repo", "view", repo, "--json",
                 "description,name,primaryLanguage,stargazerCount"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
            pass
        return {}

    def _fetch_structure(self, repo: str) -> list[str]:
        """Fetch top-level directory listing."""
        try:
            result = subprocess.run(
                ["gh", "api", f"repos/{repo}/contents", "--jq", ".[].name"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[:30]
        except (subprocess.TimeoutExpired, OSError):
            pass
        return []

    def _fetch_readme(self, repo: str) -> str:
        """Fetch README content."""
        try:
            result = subprocess.run(
                ["gh", "api", f"repos/{repo}/readme", "--jq", ".content"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                import base64
                return base64.b64decode(result.stdout.strip()).decode("utf-8", errors="replace")[:5000]
        except (subprocess.TimeoutExpired, OSError):
            pass
        return ""

    def _deep_scan_code(self, repo: str, structure: list) -> str:
        """Clone repo to temp dir and extract code-level mental model."""
        import shutil
        import tempfile

        clone_dir = tempfile.mkdtemp(prefix="hal-harness-")
        try:
            result = subprocess.run(
                ["gh", "repo", "clone", repo, clone_dir, "--", "--depth=1"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                return ""

            entry_points = _read_entry_points(clone_dir)
            top_files = _scan_top_source_files(clone_dir)
            return "\n".join(entry_points + top_files)[:8000]
        except (subprocess.TimeoutExpired, OSError):
            return ""
        finally:
            shutil.rmtree(clone_dir, ignore_errors=True)

    def _analyse_with_council(
        self, repo: str, metadata: dict, structure: list, readme: str,
        code_summary: str = "",
    ) -> list[FeatureScore]:
        """Use an LLM to identify absorbable features.

        Routes through :meth:`registry.chat` with an explicit ``max_tokens``
        budget rather than :meth:`HALCouncil.deliberate`, which relies on the
        registry's default max_tokens for both member queries and synthesis
        — a default tight enough to truncate this multi-feature JSON array
        mid-element (observed: ~387 chars, no closing ``]``, ``_parse_features``
        returns ``[]``). A single capable model with ample budget produces a
        complete, parseable response; the council's diversity benefit is
        marginal for structured-output tasks where synthesising multiple
        JSON arrays is itself a truncation risk.
        """
        models = self._get_models()
        if not models:
            return []

        code_section = ""
        if code_summary:
            code_section = f"\nSource code analysis:\n{code_summary[:3000]}\n"

        prompt = (
            f"Analyse this GitHub repository for features HAL should absorb.\n\n"
            f"Repo: {repo}\n"
            f"Description: {metadata.get('description', 'N/A')}\n"
            f"Language: {metadata.get('primaryLanguage', {}).get('name', 'N/A')}\n"
            f"Stars: {metadata.get('stargazerCount', 0)}\n"
            f"Structure: {', '.join(structure[:15])}\n"
            f"README (truncated):\n{readme[:2000]}\n"
            f"{code_section}\n"
            f"HAL is a standalone LLM-agnostic agentic runtime with: "
            f"multi-provider routing, cross-LLM council, recursive agents, "
            f"cost tracking, safety hooks, memory engine, MCP server.\n\n"
            f"For each absorbable feature, output JSON:\n"
            f'[{{"name": "...", "description": "...", "applicability": 1-10, '
            f'"complexity": 1-10, "novelty": 1-10, "burden": 1-10, '
            f'"recommendation": "absorb|monitor|skip", '
            f'"hal_module": "target module", "rationale": "..."}}]\n'
            f"Output ONLY the JSON array."
        )

        try:
            response = self._registry.chat(models[0], prompt, max_tokens=2048)
        except Exception:
            return []
        return _parse_features(response.text)

    def _get_models(self) -> list[str]:
        """Get available models for analysis."""
        models = []
        for p in self._registry.list_providers():
            if p.models:
                models.append(f"{p.name}/{p.models[0]}")
        return models[:3]

    def _build_summary(self, repo: str, metadata: dict, features: list) -> str:
        """Build human-readable summary."""
        absorb = [f for f in features if f.recommendation == "absorb"]
        monitor = [f for f in features if f.recommendation == "monitor"]
        skip = [f for f in features if f.recommendation == "skip"]

        lines = [
            f"Auto-Harness Report: {repo}",
            f"  {metadata.get('description', 'No description')}",
            f"  Language: {metadata.get('primaryLanguage', {}).get('name', '?')} | Stars: {metadata.get('stargazerCount', 0)}",
            f"  Features found: {len(features)} (absorb: {len(absorb)}, monitor: {len(monitor)}, skip: {len(skip)})",
        ]
        for f in absorb:
            lines.append(f"  [ABSORB] {f.name} ({f.score}/40) -> {f.hal_module}")
        for f in monitor:
            lines.append(f"  [MONITOR] {f.name} ({f.score}/40)")
        return "\n".join(lines)

    def _save_report(self, report: HarnessReport) -> None:
        """Persist report to disk."""
        os.makedirs(_REPORTS_DIR, exist_ok=True)
        safe_name = report.repo.replace("/", "-")
        path = os.path.join(_REPORTS_DIR, f"{safe_name}.json")
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)


def _parse_features(text: str) -> list[FeatureScore]:
    """Parse LLM response into FeatureScore list."""
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(text[start:end + 1])
        if not isinstance(items, list):
            return []
        features = []
        for item in items:
            if not isinstance(item, dict) or "name" not in item:
                continue
            features.append(FeatureScore(
                name=item.get("name", ""),
                description=item.get("description", ""),
                applicability=int(item.get("applicability", 5)),
                complexity=int(item.get("complexity", 5)),
                novelty=int(item.get("novelty", 5)),
                burden=int(item.get("burden", 5)),
                recommendation=item.get("recommendation", "monitor"),
                hal_module=item.get("hal_module", ""),
                rationale=item.get("rationale", ""),
            ))
        return features
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


_ENTRY_POINT_NAMES = frozenset({
    "main.py", "setup.py", "pyproject.toml", "package.json",
    "Cargo.toml", "go.mod", "Makefile",
})

_SOURCE_EXTENSIONS = frozenset({".py", ".ts", ".js", ".rs", ".go", ".java"})

_SKIP_DIRS = frozenset({
    "node_modules", "venv", "__pycache__", "dist", "build", ".git",
})


def _read_entry_points(clone_dir: str) -> list[str]:
    """Read common entry point files from a cloned repo."""
    lines: list[str] = []
    for name in _ENTRY_POINT_NAMES:
        path = os.path.join(clone_dir, name)
        if os.path.exists(path):
            with open(path, "r", errors="replace") as f:
                content = f.read(2000)
            lines.append(f"=== {name} ===\n{content}\n")
    return lines


def _scan_top_source_files(clone_dir: str, limit: int = 15) -> list[str]:
    """Find largest source files and read their headers. O(n log n) via sort."""
    src_files = _collect_source_files(clone_dir)
    src_files.sort(key=lambda x: x[1], reverse=True)
    return _read_file_headers(clone_dir, src_files[:limit])


def _collect_source_files(clone_dir: str) -> list[tuple[str, int]]:
    """Walk tree once, collect source files with sizes. O(n) traversal + O(1) ext check."""
    collected: list[tuple[str, int]] = []
    for root, dirs, files in os.walk(clone_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _SKIP_DIRS]
        collected.extend(_filter_source(root, files, clone_dir))
    return collected


def _filter_source(root: str, files: list[str], base: str) -> list[tuple[str, int]]:
    """Filter source files from a single directory listing. O(n) with O(1) set lookup."""
    result: list[tuple[str, int]] = []
    for f in files:
        if os.path.splitext(f)[1] in _SOURCE_EXTENSIONS:
            full = os.path.join(root, f)
            result.append((os.path.relpath(full, base), os.path.getsize(full)))
    return result


def _read_file_headers(clone_dir: str, files: list[tuple[str, int]]) -> list[str]:
    """Read first 1500 chars of each file."""
    lines: list[str] = []
    for rel_path, size in files:
        full = os.path.join(clone_dir, rel_path)
        with open(full, "r", errors="replace") as f:
            header = f.read(1500)
        lines.append(f"=== {rel_path} ({size} bytes) ===\n{header}\n")
    return lines
