"""HAL Knowledge Absorber — universal knowledge extraction from any source.

Absorbs from papers, articles, APIs, datasets, any URL or local file.

Usage:
    from lib.hal.knowledge_absorber import KnowledgeAbsorber
    absorber = KnowledgeAbsorber(registry)
    report = absorber.absorb("https://arxiv.org/abs/2603.28990")
"""
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from lib.hal.home import HAL_HOME as _HAL_HOME

_ABSORPTIONS_DIR = os.path.join(_HAL_HOME, "state", "hal", "absorptions")


@dataclass
class KnowledgeItem:
    """A single extracted knowledge item."""
    concept: str
    description: str
    category: str
    relevance: int
    actionable: bool
    hal_module: str
    source: str


@dataclass
class AbsorptionReport:
    """Complete knowledge absorption report."""
    source: str
    source_type: str
    title: str
    absorb_time: float
    items: list
    summary: str

    def to_dict(self) -> dict:
        return {
            "source": self.source, "source_type": self.source_type,
            "title": self.title, "absorb_time": self.absorb_time,
            "summary": self.summary,
            "items": [
                {"concept": i.concept, "description": i.description,
                 "category": i.category, "relevance": i.relevance,
                 "actionable": i.actionable, "hal_module": i.hal_module}
                for i in self.items
            ],
        }


class KnowledgeAbsorber:
    """Universal knowledge extraction and absorption engine."""

    def __init__(self, registry=None):
        self._registry = registry

    def absorb(self, source: str) -> AbsorptionReport:
        """Absorb knowledge from any source. Auto-detects type."""
        start = time.time()
        source_type = _detect_source_type(source)
        title, content = _extract(source, source_type, self._registry)
        items = _analyse(self._registry, source, source_type, title, content)
        wikilinks = _auto_wikilink(items)
        actionable = sum(1 for i in items if i.actionable)
        link_note = f", {len(wikilinks)} wikilinks" if wikilinks else ""
        report = AbsorptionReport(
            source=source, source_type=source_type, title=title,
            absorb_time=time.time() - start, items=items,
            summary=f"{source_type}: {title} | {len(items)} items ({actionable} actionable{link_note})",
        )
        _save_report(report)
        return report


_AUDIO_EXTS = frozenset({".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"})
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


def _detect_source_type(source: str) -> str:
    """Auto-detect knowledge source type."""
    if "github.com" in source:
        return "github"
    if source.startswith("arxiv:") or "arxiv.org" in source:
        return "arxiv"
    if "youtube.com" in source or "youtu.be" in source:
        return "audio"
    ext = os.path.splitext(source)[1].lower()
    if ext == ".pdf":
        return "paper"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _VIDEO_EXTS:
        return "audio"  # extract audio track from video
    if os.path.exists(source):
        return "file"
    return "article"


def _extract(source: str, stype: str, registry) -> tuple[str, str]:
    """Extract title and content from source."""
    dispatch = {
        "github": lambda: _extract_github(source, registry),
        "paper": lambda: _extract_paper(source),
        "article": lambda: _extract_article(source),
        "arxiv": lambda: _extract_arxiv(source),
        "audio": lambda: _extract_audio(source),
        "file": lambda: _extract_file(source),
    }
    return dispatch.get(stype, lambda: _extract_article(source))()


def _extract_github(source: str, registry) -> tuple[str, str]:
    """Delegate GitHub repos to auto_harness."""
    from lib.hal.auto_harness import AutoHarness
    repo = source.replace("https://github.com/", "").rstrip("/")
    report = AutoHarness(registry).scan(repo)
    return report.description or repo, report.summary


def _extract_paper(source: str) -> tuple[str, str]:
    """Extract text from PDF. Direct pdfminer import with subprocess fallback."""
    if not os.path.exists(source):
        return "Unknown", ""
    # Direct import — avoids fragile subprocess shelling
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(source)[:8000]
        if text.strip():
            title = text.strip().split("\n")[0][:200]
            return title, text
    except ImportError:
        pass
    except Exception:
        pass
    # Fallback: subprocess (pdfminer installed in a different env)
    try:
        result = subprocess.run(
            ["python3", "-c",
             "from pdfminer.high_level import extract_text; "
             f"print(extract_text('{source}')[:8000])"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            title = result.stdout.strip().split("\n")[0][:200]
            return title, result.stdout[:8000]
    except (subprocess.TimeoutExpired, OSError):
        pass
    return _read_raw(source)


def _extract_article(source: str) -> tuple[str, str]:
    """Extract text from web URL."""
    try:
        req = urllib.request.Request(source, headers={"User-Agent": "HAL/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        text = _strip_html(html)
        match = re.search(r"<title>([^<]+)</title>", html)
        return (match.group(1).strip() if match else source), text[:8000]
    except (urllib.error.URLError, TimeoutError, OSError):
        return source, ""


def _extract_arxiv(source: str) -> tuple[str, str]:
    """Extract from arXiv."""
    aid = source.replace("arxiv:", "").replace("https://arxiv.org/abs/", "")
    return _extract_article(f"https://arxiv.org/abs/{aid}")


def _extract_audio(source: str) -> tuple[str, str]:
    """Transcribe audio/video via local Whisper. Graceful degradation chain.

    Priority: whisper CLI (fastest, C++ binary) -> whisper Python package ->
    ffmpeg + whisper -> raw filename if nothing available.
    """
    title = os.path.basename(source) if os.path.exists(source) else source

    # Try whisper CLI (whisper.cpp or OpenAI whisper)
    transcript = _whisper_cli(source)
    if transcript:
        return title, transcript

    # Try Python whisper package
    transcript = _whisper_python(source)
    if transcript:
        return title, transcript

    return title, f"[Audio file detected but Whisper not available. Install: pip install openai-whisper]"


def _whisper_cli(source: str) -> str:
    """Try whisper CLI transcription. Returns empty string on failure."""
    if not os.path.isfile(source):
        return ""
    try:
        result = subprocess.run(
            ["whisper", source, "--model", "base", "--output_format", "txt",
             "--output_dir", "/tmp"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            txt_path = os.path.join("/tmp", os.path.splitext(os.path.basename(source))[0] + ".txt")
            if os.path.isfile(txt_path):
                with open(txt_path, "r") as f:
                    return f.read()[:8000]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def _whisper_python(source: str) -> str:
    """Try Python whisper package. Returns empty string on failure."""
    if not os.path.isfile(source):
        return ""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(source)
        return result.get("text", "")[:8000]
    except ImportError:
        return ""
    except Exception:
        return ""


def _extract_file(source: str) -> tuple[str, str]:
    """Extract from local file."""
    if source.endswith(".pdf"):
        return _extract_paper(source)
    try:
        with open(source, "r", errors="replace") as f:
            return os.path.basename(source), f.read(10000)
    except OSError:
        return source, ""


def _analyse(registry, source, stype, title, content) -> list[KnowledgeItem]:
    """Use LLM to extract structured knowledge."""
    if not registry or not content:
        return []
    models = [f"{p.name}/{p.models[0]}" for p in registry.list_providers() if p.models]
    if not models:
        return []
    prompt = (
        f"Extract knowledge from this {stype}.\nTitle: {title}\n"
        f"Content:\n{content[:3000]}\n\n"
        f"HAL is a standalone LLM-agnostic agentic runtime.\n"
        f"Output ONLY JSON: "
        f'[{{"concept":"...","description":"...","category":"technique|pattern|'
        f'api|architecture|threat|data","relevance":1-10,"actionable":true/false,'
        f'"hal_module":"target or empty"}}]'
    )
    try:
        resp = registry.chat(models[0], prompt, max_tokens=1500)
        return _parse_items(resp.text, source)
    except Exception:
        return []


def _strip_html(html: str) -> str:
    """Strip HTML to plain text."""
    t = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    t = re.sub(r"<style[^>]*>.*?</style>", "", t, flags=re.DOTALL)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _read_raw(path: str) -> tuple[str, str]:
    """Read printable text from binary file."""
    try:
        with open(path, "rb") as f:
            raw = f.read(50000)
        text = raw.decode("latin-1", errors="replace")
        parts = re.findall(r"[\x20-\x7E]{4,}", text)
        return os.path.basename(path), " ".join(parts)[:8000]
    except OSError:
        return path, ""


def _parse_items(text: str, source: str) -> list[KnowledgeItem]:
    """Parse LLM response into KnowledgeItem list."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(text[start:end + 1])
        return [
            KnowledgeItem(
                concept=i.get("concept", ""), description=i.get("description", ""),
                category=i.get("category", "technique"),
                relevance=int(i.get("relevance", 5)),
                actionable=bool(i.get("actionable", False)),
                hal_module=i.get("hal_module", ""), source=source,
            )
            for i in items if isinstance(i, dict) and "concept" in i
        ]
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def _auto_wikilink(items: list[KnowledgeItem]) -> list[str]:
    """Generate wikilinks from extracted concepts to existing KONO entries.

    Cross-references each concept against the KONO knowledge base. Returns
    a list of `[[concept]]` wikilink strings for concepts that match existing
    entries, enabling Obsidian-style graph navigation.
    """
    if not items:
        return []
    try:
        from lib.hal.kono_bridge import recall_for_query
    except ImportError:
        return []

    wikilinks: list[str] = []
    for item in items:
        if not item.concept:
            continue
        recall = recall_for_query(item.concept)
        if recall and len(recall) > 20:
            wikilinks.append(f"[[{item.concept}]]")
    return wikilinks


def _save_report(report: AbsorptionReport) -> None:
    """Persist absorption report."""
    os.makedirs(_ABSORPTIONS_DIR, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9-]", "-", report.source)[:50]
    path = os.path.join(_ABSORPTIONS_DIR, f"{safe}-{int(time.time())}.json")
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
