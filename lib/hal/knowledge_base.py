"""HAL Knowledge Base — Karpathy LLM wiki architecture.

raw/ → LLM-compiled wiki of .md articles → Q&A → outputs filed back.
The LLM maintains all wiki content; humans rarely write to it directly.

Key insight (Karpathy): auto-maintained index files + brief summaries
beat RAG at ~100-article, ~400K-word scale. No vector DB needed.

Directory layout:
    knowledge/
        raw/           # ingested source material (untouched by LLM)
        wiki/          # LLM-compiled .md articles
            index.md   # auto-maintained master index (all articles + summaries)
            concepts/  # one .md per concept
            sources/   # one .md per source document
        outputs/       # query outputs (markdown, Marp slides)
"""
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

_DEFAULT_KB_ROOT = os.path.join(os.path.expanduser("~/.hal"), "knowledge")


@dataclass
class WikiArticle:
    """A single compiled wiki article."""

    path: Path
    title: str
    summary: str = ""
    word_count: int = 0
    backlinks: list[str] = field(default_factory=list)
    last_updated: float = 0.0

    @classmethod
    def from_file(cls, path: Path) -> "WikiArticle":
        """Parse a wiki article from a .md file."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return cls(path=path, title=path.stem)
        title = path.stem.replace("-", " ").replace("_", " ").title()
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        summary = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                summary = stripped[:200]
                break
        backlinks = re.findall(r"\[\[([^\]]+)\]\]", text)
        word_count = len(text.split())
        stat = path.stat() if path.exists() else None
        return cls(
            path=path,
            title=title,
            summary=summary,
            word_count=word_count,
            backlinks=backlinks,
            last_updated=stat.st_mtime if stat else 0.0,
        )


@dataclass
class LintIssue:
    """A knowledge base health issue found during linting."""

    article: str
    issue_type: str      # "inconsistency" | "missing_data" | "broken_link" | "candidate"
    description: str
    severity: str = "info"  # "info" | "warning" | "error"


@dataclass
class KBResponse:
    """Response from a knowledge base query."""

    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    output_path: Path | None = None
    filed_back: bool = False


class KnowledgeBase:
    """HAL knowledge base implementing Karpathy's LLM wiki architecture.

    The LLM compiles and maintains all wiki content. Humans provide raw
    source material and queries; the LLM does the rest.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root or _DEFAULT_KB_ROOT)
        self._raw = self._root / "raw"
        self._wiki = self._root / "wiki"
        self._outputs = self._root / "outputs"
        self._concepts = self._wiki / "concepts"
        self._sources = self._wiki / "sources"
        self._index = self._wiki / "index.md"
        self._ensure_dirs()

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        """Create directory structure if not present."""
        for d in (self._raw, self._wiki, self._outputs, self._concepts, self._sources):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, source: str | Path, name: str | None = None) -> Path:
        """Copy source material into raw/. Accepts local file or URL stub.

        Returns the path to the ingested file.
        """
        source = Path(source)
        dest_name = name or source.name
        dest = self._raw / dest_name
        if source.exists():
            dest.write_bytes(source.read_bytes())
            _log.info("Ingested %s → %s", source, dest)
        else:
            dest.write_text(str(source), encoding="utf-8")
            _log.info("Ingested content stub → %s", dest)
        return dest

    def list_raw(self) -> list[Path]:
        """List all files in raw/."""
        return sorted(self._raw.iterdir()) if self._raw.exists() else []

    # ------------------------------------------------------------------
    # Wiki compilation
    # ------------------------------------------------------------------

    def compile(self, llm_fn=None) -> list[WikiArticle]:
        """LLM compiles raw/ → wiki/ articles.

        llm_fn: callable(prompt: str) -> str. If None, writes placeholder stubs.
        Returns the list of articles written.
        """
        raw_files = self.list_raw()
        if not raw_files:
            _log.info("No raw files to compile.")
            return []

        articles = []
        for raw_file in raw_files:
            article_path = self._sources / (raw_file.stem + ".md")
            content = self._compile_one(raw_file, llm_fn)
            article_path.write_text(content, encoding="utf-8")
            articles.append(WikiArticle.from_file(article_path))
            _log.info("Compiled %s → %s", raw_file.name, article_path.name)

        self._rebuild_index(articles)
        return articles

    def _compile_one(self, raw_file: Path, llm_fn) -> str:
        """Compile a single raw file to wiki markdown."""
        if llm_fn is None:
            return f"# {raw_file.stem}\n\n*Stub — run compile(llm_fn=...) to populate.*\n"
        prompt = (
            f"Summarise the following source document into a wiki article.\n"
            f"Include: title (# heading), 2-3 paragraph summary, key concepts, "
            f"backlinks to related concepts using [[ConceptName]] syntax.\n\n"
            f"Source: {raw_file.name}\n\n{self._read_raw(raw_file)}"
        )
        return llm_fn(prompt)

    def _read_raw(self, path: Path, max_chars: int = 8_000) -> str:
        """Read a raw file, truncating to max_chars."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return text[:max_chars]
        except OSError:
            return ""

    def _rebuild_index(self, articles: list[WikiArticle]) -> None:
        """Rewrite index.md with titles and one-line summaries of all articles."""
        lines = ["# HAL Knowledge Base Index\n", f"*{len(articles)} articles*\n\n"]
        for art in sorted(articles, key=lambda a: a.title):
            summary = art.summary[:100] + "…" if len(art.summary) > 100 else art.summary
            lines.append(f"- **{art.title}** — {summary}\n")
        self._index.write_text("".join(lines), encoding="utf-8")
        _log.info("Rebuilt index.md (%d articles)", len(articles))

    # ------------------------------------------------------------------
    # Article access
    # ------------------------------------------------------------------

    def articles(self) -> list[WikiArticle]:
        """Return all compiled wiki articles."""
        result = []
        for d in (self._sources, self._concepts):
            if d.exists():
                result.extend(WikiArticle.from_file(p) for p in sorted(d.glob("*.md")))
        return result

    def search(self, query: str) -> list[WikiArticle]:
        """Naive full-text search over wiki articles. Returns ranked results."""
        query_lower = query.lower()
        hits = []
        for article in self.articles():
            try:
                text = article.path.read_text(encoding="utf-8")
            except OSError:
                continue
            count = text.lower().count(query_lower)
            if count > 0:
                hits.append((count, article))
        return [a for _, a in sorted(hits, key=lambda x: -x[0])]

    # ------------------------------------------------------------------
    # Q&A
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        llm_fn=None,
        file_back: bool = False,
        output_format: str = "markdown",
    ) -> KBResponse:
        """Answer a question against the wiki.

        llm_fn: callable(prompt: str) -> str.
        file_back: if True, files the output back into wiki/outputs/ as a new article.
        output_format: "markdown" | "marp" (Marp slide deck format).
        """
        context = self._build_query_context(question)
        answer = self._run_query(question, context, llm_fn, output_format)
        out_path = self._write_output(question, answer)
        filed_back = self._maybe_file_back(question, answer, file_back, llm_fn)
        return KBResponse(
            question=question,
            answer=answer,
            sources=[str(a.path) for a in self.search(question)[:5]],
            output_path=out_path,
            filed_back=filed_back,
        )

    def _run_query(self, question: str, context: str, llm_fn, output_format: str) -> str:
        """Generate an answer using the LLM or return a stub."""
        if llm_fn is None:
            return f"*No LLM provided. Context collected ({len(context)} chars).*"
        format_hint = (
            "Format the output as a Marp slide deck (---\\n# Title\\n\\n* bullet)."
            if output_format == "marp"
            else "Format the output as clean Markdown."
        )
        prompt = (
            f"You are a research assistant with access to the following knowledge base.\n"
            f"{format_hint}\n\nKnowledge Base:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        return llm_fn(prompt)

    def _write_output(self, question: str, answer: str) -> Path:
        """Write query output to outputs/ and return the path."""
        slug = re.sub(r"[^a-z0-9]+", "-", question.lower())[:50]
        out_path = self._outputs / f"{slug}-{int(time.time())}.md"
        out_path.write_text(f"# Query: {question}\n\n{answer}\n", encoding="utf-8")
        return out_path

    def _maybe_file_back(self, question: str, answer: str, file_back: bool, llm_fn) -> bool:
        """Optionally file the query output back into the wiki as a concept article."""
        if not (file_back and llm_fn is not None):
            return False
        slug = re.sub(r"[^a-z0-9]+", "-", question.lower())[:50]
        filed_path = self._concepts / f"query-{slug}.md"
        filed_path.write_text(f"# {question}\n\n{answer}\n", encoding="utf-8")
        _log.info("Filed query output back into wiki: %s", filed_path.name)
        return True

    def _build_query_context(self, question: str) -> str:
        """Assemble context: index + top matching articles."""
        parts = []
        if self._index.exists():
            parts.append(f"## Index\n{self._index.read_text(encoding='utf-8')}\n")
        for article in self.search(question)[:5]:
            try:
                text = article.path.read_text(encoding="utf-8")[:2_000]
                parts.append(f"## {article.title}\n{text}\n")
            except OSError:
                pass
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Lint
    # ------------------------------------------------------------------

    def lint(self) -> list[LintIssue]:
        """Run health checks: broken backlinks, stubs, short articles."""
        all_articles = self.articles()
        titles = {a.title for a in all_articles}
        issues = []
        for article in all_articles:
            issues.extend(self._lint_article(article, titles))
        _log.info("Lint complete: %d issues in %d articles", len(issues), len(all_articles))
        return issues

    def _lint_article(self, article: WikiArticle, known_titles: set) -> list[LintIssue]:
        """Run all lint checks on a single article."""
        issues = []
        for backlink in article.backlinks:
            if backlink not in known_titles:
                issues.append(LintIssue(
                    article=article.title,
                    issue_type="broken_link",
                    description=f"Backlink [[{backlink}]] has no corresponding article.",
                    severity="warning",
                ))
        if 0 < article.word_count < 50:
            issues.append(LintIssue(
                article=article.title,
                issue_type="missing_data",
                description=f"Article is very short ({article.word_count} words). Needs expansion.",
                severity="info",
            ))
        try:
            if "*Stub" in article.path.read_text(encoding="utf-8"):
                issues.append(LintIssue(
                    article=article.title,
                    issue_type="missing_data",
                    description="Article is a stub — run compile(llm_fn=...) to populate.",
                    severity="warning",
                ))
        except OSError:
            pass
        return issues


def knowledge_stats(kb: "KnowledgeBase") -> dict:
    """Return statistics for a KnowledgeBase instance.

    Extracted as a module-level function to keep KnowledgeBase at ≤7 public
    methods (ISP compliance). Stats are derived, not behaviour — they belong
    outside the class interface.
    """
    all_articles = kb.articles()
    return {
        "raw_files": len(kb.list_raw()),
        "articles": len(all_articles),
        "total_words": sum(a.word_count for a in all_articles),
        "outputs": len(list(kb._outputs.glob("*.md"))) if kb._outputs.exists() else 0,
    }
