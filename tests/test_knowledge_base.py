"""Tests for lib.hal.knowledge_base — Karpathy LLM wiki architecture."""
import tempfile
import unittest
from pathlib import Path

from lib.hal.knowledge_base import KBResponse, KnowledgeBase, LintIssue, WikiArticle, knowledge_stats


def _kb(tmp: str) -> KnowledgeBase:
    return KnowledgeBase(root=tmp)


class TestDirectorySetup(unittest.TestCase):
    def test_dirs_created_on_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            root = Path(tmp)
            self.assertTrue((root / "raw").is_dir())
            self.assertTrue((root / "wiki").is_dir())
            self.assertTrue((root / "wiki" / "concepts").is_dir())
            self.assertTrue((root / "wiki" / "sources").is_dir())
            self.assertTrue((root / "outputs").is_dir())


class TestIngest(unittest.TestCase):
    def test_ingest_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            src = Path(tmp) / "source.txt"
            src.write_text("hello world", encoding="utf-8")
            dest = kb.ingest(src)
            self.assertTrue(dest.exists())
            self.assertEqual(dest.read_text(encoding="utf-8"), "hello world")

    def test_ingest_nonexistent_path_writes_stub(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            dest = kb.ingest("/nonexistent/file.txt", name="stub.txt")
            self.assertTrue(dest.exists())

    def test_ingest_custom_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            src = Path(tmp) / "a.txt"
            src.write_text("data", encoding="utf-8")
            dest = kb.ingest(src, name="renamed.txt")
            self.assertEqual(dest.name, "renamed.txt")

    def test_ingest_returns_path_in_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            src = Path(tmp) / "x.md"
            src.write_text("# X", encoding="utf-8")
            dest = kb.ingest(src)
            self.assertIn("raw", str(dest))


class TestListRaw(unittest.TestCase):
    def test_empty_kb(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            self.assertEqual(kb.list_raw(), [])

    def test_lists_ingested_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            src = Path(tmp) / "doc.txt"
            src.write_text("content", encoding="utf-8")
            kb.ingest(src)
            raw = kb.list_raw()
            self.assertEqual(len(raw), 1)
            self.assertEqual(raw[0].name, "doc.txt")

    def test_lists_multiple_files_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            for name in ("c.txt", "a.txt", "b.txt"):
                f = Path(tmp) / name
                f.write_text("x", encoding="utf-8")
                kb.ingest(f)
            names = [p.name for p in kb.list_raw()]
            self.assertEqual(names, sorted(names))


class TestCompile(unittest.TestCase):
    def test_compile_no_raw_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            self.assertEqual(kb.compile(), [])

    def test_compile_stub_without_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            src = Path(tmp) / "topic.txt"
            src.write_text("some content", encoding="utf-8")
            kb.ingest(src)
            articles = kb.compile()
            self.assertEqual(len(articles), 1)
            self.assertIn("Stub", articles[0].path.read_text(encoding="utf-8"))

    def test_compile_with_llm_fn(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            src = Path(tmp) / "topic.txt"
            src.write_text("raw text", encoding="utf-8")
            kb.ingest(src)
            articles = kb.compile(llm_fn=lambda p: "# Generated\n\nBody text here.")
            self.assertEqual(len(articles), 1)
            self.assertIn("Generated", articles[0].path.read_text(encoding="utf-8"))

    def test_compile_writes_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            src = Path(tmp) / "topic.txt"
            src.write_text("content", encoding="utf-8")
            kb.ingest(src)
            kb.compile()
            index = Path(tmp) / "wiki" / "index.md"
            self.assertTrue(index.exists())

    def test_compile_multiple_raw_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            for i in range(3):
                f = Path(tmp) / f"doc{i}.txt"
                f.write_text(f"doc {i}", encoding="utf-8")
                kb.ingest(f)
            articles = kb.compile()
            self.assertEqual(len(articles), 3)

    def test_compile_returns_wiki_articles(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            src = Path(tmp) / "x.txt"
            src.write_text("text", encoding="utf-8")
            kb.ingest(src)
            articles = kb.compile()
            self.assertIsInstance(articles[0], WikiArticle)


class TestArticles(unittest.TestCase):
    def test_articles_empty_before_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            self.assertEqual(kb.articles(), [])

    def test_articles_after_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            src = Path(tmp) / "a.txt"
            src.write_text("content", encoding="utf-8")
            kb.ingest(src)
            kb.compile()
            self.assertEqual(len(kb.articles()), 1)

    def test_articles_includes_manually_placed_concept(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            concept = Path(tmp) / "wiki" / "concepts" / "test-concept.md"
            concept.write_text("# Test Concept\n\nSome body text.", encoding="utf-8")
            arts = kb.articles()
            self.assertEqual(len(arts), 1)
            self.assertEqual(arts[0].title, "Test Concept")

    def test_articles_returns_wiki_article_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            src = Path(tmp) / "x.txt"
            src.write_text("x", encoding="utf-8")
            kb.ingest(src)
            kb.compile()
            for art in kb.articles():
                self.assertIsInstance(art, WikiArticle)


class TestSearch(unittest.TestCase):
    def test_search_empty_kb(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            self.assertEqual(kb.search("anything"), [])

    def test_search_finds_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            concept = Path(tmp) / "wiki" / "concepts" / "python.md"
            concept.write_text("# Python\n\nPython is a language.", encoding="utf-8")
            results = kb.search("python")
            self.assertEqual(len(results), 1)

    def test_search_no_match_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            concept = Path(tmp) / "wiki" / "concepts" / "python.md"
            concept.write_text("# Python\n\nPython is a language.", encoding="utf-8")
            results = kb.search("javascript")
            self.assertEqual(results, [])

    def test_search_ranks_by_frequency(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            low = Path(tmp) / "wiki" / "concepts" / "low.md"
            high = Path(tmp) / "wiki" / "concepts" / "high.md"
            low.write_text("# Low\n\nfoo mentioned once.", encoding="utf-8")
            high.write_text("# High\n\nfoo foo foo foo foo.", encoding="utf-8")
            results = kb.search("foo")
            self.assertEqual(results[0].title, "High")

    def test_search_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            concept = Path(tmp) / "wiki" / "concepts" / "rust.md"
            concept.write_text("# Rust\n\nRust is fast.", encoding="utf-8")
            self.assertEqual(len(kb.search("RUST")), 1)
            self.assertEqual(len(kb.search("rust")), 1)


class TestQuery(unittest.TestCase):
    def test_query_no_llm_returns_stub(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            resp = kb.query("What is HAL?")
            self.assertIsInstance(resp, KBResponse)
            self.assertIn("No LLM", resp.answer)

    def test_query_with_llm_fn(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            resp = kb.query("What is X?", llm_fn=lambda p: "X is a thing.")
            self.assertEqual(resp.answer, "X is a thing.")

    def test_query_writes_output_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            resp = kb.query("test question")
            self.assertIsNotNone(resp.output_path)
            self.assertTrue(resp.output_path.exists())

    def test_query_filed_back_false_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            resp = kb.query("Q?", llm_fn=lambda p: "A.")
            self.assertFalse(resp.filed_back)

    def test_query_file_back_creates_concept_article(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            resp = kb.query("What is X?", llm_fn=lambda p: "X is Y.", file_back=True)
            self.assertTrue(resp.filed_back)
            concepts = list((Path(tmp) / "wiki" / "concepts").glob("*.md"))
            self.assertEqual(len(concepts), 1)

    def test_query_file_back_requires_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            resp = kb.query("Q?", llm_fn=None, file_back=True)
            self.assertFalse(resp.filed_back)

    def test_query_sources_populated_when_articles_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            concept = Path(tmp) / "wiki" / "concepts" / "hal.md"
            concept.write_text("# HAL\n\nHAL is an AI.", encoding="utf-8")
            resp = kb.query("hal")
            self.assertTrue(len(resp.sources) > 0)

    def test_query_marp_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            called_with = []
            def llm(p):
                called_with.append(p)
                return "slides"
            kb.query("Q?", llm_fn=llm, output_format="marp")
            self.assertIn("Marp", called_with[0])

    def test_query_response_question_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            resp = kb.query("What is HAL?")
            self.assertEqual(resp.question, "What is HAL?")


class TestLint(unittest.TestCase):
    def test_lint_empty_kb_no_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            self.assertEqual(kb.lint(), [])

    def test_lint_detects_broken_backlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            concept = Path(tmp) / "wiki" / "concepts" / "alpha.md"
            concept.write_text("# Alpha\n\nSee [[Nonexistent]] for more.", encoding="utf-8")
            issues = kb.lint()
            types = [i.issue_type for i in issues]
            self.assertIn("broken_link", types)

    def test_lint_no_issue_for_valid_backlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            a = Path(tmp) / "wiki" / "concepts" / "alpha.md"
            b = Path(tmp) / "wiki" / "concepts" / "beta.md"
            a.write_text("# Alpha\n\nSee [[Beta]] for more.", encoding="utf-8")
            b.write_text("# Beta\n\nBeta content here.", encoding="utf-8")
            issues = kb.lint()
            broken = [i for i in issues if i.issue_type == "broken_link"]
            self.assertEqual(broken, [])

    def test_lint_detects_stub_article(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            src = Path(tmp) / "x.txt"
            src.write_text("content", encoding="utf-8")
            kb.ingest(src)
            kb.compile()  # no llm_fn -> writes stub
            issues = kb.lint()
            stubs = [i for i in issues if "stub" in i.description.lower()]
            self.assertTrue(len(stubs) > 0)

    def test_lint_detects_short_article(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            concept = Path(tmp) / "wiki" / "concepts" / "tiny.md"
            concept.write_text("# Tiny\n\nShort.", encoding="utf-8")
            issues = kb.lint()
            short = [i for i in issues if i.issue_type == "missing_data" and "short" in i.description]
            self.assertTrue(len(short) > 0)

    def test_lint_returns_lint_issue_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            concept = Path(tmp) / "wiki" / "concepts" / "a.md"
            concept.write_text("# A\n\n[[Ghost]]", encoding="utf-8")
            for issue in kb.lint():
                self.assertIsInstance(issue, LintIssue)

    def test_lint_issue_has_severity(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            concept = Path(tmp) / "wiki" / "concepts" / "a.md"
            concept.write_text("# A\n\n[[Ghost]]", encoding="utf-8")
            issues = kb.lint()
            self.assertTrue(all(i.severity in ("info", "warning", "error") for i in issues))


class TestStats(unittest.TestCase):
    def test_stats_empty_kb(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            s = knowledge_stats(kb)
            self.assertEqual(s["raw_files"], 0)
            self.assertEqual(s["articles"], 0)
            self.assertEqual(s["total_words"], 0)
            self.assertEqual(s["outputs"], 0)

    def test_stats_after_ingest(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            src = Path(tmp) / "doc.txt"
            src.write_text("content", encoding="utf-8")
            kb.ingest(src)
            self.assertEqual(knowledge_stats(kb)["raw_files"], 1)

    def test_stats_after_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            src = Path(tmp) / "doc.txt"
            src.write_text("content", encoding="utf-8")
            kb.ingest(src)
            kb.compile()
            self.assertEqual(knowledge_stats(kb)["articles"], 1)

    def test_stats_total_words_nonzero_after_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            src = Path(tmp) / "doc.txt"
            src.write_text("content", encoding="utf-8")
            kb.ingest(src)
            kb.compile(llm_fn=lambda p: "# Topic\n\nThis is a meaningful article with many words.")
            self.assertGreater(knowledge_stats(kb)["total_words"], 0)

    def test_stats_outputs_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            kb.query("Q1")
            kb.query("Q2")
            self.assertEqual(knowledge_stats(kb)["outputs"], 2)

    def test_stats_keys_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp)
            s = knowledge_stats(kb)
            for key in ("raw_files", "articles", "total_words", "outputs"):
                self.assertIn(key, s)


class TestWikiArticle(unittest.TestCase):
    def test_from_file_parses_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "my-article.md"
            p.write_text("# My Article Title\n\nBody.", encoding="utf-8")
            art = WikiArticle.from_file(p)
            self.assertEqual(art.title, "My Article Title")

    def test_from_file_fallback_title_from_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "my-article.md"
            p.write_text("No heading here.", encoding="utf-8")
            art = WikiArticle.from_file(p)
            self.assertEqual(art.title, "My Article")

    def test_from_file_extracts_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.md"
            p.write_text("# Title\n\nThis is the summary line.", encoding="utf-8")
            art = WikiArticle.from_file(p)
            self.assertIn("summary", art.summary.lower())

    def test_from_file_extracts_backlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.md"
            p.write_text("# A\n\nSee [[ConceptX]] and [[ConceptY]].", encoding="utf-8")
            art = WikiArticle.from_file(p)
            self.assertIn("ConceptX", art.backlinks)
            self.assertIn("ConceptY", art.backlinks)

    def test_from_file_word_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.md"
            p.write_text("# Title\n\none two three four five", encoding="utf-8")
            art = WikiArticle.from_file(p)
            self.assertGreater(art.word_count, 0)

    def test_from_file_missing_file_returns_default(self):
        art = WikiArticle.from_file(Path("/nonexistent/path.md"))
        self.assertEqual(art.title, "path")

    def test_from_file_last_updated_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.md"
            p.write_text("# A\n\nContent.", encoding="utf-8")
            art = WikiArticle.from_file(p)
            self.assertGreater(art.last_updated, 0.0)


class TestLintIssue(unittest.TestCase):
    def test_default_severity_info(self):
        issue = LintIssue(article="A", issue_type="broken_link", description="desc")
        self.assertEqual(issue.severity, "info")

    def test_custom_severity(self):
        issue = LintIssue(article="A", issue_type="broken_link", description="desc", severity="error")
        self.assertEqual(issue.severity, "error")


class TestKBResponse(unittest.TestCase):
    def test_defaults(self):
        resp = KBResponse(question="Q", answer="A")
        self.assertEqual(resp.sources, [])
        self.assertIsNone(resp.output_path)
        self.assertFalse(resp.filed_back)

    def test_no_kblinter_import(self):
        """KBLinter must not exist — lint() lives on KnowledgeBase."""
        import lib.hal.knowledge_base as mod
        self.assertFalse(hasattr(mod, "KBLinter"))


if __name__ == "__main__":
    unittest.main()
