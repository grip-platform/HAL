"""Evidence appendix for the AGORA judge (reviewer-reliability fix).

Four documented PRs (HAL #501/#503/#505/#512) had REQUEST_CHANGES verdicts
built on speaker concerns whose OWN cited file:line context refuted them —
the judge saw only the speakers' prose, and prompt rule 3 made any
well-formed citation force the verdict. The judge prompt now carries the
actual code at every cited line, and rule 6 instructs it to refute
contradicted citations.

Deterministic — no network; the dispatch test stubs the transport seam.
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from lib.agora_pr_review import (
    _EVIDENCE_MAX_CHARS,
    _EVIDENCE_MAX_REFS,
    _JUDGE_PROMPT_TEMPLATE,
    _build_evidence_appendix,
    _dispatch_judge,
    _extract_cited_refs,
)


# --- extraction --------------------------------------------------------------

def test_extracts_and_dedupes_in_order():
    text_a = "Bug in lib/foo.py:42 and again lib/foo.py:42; also config/bar.yaml:7."
    text_c = "See lib/foo.py:42 plus scripts/run.sh:3."
    assert _extract_cited_refs(text_a, "", text_c) == [
        ("lib/foo.py", 42),
        ("config/bar.yaml", 7),
        ("scripts/run.sh", 3),
    ]


def test_extraction_ignores_non_file_colon_numbers():
    # Model tags and timestamps must not read files.
    refs = _extract_cited_refs("use qwen2.5:7b at 12:30, ratio 3:1")
    assert refs == []


def test_extraction_caps_ref_count():
    text = " ".join(f"lib/f{i}.py:{i + 1}" for i in range(30))
    assert len(_extract_cited_refs(text)) == _EVIDENCE_MAX_REFS


# --- appendix build ----------------------------------------------------------

@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    target = tmp_path / "lib" / "sample.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "\n".join(f"line {n}" for n in range(1, 31)), encoding="utf-8")
    return tmp_path


def test_appendix_marks_cited_line_with_context(repo: Path):
    out = _build_evidence_appendix([("lib/sample.py", 15)], repo)
    assert "--- lib/sample.py:15 ---" in out
    assert ">>   15 | line 15" in out
    assert "    7 | line 7" in out and "   23 | line 23" in out
    assert "line 6" not in out and "line 24" not in out


def test_appendix_clamps_beyond_eof(repo: Path):
    out = _build_evidence_appendix([("lib/sample.py", 999)], repo)
    assert "beyond end-of-file at 30" in out
    assert ">>   30 | line 30" in out


def test_appendix_missing_file_is_unverified(repo: Path):
    out = _build_evidence_appendix([("lib/ghost.py", 5)], repo)
    assert "FILE NOT FOUND" in out and "unverified" in out


def test_appendix_rejects_traversal_and_absolute(repo: Path):
    # Speaker text is untrusted content driving file reads — citations must
    # never escape the checkout (untrusted-content-capability-lock).
    out = _build_evidence_appendix(
        [("../../etc/passwd.txt", 1), ("/etc/hosts.txt", 1)], repo)
    assert out.count("INVALID PATH") == 2
    assert "root:" not in out


def test_appendix_empty_refs_sentinel(repo: Path):
    out = _build_evidence_appendix([], repo)
    assert "no file:line citations" in out


def test_appendix_truncates_at_size_cap_and_reports_dropped_count(repo: Path):
    big = repo / "lib" / "big.py"
    big.write_text("\n".join("x" * 120 for _ in range(400)), encoding="utf-8")
    refs = [("lib/big.py", n * 20) for n in range(1, 13)]
    out = _build_evidence_appendix(refs, repo)
    assert len(out) <= _EVIDENCE_MAX_CHARS + 120
    assert "characters dropped" in out and f"{_EVIDENCE_MAX_CHARS}-character cap" in out
    # The count itself must be a real number, not a placeholder.
    import re as _re
    m = _re.search(r"truncated: (\d+) characters dropped", out)
    assert m and int(m.group(1)) > 0


def test_appendix_rejects_symlink_escaping_checkout(repo: Path, tmp_path_factory):
    # A symlink INSIDE the repo pointing OUTSIDE must not leak content:
    # _cited_block resolves BEFORE the containment check, so the resolved
    # target lands outside root and renders INVALID PATH.
    outside = tmp_path_factory.mktemp("outside") / "secret.py"
    outside.write_text("TOP-SECRET-CONTENT\n", encoding="utf-8")
    link = repo / "lib" / "sneaky.py"
    link.symlink_to(outside)
    out = _build_evidence_appendix([("lib/sneaky.py", 1)], repo)
    assert "INVALID PATH" in out
    assert "TOP-SECRET-CONTENT" not in out


# --- prompt threading --------------------------------------------------------

def test_template_carries_appendix_placeholder_and_rule6():
    assert "{evidence_appendix}" in _JUDGE_PROMPT_TEMPLATE
    assert "EVIDENCE APPENDIX" in _JUDGE_PROMPT_TEMPLATE
    assert "REFUTED" in _JUDGE_PROMPT_TEMPLATE


def test_dispatch_judge_threads_real_code_into_prompt(repo: Path):
    captured = {}

    def fake_call(envelope, hal_bin, timeout):
        captured["prompt"] = envelope.get("content", "")
        return ""  # empty judge output -> fail-closed JudgeOutput path

    with patch("lib.agora_pr_review._call_happi", side_effect=fake_call):
        _dispatch_judge(
            "test PR",
            "Speaker A prose.",
            "Speaker B prose.",
            "Concern at lib/sample.py:15 — missing fallback.",
            repo_root=repo,
        )
    prompt = captured["prompt"]
    assert ">>   15 | line 15" in prompt, "judge must see the actual cited code"
    assert "EVIDENCE APPENDIX" in prompt


def test_dispatch_judge_no_citations_gets_sentinel(repo: Path):
    captured = {}

    def fake_call(envelope, hal_bin, timeout):
        captured["prompt"] = envelope.get("content", "")
        return ""

    with patch("lib.agora_pr_review._call_happi", side_effect=fake_call):
        _dispatch_judge("t", "prose", "prose", "prose", repo_root=repo)
    assert "no file:line citations" in captured["prompt"]
