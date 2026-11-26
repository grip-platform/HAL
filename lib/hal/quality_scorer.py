"""HAL Response Quality Scorer — rate LLM response quality automatically.

Evaluates responses on 4 dimensions: relevance, completeness, structure,
and conciseness. Each dimension scores 0-25, total 0-100.

Usage:
    from lib.hal.quality_scorer import QualityScorer

    scorer = QualityScorer()
    score = scorer.score(query="How do I sort?", response="Use sorted()...")
    print(f"Quality: {score.overall}/100 ({score.grade})")
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityScore:
    """Multi-dimensional quality score for an LLM response."""
    relevance: int  # 0-25: does the response address the query?
    completeness: int  # 0-25: are all parts addressed?
    structure: int  # 0-25: formatting quality
    conciseness: int  # 0-25: information density

    @property
    def overall(self) -> int:
        return self.relevance + self.completeness + self.structure + self.conciseness

    @property
    def grade(self) -> str:
        thresholds = ((90, "A"), (75, "B"), (60, "C"), (40, "D"))
        total = self.overall
        for threshold, letter in thresholds:
            if total >= threshold:
                return letter
        return "F"

    def to_dict(self) -> dict:
        return {
            "relevance": self.relevance, "completeness": self.completeness,
            "structure": self.structure, "conciseness": self.conciseness,
            "overall": self.overall, "grade": self.grade,
        }


class QualityScorer:
    """Mechanical response quality evaluation."""

    def score(self, query: str, response: str) -> QualityScore:
        """Score a response against its query."""
        return QualityScore(
            relevance=_score_relevance(query, response),
            completeness=_score_completeness(query, response),
            structure=_score_structure(response),
            conciseness=_score_conciseness(query, response),
        )


_WORD_RE = re.compile(r"\b\w{4,}\b")
_LIST_RE = re.compile(r"^[-*\d]+\.", re.MULTILINE)
_HEADER_RE = re.compile(r"^#+\s", re.MULTILINE)


def _word_overlap_ratio(query: str, response: str) -> float:
    """Ratio of query words found in response. O(n) via set intersection."""
    q_words = set(_WORD_RE.findall(query.lower()))
    if not q_words:
        return 0.5
    r_words = set(_WORD_RE.findall(response.lower()))
    return len(q_words & r_words) / len(q_words)


def _score_relevance(query: str, response: str) -> int:
    """Score relevance via word overlap. 0-25."""
    if not response.strip():
        return 0
    ratio = _word_overlap_ratio(query, response)
    if ratio >= 0.5:
        return 25
    if ratio >= 0.3:
        return 20
    if ratio >= 0.1:
        return 15
    return 5


def _score_completeness(query: str, response: str) -> int:
    """Score completeness by response substance per question. 0-25."""
    if not response.strip():
        return 0
    questions = max(query.count("?"), 1)
    words_per_q = len(response.split()) / questions
    if words_per_q >= 30:
        return 25
    if words_per_q >= 15:
        return 20
    if words_per_q >= 5:
        return 15
    return 5


def _score_structure(response: str) -> int:
    """Score formatting quality. 0-25."""
    score = 10
    if "```" in response:
        score += 5
    if _LIST_RE.search(response):
        score += 3
    if _HEADER_RE.search(response):
        score += 3
    if response.count("\n\n") >= 2:
        score += 2
    if "**" in response:
        score += 2
    return min(25, score)


def _score_conciseness(query: str, response: str) -> int:
    """Score information density by response-to-query length ratio. 0-25."""
    if not response.strip():
        return 0
    ratio = len(response.split()) / max(len(query.split()), 1)
    if 3 <= ratio <= 10:
        return 25
    if 2 <= ratio <= 20:
        return 20
    if 1 <= ratio <= 30:
        return 15
    return 10
