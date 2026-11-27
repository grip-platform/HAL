"""Decide whether two prompts are the same request, and whether two differ in structure (HAL#965).

A response cache may hand one prompt's answer to another only when both ask for the same
thing. Vector similarity cannot certify that: embeddings place opposites close together.
HAL#965 measured "undo my last commit but keep the changes" against "... and discard the
changes" at cosine 0.973 with nomic-embed-text, and 6 of 20 such near-misses cleared the old
0.92 threshold.

Two prompts are equivalent only when they are the same text, character for character. Nothing
is ignored, because every rule that ignored something was shown to merge two different
requests: case ("Open Foo.py" and "Open foo.py"), articles ("Explain a bug" and "Explain the
bug"), quote marks ("print('x')" and "print(x)"), punctuation ("x != 5" and "x = 5"),
compatibility forms ("10²" and "102"), whitespace at the ends (an indented Python line and the
same line unindented) and Unicode encoding (``len("é")`` is 1 with a precomposed é and 2 with
an e and a combining accent). A prompt in other words can still reuse an answer, but only
through the cache's verifier, which asks a local judge.

``structural_difference`` is the deterministic veto that runs before that judge. It can only
refuse, so it reads generously: it lower-cases, expands contractions and skips the request
words "please", "kindly" and "me" before it compares numbers, negations and word order. It
also refuses two prompts that differ only invisibly, in whitespace or Unicode encoding,
because a judge cannot reliably see such a difference.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_CONTRACTIONS = {
    "can't": "can not", "won't": "will not", "shan't": "shall not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
    "mustn't": "must not", "needn't": "need not",
    "what's": "what is", "that's": "that is", "it's": "it is", "there's": "there is",
    "here's": "here is", "who's": "who is", "where's": "where is", "how's": "how is",
    "let's": "let us", "i'm": "i am", "you're": "you are", "we're": "we are",
    "they're": "they are", "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "i will", "you'll": "you will", "we'll": "we will",
    "i'd": "i would", "you'd": "you would",
}
# Request words the veto skips: their presence never changes which answer is right.
_IGNORED_WORDS = frozenset({"please", "kindly", "me"})

_TYPOGRAPHIC = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_CONTRACTION_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b",
                             re.IGNORECASE)
# Sentence punctuation only: a mark counts as punctuation only before a space or the end, so
# "file.py", "v1.2", "host:port" and the operator in "x != 5" keep theirs.
_SENTENCE_PUNCT_RE = re.compile(r"[?!,;.:](?=\s|$)")
_QUOTE_RE = re.compile(r"[\"'`]")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]+")
_DIGITS_RE = re.compile(r"\d+")
_NEGATIONS = frozenset({"not", "no", "never", "without", "none", "nothing", "neither", "nor"})


def canonical_form(text: str) -> str:
    """The key two prompts share exactly when they are the same text; "" for a blank prompt."""
    return text if text.strip() else ""


def equivalent(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` are the same text under :func:`canonical_form`."""
    key = canonical_form(a)
    return bool(key) and key == canonical_form(b)


def normalise(text: str) -> str:
    """The veto's generous reading: lower case, contractions expanded, sentence punctuation gone.

    Every quote mark becomes one standalone marker, so where a quote falls still counts.
    """
    text = unicodedata.normalize("NFC", text).translate(_TYPOGRAPHIC)
    text = _CONTRACTION_RE.sub(lambda m: _CONTRACTIONS[m.group(1).casefold()], text)
    text = _SENTENCE_PUNCT_RE.sub(" ", text)
    text = _QUOTE_RE.sub(' " ', text)
    return " ".join(text.split()).casefold()


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(normalise(text)) if t not in _IGNORED_WORDS]


def structural_difference(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` differ in a way that always changes the answer.

    Four checks that need no model. On :func:`normalise`'s reading: the numbers differ (every
    run of digits counts, so "600s" differs from "60s", and "10²" from "102"), the count of
    negations differs, or the same words and quote marks appear in another order ("JSON to
    YAML" against "yaml to json"). On the raw text: the two differ only in whitespace or Unicode
    encoding, which a judge cannot reliably see. A verifier never gets to approve such a pair.
    """
    ta, tb = _tokens(a), _tokens(b)
    numbers = Counter(_DIGITS_RE.findall(" ".join(ta))) != Counter(_DIGITS_RE.findall(" ".join(tb)))
    negations = sum(t in _NEGATIONS for t in ta) != sum(t in _NEGATIONS for t in tb)
    reordered = ta != tb and Counter(ta) == Counter(tb)
    invisible = a != b and unicodedata.normalize("NFC", a).split() == unicodedata.normalize("NFC", b).split()
    return numbers or negations or reordered or invisible
