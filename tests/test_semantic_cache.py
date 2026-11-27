"""Tests for the semantic response cache: one answer per request, never a near-miss (HAL#965)."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from lib.hal.prompt_equivalence import canonical_form, equivalent, normalise, structural_difference
from lib.hal.semantic_cache import CacheLookup, SemanticCache

REPO = Path(__file__).resolve().parents[1]

# HAL#965's near-miss pairs: shared words, a different right answer. Under the old cosine
# design, nomic-embed-text returned a wrong answer for 6 of these at 0.92, and the hashed
# bag-of-words fallback for 2 or 3. Each must now miss.
NEAR_MISSES = [
    ("How do I restart the HAL gateway?", "How do I stop the HAL gateway?"),
    ("Show me the first 20 lines of the server log", "Show me the last 20 lines of the server log"),
    ("List all open pull requests in the repo", "List all closed pull requests in the repo"),
    ("Convert this JSON file to YAML", "Convert this YAML file to JSON"),
    ("How much disk space is left on this machine?", "How much memory is left on this machine?"),
    ("What is the default timeout for provider calls?", "What is the maximum timeout for provider calls?"),
    ("Write a Python function that reverses a string", "Write a Python function that reverses a list"),
    ("How do I undo my last git commit but keep the changes?",
     "How do I undo my last git commit and discard the changes?"),
    ("Which port does the HAL gateway listen on?", "Which port does the HAL server listen on?"),
    ("Translate 'good morning' into French", "Translate 'good morning' into German"),
    ("What time is it in Tokyo right now?", "What time is it in London right now?"),
    ("Rename the variable foo to bar in this file", "Rename the variable bar to foo in this file"),
    ("Add a test for the timeout guard", "Remove the test for the timeout guard"),
    ("Is the build passing on main?", "Is the build failing on main?"),
    ("Merge pull request 123", "Close pull request 123"),
    ("Count the lines of code in the lib folder", "Count the lines of code in the tests folder"),
    ("Increase the cache TTL to 600 seconds", "Decrease the cache TTL to 60 seconds"),
    ("Explain how to enable the semantic cache", "Explain how to disable the semantic cache"),
    ("Summarise the changes in the last commit", "Summarise the changes in the first commit"),
    ("Delete the temporary files in the scratch folder", "List the temporary files in the scratch folder"),
]


class _FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.t = start

    def advance(self, seconds: float):
        self.t += seconds

    def __call__(self) -> float:
        return self.t


class TestCanonicalForm(unittest.TestCase):
    def test_only_identical_text_is_equivalent(self):
        self.assertTrue(equivalent("What is Python?", "What is Python?"))
        self.assertTrue(equivalent("  indented\n", "  indented\n"))

    def test_every_difference_inside_a_prompt_counts(self):
        pairs = [
            ("What is Python?", "what is python?"),                         # case
            ("Open Foo.py", "Open foo.py"),                                 # case naming a file
            ("Open fooBar.py", "Open foobar.py"),                           # case after the first letter
            ("Explain a bug", "Explain the bug"),                           # article
            ('Search for "red blue" green', 'Search for red "blue green"'),  # quote position
            ("print('x')", "print(x)"),                                     # quotes in code
            ("show rows where x != 5", "show rows where x = 5"),            # operator
            ("swap a and b", "swap and b"),                                 # a bare identifier
            ("Restart the gateway!", "Restart the gateway?"),               # command or question
            ("What is 10\u00b2?", "What is 102?"),                          # NFKC would merge these
            ("Email me the report", "Email the report"),                    # who it goes to
            ("if x:\n    a()\n    b()", "if x:\n    a()\nb()"),             # indentation in code
            ("    print('x')", "print('x')"),                               # indentation of one line
            ("  What is Python?\n", "What is Python?"),                     # whitespace at the ends
            ('len("\u00e9")', 'len("e\u0301")'),                            # Unicode encoding: 1 against 2
            ("Don't merge it", "Do not merge it"),                          # other words: the judge's call
        ]
        for a, b in pairs:
            self.assertFalse(equivalent(a, b), (a, b))

    def test_a_blank_prompt_has_no_canonical_form(self):
        self.assertEqual(canonical_form(" \n\t"), "")
        self.assertFalse(equivalent("", ""))
        self.assertFalse(equivalent("  ", "\n"))

    def test_canonical_form_is_the_same_in_every_process(self):
        """The old bag-of-words embedder used the per-process salted hash(); this must not."""
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from lib.hal.prompt_equivalence import canonical_form as c; "
            "print('|'.join(c(a) + '~' + c(b) for a, b in " + repr(NEAR_MISSES[:5]) + "))"
        )
        outputs = {
            subprocess.run([sys.executable, "-c", code, str(REPO)], capture_output=True, text=True,
                           check=True, env={"PYTHONHASHSEED": seed}).stdout
            for seed in ("1", "2", "3")
        }
        self.assertEqual(len(outputs), 1, outputs)


class TestStructuralVeto(unittest.TestCase):
    def test_the_veto_reads_generously(self):
        self.assertEqual(normalise("What's the timeout?"), "what is the timeout")
        self.assertEqual(normalise("Don’t merge it!"), "do not merge it")

    def test_the_veto_catches_numbers_negations_and_order(self):
        pairs = [
            ("Set the TTL to 600s", "Set the TTL to 60s"),
            ("What is 10²?", "What is 102?"),
            ("Is it passing?", "Is it not passing?"),
            ("Don't delete the branch", "Delete the branch"),
            ("Convert this JSON file to YAML", "convert this yaml file to json"),
            ('Search for "red blue" green', 'Search for red "blue green"'),
        ]
        for a, b in pairs:
            self.assertTrue(structural_difference(a, b), (a, b))

    def test_the_veto_refuses_differences_a_judge_cannot_see(self):
        pairs = [
            ("    print('x')", "print('x')"),
            ('len("é")', 'len("é")'),
            ("What is Python?\n", "What is Python?"),
        ]
        for a, b in pairs:
            self.assertTrue(structural_difference(a, b), (a, b))

    def test_the_veto_lets_other_wording_through_to_the_judge(self):
        pairs = [
            ("Show the last 20 lines", "Print the final 20 lines"),
            ("What's the timeout?", "What is the timeout"),
            ("Please show me the logs", "show the logs"),
            ("What is Python?", "what is python"),
        ]
        for a, b in pairs:
            self.assertFalse(structural_difference(a, b), (a, b))


class TestSemanticCache(unittest.TestCase):
    def setUp(self):
        self.clock = _FakeClock()
        self.cache = SemanticCache(ttl_seconds=300, max_entries=3, clock=self.clock)

    def test_exact_repeat_hits(self):
        self.cache.put("m", "What is Python?", "a language")
        response, lookup = self.cache.peek("m", "What is Python?")
        self.assertEqual(response, "a language")
        self.assertEqual(lookup, CacheLookup(hit=True, match="exact", matched_prompt="What is Python?"))

    def test_other_text_misses_without_a_judge(self):
        self.cache.put("m", "Show the logs", "here are the logs")
        self.assertIsNone(self.cache.get("m", "Please show me the logs"))
        self.assertIsNone(self.cache.get("m", "show the logs"))
        self.assertIsNone(self.cache.get("m", "Show the logs\n"))

    def test_every_hal965_near_miss_misses(self):
        for stored, asked in NEAR_MISSES:
            cache = SemanticCache(clock=self.clock)
            cache.put("m", stored, "answer for: " + stored)
            self.assertIsNone(cache.get("m", asked), (stored, asked))

    def test_a_paraphrase_in_other_words_misses_without_a_judge(self):
        self.cache.put("m", "Show me the last 20 lines of the server log", "...")
        self.assertIsNone(self.cache.get("m", "Print the final twenty lines of the server log"))

    def test_model_scope_isolation(self):
        self.cache.put("model-a", "What is Python?", "from a")
        self.assertIsNone(self.cache.get("model-b", "What is Python?"))
        self.assertEqual(self.cache.get("model-a", "What is Python?"), "from a")

    def test_ttl_expiry(self):
        self.cache.put("m", "What is Python?", "a language")
        self.clock.advance(301)
        self.assertIsNone(self.cache.get("m", "What is Python?"))
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_max_entries_fifo_eviction(self):
        for i in range(4):
            self.cache.put("m", f"question {i}", f"answer {i}")
        self.assertIsNone(self.cache.get("m", "question 0"))
        self.assertEqual(self.cache.get("m", "question 3"), "answer 3")
        self.assertEqual(self.cache.stats()["entries"], 3)

    def test_putting_the_same_request_again_replaces_its_answer(self):
        self.cache.put("m", "What is Python?", "old")
        self.cache.put("m", "What is Python?", "new")
        self.assertEqual(self.cache.get("m", "What is Python?"), "new")
        self.assertEqual(self.cache.stats()["entries"], 1)

    def test_invalidate_removes_only_the_same_prompt(self):
        self.cache.put("m", "Show the logs", "logs")
        self.assertFalse(self.cache.invalidate("m", "show the logs"))
        self.assertFalse(self.cache.invalidate("m", " Show the logs\n"))
        self.assertTrue(self.cache.invalidate("m", "Show the logs"))
        self.assertIsNone(self.cache.get("m", "Show the logs"))
        self.assertFalse(self.cache.invalidate("m", "Show the logs"))

    def test_clear(self):
        self.cache.put("m", "a question", "an answer")
        self.cache.get("m", "a question")
        self.assertEqual(self.cache.clear(), 1)
        stats = self.cache.stats()
        self.assertEqual((stats["entries"], stats["total_hits"]), (0, 0))

    def test_stats_track_hit_miss_rate(self):
        self.cache.put("m", "a question", "an answer")
        self.cache.get("m", "a question")
        self.cache.get("m", "another question")
        stats = self.cache.stats()
        self.assertEqual((stats["total_hits"], stats["total_misses"]), (1, 1))
        self.assertAlmostEqual(stats["hit_rate"], 0.5)
        self.assertEqual(stats["matcher"], "prompt_equivalence")

    def test_peek_on_a_miss_returns_a_lookup(self):
        self.assertEqual(self.cache.peek("m", "unseen"),
                         (None, CacheLookup(hit=False, match=None, matched_prompt=None)))

    def test_a_blank_prompt_is_never_stored(self):
        self.cache.put("m", "  \n", "anything")
        self.assertEqual(self.cache.stats()["entries"], 0)
        self.assertIsNone(self.cache.get("m", ""))

    def test_invalid_max_entries_rejected(self):
        with self.assertRaises(ValueError):
            SemanticCache(max_entries=0)

    def test_construction_defaults(self):
        stats = SemanticCache().stats()
        self.assertEqual((stats["ttl_seconds"], stats["max_entries"]), (300, 100))


class _FakeVerifier:
    """Canned embeddings and a canned judge, recording every text embedded and pair judged."""

    name = "fake"

    def __init__(self, vectors: dict, same: set):
        self.vectors = vectors
        self.same = same
        self.asked: list = []
        self.embedded: list = []

    def embed(self, text):
        self.embedded.append(text)
        return self.vectors.get(text)

    def same_request(self, a, b):
        self.asked.append((a, b))
        return frozenset((a, b)) in self.same


STORED = "Show me the last 20 lines of the server log"
PARAPHRASE = "Print the final 20 lines of the server log"
ORIGIN, CLOSE, FAR = [1.0, 0.0], [0.9, 0.4358898943540674], [0.5, 0.8660254037844386]


class TestVerifiedTier(unittest.TestCase):
    def _cache(self, vectors, same=(), clock=None):
        self.verifier = _FakeVerifier(vectors, {frozenset(p) for p in same})
        return SemanticCache(verifier=self.verifier, clock=clock or _FakeClock())

    def test_a_confirmed_paraphrase_hits_as_verified(self):
        cache = self._cache({STORED: ORIGIN, PARAPHRASE: CLOSE}, same=[(PARAPHRASE, STORED)])
        cache.put("m", STORED, "the lines")
        response, lookup = cache.peek("m", PARAPHRASE)
        self.assertEqual((response, lookup.match, lookup.matched_prompt), ("the lines", "verified", STORED))

    def test_a_paraphrase_the_judge_rejects_misses(self):
        cache = self._cache({STORED: ORIGIN, PARAPHRASE: CLOSE})
        cache.put("m", STORED, "the lines")
        self.assertIsNone(cache.get("m", PARAPHRASE))
        self.assertEqual(self.verifier.asked, [(PARAPHRASE, STORED)])

    def test_below_the_floor_the_judge_is_never_asked(self):
        cache = self._cache({STORED: ORIGIN, PARAPHRASE: FAR}, same=[(PARAPHRASE, STORED)])
        cache.put("m", STORED, "the lines")
        self.assertIsNone(cache.get("m", PARAPHRASE))
        self.assertEqual(self.verifier.asked, [])

    def test_the_veto_decides_before_the_judge(self):
        pairs = [("Raise the TTL to 600 seconds", "Raise the TTL to 60 seconds"),
                 ("Is the build passing?", "Is the build not passing?"),
                 ("Convert this JSON file to YAML", "Convert this YAML file to JSON")]
        for stored, asked in pairs:
            cache = self._cache({stored: ORIGIN, asked: CLOSE}, same=[(asked, stored)])
            cache.put("m", stored, "answer")
            self.assertIsNone(cache.get("m", asked), (stored, asked))
            self.assertEqual(self.verifier.asked, [], (stored, asked))

    def test_without_an_embedding_there_is_no_candidate(self):
        cache = self._cache({STORED: ORIGIN}, same=[(PARAPHRASE, STORED)])
        cache.put("m", STORED, "the lines")
        self.assertIsNone(cache.get("m", PARAPHRASE))
        self.assertEqual(self.verifier.asked, [])

    def test_the_closest_candidate_is_the_one_judged(self):
        other = "Show me the server log"
        cache = self._cache({STORED: ORIGIN, other: [0.6, 0.8], PARAPHRASE: CLOSE},
                            same=[(PARAPHRASE, STORED)])
        cache.put("m", other, "the whole log")
        cache.put("m", STORED, "the lines")
        self.assertEqual(cache.get("m", PARAPHRASE), "the lines")
        self.assertEqual(self.verifier.asked, [(PARAPHRASE, STORED)])

    def test_verified_hits_respect_model_scope_and_ttl(self):
        clock = _FakeClock()
        cache = self._cache({STORED: ORIGIN, PARAPHRASE: CLOSE}, same=[(PARAPHRASE, STORED)], clock=clock)
        cache.put("m", STORED, "the lines")
        self.assertIsNone(cache.get("other-model", PARAPHRASE))
        clock.advance(301)
        self.assertIsNone(cache.get("m", PARAPHRASE))
        self.assertEqual(self.verifier.asked, [])

    def test_stats_name_the_verifier_and_floor(self):
        stats = self._cache({}).stats()
        self.assertEqual((stats["verifier"], stats["candidate_floor"]), ("fake", 0.70))

    def test_invalid_candidate_floor_rejected(self):
        for floor in (0.0, 1.5):
            with self.assertRaises(ValueError):
                SemanticCache(candidate_floor=floor)

    def test_an_answer_invalidated_during_the_judge_call_is_not_served(self):
        cache = self._cache({STORED: ORIGIN, PARAPHRASE: CLOSE}, same=[(PARAPHRASE, STORED)])
        cache.put("m", STORED, "the lines")
        judge = self.verifier.same_request
        self.verifier.same_request = lambda a, b: (cache.invalidate("m", STORED), judge(a, b))[1]
        self.assertIsNone(cache.get("m", PARAPHRASE))
        self.assertEqual(self.verifier.asked, [(PARAPHRASE, STORED)])
        self.assertEqual(cache.stats()["total_misses"], 1)

    def test_a_stored_vector_of_another_size_is_never_a_candidate(self):
        cache = self._cache({STORED: [1.0, 0.0, 0.0], PARAPHRASE: CLOSE}, same=[(PARAPHRASE, STORED)])
        cache.put("m", STORED, "the lines")
        self.assertIsNone(cache.get("m", PARAPHRASE))
        self.assertEqual(self.verifier.asked, [])

    def test_no_embedding_call_when_nothing_could_match(self):
        cache = self._cache({STORED: ORIGIN, PARAPHRASE: CLOSE}, same=[(PARAPHRASE, STORED)])
        self.assertIsNone(cache.get("m", PARAPHRASE))  # empty cache
        cache.put("other-model", STORED, "the lines")
        self.assertIsNone(cache.get("m", PARAPHRASE))  # entries only for another model
        self.assertEqual((self.verifier.embedded, self.verifier.asked), ([STORED], []))

    def test_a_change_of_case_can_still_hit_through_the_judge(self):
        cache = self._cache({"What is Python?": ORIGIN, "what is python?": CLOSE},
                            same=[("what is python?", "What is Python?")])
        cache.put("m", "What is Python?", "a language")
        response, lookup = cache.peek("m", "what is python?")
        self.assertEqual((response, lookup.match), ("a language", "verified"))


# HAL#965's same-meaning pairs: one answer is right for both.
PARAPHRASES = [
    ("How do I restart the HAL gateway?", "What's the command to restart the HAL gateway service?"),
    ("Show me the last 20 lines of the server log", "Print the final twenty lines of the server log"),
    ("What does the semantic cache threshold do?", "Explain what the semantic cache's similarity threshold controls"),
    ("List all open pull requests in the repo", "Show every pull request that is still open in this repository"),
    ("Why is my pytest run failing with an import error?", "My pytest run fails on an import error, why?"),
    ("Convert this JSON file to YAML", "Turn this JSON file into YAML"),
    ("How much disk space is left on this machine?", "Check how much free disk space this machine has"),
    ("Summarise the changes in the last commit", "Give me a summary of what the most recent commit changed"),
    ("What is the default timeout for provider calls?", "What's the provider call timeout by default?"),
    ("Write a Python function that reverses a string", "Write a Python function to reverse a string"),
    ("How do I undo my last git commit but keep the changes?", "Undo the latest git commit while keeping my changes"),
    ("Which port does the HAL gateway listen on?", "What port is the HAL gateway listening on?"),
    ("Explain the difference between a list and a tuple in Python",
     "What's the difference between Python lists and tuples?"),
    ("Translate 'good morning' into French", "How do you say 'good morning' in French?"),
    ("Find all TODO comments in the codebase", "Search the codebase for every TODO comment"),
    ("What time is it in Tokyo right now?", "Current time in Tokyo?"),
    ("How do I create a virtual environment in Python?", "Create a Python virtual environment, how?"),
    ("Rename the variable foo to bar in this file", "In this file, rename foo to bar"),
    ("Is the remote server reachable over the VPN?", "Can I reach the remote server through the VPN?"),
    ("Count the lines of code in the lib folder", "How many lines of code are in lib?"),
]


@unittest.skipUnless(os.environ.get("HAL_TEST_LIVE") == "1", "live: needs HAL_TEST_LIVE=1 and a local Ollama")
class TestLiveVerifier(unittest.TestCase):
    """HAL#965's measurement as an executable anchor, against the real local judge.

    Measured on 2026-11-26 with qwen3:8b: 18 of 20 paraphrases hit, 0 of 20 near-misses in
    either direction. The paraphrase floor below is deliberately loose, because recall varies
    by model and by which prompt is stored first; the near-miss assertion is not loose.
    """

    @classmethod
    def setUpClass(cls):
        from lib.hal.cache_verifier import OllamaVerifier
        # A long timeout: this measures the judge's decisions, not its latency. At the 5 s
        # production timeout, one slow call rests the verifier and every later pair misses.
        cls.verifier = OllamaVerifier(os.environ.get("HAL_TEST_JUDGE", "qwen3:8b"), timeout_s=60.0)
        if not cls.verifier.warm():
            raise unittest.SkipTest("the local Ollama models did not load")

    def _hits(self, pairs):
        hits = []
        for stored, asked in pairs:
            cache = SemanticCache(verifier=self.verifier)
            cache.put("m", stored, "answer")
            if cache.get("m", asked) is not None:
                hits.append((stored, asked))
        return hits

    def test_no_near_miss_gets_an_answer_in_either_direction(self):
        self.assertEqual(self._hits(NEAR_MISSES) + self._hits([(b, a) for a, b in NEAR_MISSES]), [])

    def test_most_paraphrases_hit(self):
        hits = self._hits(PARAPHRASES)
        self.assertGreaterEqual(len(hits), 12, f"{len(hits)} of {len(PARAPHRASES)} paraphrases hit")


if __name__ == "__main__":
    unittest.main()
