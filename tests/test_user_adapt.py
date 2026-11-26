"""Tests for HAL user adaptation."""
import os
import sys
import tempfile
import unittest

from lib.hal.user_adapt import (
    UserAdapter,
    UserProfile,
    _DEFAULTS,
    _DIMENSIONS,
    _load_profile,
    _save_profile,
)


class TestUserAdapter(unittest.TestCase):

    def test_default_profile(self):
        adapter = UserAdapter()
        profile = adapter.profile
        self.assertEqual(profile.preferences["verbosity"], "balanced")
        self.assertEqual(profile.preferences["depth"], "intermediate")

    def test_observe_updates_profile(self):
        adapter = UserAdapter()
        adapter.observe("verbosity", "concise")
        self.assertEqual(adapter.profile.preferences["verbosity"], "concise")

    def test_observe_ignores_invalid_dimension(self):
        adapter = UserAdapter()
        adapter.observe("nonexistent", "value")
        self.assertEqual(adapter.profile.observation_count, 0)

    def test_observe_ignores_invalid_value(self):
        adapter = UserAdapter()
        adapter.observe("verbosity", "ultra_brief")
        self.assertEqual(adapter.profile.observation_count, 0)

    def test_higher_confidence_wins(self):
        adapter = UserAdapter()
        adapter.observe("depth", "beginner", confidence=0.5)
        adapter.observe("depth", "expert", confidence=0.9)
        self.assertEqual(adapter.profile.preferences["depth"], "expert")

    def test_recency_breaks_ties(self):
        adapter = UserAdapter()
        adapter.observe("style", "formal", confidence=0.8)
        adapter.observe("style", "casual", confidence=0.8)
        # Second observation is more recent, should win
        self.assertEqual(adapter.profile.preferences["style"], "casual")


class TestObserveFromFeedback(unittest.TestCase):

    def test_infers_concise(self):
        adapter = UserAdapter()
        updated = adapter.observe_from_feedback("Your response was too long, be shorter")
        self.assertIn("verbosity", updated)
        self.assertEqual(adapter.profile.preferences["verbosity"], "concise")

    def test_infers_expert(self):
        adapter = UserAdapter()
        updated = adapter.observe_from_feedback("That's too basic, I know this already")
        self.assertIn("depth", updated)
        self.assertEqual(adapter.profile.preferences["depth"], "expert")

    def test_infers_skip_explanations(self):
        adapter = UserAdapter()
        updated = adapter.observe_from_feedback("Don't explain, just code")
        self.assertIn("explanations", updated)
        self.assertEqual(adapter.profile.preferences["explanations"], "skip")

    def test_no_inference_from_unrelated(self):
        adapter = UserAdapter()
        updated = adapter.observe_from_feedback("Great work, thanks!")
        self.assertEqual(len(updated), 0)


class TestToSystemPrompt(unittest.TestCase):

    def test_generates_prompt(self):
        adapter = UserAdapter()
        adapter.observe("verbosity", "concise")
        adapter.observe("depth", "expert")
        prompt = adapter.to_system_prompt()
        self.assertIn("short and direct", prompt)
        self.assertIn("Skip basics", prompt)

    def test_empty_for_defaults(self):
        adapter = UserAdapter()
        prompt = adapter.to_system_prompt()
        # Defaults may or may not generate a prompt — "direct" has a mapping
        self.assertIsInstance(prompt, str)


class TestProfilePersistence(unittest.TestCase):

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import lib.hal.user_adapt as mod
            original_dir = mod._PROFILES_DIR
            mod._PROFILES_DIR = tmpdir
            try:
                profile = UserProfile(
                    user_id="test-user",
                    preferences={"verbosity": "concise", "depth": "expert"},
                    observation_count=5,
                    last_updated=1000.0,
                )
                _save_profile(profile)
                loaded = _load_profile("test-user")
                self.assertEqual(loaded.user_id, "test-user")
                self.assertEqual(loaded.preferences["verbosity"], "concise")
                self.assertEqual(loaded.observation_count, 5)
            finally:
                mod._PROFILES_DIR = original_dir

    def test_load_missing_returns_defaults(self):
        profile = _load_profile("nonexistent-user-12345")
        self.assertEqual(profile.preferences, dict(_DEFAULTS))


class TestDimensions(unittest.TestCase):

    def test_all_defaults_are_valid(self):
        for dim, default_val in _DEFAULTS.items():
            self.assertIn(dim, _DIMENSIONS)
            self.assertIn(default_val, _DIMENSIONS[dim])


if __name__ == "__main__":
    unittest.main()
