"""Tests for subscribe_wizard.py — token-already-active pre-check."""
import sys
import os
import unittest
from unittest.mock import MagicMock

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.subscribe_wizard import (
    _token_already_active,
    run_wizard,
    EXIT_OK,
    EXIT_USER_CANCELLED,
    DEFAULT_PROVIDER,
)


class TestTokenAlreadyActive(unittest.TestCase):
    def test_returns_true_when_loader_returns_truthy(self):
        self.assertTrue(_token_already_active(lambda: MagicMock()))

    def test_returns_false_when_loader_returns_none(self):
        self.assertFalse(_token_already_active(lambda: None))

    def test_returns_false_when_loader_returns_falsy(self):
        self.assertFalse(_token_already_active(lambda: False))

    def test_returns_false_when_loader_raises(self):
        def bad_loader():
            raise RuntimeError("keychain error")
        self.assertFalse(_token_already_active(bad_loader))


class TestRunWizardPrecheck(unittest.TestCase):
    """Pre-check UX paths exercised via _active_loader injection."""

    def _base_kwargs(self, active: bool, choice: str) -> dict:
        return {
            "writer": MagicMock(return_value=(True, "ok")),
            "verifier": lambda: True,
            "printer": lambda *a: None,
            "platform_check": lambda: True,
            "_active_loader": (lambda: MagicMock()) if active else (lambda: None),
            "reader": lambda prompt: choice,
        }

    def test_skip_returns_exit_ok_without_writing(self):
        kwargs = self._base_kwargs(active=True, choice="s")
        result = run_wizard(DEFAULT_PROVIDER, **kwargs)
        self.assertEqual(result, EXIT_OK)
        kwargs["writer"].assert_not_called()

    def test_blank_choice_defaults_to_skip(self):
        kwargs = self._base_kwargs(active=True, choice="")
        result = run_wizard(DEFAULT_PROVIDER, **kwargs)
        self.assertEqual(result, EXIT_OK)
        kwargs["writer"].assert_not_called()

    def test_rotate_proceeds_to_token_capture_and_writes(self):
        token = "sk-" + "x" * 30
        responses = iter(["r", token])
        writer = MagicMock(return_value=(True, "ok"))
        result = run_wizard(
            DEFAULT_PROVIDER,
            reader=lambda prompt: next(responses),
            writer=writer,
            verifier=lambda: True,
            printer=lambda *a: None,
            platform_check=lambda: True,
            _active_loader=lambda: MagicMock(),
        )
        self.assertEqual(result, EXIT_OK)
        writer.assert_called_once()

    def test_cancel_on_precheck_returns_user_cancelled(self):
        def raise_interrupt(prompt):
            raise KeyboardInterrupt

        result = run_wizard(
            DEFAULT_PROVIDER,
            reader=raise_interrupt,
            writer=MagicMock(return_value=(True, "ok")),
            verifier=lambda: True,
            printer=lambda *a: None,
            platform_check=lambda: True,
            _active_loader=lambda: MagicMock(),
        )
        self.assertEqual(result, EXIT_USER_CANCELLED)

    def test_no_active_token_skips_precheck_and_captures(self):
        token = "sk-" + "y" * 30
        writer = MagicMock(return_value=(True, "ok"))
        result = run_wizard(
            DEFAULT_PROVIDER,
            reader=lambda prompt: token,
            writer=writer,
            verifier=lambda: True,
            printer=lambda *a: None,
            platform_check=lambda: True,
            _active_loader=lambda: None,
        )
        self.assertEqual(result, EXIT_OK)
        writer.assert_called_once()
