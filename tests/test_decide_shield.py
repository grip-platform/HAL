"""HAL loads GRIP's decision shield, and refuses to send without it (lib/hal/decide/shield.py).

Each test points CLAUDE_DIR at a made-up GRIP checkout under tmp_path, so the
shield found is the one the test wrote. Each test names the mutation that turns it red.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lib.hal.decide import shield as decision_shield

OLD_GRIP = "def shield_payload(value, *, to=None, env=None):\n    return value\n"
NEW_GRIP = OLD_GRIP + (
    "def shield_decision_payload(value, *, to=None, env=None):\n"
    "    return {**value, 'state': '[scrubbed]', 'sent_to': to}\n")


@pytest.fixture
def grip(tmp_path, monkeypatch):
    """Write a GRIP checkout whose review_redaction.py holds ``source``; return its root.

    Any directory already on sys.path that holds a review_redaction.py (a real
    GRIP another test added) is dropped for the test, so the only shield to be
    found is the one written here."""
    monkeypatch.setattr(sys, "path", [p for p in sys.path
                                      if not (Path(p) / "review_redaction.py").exists()])
    monkeypatch.delitem(sys.modules, "review_redaction", raising=False)
    decision_shield.reset()

    def make(source, name="grip"):
        root = tmp_path / name
        (root / "lib").mkdir(parents=True, exist_ok=True)
        (root / "lib" / "review_redaction.py").write_text(source, encoding="utf-8")
        monkeypatch.setenv("CLAUDE_DIR", str(root))
        sys.modules.pop("review_redaction", None)
        return root

    yield make
    sys.modules.pop("review_redaction", None)
    decision_shield.reset()


def test_a_grip_with_the_decision_shield_scrubs_the_body(grip):
    grip(NEW_GRIP)
    out = decision_shield.scrub({"state": "raw", "questions": {}}, to="https://api.typesafe.ai/v1/systemone")
    assert out == {"state": "[scrubbed]", "questions": {}, "sent_to": "https://api.typesafe.ai/v1/systemone"}


def test_a_grip_too_old_for_the_decision_shield_refuses_the_send(grip):
    """Mutation: fall back to shield_payload, which would let ``state`` out whole."""
    grip(OLD_GRIP)
    with pytest.raises(decision_shield.ShieldUnavailable):
        decision_shield.scrub({"state": "raw", "questions": {}}, to="https://api.typesafe.ai/v1/systemone")


def test_no_grip_at_all_refuses_the_send(grip, tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_DIR", str(tmp_path / "nowhere"))
    with pytest.raises(decision_shield.ShieldUnavailable):
        decision_shield.scrub({"state": "raw", "questions": {}}, to="https://api.typesafe.ai/v1/systemone")


def test_a_failed_load_is_retried_after_a_minute_not_cached_forever(grip):
    """Mutation: remember a failed load for the life of the process."""
    root = grip(OLD_GRIP)
    assert decision_shield.load(now=0.0) is None
    (root / "lib" / "review_redaction.py").write_text(NEW_GRIP, encoding="utf-8")
    sys.modules.pop("review_redaction", None)
    assert decision_shield.load(now=30.0) is None, "inside the wait, no reload is attempted"
    assert decision_shield.load(now=61.0) is not None, "after the wait, the updated GRIP is found"
