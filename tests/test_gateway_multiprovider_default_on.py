"""Goodhart-resistant: HAL_GATEWAY_MULTIPROVIDER defaults ON (sovereign default).

The sovereign switch — a known non-Anthropic model routes to its OWN provider by
default, with NO env var set. Mutation-killing:
  * test_default_is_on fails if anyone reverts to the old opt-in (unset => OFF).
  * the kill-switch tests fail if the falsey set is broadened/narrowed.
  * test_claude_stays_verbatim_even_on_default proves the flip CANNOT break the
    Anthropic-verbatim path (#3274) — same_wire_provider must still return None
    for claude-*/unknown even with the default ON.
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.hal.gateway_egress import multiprovider_enabled, same_wire_provider


@pytest.fixture(autouse=True)
def _no_flag(monkeypatch):
    monkeypatch.delenv("HAL_GATEWAY_MULTIPROVIDER", raising=False)
    yield


def test_default_is_on():
    # THE sovereign default: absent env => multi-provider routing ON.
    assert multiprovider_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "OFF", "False", " off "])
def test_explicit_falsey_is_killswitch(monkeypatch, val):
    monkeypatch.setenv("HAL_GATEWAY_MULTIPROVIDER", val)
    assert multiprovider_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "on", "yes", "anything", ""])
def test_truthy_or_unknown_is_on(monkeypatch, val):
    monkeypatch.setenv("HAL_GATEWAY_MULTIPROVIDER", val)
    assert multiprovider_enabled() is True


def test_claude_stays_verbatim_even_on_default():
    # SAFETY: default ON must NOT route claude-*/unknown away from Anthropic.
    assert same_wire_provider("claude-opus-4-8") is None
    assert same_wire_provider("claude-3-5-sonnet-20241022") is None
    assert same_wire_provider("some-unknown-model-xyz") is None
