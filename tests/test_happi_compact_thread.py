"""FU3: opt-in compact threading through the happi provider envelope.

`_build_cch_call_args` adds `compact=True` to the CCH build_kwargs ONLY when the
envelope explicitly requests it, preserving the opt-in / default-off contract.
The downstream model-capability gate (`_effective_compact`) is unchanged — this
only carries the request intent from envelope → CCHSigner.
"""
import os
import sys

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

import pytest  # noqa: E402

from lib.hal.happi.envelope import HappiEnvelope  # noqa: E402
from lib.hal.happi.providers.anthropic import (  # noqa: E402
    _build_cch_call_args,
    _flag_is_true,
)


def _env(**flags):
    return HappiEnvelope.parse({
        "v": "happi/1.0", "id": "req-test", "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "claude-opus-4-8", "max_tokens": 16, **flags},
        "auth": {"scheme": "subscription"},
    })


def test_compact_threaded_when_flag_true():
    _, _, kw = _build_cch_call_args(_env(compact=True))
    assert kw.get("compact") is True


def test_compact_threaded_when_flag_string_true():
    _, _, kw = _build_cch_call_args(_env(compact="true"))
    assert kw.get("compact") is True


def test_compact_absent_by_default():
    # No compact flag -> build_kwargs carries no compact -> byte-identical request.
    _, _, kw = _build_cch_call_args(_env())
    assert "compact" not in kw


def test_compact_not_threaded_when_false():
    _, _, kw = _build_cch_call_args(_env(compact=False))
    assert "compact" not in kw


def test_compact_not_threaded_when_string_false():
    _, _, kw = _build_cch_call_args(_env(compact="false"))
    assert "compact" not in kw


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False),
    ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("0", False), ("no", False), ("", False), (None, False),
    ("garbage", False),
])
def test_flag_is_true_table(value, expected):
    assert _flag_is_true(value) is expected
