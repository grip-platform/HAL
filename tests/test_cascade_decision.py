"""The decision signer both cascades share fails closed.

A cascade move hands a conversation to another vendor, so a target may be
contacted only after a signed decision row is written (ADR-0016 decision 5).
``sign_failover_decision`` is the one place that rule is decided for the Claude
wire and for the OpenAI rails. These tests pin its three answers: a written row
is a decision; an unwritten row is not; a signer or store that raises has not
signed either.
"""
from types import SimpleNamespace

import pytest

from lib.hal import cascade_decision, gateway_witness


def test_a_written_row_is_a_signed_decision(monkeypatch):
    calls = []

    def writes(row, **kwargs):
        calls.append((row, kwargs))
        return SimpleNamespace(written=True)

    monkeypatch.setattr(gateway_witness, 'append_thread_row', writes)
    assert cascade_decision.sign_failover_decision('groq', 'model-x', 'unknown', 's1') is True
    row, kwargs = calls[0]
    assert row == {'provider': 'groq', 'model': 'model-x', 'status': None}
    assert kwargs == {'row_class': 'failover', 'route': 'inference',
                      'session_id': 's1', 'billing': 'unknown'}


@pytest.mark.parametrize('result', [SimpleNamespace(written=False), None, object()])
def test_an_unwritten_row_is_not_a_decision(monkeypatch, result):
    monkeypatch.setattr(gateway_witness, 'append_thread_row', lambda *args, **kwargs: result)
    assert cascade_decision.sign_failover_decision('groq', 'model-x', 'unknown', None) is False


@pytest.mark.parametrize('error', [OSError('thread store closed'), RuntimeError('no signer'),
                                   ValueError('bad row')])
def test_a_signer_that_raises_has_not_signed(monkeypatch, error):
    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr(gateway_witness, 'append_thread_row', broken)
    assert cascade_decision.sign_failover_decision('groq', 'model-x', 'unknown', None) is False
