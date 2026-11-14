"""A readout's reaction-time link reads the float64 probabilities and their validity.

Through ``participate``: the link is computed from the readout's own float64
rows (only the stored ``behavior:probabilities`` copy is float32), and a
reaction time from an invalid probability row is invalid. The link keeps its
source rows as evidence.
"""
import numpy as np
import pytest
import sklearn.linear_model

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from tests.brainscore_model.test_task_responses import STIMULI, _ask, _model

pytestmark = pytest.mark.unit

ROWS = np.array([[0.2, 0.2],                       # not a probability row: sums to 0.4
                 [0.123456789, 0.876543211],
                 [0.5, 0.5],
                 [0.9, 0.1]])


def _session(monkeypatch, outputs=('behavior:probabilities', 'behavior:rt')):
    monkeypatch.setattr(sklearn.linear_model.LogisticRegression, 'predict_proba',
                        lambda self, features: ROWS[:len(features)])
    fitting = STIMULI.copy()
    fitting.identifier = 'rt-link-fitting'
    model, _ = _model(readout=True)
    task = TaskContext(response=Choice(['cat', 'dog'], probabilities=True,
                                       reaction_time='evidence_margin'),
                       fitting_stimuli=fitting)
    return _ask(model, task, list(outputs))


def test_an_invalid_probability_row_gives_an_invalid_reaction_time(monkeypatch):
    session = _session(monkeypatch)
    probabilities = session.collect('behavior:probabilities')
    rt = session.collect('behavior:rt')
    assert list(probabilities['response_state'].values) == ['invalid', 'valid', 'valid', 'valid']
    assert list(rt['response_state'].values) == ['invalid', 'valid', 'valid', 'valid']


def test_the_link_reads_the_float64_rows_not_the_stored_float32_copy(monkeypatch):
    session = _session(monkeypatch)
    assert session.collect('behavior:rt').values[1] == 0.24691357800000002
    assert session.collect('behavior:probabilities').values.dtype == np.float32
    assert session.collect('behavior:rt')['raw_answer'].values[1] == '[0.123456789, 0.876543211]'
