"""The reaction-time link keeps the values of 7bc821e on each path, with explicit validity.

Through ``participate``: a readout's link reads the readout's float64 rows (only
the stored ``behavior:probabilities`` copy is float32); a likelihood link reads
the stored float32 probabilities, as before. Both match the functions of
``7bc821e`` bit for bit on finite rows. A reaction time from an invalid
probability row is invalid, and the link keeps its source rows as evidence.
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
    task = TaskContext(response=Choice(['cat', 'dog'],
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


# --- parity with 7bc821e on both paths, for finite rows, bit for bit -----------

def _old_answers():
    """The answers module of 7bc821e: its softmax, implied_array and reaction_time."""
    from tests import reference_answers_7bc821e
    return vars(reference_answers_7bc821e)


def _likelihood_rt(scores, link):
    model, _ = _model(scores=scores)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], reaction_time=link))
    return _ask(model, task, ['behavior:rt']).collect('behavior:rt').values


def test_a_likelihood_reaction_time_keeps_its_old_value():
    scores = {s: {'cat': -1.0, 'dog': -3.0} for s in 'abcd'}
    assert _likelihood_rt(scores, 'evidence_margin')[0] == 0.23840583115816116


@pytest.mark.parametrize('link', ['evidence_margin', 'surprisal'])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_the_likelihood_link_matches_7bc821e(link, seed):
    old = _old_answers()
    logp = -np.abs(np.random.default_rng(seed).normal(size=(4, 2)) * 3)
    scores = {s: {'cat': float(row[0]), 'dog': float(row[1])} for s, row in zip('abcd', logp)}
    stored = old['implied_array'](old['softmax'](logp), STIMULI, ['cat', 'dog'], 'likelihood')
    expected = old['reaction_time'](stored, link, STIMULI).values
    np.testing.assert_array_equal(_likelihood_rt(scores, link), expected)


@pytest.mark.parametrize('link', ['evidence_margin', 'surprisal'])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_the_readout_link_matches_7bc821e(monkeypatch, link, seed):
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    old = _old_answers()
    rows = np.random.default_rng(seed).dirichlet([1.0, 1.0], size=4)
    monkeypatch.setattr(sklearn.linear_model.LogisticRegression, 'predict_proba',
                        lambda self, features: rows[:len(features)])
    fitting = STIMULI.copy()
    fitting.identifier = f'rt-parity-{seed}'
    model, _ = _model(readout=True)
    task = TaskContext(response=Choice(['cat', 'dog'], reaction_time=link),
                       fitting_stimuli=fitting)
    rt = _ask(model, task, ['behavior:probabilities', 'behavior:rt']).collect('behavior:rt')
    readout = BehavioralAssembly(rows, dims=['presentation', 'choice'],
                                 coords={'stimulus_id': ('presentation', list('abcd')),
                                         'choice': ['cat', 'dog']})
    expected = old['reaction_time'](old['implied'](readout, 'readout'), link, STIMULI).values
    np.testing.assert_array_equal(rt.values, expected)
