"""Conformance test T3 (Response Format Standard, section 11.1), for a BrainScoreModel.

1. Parity: the core conversions give the numbers of the inline code they
   replace, bit for bit. The old ``softmax`` is read from the base commit
   (``git show 7bc821e:brainscore/_model/answers.py``); the old choice was
   ``probabilities.argmax(1)`` (likelihood) and ``values.argmax(1)``
   (readout). Finite rows (with ties) match exactly, as values and dtypes.
   A source row with a non-finite value (NaN, +inf, -inf) had a choice
   before and is invalid now, as is every conversion of it: the one stated
   change (decision D3).
2. A derived answer of a BrainScoreModel always carries a record that names
   its conversion, parameters and source format.
The core version (the functions alone) is core
``tests/test_conformance_t3_conversions.py``.
"""
import ast
import subprocess
from pathlib import Path

import numpy as np
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.response_formats import argmax, response_record, softmax, softmax_argmax
from brainscore_core.responses import Choice
from tests.brainscore_model.test_task_responses import STIMULI, _ask, _model

pytestmark = pytest.mark.unit

BASE = '7bc821e'


def _old_softmax():
    source = subprocess.run(['git', 'show', f'{BASE}:brainscore/_model/answers.py'],
                            capture_output=True, text=True, check=True,
                            cwd=Path(__file__).resolve().parents[1]).stdout
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'softmax')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), f'{BASE}:answers.py', 'exec'), namespace)
    return namespace['softmax']


def _inputs():
    rng = np.random.default_rng(0)
    finite = [rng.normal(size=(500, k)) * 5 for k in (2, 3, 7)]
    finite += [x.astype(np.float32) for x in finite]
    ties = np.repeat(rng.normal(size=(50, 1)), 3, axis=1)
    ties[::2, 2] -= 1.0                           # ties between the first two options only
    ties[1::4, 0] -= 1.0                          # ties between the last two options only
    return finite + [ties]


# Rows with a non-finite value: NaN, +inf, -inf (an option of probability 0 before).
NON_FINITE = np.array([[np.nan, -1.0], [-1.0, np.nan], [np.inf, 0.0], [-np.inf, -np.inf],
                       [-np.inf, -1.0], [-1.0, -np.inf]])


@pytest.mark.parametrize('index', range(7))
def test_likelihood_conversions_match_the_replaced_code_bit_for_bit(index):
    logp = _inputs()[index]
    old_probabilities = _old_softmax()(logp)
    probabilities, valid = softmax(logp)
    np.testing.assert_array_equal(probabilities, old_probabilities)
    assert probabilities.dtype == old_probabilities.dtype
    assert valid.all()
    indices, valid = softmax_argmax(logp)
    assert valid.all()
    np.testing.assert_array_equal(indices, old_probabilities.argmax(1))


@pytest.mark.parametrize('index', range(7))
def test_readout_argmax_matches_the_replaced_code_bit_for_bit(index):
    values = _old_softmax()(_inputs()[index])     # probability rows, as a readout gives
    indices, valid = argmax(values)
    assert valid.all()
    np.testing.assert_array_equal(indices, np.asarray(values).argmax(1))


def test_stated_change_a_non_finite_source_row_is_now_invalid():
    """The one stated change against 7bc821e (decision D3): a source row with a NaN,
    +inf or -inf value is invalid, and so is every conversion of it. Before, it got a
    choice (the first NaN position, or the most probable finite option)."""
    old = _old_softmax()(NON_FINITE)
    probabilities, valid = softmax(NON_FINITE)
    np.testing.assert_array_equal(probabilities, old)        # the values are computed as before
    assert old.argmax(1).tolist() == [0, 0, 0, 0, 1, 0]      # old: every row had a choice
    assert not valid.any()
    indices, valid = softmax_argmax(NON_FINITE)
    assert indices.tolist() == [-1] * 6 and not valid.any()  # new: no choice, invalid


def test_a_likelihood_choice_is_the_same_choice_as_before():
    scores = {'a': {'cat': -1.0, 'dog': -1.0}, 'b': {'cat': -3.0, 'dog': -0.5},
              'c': {'cat': -0.5, 'dog': -3.0}, 'd': {'cat': -2.0, 'dog': -2.0}}
    model, _ = _model(scores=scores)
    session = _ask(model, TaskContext(instruction='Which?',
                                      response=Choice(['cat', 'dog'], probabilities=True)),
                   ['behavior:choice', 'behavior:probabilities'])
    logp = np.array([[scores[s]['cat'], scores[s]['dog']] for s in 'abcd'])
    old = _old_softmax()(logp)
    choice = session.collect('behavior:choice')
    np.testing.assert_array_equal(choice.values.argmax(1), old.argmax(1))   # ties: the first
    np.testing.assert_array_equal(session.collect('behavior:probabilities').values,
                                  old.astype(np.float32))
    np.testing.assert_array_equal(choice.values, np.eye(2, dtype=np.float32)[old.argmax(1)])


def test_every_derived_answer_names_its_conversion():
    scores = {s: {'cat': -1.0, 'dog': -2.0} for s in 'abcd'}
    model, _ = _model(scores=scores)
    session = _ask(model, TaskContext(instruction='Which?', response=Choice(
        ['cat', 'dog'], probabilities=True)), ['behavior:choice', 'behavior:probabilities'])
    assert session.collect('behavior:choice').attrs['response'] == response_record(
        'behavior:choice', 'likelihood', conversion='softmax_argmax',
        source_format='behavior:log_probabilities')
    assert session.collect('behavior:probabilities').attrs['response'] == response_record(
        'behavior:probabilities', 'likelihood', conversion='softmax',
        source_format='behavior:log_probabilities')

    fitting = STIMULI.copy()
    fitting.identifier = 'conformance-t3-fitting'
    readout, _ = _model(readout=True)
    choice = _ask(readout, TaskContext(response=Choice(['cat', 'dog']), fitting_stimuli=fitting),
                  ['behavior:choice']).collect('behavior:choice')
    record = choice.attrs['response']
    assert record['origin'] == 'derived' and record['conversion'] == 'argmax'
    assert record['source_format'] == 'behavior:probabilities'
    # The readout's own column order decides a tie, as before this change.
    assert record['parameters'] == {'tie': 'first', 'option_order': 'readout'}
