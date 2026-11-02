"""Check the independent Python comparison against simple RSA identities."""

import math
from pathlib import Path
import runpy

import pytest


REFERENCE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / 'examples/memo_umi/python_rsa.py'))
listener_table = REFERENCE['listener_table']


def test_literal_listener_matches_the_task_denotations():
    assert listener_table(depth=0) == {
        'green': {'green square': 0.5, 'green circle': 0.5, 'pink circle': 0.0},
        'pink': {'green square': 0.0, 'green circle': 0.0, 'pink circle': 1.0},
        'square': {'green square': 1.0, 'green circle': 0.0, 'pink circle': 0.0},
        'round': {'green square': 0.0, 'green circle': 0.5, 'pink circle': 0.5},
    }


def test_recursive_listener_matches_the_closed_form_for_green():
    # The speaker uses green with probability 1/(1+exp(beta/2)) for the square
    # and 1/2 for the circle; conditioning on green gives this closed form.
    expected = 2.0 / (3.0 + math.exp(0.5))
    actual = listener_table(depth=1, beta=1.0)['green']['green square']
    assert abs(actual - expected) <= math.ulp(expected)


@pytest.mark.parametrize('depth', [0, 1, 2, 3])
def test_recursive_tables_keep_unambiguous_choices_and_color_shape_symmetry(depth):
    table = listener_table(depth=depth)
    assert table['pink']['pink circle'] == 1.0
    assert table['square']['green square'] == 1.0
    assert table['green']['pink circle'] == table['round']['green square'] == 0.0
    assert table['green']['green square'] == table['round']['pink circle']
    assert table['green']['green circle'] == table['round']['green circle']
    assert all(math.fsum(row.values()) == 1.0 for row in table.values())
    assert list(table) == list(REFERENCE['UTTERANCES'])
    assert all(tuple(row) == REFERENCE['CHOICES'] for row in table.values())


@pytest.mark.parametrize('depth', [1, 2, 3])
def test_zero_speaker_rationality_recovers_the_literal_listener(depth):
    assert listener_table(depth=depth, beta=0.0) == listener_table(depth=0)


@pytest.mark.parametrize('arguments', [
    {'depth': -1}, {'depth': True}, {'depth': 0, 'beta': float('nan')},
    {'depth': 1, 'beta': float('inf')},
])
def test_invalid_model_configuration_is_rejected(arguments):
    with pytest.raises(ValueError):
        listener_table(**arguments)
