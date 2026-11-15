"""Each part declares the answer formats it gives, native or derived (rule 4.11; stage 2, C1).

In C1 the declaration of a part is every format it gives, native or derived;
the literal sets below are the native and derived formats of each part kind.
"""
import subprocess
from pathlib import Path

import pytest

from brainscore import BrainScoreModel
from tests.brainscore_model.test_answer_ownership import _model

pytestmark = pytest.mark.unit

FORMATS = {   # part kind: native formats | derived formats
    'answer': {'behavior', 'behavior:choice'},
    'reply': {'behavior:choice', 'behavior:rating', 'behavior:confidence', 'behavior:text'},
    'likelihood': {'behavior:probabilities', 'behavior:choice', 'behavior:rt'},
    'behavioral_readout_layer': {'behavior', 'behavior:probabilities', 'behavior:rating',
                                 'behavior:choice', 'behavior:rt'},
}


def _behavior(model):
    return {c for c in model.out_channels if c == 'behavior' or c.startswith('behavior:')}


@pytest.mark.parametrize('part', sorted(FORMATS))
def test_a_part_alone_declares_its_formats(part):
    assert _behavior(_model(part)) == FORMATS[part] | {'behavior'}


def test_answer_and_reply_are_distinct_kinds_of_generation():
    assert _behavior(_model('answer')) == {'behavior', 'behavior:choice'}
    assert 'behavior:rating' in _behavior(_model('reply'))
    assert 'behavior:rating' not in _behavior(_model('answer'))


def test_a_readout_only_model_declares_choice_probabilities_rating_rt():
    model = BrainScoreModel('m', extractor={'text': lambda *a, **k: None}, in_channels={'text'},
                            behavioral_readout_layer='late')
    assert model.out_channels >= {'behavior', 'behavior:choice', 'behavior:probabilities',
                                  'behavior:rating', 'behavior:rt'}


def test_a_reply_and_likelihood_model_declares_every_format():
    model = _model('reply', 'likelihood', owners={'behavior:choice': 'likelihood'})
    assert _behavior(model) == {'behavior', 'behavior:choice', 'behavior:probabilities',
                                'behavior:rating', 'behavior:confidence', 'behavior:text',
                                'behavior:rt'}


def test_the_replaced_names_are_gone():
    root = Path(__file__).resolve().parents[2]
    found = subprocess.run(['git', 'grep', '-n', '-e', 'answer_parts', '-e', '_FOLLOWS', '--',
                            'brainscore'], cwd=root, capture_output=True, text=True).stdout
    assert found == ''
