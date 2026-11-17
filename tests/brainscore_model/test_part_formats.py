"""Each part declares the answer formats it owns (rules 2.4, 4.11).

A part alone owns the formats it gives natively; a format it derives is
declared only when a statement names it. ``behavior:rt`` is declared with
``behavior:probabilities``. The literal sets below are the native and derived
formats of each part kind.
"""
from pathlib import Path

import pytest

import brainscore
from brainscore import BrainScoreModel
from tests.brainscore_model.test_answer_ownership import _model

pytestmark = pytest.mark.unit

NATIVE = {
    'answer': {'behavior', 'behavior:choice'},
    'reply': {'behavior:choice', 'behavior:rating', 'behavior:text'},
    'likelihood': {'behavior:log_probabilities'},
    'behavioral_readout_layer': {'behavior', 'behavior:probabilities', 'behavior:rating',
                                 'behavior:rt'},
}
DERIVED = {   # the statements that declare each derived format of the part
    'answer': {},
    'reply': {},
    'likelihood': {'behavior:probabilities': ('likelihood', 'softmax'),
                   'behavior:choice': ('likelihood', 'softmax_argmax')},
    'behavioral_readout_layer': {'behavior:choice': ('readout', 'argmax')},
}


def _behavior(model):
    return {c for c in model.out_channels if c == 'behavior' or c.startswith('behavior:')}


@pytest.mark.parametrize('part', sorted(NATIVE))
def test_a_part_alone_declares_its_native_formats(part):
    assert _behavior(_model(part)) == NATIVE[part]


@pytest.mark.parametrize('part', sorted(NATIVE))
def test_a_part_with_statements_also_declares_its_derived_formats(part):
    derived = set(DERIVED[part]) | ({'behavior:rt'} if 'behavior:probabilities' in DERIVED[part] else set())
    assert _behavior(_model(part, owners=DERIVED[part])) == NATIVE[part] | derived


def test_answer_and_reply_are_distinct_kinds_of_generation():
    assert _behavior(_model('answer')) == {'behavior', 'behavior:choice'}
    assert 'behavior:rating' in _behavior(_model('reply'))
    assert 'behavior:rating' not in _behavior(_model('answer'))


def test_a_readout_only_model_declares_probabilities_rating_rt_not_choice():
    model = BrainScoreModel('m', extractor={'text': lambda *a, **k: None}, in_channels={'text'},
                            behavioral_readout_layer='late')
    assert 'behavior:choice' not in model.out_channels
    assert model.out_channels >= {'behavior', 'behavior:probabilities', 'behavior:rating', 'behavior:rt'}


def test_a_reply_and_likelihood_model_declares_the_stated_probabilities():
    model = _model('reply', 'likelihood', owners={'behavior:probabilities': ('likelihood', 'softmax')})
    assert _behavior(model) == {'behavior:choice', 'behavior:probabilities', 'behavior:rating',
                                'behavior:text', 'behavior:rt', 'behavior:log_probabilities'}


def test_the_replaced_names_are_gone():
    # The imported package, so the check also holds for an installed copy.
    package = Path(brainscore.__file__).parent
    found = [f'{path}:{number}' for path in sorted(package.rglob('*.py'))
             for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1)
             if 'answer_parts' in line or '_FOLLOWS' in line]
    assert found == []
