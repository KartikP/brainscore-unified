"""Legacy behavioral benchmarks score an invalid generated answer at chance."""
import numpy as np
import pytest

from brainscore.brainscore_model import BrainScoreModel
from brainscore_core.legacy_facade import LegacyBenchmarkFacade
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _look_at(replies):
    ids = [f's{i}' for i in range(len(replies))]
    lookup = dict(zip(ids, replies))
    model = BrainScoreModel(identifier='answers', in_channels={'text'},
                            answer=lambda stimulus_row, instruction, label_set:
                            lookup[stimulus_row.stimulus_id])
    # A legacy label benchmark reaches the model through the facade (a choice among the labels).
    facade = LegacyBenchmarkFacade(model)
    facade.start_task('label', ['cat', 'dog'])
    return facade.look_at(StimulusSet({'sentence': ['x'] * len(ids), 'stimulus_id': ids}))


def test_invalid_answers_become_seeded_valid_labels():
    replies = ['cat'] * 19 + ['neither']   # 5% invalid, under the limit
    with pytest.warns(UserWarning, match='random valid labels'):
        first = _look_at(replies)
    second = _look_at(replies)
    assert set(first.values[0]) <= {'cat', 'dog'}                 # every row is one label
    assert list(first.values[0]) == list(second.values[0])        # seeded, so repeatable
    assert first.attrs['n_invalid_answers'] == 1


def test_valid_answers_pass_through_unchanged():
    result = _look_at(['cat', 'dog'])
    assert list(result.values[0]) == ['cat', 'dog']
    assert 'n_invalid_answers' not in result.attrs


def test_too_many_invalid_answers_raise():
    with pytest.raises(ValueError, match='outside the label set'):
        _look_at(['cat'] * 8 + ['neither'] * 2)   # 20% invalid
