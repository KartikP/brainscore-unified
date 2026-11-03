"""ROAR scores an invalid generated answer as wrong, not as the first label."""
import pytest

from brainscore.benchmarks.roar_yeatman2021.benchmark import Yeatman2021LexicalDecision
from brainscore_core.brainscore_model import BrainScoreModel
from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _benchmark():
    # Skip __init__: it downloads the ROAR data. __call__ needs only these fields.
    benchmark = object.__new__(Yeatman2021LexicalDecision)
    benchmark._test_stimuli = StimulusSet({
        'stimulus_id': ['w1', 'w2', 'p1', 'p2'], 'sentence': ['cat', 'dog', 'florp', 'blick'],
        'image_label': ['real', 'real', 'pseudo', 'pseudo']})
    benchmark._train_stimuli = None
    benchmark._ceiling = Score(1.0)
    benchmark.prefer_path = 'generation'
    return benchmark


def test_invalid_answers_count_as_wrong():
    # Always answering 'real' when unsure used to earn credit on every real word.
    replies = {'w1': 'real', 'w2': 'unsure', 'p1': 'pseudo', 'p2': 'unsure'}
    model = BrainScoreModel(identifier='answers', supported_modalities={'text'},
                            answer=lambda stimulus_row, instruction, label_set:
                            replies[stimulus_row.stimulus_id])
    with pytest.warns(UserWarning, match='invalid'):
        score = _benchmark()(model)
    assert float(score.attrs['raw']) == pytest.approx(.5)
    assert score.attrs['n_invalid_answers'] == 2
    assert score.attrs['accuracy_real'] == pytest.approx(.5)
