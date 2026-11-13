"""ROAR scores an invalid generated answer as wrong, not as the first label."""
import pytest

from brainscore.benchmarks.roar_yeatman2021.benchmark import Yeatman2021LexicalDecision
from brainscore.brainscore_model import BrainScoreModel
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
    benchmark._version = 1
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


def test_session_reading_matches_the_configured_readout():
    """ROAR reads through behavioral_response; on a BrainScoreModel the answers
    are the ones the readout configured by start_task gives."""
    import numpy as np
    from brainscore_core.contract import TaskContext
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    words = {'w1': 'cat', 'w2': 'dog', 'w3': 'sun', 'p1': 'florp', 'p2': 'blick', 'p3': 'zund'}

    def extractor(stimuli, layers):
        rng = np.random.RandomState(0)
        table = {sid: rng.randn(4) + (2. if sid.startswith('w') else -2.) for sid in words}
        rows = np.stack([table[sid] for sid in stimuli['stimulus_id']])
        return NeuroidAssembly(rows, dims=['presentation', 'neuroid'], coords={
            **{k: ('presentation', list(stimuli[k])) for k in stimuli.columns},
            'layer': ('neuroid', [layers[0]] * 4), 'neuroid_id': ('neuroid', list(range(4)))})

    def stimuli(ids):
        table = StimulusSet({'stimulus_id': ids, 'sentence': [words[i] for i in ids],
                             'image_label': ['real' if i.startswith('w') else 'pseudo' for i in ids]})
        table.identifier = f'roar-equivalence-{len(ids)}'
        return table

    def model():
        return BrainScoreModel(identifier='readout', extractor={'text': extractor},
                               region_layer_map={}, behavioral_readout_layer='L')

    benchmark = _benchmark()
    benchmark.prefer_path = 'readout'
    benchmark._train_stimuli = stimuli(['w1', 'w2', 'p1', 'p2'])
    benchmark._test_stimuli = stimuli(['w3', 'p3', 'w1', 'p1'])

    legacy = model()
    legacy.start_task(TaskContext(task_type='probabilities', label_set=['real', 'pseudo'],
                                  fitting_stimuli=benchmark._train_stimuli,
                                  instruction='Is this a real word or a pseudo word?',
                                  prefer_path='readout'))
    expected = legacy._behavioral.predict_probabilities(benchmark._test_stimuli)
    choices = list(expected['choice'].values)
    expected_correct = [choices[int(i)] == label for i, label in
                        zip(expected.values.argmax(axis=1), benchmark._test_stimuli['image_label'])]

    from brainscore_core.streaming_helpers import behavioral_response
    answered = behavioral_response(model(), benchmark._test_stimuli, TaskContext(
        task_type='probabilities', label_set=['real', 'pseudo'],
        fitting_stimuli=benchmark._train_stimuli,
        instruction='Is this a real word or a pseudo word?', prefer_path='readout'))
    np.testing.assert_array_equal(answered.values, expected.values)
    assert list(answered['stimulus_id'].values) == list(expected['stimulus_id'].values)
    assert list(answered['choice'].values) == choices

    score = benchmark(model())
    assert float(score.attrs['raw']) == np.mean(expected_correct)
    assert score.attrs['n_test_stimuli'] == len(expected)
