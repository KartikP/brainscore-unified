"""A BrainScoreModel answer reaches an old benchmark with every stimulus-set column."""
import pandas as pd

from brainscore.brainscore_model import BrainScoreModel
from brainscore_core.legacy_facade import LegacyBenchmarkFacade
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def test_old_metrics_read_the_stimulus_columns_of_a_brainscore_model_answer():
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': ['a', 'b', 'c', 'd'], 'sentence': ['x'] * 4, 'image_file_name': list('abcd'),
        'animal': ['cat', 'dog', 'cat', 'dog'], 'image_type': ['w', 'w', 'f', 'f']}))
    stimuli.identifier = 'bsm-columns'
    model = BrainScoreModel(identifier='columns', in_channels={'text'},
                            answer=lambda stimulus_row, instruction, label_set: 'cat')
    facade = LegacyBenchmarkFacade(model)
    facade.start_task('label', ['cat', 'dog'])
    answer = facade.look_at(stimuli)
    assert set(answer.indexes['presentation'].names) == set(stimuli.columns)
    accuracy = {cat: float((answer.sel(animal=cat, image_type='w').values == 'cat').mean())
                for cat in sorted(set(answer['animal'].values))}      # as the Baker metric reads 'animal'
    assert accuracy == {'cat': 1.0, 'dog': 1.0}
    assert list(answer.sortby('stimulus_id')['stimulus_id'].values) == ['a', 'b', 'c', 'd']   # as BMD
