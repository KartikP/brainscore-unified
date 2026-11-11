"""The lazy table session marks trials like the table session: state never leaks between trials."""
import numpy as np
import pytest

from brainscore import BrainScoreModel, LayerSelector
from brainscore_core._sessions.perception import streaming_stimulus_session
from brainscore_core.streaming_helpers import table_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


class Counter:
    """Stateful extractor: its output is how many stimuli it saw since the last reset."""
    identifier = 'counter'

    def __init__(self):
        self.seen = 0

    def reset(self):
        self.seen = 0

    def __call__(self, stimuli, layers, **kwargs):
        out = np.array([[float(self.seen + i)] for i in range(len(stimuli))])
        self.seen += len(stimuli)
        return NeuroidAssembly(out, dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
            'neuroid_id': ('neuroid', ['late.0']), 'layer': ('neuroid', ['late'])})


def _counts(build, order):
    stimuli = StimulusSet({'stimulus_id': [f's{i}' for i in order], 'sentence': list('abc')})
    stimuli.identifier = f'stream-memory-{order}'
    model = BrainScoreModel('counts', extractor={'text': Counter()}, memory='trial_history',
                            region_layer_map={'A': LayerSelector('late', modality='text')})
    session = build(stimuli, record='A')
    model.participate(session)
    return [float(v) for v in session.collect('neural:A').values.ravel()]


@pytest.mark.parametrize('order', [[0, 1, 2], [2, 0, 1]])
def test_independent_trials_start_at_zero_in_both_sessions(order):
    table = _counts(lambda s, record: table_session(s, [f'neural:{record}'], record=record), order)
    lazy = _counts(streaming_stimulus_session, order)
    assert table == lazy == [0.0] * 3
