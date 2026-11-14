"""Contract v3 rule 5.5: a stream the subject cannot take in never enters it."""
import numpy as np
import pandas as pd

from brainscore import BrainScoreModel
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import neural_response
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


class _VisionExtractor:
    def __init__(self):
        self.seen = []

    def __call__(self, stimuli, layers=None, **kwargs):
        self.seen.append(stimuli)
        n = len(stimuli)
        return NeuroidAssembly(
            np.ones((n, 2)), dims=['presentation', 'neuroid'],
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'neuroid_id': ('neuroid', [0, 1]),
                    'layer': ('neuroid', [layers[0]] * 2)})


def _model():
    extractor = _VisionExtractor()
    model = BrainScoreModel('vision-only', region_layer_map={'IT': 'layer'},
                            extractor={'vision': extractor})
    return model, extractor


def _movie():
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': ['clip0', 'clip1'],
        'image_path': ['/tmp/0.png', '/tmp/1.png'],
        'sentence': ['subtitle zero', 'subtitle one'],
        'audio_path': ['/tmp/0.wav', '/tmp/1.wav'],
        'object_name': ['cat', 'dog'],
    }))
    stimuli.identifier = 'movie-clips'
    return stimuli


def test_undeclared_modalities_never_reach_the_extractor():
    model, extractor = _model()
    response = neural_response(model, _movie(), ['neural:IT'])
    assert response.sizes['presentation'] == 2
    seen, = extractor.seen
    assert sorted(seen.columns) == ['image_path', 'object_name', 'stimulus_id']
    assert seen.identifier == 'movie-clips'  # cache identity survives the projection


def test_stimulus_set_keeps_its_identity_when_nothing_is_dropped():
    model, extractor = _model()
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a'], 'image_path': ['/tmp/a.png']}))
    stimuli.identifier = 'one-image'
    neural_response(model, stimuli, ['neural:IT'])
    assert extractor.seen[0] is stimuli


def test_undeclared_events_are_skipped_and_declared_ones_delivered():
    model, extractor = _model()
    session = Session([
        StreamEvent('text', 'not for this subject', 0.0,
                    meta={'stimulus_id': 't', 'column': 'sentence'}),
        StreamEvent('vision:image', '/tmp/a.png', 0.0,
                    meta={'stimulus_id': 'a', 'column': 'image_path'}),
    ], ['neural:IT'])
    model.participate(session)
    seen, = extractor.seen
    assert list(seen['stimulus_id']) == ['a']
    assert session.collect('neural:IT').sizes['presentation'] == 1


def test_rows_pass_through_to_the_policy():
    from brainscore_core.streaming import respond_event
    rows = []
    model = BrainScoreModel('policy', policy=lambda row: rows.append(row) or {'motor': 1})
    session = Session([StreamEvent('observation', {'value': 1}, None), respond_event()], ['motor'])
    model.participate(session)
    assert rows == [{'observation': {'value': 1}}]
    assert session.emitted[0].payload == 1
