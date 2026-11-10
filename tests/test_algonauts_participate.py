"""Algonauts 2025 reads any contract subject through participate only.

The subject declares channels and implements participate and reset only: no
process(), no start_recording(), no private attributes. Tiny synthetic frames;
no movie, fMRI or model weights.
"""
import numpy as np
import pandas as pd
import pytest

from brainscore_core.contract import Subject
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import drain_session_inputs, session_stimuli
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends

FRAMES = [f'clip_t{index:04d}' for index in range(6)]
UNITS = {'V4': ['a', 'b', 'shared'], 'IT': ['shared', 'c']}


def _value(frame: str, unit: str) -> float:
    return float(FRAMES.index(frame) * 10 + sorted({u for us in UNITS.values() for u in us}).index(unit))


class Recorder(Subject):
    """Answers every requested neural:<region>, optionally in reversed order."""

    identifier = 'participate-only'
    in_channels = frozenset({'vision:image'})
    out_channels = frozenset(f'neural:{region}' for region in UNITS)

    def __init__(self, reverse=False):
        self.reverse = reverse
        self.requests = []

    def participate(self, session):
        stimuli = session_stimuli(session, drain_session_inputs(session))
        ids = [str(stimulus_id) for stimulus_id in stimuli['stimulus_id']]
        if self.reverse:
            ids = ids[::-1]
        channels = list(session.requested_output_channels)
        self.requests.append(channels)
        for channel in channels:
            units = UNITS[channel.split(':', 1)[1]]
            values = np.array([[_value(frame, unit) for unit in units] for frame in ids])
            session.respond(StreamEvent(channel, NeuroidAssembly(
                values, coords={'stimulus_id': ('presentation', ids),
                                'neuroid_id': ('neuroid', units)},
                dims=['presentation', 'neuroid']), t_ms=None))


def _frames():
    frame = pd.DataFrame({'stimulus_id': FRAMES, 'frame_id': FRAMES,
                          'image_file_name': [f'/frames/{f}.jpg' for f in FRAMES]})
    frames = StimulusSet(frame)
    frames.identifier = 'synthetic-frames'
    frames.stimulus_paths = dict(zip(FRAMES, frame['image_file_name']))
    return frames


@pytest.fixture
def benchmark(tmp_path):
    return Algonauts2025Friends(subject=1, assembly_root=tmp_path)


def test_one_region_is_recorded_through_participate(benchmark):
    subject = Recorder()
    features, frame_ids = benchmark._extract_per_TR_features(subject, _frames())
    assert subject.requests == [['neural:IT']]
    assert frame_ids == FRAMES
    expected = [[_value(frame, unit) for unit in UNITS['IT']] for frame in FRAMES]
    np.testing.assert_array_equal(features, np.asarray(expected, dtype=np.float32))


def test_answers_are_put_in_frame_order_by_stimulus_id(benchmark):
    features, _ = benchmark._extract_per_TR_features(Recorder(reverse=True), _frames())
    expected = [[_value(frame, unit) for unit in UNITS['IT']] for frame in FRAMES]
    np.testing.assert_array_equal(features, np.asarray(expected, dtype=np.float32))


# ── Old path vs participate on the SAME subject (Contract Rule 10.1) ──────

from brainscore import BrainScoreModel, CompositeSelector
from brainscore.benchmarks.algonauts2025 import benchmark as algonauts_module


class _Extractor:
    """Deterministic activity per (frame, layer); keeps the stimulus columns."""

    identifier = 'synthetic-algonauts-extractor'

    def __call__(self, stimuli, layers):
        blocks = [np.array([[float(hash((layer, frame, unit)) % 997) for unit in range(4)]
                            for frame in stimuli['stimulus_id']]) for layer in layers]
        return NeuroidAssembly(
            np.concatenate(blocks, axis=1), dims=['presentation', 'neuroid'], coords={
                **{column: ('presentation', list(stimuli[column])) for column in stimuli.columns},
                'layer': ('neuroid', [layer for layer in layers for _ in range(4)]),
                'neuroid_id': ('neuroid', [f'{layer}.{unit}' for layer in layers for unit in range(4)]),
            })


def _old_record_features(subject, stimuli, region, *, time_bins):
    """What the benchmark did before it ran through participate."""
    subject.start_recording(region, time_bins=time_bins)
    return algonauts_module._as_features(subject.process(stimuli))


REGION_MAPS = {
    'one layer per region': {'V1': 'early', 'V4': 'mid', 'IT': 'late'},
    'regions share a layer': {'V4': 'late', 'V1': 'early', 'IT': 'late'},
    'composite overlaps a region': {
        'V4': 'mid', 'IT': CompositeSelector((('mid', (0, 2)), ('late', None))),
        'V1': 'early'},
}


@pytest.mark.parametrize('region_map', REGION_MAPS.values(), ids=REGION_MAPS.keys())
@pytest.mark.parametrize('target', ['IT', 'all'])
def test_old_and_new_path_give_identical_features(benchmark, monkeypatch, region_map, target):
    model = BrainScoreModel('synthetic', region_layer_map=region_map,
                            preprocessors={'vision': _Extractor(), 'text': _Extractor()})
    new, new_ids = benchmark._extract_per_TR_features(model, _frames(), recording_target=target)
    with monkeypatch.context() as patch:
        patch.setattr(algonauts_module, '_record_features', _old_record_features)
        old, old_ids = benchmark._extract_per_TR_features(model, _frames(), recording_target=target)
    assert new.dtype == old.dtype and new.shape == old.shape
    np.testing.assert_array_equal(new, old)
    assert new_ids == old_ids


class _WrongIds(Recorder):
    """Answers with the right number of rows but one wrong stimulus id."""

    def participate(self, session):
        stimuli = session_stimuli(session, drain_session_inputs(session))
        ids = [str(s) for s in stimuli['stimulus_id']][:-1] + ['not_a_frame']
        for channel in session.requested_output_channels:
            session.respond(StreamEvent(channel, NeuroidAssembly(
                np.zeros((len(ids), 2)),
                coords={'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['a', 'b'])},
                dims=['presentation', 'neuroid']), t_ms=None))


def test_a_response_for_other_frames_is_refused(benchmark):
    with pytest.raises(ValueError, match='exactly once'):
        benchmark._extract_per_TR_features(_WrongIds(), _frames())
