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
    in_channels = frozenset({'vision:still'})
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


def test_all_records_every_declared_region_once(benchmark):
    subject = Recorder()
    features, _ = benchmark._extract_per_TR_features(
        subject, _frames(), recording_target='all')
    assert len(subject.requests) == 1                      # one session for every region
    assert sorted(subject.requests[0]) == ['neural:IT', 'neural:V4']
    units = ['shared', 'c', 'a', 'b']                      # IT then V4 (sorted); 'shared' once
    expected = [[_value(frame, unit) for unit in units] for frame in FRAMES]
    np.testing.assert_array_equal(features, np.asarray(expected, dtype=np.float32))
