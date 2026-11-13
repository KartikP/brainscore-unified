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

from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025

FRAMES = [f'clip_t{index:04d}' for index in range(6)]
UNITS = {'V1': ['v', 'shared'], 'V4': ['a', 'b', 'shared'], 'IT': ['shared', 'c']}


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
        channels = list(session.outputs)
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
    return Algonauts2025(participant=1, assembly_root=tmp_path)


def test_every_region_is_recorded_through_participate_on_its_own_channel(benchmark):
    subject = Recorder()
    regions = ['IT', 'V4']
    bands = benchmark._record_bands(subject, _frames(), regions)
    assert subject.requests == [['neural:IT'], ['neural:V4']]     # one session per region
    for region in regions:
        expected = [[_value(frame, unit) for unit in UNITS[region]] for frame in FRAMES]
        np.testing.assert_array_equal(bands[region][0], np.asarray(expected, dtype=np.float32))


def test_answers_are_put_in_frame_order_by_stimulus_id(benchmark):
    bands = benchmark._record_bands(Recorder(reverse=True), _frames(), ['IT'])
    expected = [[_value(frame, unit) for unit in UNITS['IT']] for frame in FRAMES]
    np.testing.assert_array_equal(bands['IT'][0], np.asarray(expected, dtype=np.float32))


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


def _old_features(subject, stimuli, region, *, time_bins):
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
def test_old_path_and_participate_give_identical_bands(benchmark, region_map):
    model = BrainScoreModel('synthetic', region_layer_map=region_map,
                            preprocessors={'vision': _Extractor()})
    regions = sorted(region_map)
    bands = benchmark._record_bands(model, _frames(), regions)
    for region in regions:
        old, old_ids = _old_features(model, _frames(), region,
                                     time_bins=[(0, int(algonauts_module.TR_SEC * 1000))])
        new, new_ids = bands[region]
        assert new.dtype == old.dtype and new.shape == old.shape
        np.testing.assert_array_equal(new, old)
        np.testing.assert_array_equal(new_ids, old_ids)


class _WrongIds(Recorder):
    """Answers with the right number of rows but one wrong stimulus id."""

    def participate(self, session):
        stimuli = session_stimuli(session, drain_session_inputs(session))
        ids = [str(s) for s in stimuli['stimulus_id']][:-1] + ['not_a_frame']
        for channel in session.outputs:
            session.respond(StreamEvent(channel, NeuroidAssembly(
                np.zeros((len(ids), 2)),
                coords={'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['a', 'b'])},
                dims=['presentation', 'neuroid']), t_ms=None))


def test_a_response_for_other_frames_is_refused(benchmark):
    with pytest.raises(ValueError, match='exactly once'):
        benchmark._record_bands(_WrongIds(), _frames(), ['IT'])


# ── Declarations and a whole score through participate ────────────

import xarray as xr

from tests.benchmark_declarations import assert_declares, assert_preflight
from brainscore.benchmarks.algonauts2025.benchmark import (
    Algonauts2025)

CLIPS = [f'clip{index}' for index in range(8)]   # 8 runs: 5 outer folds, 5 inner
TRS = 10

CHANNELS = {'image': ('vision:image',), 'video': ('vision:video',),
            'video+audio': ('vision:video', 'audio')}



@pytest.mark.parametrize('split', ['friends', 'friends_s7', 'ood'])
def test_every_split_declares_vision_in_and_the_named_regions_out(split, tmp_path):
    benchmark = Algonauts2025(participant=1, split=split, assembly_root=tmp_path)
    assert_declares(benchmark, {'vision:image'}, {'neural:V1', 'neural:V4', 'neural:IT'})
    assert_preflight(benchmark)


def test_a_whole_score_comes_from_a_participate_only_subject(tmp_path, monkeypatch):
    benchmark = Algonauts2025(participant=1, assembly_root=tmp_path, stimulus_window=2,
                                     hrf_delay=1, excluded_samples_start=2, excluded_samples_end=0)
    stimulus = [clip for clip in CLIPS for _ in range(TRS)]
    steps = list(range(TRS)) * len(CLIPS)
    values = np.random.default_rng(0).normal(size=(len(stimulus), 4))
    benchmark._assembly = xr.DataArray(
        values, dims=('presentation', 'neuroid'),
        coords={'stimulus_id': ('presentation', stimulus), 'run': ('presentation', stimulus),
                't_within_run': ('presentation', steps)})
    rows = [{'stimulus_id': f'{clip}_t{step:04d}', 'frame_id': f'{clip}_t{step:04d}',
             'clip_id': clip, 't_within_run': step, 'image_file_name': f'/frames/{clip}/{step}.jpg'}
            for clip in CLIPS for step in range(TRS)]
    frames = StimulusSet(pd.DataFrame(rows))
    frames.identifier = 'synthetic-algonauts-frames'
    frames.stimulus_paths = {row['stimulus_id']: row['image_file_name'] for row in rows}
    monkeypatch.setattr(benchmark, '_expand_to_per_TR_frames', lambda: frames)

    class Frames(Recorder):
        def participate(self, session):
            stimuli = session_stimuli(session, drain_session_inputs(session))
            ids = [str(stimulus_id) for stimulus_id in stimuli['stimulus_id']]
            self.requests.append(list(session.outputs))
            rng = np.random.default_rng(1)
            session.respond(StreamEvent(list(session.outputs)[0], NeuroidAssembly(
                rng.normal(size=(len(ids), 3)),
                coords={'stimulus_id': ('presentation', ids),
                        'neuroid_id': ('neuroid', ['u0', 'u1', 'u2'])},
                dims=['presentation', 'neuroid']), t_ms=None))

    subject = Frames()
    score = benchmark(subject)
    assert np.isfinite(float(score))
    assert subject.requests == [['neural:IT'], ['neural:V1'], ['neural:V4']]
    assert score.attrs['bands'] == ['IT', 'V1', 'V4'] and 'mode' not in score.attrs


# ── The video form ───────────────────────────────────────────────

from brainscore_core.compatibility import CompatibilityError, check_compatibility


def _clip_table(clips, trs):
    rows = [{'stimulus_id': f'{clip}_t{step:04d}', 'frame_id': f'{clip}_t{step:04d}',
             'clip_id': clip, 't_within_run': step,
             'video_path': f'/clips/{clip}/{step:04d}.mp4'}
            for clip in clips for step in range(trs)]
    table = StimulusSet(pd.DataFrame(rows))
    table.identifier = 'synthetic-algonauts-clips'
    table.stimulus_paths = {row['stimulus_id']: row['video_path'] for row in rows}
    return table


class _VideoOnly(Subject):
    """Declares only vision:video; answers one row per clip it receives."""

    identifier = 'video-only'
    in_channels = frozenset({'vision:video'})
    out_channels = frozenset({'neural:V1', 'neural:V4', 'neural:IT'})

    def __init__(self):
        self.paths = []

    def participate(self, session):
        events = drain_session_inputs(session)
        assert {event.channel for event in events} == {'vision:video'}
        stimuli = session_stimuli(session, events)
        ids = [str(s) for s in stimuli['stimulus_id']]
        self.paths += list(stimuli['video_path'])
        rng = np.random.default_rng(1)
        session.respond(StreamEvent(list(session.outputs)[0], NeuroidAssembly(
            rng.normal(size=(len(ids), 3)),
            coords={'stimulus_id': ('presentation', ids),
                    'neuroid_id': ('neuroid', ['u0', 'u1', 'u2'])},
            dims=['presentation', 'neuroid']), t_ms=None))


def _score_benchmark(tmp_path, monkeypatch, form, table):
    benchmark = Algonauts2025(participant=1, channels=CHANNELS[form], assembly_root=tmp_path,
                              stimulus_window=2, hrf_delay=1,
                              excluded_samples_start=2, excluded_samples_end=0)
    stimulus = [clip for clip in CLIPS for _ in range(TRS)]
    steps = list(range(TRS)) * len(CLIPS)
    values = np.random.default_rng(0).normal(size=(len(stimulus), 4))
    benchmark._assembly = xr.DataArray(
        values, dims=('presentation', 'neuroid'),
        coords={'stimulus_id': ('presentation', stimulus), 'run': ('presentation', stimulus),
                't_within_run': ('presentation', steps)})
    name = '_expand_to_per_TR_clips' if form == 'video' else '_expand_to_per_TR_frames'
    monkeypatch.setattr(benchmark, name, lambda: table)
    return benchmark


def test_a_video_only_subject_gets_one_answer_per_tr_and_a_score(tmp_path, monkeypatch):
    table = _clip_table(CLIPS, TRS)
    benchmark = _score_benchmark(tmp_path, monkeypatch, 'video', table)
    subject = _VideoOnly()
    score = benchmark(subject)
    assert np.isfinite(float(score))
    # One clip per TR, once in each region's session (V1, V4, IT).
    assert subject.paths == list(table['video_path']) * 3
    assert score.attrs['bands'] == ['IT', 'V1', 'V4']
    assert score.attrs['input_channels'] == ['vision:video']
    assert score.attrs['frames_per_clip'] == 30


def test_an_image_only_subject_is_not_applicable_to_video_and_scores_on_frames(tmp_path, monkeypatch):
    video = Algonauts2025(participant=1, channels=('vision:video',), assembly_root=tmp_path)
    frame = Algonauts2025(participant=1, assembly_root=tmp_path)
    image_only = Recorder()
    assert image_only.in_channels == frozenset({'vision:image'})
    with pytest.raises(CompatibilityError, match='takes in none of the inputs'):
        check_compatibility(image_only, video)
    assert check_compatibility(image_only, frame).kind == 'full'
    # A video-only subject is not applicable to the frame form either.
    with pytest.raises(CompatibilityError, match='takes in none of the inputs'):
        check_compatibility(_VideoOnly(), frame)

    rows = [{'stimulus_id': f'{clip}_t{step:04d}', 'frame_id': f'{clip}_t{step:04d}',
             'clip_id': clip, 't_within_run': step, 'image_file_name': f'/frames/{clip}/{step}.jpg'}
            for clip in CLIPS for step in range(TRS)]
    frames = StimulusSet(pd.DataFrame(rows))
    frames.identifier = 'synthetic-algonauts-frames'
    frames.stimulus_paths = {r['stimulus_id']: r['image_file_name'] for r in rows}
    benchmark = _score_benchmark(tmp_path, monkeypatch, 'image', frames)

    class Image(Recorder):
        def participate(self, session):
            stimuli = session_stimuli(session, drain_session_inputs(session))
            ids = [str(s) for s in stimuli['stimulus_id']]
            session.respond(StreamEvent(list(session.outputs)[0], NeuroidAssembly(
                np.random.default_rng(1).normal(size=(len(ids), 3)),
                coords={'stimulus_id': ('presentation', ids),
                        'neuroid_id': ('neuroid', ['u0', 'u1', 'u2'])},
                dims=['presentation', 'neuroid']), t_ms=None))

    score = benchmark(Image())
    assert np.isfinite(float(score))
    assert score.attrs['input_channels'] == ['vision:image'] and score.attrs['bands'] == ['IT', 'V1', 'V4']
    assert 'frames_per_clip' not in score.attrs
