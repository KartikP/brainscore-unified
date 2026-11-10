"""Lahner2024 variants run any contract subject through participate only.

The subjects here declare channels and implement participate and reset only:
no process(), no start_recording(), no private attributes. Their responses
carry only stimulus_id on the presentation axis, so the benchmark must join
clip and frame time from its own stimulus tables.

Tiny synthetic stimuli and responses; no model weights, video or BOLD data.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from brainscore_core.contract import Subject
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import drain_session_inputs, session_stimuli
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

from brainscore.benchmarks.lahner2024 import _util
from brainscore.benchmarks.lahner2024.benchmark import Lahner2024BOLDMoments
from brainscore.benchmarks.lahner2024.benchmark_multimodal import (
    Lahner2024BOLDMoments_multimodal,
)
from brainscore.benchmarks.lahner2024.benchmark_timeresolved import (
    Lahner2024BOLDMoments_timeresolved,
)
from brainscore.benchmarks.lahner2024.benchmark_timeresolved_multimodal import (
    Lahner2024BOLDMoments_timeresolved_multimodal,
)

CLIPS = [f'stimulus_{index:04d}' for index in range(1, 11)]
SAMPLE_TIMES_MS = (500.0, 1500.0, 2500.0)
N_UNITS = 4


def _unit_values(stimulus_id: str, channel: str) -> np.ndarray:
    """Deterministic, distinct activity for each (stimulus, channel)."""
    seed = sum(ord(character) for character in stimulus_id + channel)
    return np.random.default_rng(seed).normal(size=N_UNITS).astype(np.float32)


class Recorder(Subject):
    """Records any requested neural:<region> for whatever it is shown."""

    identifier = 'participate-only'

    def __init__(self, in_channels, regions=('IT',)):
        self._in_channels = frozenset(in_channels)
        self._out_channels = frozenset(f'neural:{region}' for region in regions)
        self.seen = []   # (requested channels, stimulus ids) per session

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def participate(self, session):
        stimuli = session_stimuli(session, drain_session_inputs(session))
        ids = [str(stimulus_id) for stimulus_id in stimuli['stimulus_id']]
        channels = list(session.requested_output_channels)
        self.seen.append((channels, ids))
        for channel in channels:
            values = np.stack([_unit_values(stimulus_id, channel) for stimulus_id in ids])
            session.respond(StreamEvent(channel, NeuroidAssembly(
                values,
                coords={'stimulus_id': ('presentation', ids),
                        'neuroid_id': ('neuroid', [f'{channel}.{unit}' for unit in range(N_UNITS)])},
                dims=['presentation', 'neuroid']), t_ms=None))


def _videos():
    frame = pd.DataFrame({'stimulus_id': CLIPS,
                          'video_path': [f'/clips/{clip}.mp4' for clip in CLIPS]})
    videos = StimulusSet(frame)
    videos.identifier = 'synthetic-videos'
    videos.stimulus_paths = dict(zip(frame['stimulus_id'], frame['video_path']))
    return videos


def _frames():
    rows = [{'stimulus_id': f'{clip}_t{int(t_ms)}', 'clip_id': clip, 'frame_time_ms': t_ms,
             'image_file_name': f'/frames/{clip}_{int(t_ms)}.jpg'}
            for clip in CLIPS for t_ms in SAMPLE_TIMES_MS]
    frames = StimulusSet(pd.DataFrame(rows))
    frames.identifier = 'synthetic-videos-frames'
    frames.stimulus_paths = {row['stimulus_id']: row['image_file_name'] for row in rows}
    return frames


def _audio():
    frame = pd.DataFrame({'stimulus_id': CLIPS,
                          'audio_path': [f'/audio/{clip}.wav' for clip in CLIPS]})
    audio = StimulusSet(frame)
    audio.identifier = 'synthetic-audio'
    audio.stimulus_paths = dict(zip(frame['stimulus_id'], frame['audio_path']))
    return audio


def _neural(n_voxels=3):
    values = np.random.default_rng(0).normal(size=(n_voxels, len(CLIPS)))
    return xr.DataArray(values, dims=('neuroid', 'stimulus_id'),
                        coords={'stimulus_id': CLIPS})


def _glm(benchmark, monkeypatch):
    monkeypatch.setattr(benchmark, '_videos_stimulus_set', _videos)
    monkeypatch.setattr(benchmark, '_video_stim_set', _videos, raising=False)
    monkeypatch.setattr(benchmark, '_expand_videos', _frames)
    monkeypatch.setattr(benchmark, '_audio_stim_set', _audio, raising=False)
    monkeypatch.setattr(benchmark, '_average_repetitions', _neural)
    monkeypatch.setattr(benchmark, '_get_voxel_mask', lambda: None)
    return benchmark


# ── shared helpers ────────────────────────────────────────────────


def test_frames_are_pooled_per_clip_from_the_benchmark_table():
    subject = Recorder({'vision:still'})
    pooled = _util.record_frames_per_clip(subject, _frames(), 'IT', [(0, 3000)])
    assert list(pooled.indexes['presentation'].get_level_values('clip_id')) == CLIPS
    expected = np.mean([_unit_values(f'{CLIPS[0]}_t{int(t)}', 'neural:IT')
                        for t in SAMPLE_TIMES_MS], axis=0)
    np.testing.assert_allclose(pooled.values[0, 0, :], expected, rtol=1e-6)


def test_routing_reads_declared_channels_only():
    assert _util.takes_audiovisual(Recorder({'vision:clip', 'audio'}, regions=('IT', 'A1')))
    assert _util.takes_audiovisual(Recorder({'vision', 'audio'}, regions=('IT', 'A1')))
    assert not _util.takes_audiovisual(Recorder({'vision:clip'}, regions=('IT', 'A1')))
    assert not _util.takes_audiovisual(Recorder({'vision:clip', 'audio'}, regions=('IT',)))


# ── GLM-beta variant ──────────────────────────────────────────────


@pytest.mark.parametrize('in_channels, pipeline, rows', [
    ({'vision:clip'}, 'video_native', len(CLIPS)),
    ({'vision:still'}, 'frame_aggregation', len(CLIPS) * len(SAMPLE_TIMES_MS)),
])
def test_glm_variant_scores_a_participate_only_subject(monkeypatch, in_channels, pipeline, rows):
    benchmark = _glm(Lahner2024BOLDMoments(sample_times_ms=list(SAMPLE_TIMES_MS)), monkeypatch)
    subject = Recorder(in_channels)
    score = benchmark(subject)
    assert np.isfinite(float(score))
    assert score.attrs['pipeline'] == pipeline
    assert score.attrs['n_videos'] == len(CLIPS)
    assert [channels for channels, _ in subject.seen] == [['neural:IT']]
    assert len(subject.seen[0][1]) == rows


# ── GLM-beta multimodal variant ───────────────────────────────────


def test_multimodal_variant_scores_a_participate_only_subject(monkeypatch):
    benchmark = _glm(Lahner2024BOLDMoments_multimodal(), monkeypatch)
    subject = Recorder({'vision:clip', 'audio'}, regions=('IT', 'A1'))
    score = benchmark(subject)
    assert np.isfinite(float(score))
    assert [channels for channels, _ in subject.seen] == [['neural:IT'], ['neural:A1']]


def test_multimodal_variant_is_refused_before_any_compute(monkeypatch):
    benchmark = _glm(Lahner2024BOLDMoments_multimodal(), monkeypatch)
    subject = Recorder({'vision:clip'}, regions=('IT',))
    with pytest.raises(ValueError, match='both'):
        benchmark(subject)
    assert subject.seen == []


# ── TR-resolved variants: feature extraction ──────────────────────


def _tr(benchmark, monkeypatch):
    benchmark._events = pd.DataFrame({'stimulus_id': CLIPS})
    helper = benchmark._stim_helper
    monkeypatch.setattr(type(helper), 'stimulus_set', property(lambda self: _videos()))
    monkeypatch.setattr(helper, '_videos_stimulus_set', _videos)
    monkeypatch.setattr(helper, '_expand_videos', _frames)
    return benchmark


@pytest.mark.parametrize('in_channels', [{'vision:clip'}, {'vision:still'}])
def test_tr_variant_extracts_features_through_participate(monkeypatch, in_channels):
    benchmark = _tr(Lahner2024BOLDMoments_timeresolved(), monkeypatch)
    features, stimulus_ids = benchmark._extract_per_stimulus_features(Recorder(in_channels))
    assert features.shape == (len(CLIPS), N_UNITS)
    assert [str(s) for s in stimulus_ids] == CLIPS


def test_tr_multimodal_variant_extracts_both_towers_through_participate(monkeypatch):
    benchmark = _tr(Lahner2024BOLDMoments_timeresolved_multimodal(), monkeypatch)
    monkeypatch.setattr(benchmark, '_video_stim_set_for', lambda ids: _videos())
    monkeypatch.setattr(benchmark, '_audio_stim_set_for', lambda ids: _audio())
    subject = Recorder({'vision:clip', 'audio'}, regions=('IT', 'A1'))
    features, stimulus_ids = benchmark._extract_per_stimulus_features(subject)
    assert features.shape == (len(CLIPS), 2 * N_UNITS)
    assert list(benchmark._modality_per_feature) == ['video'] * N_UNITS + ['audio'] * N_UNITS
    assert [channels for channels, _ in subject.seen] == [['neural:IT'], ['neural:A1']]


def test_tr_multimodal_variant_falls_back_to_one_tower_from_declarations(monkeypatch):
    benchmark = _tr(Lahner2024BOLDMoments_timeresolved_multimodal(), monkeypatch)
    subject = Recorder({'vision:clip'}, regions=('IT',))
    features, _ = benchmark._extract_per_stimulus_features(subject)
    assert benchmark._modality_per_feature is None
    assert features.shape == (len(CLIPS), N_UNITS)
    assert [channels for channels, _ in subject.seen] == [['neural:IT']]
