"""Lahner2024 benchmarks run any contract subject through participate only.

The subjects here declare channels and implement participate and reset only:
no process(), no start_recording(), no private attributes. Their responses
carry only stimulus_id on the presentation axis, so the benchmark must join
clip and frame time from its own stimulus tables.

Synthetic data behind the registered loaders (``tests/lahner_synthetic.py``);
no model weights, video or BOLD data.
"""
import numpy as np
import pytest

import brainscore
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import drain_session_inputs, session_stimuli
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

from brainscore.benchmarks.lahner2024 import _util, benchmark as glm_module
from brainscore.benchmarks.lahner2024 import benchmark_timeresolved as tr_module
from brainscore.benchmarks.lahner2024 import (
    Lahner2024BOLDMoments, Lahner2024BOLDMoments_timeresolved, RunBlocks)
from tests.lahner_synthetic import CLIPS, SAMPLE_TIMES, Recorder, install, unit_values

AUDIOVISUAL = ('vision:image', 'vision:video', 'audio')


def _frames():
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    import pandas as pd
    rows = [{'stimulus_id': f'{clip}_t{t}', 'clip_id': clip, 'frame_time_ms': float(t),
             'image_file_name': f'/frames/{clip}_{t}.png'} for clip in CLIPS for t in SAMPLE_TIMES]
    frames = StimulusSet(pd.DataFrame(rows))
    frames.identifier = 'synthetic-frames'
    frames.stimulus_paths = {row['stimulus_id']: row['image_file_name'] for row in rows}
    return frames


def _tr(**kwargs):
    return Lahner2024BOLDMoments_timeresolved(folds=RunBlocks(2), **kwargs)


# ── shared helpers ────────────────────────────────────────────────


def test_frames_are_pooled_per_clip_from_the_benchmark_table():
    subject = Recorder({'vision:image'})
    pooled = _util.record_frames_per_clip(subject, _frames(), 'IT')
    assert list(pooled.indexes['presentation'].get_level_values('clip_id')) == CLIPS
    frames = [unit_values(f'{CLIPS[0]}_t{t}', 'neural:IT')[0] for t in SAMPLE_TIMES]
    np.testing.assert_array_equal(pooled.values[0, 0, :], np.stack(frames).mean(axis=0))


def test_routing_reads_declared_channels_only():
    def taken(in_channels):
        return _util.taken_inputs(Recorder(in_channels), AUDIOVISUAL)
    assert taken({'vision:video', 'audio'}) == {'vision:video', 'audio'}
    assert taken({'vision:image', 'audio'}) == {'vision:image', 'audio'}
    assert taken({'vision:video'}) == {'vision:video'}
    assert taken({'vision', 'audio'}) == {'audio'}   # a bare name is no form


# ── GLM betas ─────────────────────────────────────────────────────


@pytest.mark.parametrize('in_channels, pipeline, rows, channel', [
    ({'vision:video'}, 'video_native', len(CLIPS), 'vision:video'),
    ({'vision:image'}, 'frame_aggregation', len(CLIPS) * len(SAMPLE_TIMES), 'vision:image'),
])
def test_glm_scores_a_participate_only_subject(monkeypatch, tmp_path, in_channels, pipeline,
                                               rows, channel):
    install(monkeypatch, tmp_path)
    subject = Recorder(in_channels, time_bins=3 if 'vision:video' in in_channels else None)
    score = Lahner2024BOLDMoments()(subject)
    assert np.isfinite(float(score))
    assert score.attrs['pipeline'] == pipeline
    assert score.attrs['n_videos'] == len(CLIPS)
    [(inputs, outputs, ids, _)] = subject.seen
    assert (inputs, outputs, len(ids)) == ([channel], ['neural:IT'], rows)


def test_audiovisual_glm_records_both_streams(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path)
    subject = Recorder({'vision:video', 'audio'}, regions=('IT', 'A1'))
    score = Lahner2024BOLDMoments(channels=AUDIOVISUAL)(subject)
    assert np.isfinite(float(score))
    assert [(inputs, outputs) for inputs, outputs, _, _ in subject.seen] == [
        (['vision:video'], ['neural:IT']), (['audio'], ['neural:A1'])]
    assert (score.attrs['n_features_video'], score.attrs['n_features_audio']) == (4, 4)
    assert score.attrs['scored_inputs'] == ['audio', 'vision:video']
    assert (score.attrs['mode'], score.attrs['pipeline']) == ('concat', 'multimodal_av_concat')


@pytest.mark.parametrize('build', [lambda: Lahner2024BOLDMoments(channels=AUDIOVISUAL),
                                   lambda: _tr(channels=AUDIOVISUAL)], ids=['glm', 'tr'])
def test_audiovisual_scores_the_visual_stream_alone_with_a_true_label(monkeypatch, tmp_path, build):
    """A partial match runs (rule 5.5) on the stream the subject takes; the score
    names it and carries no two-stream label. Both classes give the same answer."""
    install(monkeypatch, tmp_path)
    subject = Recorder({'vision:video'}, regions=('IT', 'A1'), time_bins=3)
    score = build()(subject)
    assert np.isfinite(float(score))
    assert [(inputs, outputs) for inputs, outputs, _, _ in subject.seen] == [
        (['vision:video'], ['neural:IT'])]
    assert score.attrs['scored_inputs'] == ['vision:video']
    assert 'mode' not in score.attrs and 'n_features_audio' not in score.attrs
    assert score.attrs['pipeline'] in ('video_native', 'continuous_time_encoding')


@pytest.mark.parametrize('build', [lambda: Lahner2024BOLDMoments(channels=AUDIOVISUAL),
                                   lambda: _tr(channels=AUDIOVISUAL)], ids=['glm', 'tr'])
def test_audiovisual_scores_the_sound_alone_for_a_subject_without_a_visual_form(
        monkeypatch, tmp_path, build):
    install(monkeypatch, tmp_path)
    subject = Recorder({'audio'}, regions=('IT', 'A1'))
    score = build()(subject)
    assert np.isfinite(float(score))
    assert [(inputs, outputs) for inputs, outputs, _, _ in subject.seen] == [(['audio'], ['neural:A1'])]
    assert score.attrs['scored_inputs'] == ['audio'] and score.attrs['n_features_audio'] == 4
    assert 'mode' not in score.attrs and 'n_features_video' not in score.attrs
    assert 'multimodal' not in score.attrs.get('pipeline', '')


def test_unknown_channels_are_refused():
    with pytest.raises(ValueError, match='channels must be one of'):
        Lahner2024BOLDMoments(channels=('vision:image',))


# ── TR time series ────────────────────────────────────────────────


@pytest.mark.parametrize('in_channels', [{'vision:video'}, {'vision:image'}])
def test_tr_records_each_shown_clip_once(monkeypatch, tmp_path, in_channels):
    install(monkeypatch, tmp_path)
    subject = Recorder(in_channels)
    score = _tr()(subject)
    assert np.isfinite(float(score))
    [(_, outputs, ids, _)] = subject.seen
    shown = {clip for clip in CLIPS}   # every clip appears in some run
    clips = ids if 'vision:video' in in_channels else sorted({i.split('_t')[0] for i in ids})
    assert outputs == ['neural:IT'] and set(clips) == shown


def test_audiovisual_tr_records_both_streams(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path)
    subject = Recorder({'vision:video', 'audio'}, regions=('IT', 'A1'))
    score = _tr(channels=AUDIOVISUAL)(subject)
    assert [(inputs, outputs) for inputs, outputs, _, _ in subject.seen] == [
        (['vision:video'], ['neural:IT']), (['audio'], ['neural:A1'])]
    assert (score.attrs['n_features_video'], score.attrs['n_features_audio']) == (4, 4)
    assert score.attrs['pipeline'] == 'tr_resolved_multimodal_concat'


# ── Old path vs participate on the SAME subject (Contract Rule 10.1) ──────
#
# The old code called start_recording + process, then temporal_bin on the frame
# response, and routed on the declared inputs. The functions below are those
# removed lines. Each benchmark runs once through participate and once with them
# patched in, on the same BrainScoreModel; results must be identical.

from brainscore import BrainScoreModel, LayerSelector
from brainscore.temporal import temporal_bin


class _Extractor:
    """Deterministic activity per (stimulus, layer); keeps the stimulus columns."""

    identifier = 'synthetic-lahner-extractor'

    def __init__(self, width, salt):
        self.width, self.salt = width, salt

    def __call__(self, stimuli, layers):
        blocks = [np.stack([unit_values(f'{self.salt}{layer}{sid}', 'x', self.width)[0]
                            for sid in stimuli['stimulus_id']]) for layer in layers]
        return NeuroidAssembly(
            np.concatenate(blocks, axis=1), dims=['presentation', 'neuroid'], coords={
                **{column: ('presentation', list(stimuli[column])) for column in stimuli.columns},
                'layer': ('neuroid', [layer for layer in layers for _ in range(self.width)]),
                'neuroid_id': ('neuroid', [f'{layer}.{i}' for layer in layers
                                           for i in range(self.width)]),
            })


def _model(visual_key):
    """An audio-visual model; ``vision`` + ``text`` takes stills, ``video`` clips."""
    extractors = {visual_key: _Extractor(3, 'v'), 'audio': _Extractor(2, 'a')}
    if visual_key == 'vision':
        extractors['text'] = _Extractor(3, 't')   # vision + text, like CLIP
    return BrainScoreModel(f'synthetic-{visual_key}',
                           region_layer_map={
                               'IT': LayerSelector(name='visual.layer', modality=visual_key),
                               'A1': LayerSelector(name='audio.layer', modality='audio')},
                           extractor=extractors)


def _old_record_clips(subject, clips, region):
    subject.start_recording(region, time_bins=_util.TIME_BINS)
    return subject.process(clips)


def _old_record_frames_per_clip(subject, frames, region):
    subject.start_recording(region, time_bins=_util.TIME_BINS)
    return temporal_bin(subject.process(frames), time_bins=_util.TIME_BINS)


def _both(monkeypatch, run):
    new = run()
    with monkeypatch.context() as patch:
        for module in (_util, glm_module, tr_module):
            for name, old in (('record_clips', _old_record_clips),
                              ('record_frames_per_clip', _old_record_frames_per_clip)):
                if hasattr(module, name):
                    patch.setattr(module, name, old)
        old = run()
    return old, new


def _same_score(old, new):
    assert repr(float(old)) == repr(float(new))
    np.testing.assert_array_equal(np.asarray(old.attrs['mean_r']), np.asarray(new.attrs['mean_r']))
    assert old.attrs['pipeline'] == new.attrs['pipeline']


@pytest.mark.parametrize('visual_key', ['vision', 'video'])
@pytest.mark.parametrize('build', [
    lambda: Lahner2024BOLDMoments(),
    lambda: Lahner2024BOLDMoments(channels=AUDIOVISUAL),
    lambda: _tr(),
    lambda: _tr(channels=AUDIOVISUAL),
], ids=['glm', 'glm-audiovisual', 'tr', 'tr-audiovisual'])
def test_old_and_new_path_give_the_same_score(monkeypatch, tmp_path, visual_key, build):
    install(monkeypatch, tmp_path)
    benchmark, model = build(), _model(visual_key)
    old, new = _both(monkeypatch, lambda: benchmark(model))
    _same_score(old, new)


def test_frame_pooling_is_identical_to_the_old_temporal_bin():
    model = _model('vision')
    old = _old_record_frames_per_clip(model, _frames(), 'IT')
    new = _util.record_frames_per_clip(model, _frames(), 'IT')
    np.testing.assert_array_equal(old.values, new.values)
    assert list(old['clip_id'].values) == list(new['clip_id'].values)


# ── Response ids are checked ──────────────────────────────────────


class _WrongIds(Recorder):
    """Answers with the right number of rows but other stimulus ids."""

    def participate(self, session):
        stimuli = session_stimuli(session, drain_session_inputs(session))
        ids = [f'other_{index}' for index in range(len(stimuli))]
        for channel in session.outputs:
            session.respond(StreamEvent(channel, NeuroidAssembly(
                np.zeros((len(ids), 4)),
                coords={'stimulus_id': ('presentation', ids),
                        'neuroid_id': ('neuroid', list(range(4)))},
                dims=['presentation', 'neuroid']), t_ms=None))


@pytest.mark.parametrize('in_channels', [{'vision:video'}, {'vision:image'}])
def test_a_response_for_other_stimuli_is_refused(monkeypatch, tmp_path, in_channels):
    install(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match='exactly once'):
        Lahner2024BOLDMoments()(_WrongIds(in_channels))


# ── Declarations (Contract Rules 5.1, 5.2, 5.12) ──────────────────

from tests.benchmark_declarations import assert_declares, assert_preflight

SINGLE_TOWER = [
    'Lahner2024-fMRI-naturalistic',
    'Lahner2024-fMRI-naturalistic-visualROI',
    'Lahner2024-fMRI-naturalistic-timeresolved',
    'Lahner2024-fMRI-naturalistic-timeresolved-visualROI',
    'Lahner2024-fMRI-naturalistic-timeresolved-improved',
    'Lahner2024-fMRI-naturalistic-timeresolved-improved-visualROI',
]
TWO_TOWERS = [
    'Lahner2024-fMRI-naturalistic-multimodal',
    'Lahner2024-fMRI-naturalistic-multimodal-visualROI',
    'Lahner2024-fMRI-naturalistic-multimodal-auditoryROI',
    'Lahner2024-fMRI-naturalistic-timeresolved-multimodal',
    'Lahner2024-fMRI-naturalistic-timeresolved-multimodal-visualROI',
    'Lahner2024-fMRI-naturalistic-timeresolved-multimodal-auditoryROI',
]


@pytest.mark.parametrize('identifier', SINGLE_TOWER)
def test_single_tower_variants_declare_vision_in_and_it_out(identifier):
    benchmark = brainscore.benchmark_registry[identifier]()
    assert_declares(benchmark, {'vision:image', 'vision:video'}, {'neural:IT'})
    assert_preflight(benchmark)


@pytest.mark.parametrize('identifier', TWO_TOWERS)
def test_audio_visual_variants_declare_both_towers(identifier):
    benchmark = brainscore.benchmark_registry[identifier]()
    assert_declares(benchmark, {'vision:image', 'vision:video', 'audio'}, {'neural:IT', 'neural:A1'})
    assert_preflight(benchmark)
