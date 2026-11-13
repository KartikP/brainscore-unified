"""The video+audio form of the Algonauts 2025 benchmark, on tiny synthetic data.

No model weights, no downloads, no real movies. The ffmpeg tests make a short movie
with ffmpeg's ``testsrc`` and ``sine`` and skip when ffmpeg is absent.
"""
import shutil
import subprocess
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import brainscore
from brainscore import BrainScoreModel, LayerSelector
from brainscore.benchmarks.algonauts2025 import benchmark as algo
from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends, TR_SEC
from brainscore_core.compatibility import CompatibilityError, check_compatibility
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

from tests.benchmark_declarations import Declared, assert_declares, assert_preflight

needs_ffmpeg = pytest.mark.skipif(
    shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None,
    reason='ffmpeg and ffprobe are needed')

SAMPLES = round(TR_SEC * 16000)


def _benchmark(tmp_path, **kwargs):
    return Algonauts2025Friends(participant=1, form='video+audio', assembly_root=tmp_path, **kwargs)


def _samples(path):
    with wave.open(str(path)) as wav:
        return wav.getnframes(), wav.getframerate(), wav.getnchannels()


# ── The sound command and the cutter ──────────────────────────────


def test_sound_command_cuts_16khz_mono_chunks_of_one_tr():
    cmd = algo._sound_ffmpeg_command('/m.mkv', '/out/%04d.wav', n_TRs=7)
    assert cmd[:6] == ['ffmpeg', '-y', '-loglevel', 'error', '-i', '/m.mkv']
    assert cmd[cmd.index('-t') + 1] == f'{TR_SEC * 7:.4f}'
    assert '-vn' in cmd
    audio_filter = cmd[cmd.index('-af') + 1]
    assert 'aresample=16000' in audio_filter and 'channel_layouts=mono' in audio_filter
    assert f'asetnsamples=n={SAMPLES}:p=0' in audio_filter
    assert cmd[cmd.index('-f') + 1] == 'segment'
    assert cmd[cmd.index('-segment_time') + 1] == f'{TR_SEC}'
    assert cmd[cmd.index('-segment_format') + 1] == 'wav'
    assert cmd[-1] == '/out/%04d.wav'
    assert '-t' not in algo._sound_ffmpeg_command('/m.mkv', '/o/%04d.wav')


N_FULL = 6
MOVIE_SECONDS = N_FULL * TR_SEC + 0.7   # ends in the middle of a TR


@pytest.fixture(scope='module')
def movie(tmp_path_factory):
    path = tmp_path_factory.mktemp('movie') / 'movie.mkv'
    subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error',
         '-f', 'lavfi', '-i', f'testsrc=duration={MOVIE_SECONDS}:size=320x240:rate=30000/1001',
         '-f', 'lavfi', '-i', f'sine=frequency=440:duration={MOVIE_SECONDS}:sample_rate=44100',
         '-shortest', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ac', '2',
         str(path)], check=True)
    return path


@needs_ffmpeg
def test_every_sound_chunk_has_one_tr_of_16khz_mono_samples(movie, tmp_path):
    out = tmp_path / 'sound'
    out.mkdir()
    algo._ffmpeg_cut_sound_one((str(movie), str(out), N_FULL + 3))
    chunks = sorted(out.glob('*.wav'))
    # The 0.7 s rest is no chunk; nothing past the movie is cut.
    assert [c.name for c in chunks] == [f'{t:04d}.wav' for t in range(N_FULL)]
    for chunk in chunks:
        assert _samples(chunk) == (SAMPLES, 16000, 1)
    assert not (out / '_cut').exists()


@needs_ffmpeg
def test_the_benchmark_cuts_clips_and_sound_into_one_table(movie, tmp_path):
    benchmark = _benchmark(tmp_path)
    benchmark._stimulus_set = pd.DataFrame({'stimulus_id': ['m1'], 'video_path': [str(movie)]})
    n_TRs = N_FULL + 2                   # the scan goes on for 2 TRs after the movie ends
    benchmark._stim_id_to_n_TRs = lambda: {'m1': n_TRs}
    table = benchmark._expand_to_per_TR_stimuli()
    assert isinstance(table, StimulusSet)
    assert list(table['stimulus_id']) == [f'm1_t{t:04d}' for t in range(n_TRs)]
    assert list(table['audio_path']) == [
        str(tmp_path / 'sound' / 'm1' / f'{t:04d}.wav') for t in range(n_TRs)]
    assert list(table['video_path']) == [
        str(tmp_path / 'clips' / 'm1' / f'{t:04d}.mp4') for t in range(n_TRs)]
    for path in table['audio_path']:
        assert _samples(path) == (SAMPLES, 16000, 1)
    last = Path(table['audio_path'].iloc[N_FULL - 1]).read_bytes()
    for t in (N_FULL, N_FULL + 1):
        assert Path(table['audio_path'].iloc[t]).read_bytes() == last   # copies of the last chunk
    assert table.identifier.startswith('algonauts2025-friends-sub01-clips-sound-')


def _place_files(root, folder, stim_id, ext, n):
    (root / folder / stim_id).mkdir(parents=True, exist_ok=True)
    for t in range(n):
        (root / folder / stim_id / f'{t:04d}{ext}').write_bytes(b'')


def test_the_table_has_a_video_and_an_audio_column_per_tr(tmp_path):
    benchmark = _benchmark(tmp_path)
    (tmp_path / 'm1.mkv').write_bytes(b'')
    benchmark._stimulus_set = pd.DataFrame({'stimulus_id': ['m1'],
                                            'video_path': [str(tmp_path / 'm1.mkv')]})
    _place_files(tmp_path, 'clips', 'm1', '.mp4', 3)
    _place_files(tmp_path, 'sound', 'm1', '.wav', 3)
    with patch.object(Algonauts2025Friends, '_stim_id_to_n_TRs', return_value={'m1': 3}), \
            patch.object(Algonauts2025Friends, '_run_pool') as pool:
        table = benchmark._expand_to_per_TR_stimuli()
    pool.assert_not_called()                      # every file is there: nothing is cut
    assert set(table.columns) == {'stimulus_id', 'frame_id', 'clip_id', 't_within_run',
                                  'video_path', 'audio_path'}
    assert len(table) == 3
    assert list(table['audio_path']) == [str(tmp_path / 'sound' / 'm1' / f'{t:04d}.wav')
                                         for t in range(3)]


def test_a_movie_whose_sound_could_not_be_cut_stops_the_run(tmp_path):
    benchmark = _benchmark(tmp_path)
    (tmp_path / 'm1.mkv').write_bytes(b'')
    benchmark._stimulus_set = pd.DataFrame({'stimulus_id': ['m1'],
                                            'video_path': [str(tmp_path / 'm1.mkv')]})
    _place_files(tmp_path, 'clips', 'm1', '.mp4', 2)   # clips are there, sound is not
    with patch.object(Algonauts2025Friends, '_stim_id_to_n_TRs', return_value={'m1': 2}), \
            patch.object(Algonauts2025Friends, '_run_pool'), \
            pytest.raises(FileNotFoundError, match='sound chunks were not cut'):
        benchmark._expand_to_per_TR_stimuli()


# ── Declarations and pre-flight ───────────────────────────────────

VIDEO_REGIONS = {'neural:V1', 'neural:V4', 'neural:IT'}
AV_REGIONS = VIDEO_REGIONS | {'neural:A1'}


def test_registered_identifier_loads_lazily_and_declares_its_channels(tmp_path):
    assert 'Algonauts2025-friends-sub01-video-audio' in brainscore.benchmark_registry
    assert 'Algonauts2025-friends-sub01-wholebrain' not in brainscore.benchmark_registry
    with patch.object(algo, 'load_dataset', side_effect=AssertionError('loaded data')), \
            patch.object(algo, 'load_stimulus_set', side_effect=AssertionError('loaded data')):
        benchmark = brainscore.benchmark_registry['Algonauts2025-friends-sub01-video-audio']()
    assert benchmark.identifier == 'Algonauts2025-friends-sub01-video-audio'
    assert_declares(benchmark, {'vision:video', 'audio'}, AV_REGIONS)
    assert_preflight(_benchmark(tmp_path))


@pytest.mark.parametrize('form, inputs, outputs', [
    ('image', {'vision:image'}, VIDEO_REGIONS),
    ('video', {'vision:video'}, VIDEO_REGIONS),
    ('video+audio', {'vision:video', 'audio'}, AV_REGIONS)])
def test_each_form_requests_named_regions_and_preflight_is_na_without_them(
        tmp_path, form, inputs, outputs):
    benchmark = Algonauts2025Friends(participant=1, form=form, assembly_root=tmp_path)
    assert benchmark.requested_output_channels == frozenset(outputs)
    model = {'image': _image_model, 'video': _video_model, 'video+audio': _av_model}[form]([])
    assert check_compatibility(model, benchmark).kind == 'full'
    # A subject without one requested region, or with no neural output, is N/A.
    for declared in (outputs - {'neural:V4'}, {'behavior'}):
        with pytest.raises(CompatibilityError, match='cannot produce output channels'):
            check_compatibility(Declared(inputs, declared), benchmark)
    # A subject that takes none of the inputs of the form is N/A.
    other = {'vision:image'} if form != 'image' else {'vision:video'}
    with pytest.raises(CompatibilityError, match='takes in none of the inputs'):
        check_compatibility(Declared(other, outputs), benchmark)


def _clips_and_pictures_model(calls):
    """The vision regions read clips; a second tower reads pictures."""
    return BrainScoreModel(
        identifier='clips-and-pictures', model=None,
        region_layer_map={region: LayerSelector(name=f'clip.{region}', modality='video')
                          for region in ('V1', 'V4', 'IT')}
                         | {'scene': LayerSelector(name='still.layer', modality='vision')},
        extractor={'video': _extractor('video_path', 1.0, VIDEO_BINS, calls),
                   'vision': _extractor('image_file_name', 3.0, 1, calls)})


def test_a_requested_region_whose_tower_gets_no_input_is_na_at_preflight(tmp_path, monkeypatch):
    # The image form sends pictures only, so no requested region can be answered.
    calls = []
    model = _clips_and_pictures_model(calls)
    image = Algonauts2025Friends(participant=1, assembly_root=tmp_path)
    na = r"answers 'neural:IT' only from the inputs \{'vision:video'\}"
    with pytest.raises(CompatibilityError, match=na):
        check_compatibility(model, image)
    # Through the benchmark: N/A before any cutting.
    image._expand_to_per_TR_stimuli = lambda: pytest.fail('nothing may be cut')
    with pytest.raises(CompatibilityError, match=na):
        image(model)
    # Through score(): N/A before the memory probe and before any cutting.
    from brainscore_core import memory
    monkeypatch.setattr(memory, 'check_memory',
                        lambda *args, **kwargs: pytest.fail('no memory probe'))
    with pytest.raises(CompatibilityError, match=na):
        brainscore.score(model, image)
    assert calls == []


def test_a_video_only_subject_runs_on_part_of_the_inputs_and_a_sound_subject_needs_sound(tmp_path):
    benchmark = _benchmark(tmp_path)
    match = check_compatibility(Declared({'vision:video'}, AV_REGIONS), benchmark)
    assert match.kind == 'partial' and match.dropped_inputs == frozenset({'audio'})
    # A model that needs sound gets N/A on the video form, which sends none.
    video = Algonauts2025Friends(participant=1, form='video', assembly_root=tmp_path)
    with pytest.raises(CompatibilityError, match='hard-requires'):
        check_compatibility(_av_model([]), video)
    assert check_compatibility(_av_model([]), benchmark).kind == 'full'


def test_the_constructor_has_no_mode_and_no_record_argument(tmp_path):
    for name in ('mode', 'record'):
        with pytest.raises(TypeError):
            Algonauts2025Friends(participant=1, assembly_root=tmp_path, **{name: 'banded'})


# ── Fake towers with different time axes ──────────────────────────


N_UNITS = 3
VIDEO_BINS, AUDIO_BINS = 4, 7       # the two towers answer on different time axes


def _values(paths, salt, n_bins):
    """``(rows, n_bins, N_UNITS)``; values depend on the file, so a wrong row shows up."""
    return np.array([[[float(sum(map(ord, p)) % 97) + salt * unit + 0.5 * step
                       for unit in range(N_UNITS)] for step in range(n_bins)]
                     for p in paths], dtype=np.float32)


def _extractor(column, salt, n_bins, calls):
    """One tower: one answer per row, with ``n_bins`` time bins over the 1.49 s."""
    def extract(stimuli, layers):
        calls.append((column, list(stimuli['stimulus_id']), sorted(stimuli.columns)))
        edges = np.linspace(0, TR_SEC * 1000, n_bins + 1)
        values = np.concatenate([_values(stimuli[column], salt, n_bins) + 10.0 * len(layer)
                                 for layer in layers], axis=2)
        return NeuroidAssembly(
            values, dims=['presentation', 'time_bin', 'neuroid'], coords={
                'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                'time_bin_start_ms': ('time_bin', edges[:-1]),
                'time_bin_end_ms': ('time_bin', edges[1:]),
                'neuroid_id': ('neuroid', [f'{layer}.{u}' for layer in layers
                                           for u in range(N_UNITS)]),
                'layer': ('neuroid', [layer for layer in layers for _ in range(N_UNITS)])})
    return extract


VIDEO_LAYERS = {'V1': 'backbone.blocks.4', 'V4': 'backbone.blocks.15',
                'IT': 'backbone.blocks.16'}


def _av_regions():
    return ({region: LayerSelector(name=layer, modality='video')
             for region, layer in VIDEO_LAYERS.items()}
            | {'A1': LayerSelector(name='encoder.layers.6', modality='audio')})


def _av_model(calls):
    """Declares what vjepa1-wav2vec2 declares (three video regions, one audio), with tiny towers."""
    return BrainScoreModel(
        identifier='fake-av', model=None, region_layer_map=_av_regions(),
        extractor={'video': _extractor('video_path', 1.0, VIDEO_BINS, calls),
                   'audio': _extractor('audio_path', 2.0, AUDIO_BINS, calls)},
        required_modalities={'video', 'audio'})


def _av_model_without_requirement(calls):
    return BrainScoreModel(
        identifier='fake-av-free', model=None, region_layer_map=_av_regions(),
        extractor={'video': _extractor('video_path', 1.0, VIDEO_BINS, calls),
                   'audio': _extractor('audio_path', 2.0, AUDIO_BINS, calls)})


def _video_model(calls):
    return BrainScoreModel(
        identifier='fake-video', model=None,
        region_layer_map={region: LayerSelector(name=layer, modality='video')
                          for region, layer in VIDEO_LAYERS.items()},
        extractor={'video': _extractor('video_path', 1.0, VIDEO_BINS, calls)})


IMAGE_LAYERS = {'V1': 'layer.2', 'V4': 'layer.6', 'IT': 'layer.10'}


def _image_model(calls=None):
    calls = [] if calls is None else calls
    return BrainScoreModel(
        identifier='fake-image', model=None,
        region_layer_map={region: LayerSelector(name=layer, modality='vision')
                          for region, layer in IMAGE_LAYERS.items()},
        extractor={'vision': _extractor('image_file_name', 3.0, 1, calls)})


def _vision_language_model(calls):
    """A vision-language fake: three vision regions and one text region."""
    def text(stimuli, layers):
        calls.append(('text', list(stimuli['stimulus_id']), sorted(stimuli.columns)))
        raise AssertionError('the image form asks for no text region')
    return BrainScoreModel(
        identifier='fake-vlm', model=None,
        region_layer_map={region: LayerSelector(name=layer, modality='vision')
                          for region, layer in IMAGE_LAYERS.items()}
                         | {'language_system': LayerSelector(name='text.layer.10',
                                                             modality='text')},
        extractor={'vision': _extractor('image_file_name', 3.0, 1, calls), 'text': text})


MOVIES = [f'friends_s01e0{e}{half}' for e in range(1, 5) for half in 'ab']   # 8 runs
N_TRS = 16
N_PARCELS = 4


def _table(movies=MOVIES, n=N_TRS, columns=('video_path', 'audio_path')):
    paths = {'video_path': '/clips/{m}/{t:04d}.mp4', 'audio_path': '/sound/{m}/{t:04d}.wav',
             'image_file_name': '/frames/{m}/{t:04d}.jpg'}
    rows = [{'stimulus_id': f'{m}_t{t:04d}', 'frame_id': f'{m}_t{t:04d}', 'clip_id': m,
             't_within_run': t, **{c: paths[c].format(m=m, t=t) for c in columns}}
            for m in movies for t in range(n)]
    table = StimulusSet(pd.DataFrame(rows))
    table.identifier = 'synthetic-' + '-'.join(columns)
    table.stimulus_paths = {r['stimulus_id']: r[columns[0]] for r in rows}
    return table


def _assembly(movies, n=N_TRS):
    stim = [m for m in movies for _ in range(n)]
    values = np.random.default_rng(0).normal(size=(len(stim), N_PARCELS)).astype(np.float32)
    return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', stim),
        'run': ('presentation', ['1'] * len(stim)),
        't_within_run': ('presentation', [t for _ in movies for t in range(n)]),
        'parcel_id': ('neuroid', list(range(N_PARCELS)))})


# ── Every requested region is one band ───────────────────────────


def test_every_requested_region_is_asked_for_on_its_own_channel_and_is_one_band(tmp_path):
    calls = []
    benchmark = _benchmark(tmp_path)
    benchmark._assembly = _assembly(MOVIES)
    regions = list(benchmark.FORM_REGIONS['video+audio'])
    assert regions == ['A1', 'IT', 'V1', 'V4']
    bands = benchmark._record_bands(_av_model(calls), _table(), regions)
    assert list(bands) == regions
    # Each band is its own region, averaged over that region's own time bins.
    table = _table()
    for region, column, salt, n_bins, layer in (
            ('A1', 'audio_path', 2.0, AUDIO_BINS, 'encoder.layers.6'),
            *((r, 'video_path', 1.0, VIDEO_BINS, l) for r, l in VIDEO_LAYERS.items())):
        features, ids = bands[region]
        expected = _values(table[column], salt, n_bins).mean(axis=1) + 10.0 * len(layer)
        np.testing.assert_allclose(features, expected, rtol=1e-6)
        assert list(ids) == [f'{layer}.{u}' for u in range(N_UNITS)]
    assert {column for column, *_ in calls} == {'video_path', 'audio_path'}


def test_output_inputs_lists_the_inputs_of_each_region():
    vlm = _vision_language_model([])
    assert vlm.output_inputs == {
        **{f'neural:{r}': frozenset({'vision:image'}) for r in IMAGE_LAYERS},
        'neural:language_system': frozenset({'text'})}
    av = _av_model([]).output_inputs
    assert av['neural:A1'] == frozenset({'audio'}) and av['neural:IT'] == frozenset({'vision:video'})


def test_a_region_the_benchmark_does_not_name_is_not_asked_for(tmp_path):
    calls = []
    model = _vision_language_model(calls)
    benchmark = Algonauts2025Friends(participant=1, assembly_root=tmp_path)
    assert check_compatibility(model, benchmark).kind == 'full'   # the form sends only images
    _, score = _score(tmp_path, False, calls, form='image', columns=('image_file_name',),
                      model=model)
    assert score.attrs['bands'] == ['IT', 'V1', 'V4'] and np.isfinite(float(score))
    assert {column for column, *_ in calls} == {'image_file_name'}    # no text request


def test_on_the_video_form_an_audiovisual_model_is_na_or_runs_with_the_video_regions(tmp_path):
    video = Algonauts2025Friends(participant=1, form='video', assembly_root=tmp_path)
    with pytest.raises(CompatibilityError, match='hard-requires'):
        check_compatibility(_av_model([]), video)
    calls = []
    model = _av_model_without_requirement(calls)
    assert check_compatibility(model, video).kind == 'full'
    _, score = _score(tmp_path, False, calls, form='video', columns=('video_path',), model=model)
    assert score.attrs['bands'] == ['IT', 'V1', 'V4']
    assert {column for column, *_ in calls} == {'video_path'}


@pytest.mark.parametrize('form, columns, make', [
    ('image', ('image_file_name',), _vision_language_model),
    ('video', ('video_path',), _av_model_without_requirement)])
def test_the_memory_probe_records_a_requested_region(tmp_path, form, columns, make):
    from brainscore_core.memory import _probe_feature_dim
    benchmark = Algonauts2025Friends(participant=1, form=form, assembly_root=tmp_path)
    width, region = _probe_feature_dim(make([]), benchmark, None,
                                       probe_stimuli=_table(MOVIES[:1], n=1, columns=columns))
    assert f'neural:{region}' in benchmark.requested_output_channels and width == N_UNITS


# ── Stream and table ──────────────────────────────────────────────


def _score(tmp_path, streaming, calls, form='video+audio', columns=('video_path', 'audio_path'),
           model=None):
    benchmark = Algonauts2025Friends(participant=1, form=form, assembly_root=tmp_path,
                                     streaming=streaming)
    benchmark._assembly = _assembly(MOVIES)
    benchmark._expand_to_per_TR_stimuli = lambda: _table(columns=columns)
    return benchmark, benchmark(model if model is not None else _av_model(calls))


def test_stream_and_table_give_equal_bands_and_scores(tmp_path):
    table_calls, stream_calls = [], []
    benchmark = _benchmark(tmp_path)
    regions = list(benchmark.FORM_REGIONS['video+audio'])
    table_bands = benchmark._record_bands(_av_model(table_calls), _table(), regions)
    streamed = _benchmark(tmp_path, streaming=True)
    stream_bands = streamed._record_bands(_av_model(stream_calls), _table(), regions)
    assert list(table_bands) == list(stream_bands) == regions
    for band in regions:
        np.testing.assert_array_equal(stream_bands[band][0], table_bands[band][0])
        np.testing.assert_array_equal(stream_bands[band][1], table_bands[band][1])
    # The table runs each region once on every row; the stream once per row and region.
    assert len(table_calls) == 4 and len(stream_calls) == 4 * len(_table())

    _, table_score = _score(tmp_path, False, [])
    _, stream_score = _score(tmp_path, True, [])
    assert float(table_score) == float(stream_score)
    np.testing.assert_array_equal(table_score.attrs['per_parcel_r'],
                                  stream_score.attrs['per_parcel_r'])
    assert table_score.attrs['penalties_per_fold'] == stream_score.attrs['penalties_per_fold']
    assert table_score.attrs['presentation'] == 'table'
    assert stream_score.attrs['presentation'] == 'stream'
    assert stream_score.attrs['bands'] == regions
    assert stream_score.attrs['band_channels'] == {r: f'neural:{r}' for r in regions}
    assert stream_score.attrs['form'] == 'video+audio'
    assert stream_score.attrs['feature_projection_protocol'] == (
        'centred_svd_and_scale_fitted_on_training_stimuli')
    folds = stream_score.attrs['penalties_per_fold']
    assert len(folds) == 5 and all(set(entry['penalties']) == set(regions)
                                   for fold in folds for entry in fold['tuples'])
    assert stream_score.attrs['normalized'] is False
    assert stream_score.attrs['ceiling_status'] == 'not_estimated'
    assert float(stream_score) == float(stream_score.attrs['raw'])
    assert stream_score.attrs['components_per_band'] == {r: N_UNITS for r in regions}
    assert stream_score.attrs['cv_groups'] == 'stimulus_id'
    assert 'movies' not in stream_score.attrs and 'mode' not in stream_score.attrs


def test_streaming_works_for_the_image_form_too(tmp_path):
    _, table_score = _score(tmp_path, False, [], form='image', columns=('image_file_name',),
                            model=_image_model())
    _, stream_score = _score(tmp_path, True, [], form='image', columns=('image_file_name',),
                             model=_image_model())
    np.testing.assert_array_equal(table_score.attrs['per_parcel_r'],
                                  stream_score.attrs['per_parcel_r'])
    assert stream_score.attrs['bands'] == ['IT', 'V1', 'V4']
    assert stream_score.attrs['form'] == 'image' and 'clip_seconds' not in stream_score.attrs


def test_in_the_stream_each_region_gets_one_row_with_both_columns_at_a_time(tmp_path):
    calls = []
    benchmark = _benchmark(tmp_path, streaming=True)
    table = _table(MOVIES[:1], n=4)
    benchmark._record_bands(_av_model(calls), table, ['A1', 'IT'])
    ids = list(table['stimulus_id'])
    # One session per region; in each, row by row and in order.
    assert [(column, rows) for column, rows, _ in calls] == [
        (column, [sid]) for column in ('audio_path', 'video_path') for sid in ids]
    for _, _, columns in calls:
        assert {'video_path', 'audio_path'} <= set(columns)


# ── movies ────────────────────────────────────────────────────────


def test_movies_limit_the_assembly_the_tables_and_the_cut_jobs(tmp_path):
    named = MOVIES[:7]
    benchmark = _benchmark(tmp_path, movies=named)
    with patch.object(algo, 'load_dataset', return_value=_assembly(MOVIES)):
        assembly = benchmark.assembly
    assert sorted(set(assembly['stimulus_id'].values)) == sorted(named)
    assert assembly.sizes['presentation'] == len(named) * N_TRS
    for movie in MOVIES:
        (tmp_path / f'{movie}.mkv').write_bytes(b'')
    benchmark._stimulus_set = pd.DataFrame({
        'stimulus_id': MOVIES, 'video_path': [str(tmp_path / f'{m}.mkv') for m in MOVIES]})
    jobs = []

    def cut(worker, planned, n_workers):
        jobs.append((worker, [Path(job[0]).stem for job in planned]))
        for video, out_dir, *_ in planned:   # stand in for ffmpeg
            ext = '.mp4' if worker is algo._ffmpeg_cut_clips_one else '.wav'
            for t in range(N_TRS):
                (Path(out_dir) / f'{t:04d}{ext}').write_bytes(b'')

    with patch.object(Algonauts2025Friends, '_run_pool', side_effect=cut):
        table = benchmark._expand_to_per_TR_stimuli()
    assert jobs == [(algo._ffmpeg_cut_clips_one, named), (algo._ffmpeg_cut_sound_one, named)]
    assert sorted(set(table['clip_id'])) == sorted(named)
    assert len(table) == len(named) * N_TRS
    assert not (tmp_path / 'clips' / MOVIES[-1]).exists()
    assert not (tmp_path / 'sound' / MOVIES[-1]).exists()


def test_the_score_names_the_movies(tmp_path):
    benchmark = _benchmark(tmp_path, movies=MOVIES[:7])
    benchmark._expand_to_per_TR_stimuli = lambda: _table(MOVIES[:7])
    with patch.object(algo, 'load_dataset', return_value=_assembly(MOVIES)):
        score = benchmark(_av_model([]))
    assert score.attrs['movies'] == MOVIES[:7]
    assert np.isfinite(float(score))


@pytest.mark.parametrize('n_movies, ok', [(6, False), (7, True)])
def test_too_few_runs_raise(tmp_path, n_movies, ok):
    benchmark = _benchmark(tmp_path, movies=MOVIES[:n_movies])
    with patch.object(algo, 'load_dataset', return_value=_assembly(MOVIES)):
        if ok:
            assert benchmark.assembly.sizes['presentation'] == n_movies * N_TRS
        else:
            with pytest.raises(ValueError, match='Name more movies'):
                benchmark.assembly


def test_one_held_out_film_can_be_selected_but_the_training_split_needs_more(tmp_path):
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025FriendsS7, Algonauts2025OOD
    for cls in (Algonauts2025FriendsS7, Algonauts2025OOD):
        benchmark = cls(participant=1, form='video+audio', assembly_root=tmp_path, movies=MOVIES[:1])
        kept = benchmark._only_movies(_assembly(MOVIES))
        assert kept.sizes['presentation'] == N_TRS
    with pytest.raises(ValueError, match='Name more movies'):
        _benchmark(tmp_path, movies=MOVIES[:1])._only_movies(_assembly(MOVIES))


def test_unknown_movies_raise(tmp_path):
    benchmark = _benchmark(tmp_path, movies=MOVIES[:6] + ['friends_s09e99a'])
    with patch.object(algo, 'load_dataset', return_value=_assembly(MOVIES)), \
            pytest.raises(ValueError, match='not in the assembly'):
        benchmark.assembly
    with pytest.raises(ValueError, match='non-empty list'):
        _benchmark(tmp_path, movies='friends_s01e01a')


def test_movies_apply_to_the_image_form_too(tmp_path):
    benchmark = Algonauts2025Friends(participant=1, movies=MOVIES[:7], assembly_root=tmp_path)
    assert benchmark.identifier == 'Algonauts2025-friends-sub01'
    with patch.object(algo, 'load_dataset', return_value=_assembly(MOVIES)):
        assert sorted(set(benchmark.assembly['stimulus_id'].values)) == sorted(MOVIES[:7])
