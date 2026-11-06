"""Replay uses exact saved stimulus IDs and native clocks without a subject."""
import json
from types import SimpleNamespace
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import xarray as xr

from brainscore.run_record import PayloadCodec
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from umi_lab.activation_data import temporal_page
from umi_lab.replay import byte_range, replay_manifest, replay_media


@pytest.fixture
def recording(tmp_path):
    root = tmp_path / 'fixture'
    inputs = root / 'inputs'; inputs.mkdir(parents=True)
    for stem in ('a', 'b'):
        (inputs / f'{stem}.mp4').write_bytes(bytes(range(64)))
        (inputs / f'{stem}.wav').write_bytes(b'RIFF-audio')
    rows = [{'stimulus_id': stem, 'video_path': f'{stem}.mp4', 'audio_path': f'{stem}.wav'} for stem in ('a', 'b')]
    request = {'mode': 'protocol', 'rows': rows, 'protocol': {
        'identifier': 'replay-fixture', 'task': {'kind': 'passive'},
        'measurements': {'regions': ['IT', 'A1']},
        'conditions': [{'name': 'baseline'}, {'name': 'changed'}], 'repetitions': 2}}
    (root / 'plan.json').write_text(json.dumps({'kind': 'workbench', 'request': request}))
    experiment = root / 'experiment'; experiment.mkdir()
    def output(times, step, ids, value):
        return NeuroidAssembly(np.arange(2 * times * 3).reshape(2, times, 3) + value,
            dims=['presentation', 'time_bin', 'neuroid'], coords={
                'stimulus_id': ('presentation', ids),
                'stimulus_path': ('presentation', [f'{value}.mp4' for value in ids]),
                'time_bin_start_ms': ('time_bin', np.arange(times) * step),
                'time_bin_end_ms': ('time_bin', (np.arange(times) + 1) * step)})
    results = [{'neural:IT': output(2, 40, ['b', 'a'], index * 100),
                'neural:A1': output(4, 20, ['a', 'b'], index * 100)} for index in range(4)]
    (experiment / 'result.json').write_text(json.dumps(PayloadCodec(experiment).encode(results)))
    return root


def test_replay_selects_condition_repetition_and_matches_stimuli_by_id(recording):
    saved = replay_manifest(recording, condition=1, repetition=1)
    assert saved['index'] == 3 and saved['conditions'] == ['baseline', 'changed']
    assert saved['repetitions'] == 2
    assert [row['id'] for row in saved['stimuli']] == ['a', 'b']
    video, audio = saved['channels']
    assert video['channel'] == 'neural:IT' and video['presentations'] == [1, 0]
    assert audio['presentations'] == [0, 1]
    assert len(video['time_bins']) == 2 and len(audio['time_bins']) == 4
    assert video['time_bins'][1]['start_ms'] == 40.
    assert audio['time_bins'][1]['start_ms'] == 20.
    page = temporal_page(recording / 'experiment', index=saved['index'], channel='neural:IT',
                         presentation=video['presentations'][0], labels=False)
    assert page['values'][0] == [306., 307., 308.] and page['labels'] == []
    assert saved['stimuli'][0]['media'][0]['url'] == '/api/runs/fixture/media?stimulus=0&kind=video'
    json.dumps(saved, allow_nan=False)


def test_valid_default_protocol_fields_and_missing_media_remain_explicit(recording):
    filename = recording / 'plan.json'; saved = json.loads(filename.read_text())
    saved['request']['protocol'].pop('conditions'); saved['request']['protocol'].pop('repetitions')
    filename.write_text(json.dumps(saved)); (recording / 'inputs/a.mp4').unlink()
    manifest = replay_manifest(recording)
    assert manifest['conditions'] == ['baseline'] and manifest['repetitions'] == 1
    assert manifest['stimuli'][0]['media'][0]['error']


def test_replay_refuses_ambiguous_ids_unknown_selection_and_large_clocks(recording, monkeypatch):
    for settings in ({'condition': 2}, {'repetition': 2}, {'repetition': -1}, {'condition': True}):
        with pytest.raises(ValueError): replay_manifest(recording, **settings)
    monkeypatch.setattr('umi_lab.replay.MAX_REPLAY_BINS', 3)
    with pytest.raises(ValueError, match='clocks exceed'): replay_manifest(recording)
    monkeypatch.setattr('umi_lab.replay.MAX_REPLAY_BINS', 65536)
    experiment = recording / 'experiment'
    assembly = xr.DataArray(np.zeros((2, 2)), dims=['presentation', 'neuroid'])
    (experiment / 'result.json').write_text(json.dumps(PayloadCodec(experiment).encode([{'neural:IT': assembly}])))
    with pytest.raises(ValueError, match='no stimulus IDs'): replay_manifest(recording)


def test_replay_media_never_exposes_other_files_or_escaping_symlinks(recording, tmp_path):
    filename, mime = replay_media(recording, stimulus=0, kind='video')
    assert filename == recording / 'inputs/a.mp4' and mime == 'video/mp4'
    for settings in ({'stimulus': 2, 'kind': 'video'}, {'stimulus': 0, 'kind': 'html'},
                     {'stimulus': -1, 'kind': 'audio'}):
        with pytest.raises(ValueError): replay_media(recording, **settings)
    (recording / 'inputs/a.mp4').unlink()
    outside = tmp_path / 'private.mp4'; outside.write_bytes(b'private')
    (recording / 'inputs/a.mp4').symlink_to(outside)
    with pytest.raises(ValueError): replay_media(recording, stimulus=0, kind='video')
    saved = json.loads((recording / 'plan.json').read_text())
    saved['request']['rows'][0]['video_path'] = '../private.mp4'
    (recording / 'plan.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError): replay_media(recording, stimulus=0, kind='video')


@pytest.mark.parametrize('header,expected', [(None, (0, 63, 200)), ('bytes=2-5', (2, 5, 206)),
    ('bytes=60-', (60, 63, 206)), ('bytes=-3', (61, 63, 206)), ('bytes=0-999', (0, 63, 206))])
def test_media_range_bounds(header, expected):
    assert byte_range(header, 64) == expected


@pytest.mark.parametrize('header', ['bytes=64-', 'bytes=4-2', 'bytes=-0', 'bytes=-', 'bytes=0-1,3-4', 'invalid'])
def test_invalid_or_unsatisfied_media_ranges(header):
    with pytest.raises(ValueError): byte_range(header, 64)


def test_replay_http_reads_saved_data_and_streams_byte_ranges_without_a_subject(recording, tmp_path):
    from umi_lab.server import make_server
    state = {'run': {'id': 'fixture', 'phase': 'complete'}}
    controller = SimpleNamespace(runs={'fixture': recording}, token='fixture', snapshot=lambda: state)
    server = make_server(controller, tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}/api/runs/fixture/'
    try:
        with urlopen(base + 'replay?condition=1&repetition=1') as response:
            assert json.load(response)['index'] == 3
        request = Request(base + 'media?stimulus=0&kind=video', headers={'Range': 'bytes=4-9'})
        with urlopen(request) as response:
            assert response.status == 206 and response.read() == bytes(range(4, 10))
            assert response.headers['Content-Range'] == 'bytes 4-9/64'
            assert response.headers['Accept-Ranges'] == 'bytes'
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + 'media?stimulus=0&kind=video', headers={'Range': 'bytes=99-'}))
        assert error.value.code == 416 and error.value.headers['Content-Range'] == 'bytes */64'
        for suffix in ('replay?condition=0&condition=1', 'replay?repetition=-1', 'media?file=../outside', 'media?kind=video'):
            with pytest.raises(HTTPError) as error: urlopen(base + suffix)
            assert error.value.code == 400
        state['run']['phase'] = 'running'
        for suffix in ('replay', 'media?stimulus=0&kind=video'):
            with pytest.raises(HTTPError) as error: urlopen(base + suffix)
            assert error.value.code == 409
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
