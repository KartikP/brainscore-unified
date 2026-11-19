import hashlib
import json
from pathlib import Path
import subprocess
import wave

import numpy as np
import pytest

from umi_lab.media import extract_audio, ffmpeg_executable


def movie(directory, audio=True, delay=False):
    directory.mkdir(exist_ok=True)
    filename = directory / ('delayed.mp4' if delay else 'movie.mp4' if audio else 'silent.mp4')
    command = [ffmpeg_executable(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-threads', '1',
               '-f', 'lavfi', '-i', 'color=c=black:s=16x16:r=8:d=1']
    if audio:
        if delay:
            command += ['-itsoffset', '0.3']
        command += ['-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=44100:duration=0.7']
    command += ['-t', '1', '-c:v', 'mpeg4']
    if audio:
        command += ['-c:a', 'aac']
    command += [str(filename)]
    subprocess.run(command, check=True, capture_output=True)
    return filename


def test_soundtrack_becomes_a_cached_waveform_without_changing_video(tmp_path, monkeypatch):
    source = movie(tmp_path / 'uploads')
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    result = extract_audio(source.parent, source.name)
    target = source.parent / result['audio_asset']
    with wave.open(str(target), 'rb') as audio:
        assert audio.getframerate() == 16000 and audio.getnchannels() == 1 and audio.getsampwidth() == 2
        assert audio.getnframes() > 10000
    assert result['source_video'] == source.name
    assert 'mono PCM16' in result['preparation']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('Cached extraction must not run FFmpeg'))
    assert extract_audio(source.parent, source.name) == result


def test_silent_movie_does_not_create_a_fake_audio_input(tmp_path):
    source = movie(tmp_path / 'uploads', audio=False)
    result = extract_audio(source.parent, source.name)
    assert result['status'] == 'no_audio' and result['audio_asset'] is None
    assert not list(source.parent.glob('*.wav'))


def test_audio_offset_preserves_a_leading_gap(tmp_path):
    source = movie(tmp_path / 'uploads', delay=True)
    result = extract_audio(source.parent, source.name)
    with wave.open(str(source.parent / result['audio_asset']), 'rb') as audio:
        values = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2')
    assert np.max(np.abs(values[:3200])) == 0
    assert np.max(np.abs(values[8000:])) > 100


@pytest.mark.parametrize('asset', ['../outside.mp4', '/outside.mp4', 'absent.mp4', 'file.wav'])
def test_extraction_accepts_only_uploaded_video_paths(tmp_path, asset):
    with pytest.raises(ValueError):
        extract_audio(tmp_path, asset)


def test_extracted_pair_satisfies_a_subject_that_requires_video_and_audio(tmp_path):
    from brainscore_core import Subject
    from umi_lab.protocol import validate_protocol
    from umi_lab.workbench import Request
    source = movie(tmp_path / 'uploads')
    result = extract_audio(source.parent, source.name)
    class AudiovisualSubject(Subject):
        identifier = 'paired-fixture'
        in_channels = {'vision:video', 'audio'}
        required_channels = {'vision:video', 'audio'}
        out_channels = {'neural:IT', 'neural:A1'}
        def participate(self, session):
            raise AssertionError('Preflight must not run the subject')
    request = Request(mode='protocol', rows=[{'video_path': source.name}], protocol={
        'identifier': 'movie', 'task': {'kind': 'passive'},
        'measurements': {'regions': ['IT', 'A1'], 'behavior': False},
        'conditions': [{'name': 'baseline'}], 'repetitions': 1})
    with pytest.raises(ValueError, match='audio'):
        validate_protocol(AudiovisualSubject(), request, source.parent)
    request.rows[0]['audio_path'] = result['audio_asset']
    summary = validate_protocol(AudiovisualSubject(), request, source.parent)
    assert summary['presentations'] == 1
    assert set(summary['channels']) == {'neural:IT', 'neural:A1'}


def test_http_extraction_returns_the_paired_asset_and_rejects_busy_requests(tmp_path):
    import threading
    from types import SimpleNamespace
    from urllib.request import Request as HttpRequest, urlopen
    from urllib.error import HTTPError
    from umi_lab.server import make_server
    source = movie(tmp_path / 'uploads')
    state = {'phase': 'ready', 'run': None}
    controller = SimpleNamespace(root=tmp_path, token='media-fixture', snapshot=lambda: state)
    server = make_server(controller, tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/api/extract-audio'
    body = json.dumps({'asset': source.name}).encode()
    headers = {'Content-Type': 'application/json'}
    try:
        with pytest.raises(HTTPError) as denied:
            urlopen(HttpRequest(url, body, headers))
        assert denied.value.code == 403
        headers['X-Lab-Token'] = controller.token
        with urlopen(HttpRequest(url, body, headers)) as response:
            result = json.load(response)
        assert (source.parent / result['audio_asset']).is_file()
        state['phase'] = 'loading'
        with pytest.raises(HTTPError) as busy:
            urlopen(HttpRequest(url, body, headers))
        assert busy.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
