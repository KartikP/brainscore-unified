import urllib.request

import pytest

from umi_lab.downloads import observe_downloads


def test_urlretrieve_preserves_file_callbacks_and_restores_hooks(tmp_path):
    source = tmp_path / 'checkpoint.bin'
    source.write_bytes(b'x' * 20000)
    target = tmp_path / 'copied.bin'
    events, callbacks = [], []
    original = urllib.request.urlretrieve
    with observe_downloads(events.append):
        urllib.request.urlretrieve(source.as_uri(), target, lambda *values: callbacks.append(values))
    assert target.read_bytes() == source.read_bytes()
    assert callbacks and urllib.request.urlretrieve is original
    final = events[-1]['download']
    assert final['downloaded_bytes'] == 20000 and final['total_bytes'] == 20000
    assert final['files_completed'] == 1 and final['file'] == 'checkpoint.bin'


def test_huggingface_byte_progress_includes_resume_even_when_terminal_is_disabled():
    from huggingface_hub.utils import tqdm
    from tqdm.std import tqdm as standard
    original = standard.update
    events = []
    with observe_downloads(events.append):
        with tqdm(unit='B', total=100, initial=40, desc='model.safetensors', disable=True) as progress:
            progress.update(60)
        with tqdm(total=3, desc='Not a byte counter', disable=True) as progress:
            progress.update(3)
    assert standard.update is original
    assert events[-1]['download']['downloaded_bytes'] == 100
    assert events[-1]['download']['files_completed'] == 1


def test_unknown_size_and_failed_load_do_not_report_success(tmp_path):
    from tqdm.std import tqdm
    events = []
    original = tqdm.close
    with pytest.raises(RuntimeError, match='fixture failure'):
        with observe_downloads(events.append):
            with tqdm(unit='B', total=None, desc='https://example.test/model.bin?secret=hidden', disable=True) as progress:
                progress.update(12)
            raise RuntimeError('fixture failure')
    assert tqdm.close is original
    assert events[-1]['download']['phase'] == 'failed'
    assert events[-1]['download']['total_bytes'] is None
    assert events[-1]['download']['files_completed'] == 0
    assert 'secret' not in str(events)


def test_updates_are_throttled_and_original_transfer_errors_propagate(tmp_path, monkeypatch):
    from umi_lab import downloads
    monkeypatch.setattr(downloads.time, 'monotonic', lambda: 1.)
    events = []
    from tqdm.std import tqdm
    with observe_downloads(events.append):
        with tqdm(unit='B', total=10000, disable=True) as progress:
            for _ in range(10000):
                progress.update(1)
    assert len(events) <= 4
    original = urllib.request.urlretrieve
    with pytest.raises(OSError):
        with observe_downloads(events.append):
            urllib.request.urlretrieve((tmp_path / 'absent.bin').as_uri(), tmp_path / 'output.bin')
    assert urllib.request.urlretrieve is original


def test_parallel_file_completion_keeps_the_other_transfer_visible():
    from umi_lab.downloads import DownloadProgress
    events = []
    progress = DownloadProgress(events.append)
    first = progress.start('first.bin', total=100)
    second = progress.start('second.bin', total=200)
    progress.update(first, 100)
    progress.finish(first, True)
    assert events[-1]['download']['phase'] == 'downloading'
    assert events[-1]['download']['file'] == 'second.bin'
    assert events[-1]['download']['active_files'] == 1
    progress.update(second, 200)
    progress.finish(second, True)
    assert events[-1]['download']['files_completed'] == 2
