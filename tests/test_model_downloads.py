"""Offline checks: no checkpoint loader or network runs without opt-in."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from brainscore.models import _downloads as downloads


@pytest.fixture
def cache(tmp_path, monkeypatch):
    from transformers.utils import hub
    monkeypatch.setattr(hub, 'TRANSFORMERS_CACHE', str(tmp_path / 'hub'), raising=False)
    monkeypatch.delenv(downloads.SKIP_ENV, raising=False)
    monkeypatch.setattr('builtins.input', Mock(side_effect=AssertionError('prompted')))
    monkeypatch.setattr(downloads.shutil, 'disk_usage',
                        Mock(return_value=SimpleNamespace(free=100 * 10**9)))
    return tmp_path / 'hub'


def write_snapshot(cache, files):
    repo = cache / 'models--example--large'
    (repo / 'refs').mkdir(parents=True)
    (repo / 'refs/main').write_text('test-commit')
    snapshot = repo / 'snapshots/test-commit'
    snapshot.mkdir(parents=True)
    for name, content in files.items():
        (snapshot / name).write_text(content)
    return snapshot


def test_cold_cache_explains_model_disk_source_destination_and_opt_out(cache):
    with pytest.raises(downloads.ModelDownloadRequired) as error:
        downloads.hf_preflight('large-model', 'example/large', 15.5)
    message = str(error.value)
    for expected in ['large-model', '15.5 GB', '100.0 GB free', '18.1 GB',
                     str(cache / 'models--example--large'),
                     'https://huggingface.co/example/large',
                     downloads.SKIP_ENV + '=1', 'HF_HUB_CACHE']:
        assert expected in message
    assert not cache.exists()  # no directories or downloads as a side effect
    downloads.shutil.disk_usage.assert_called_once_with(cache.parent)


def test_low_disk_is_explicit(cache, monkeypatch):
    monkeypatch.setattr(downloads.shutil, 'disk_usage',
                        lambda _: SimpleNamespace(free=100))
    with pytest.raises(downloads.ModelDownloadRequired, match='Insufficient free disk'):
        downloads.hf_preflight('large', 'example/large', 15.5)


def test_failed_disk_check_is_actionable(cache, monkeypatch):
    monkeypatch.setattr(downloads.shutil, 'disk_usage', Mock(side_effect=OSError('unreadable')))
    with pytest.raises(downloads.ModelDownloadRequired, match='Could not check free disk'):
        downloads.hf_preflight('large', 'example/large', 15.5)


def test_modern_transformers_cache_without_legacy_constant(cache, monkeypatch):
    from huggingface_hub import constants
    from transformers.utils import hub
    monkeypatch.delattr(hub, 'TRANSFORMERS_CACHE')
    monkeypatch.setattr(constants, 'HF_HUB_CACHE', str(cache))
    monkeypatch.setenv(downloads.SKIP_ENV, '1')
    assert downloads.hf_preflight('large', 'example/large', 15.5) == {'cache_dir': str(cache)}


@pytest.mark.parametrize('weights', ['model.safetensors', 'pytorch_model.bin'])
def test_complete_weights_load_locally_without_disk_budget(cache, weights):
    write_snapshot(cache, {'config.json': '{}', weights: 'cached-weight-placeholder'})
    assert downloads.hf_preflight('large', 'example/large', 15.5) == {
        'cache_dir': str(cache), 'local_files_only': True}
    downloads.shutil.disk_usage.assert_not_called()


@pytest.mark.parametrize('state', ['config_only', 'missing', 'empty', 'broken_link',
                                  'bad_index', 'complete'])
def test_sharded_cache_requires_every_weight_file(cache, state):
    files = {'config.json': '{}'}
    if state != 'config_only':
        files['model.safetensors.index.json'] = json.dumps({
            'weight_map': {'a': 'part-1.safetensors', 'b': 'part-2.safetensors'}})
        files['part-1.safetensors'] = 'weight-placeholder'
    if state in ('empty', 'complete'):
        files['part-2.safetensors'] = '' if state == 'empty' else 'weight-placeholder'
    if state == 'bad_index':
        files['model.safetensors.index.json'] = '{'
    snapshot = write_snapshot(cache, files)
    if state == 'broken_link':
        (snapshot / 'part-2.safetensors').symlink_to(cache / 'missing-blob')
    if state == 'complete':
        assert downloads.hf_preflight('large', 'example/large', 15.5)['local_files_only']
    else:
        with pytest.raises(downloads.ModelDownloadRequired):
            downloads.hf_preflight('large', 'example/large', 15.5)


def test_noninteractive_opt_out_skips_disk_check(cache, monkeypatch):
    monkeypatch.setenv(downloads.SKIP_ENV, '1')
    assert downloads.hf_preflight('large', 'example/large', 15.5) == {'cache_dir': str(cache)}
    downloads.file_preflight('large', 'https://example.org/weights', cache / 'weights', 5.2)
    downloads.shutil.disk_usage.assert_not_called()


@pytest.mark.parametrize('value', ['', '0', 'false'])
def test_opt_out_requires_explicit_one(cache, monkeypatch, value):
    monkeypatch.setenv(downloads.SKIP_ENV, value)
    with pytest.raises(downloads.ModelDownloadRequired):
        downloads.hf_preflight('large', 'example/large', 15.5)


@pytest.mark.parametrize('identifier', [
    'blip2-opt-2.7b', 'blip2-wav2vec2', 'qwen2.5-vl-3b',
    'qwen2.5-vl-3b-vwfa', 'qwen2.5-vl-wav2vec2', 'qwen3.6-27b', 'vjepa2-vitl',
])
def test_cold_registered_factory_stops_before_any_hf_download(cache, monkeypatch, identifier):
    import brainscore
    import transformers
    from huggingface_hub import file_download
    no_download = Mock(side_effect=AssertionError('attempted a download'))
    monkeypatch.setattr(transformers.PreTrainedModel, 'from_pretrained', no_download)
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', no_download)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', no_download)
    monkeypatch.setattr(file_download, 'http_get', no_download)
    with pytest.raises(downloads.ModelDownloadRequired, match=identifier):
        brainscore.load_model(identifier)
    no_download.assert_not_called()


@pytest.mark.parametrize('opt_out', [False, True])
def test_factory_passes_cache_settings_to_first_loader(cache, monkeypatch, opt_out):
    from brainscore.models.blip2_opt_2_7b import model
    if opt_out:
        monkeypatch.setenv(downloads.SKIP_ENV, '1')
    else:
        write_snapshot(cache, {'config.json': '{}', 'model.safetensors': 'weight-placeholder'})
        (cache / 'models--example--large').rename(cache / 'models--Salesforce--blip2-opt-2.7b')

    class LoaderReached(Exception):
        pass

    loader = Mock(side_effect=LoaderReached)
    monkeypatch.setattr(model.Blip2ForConditionalGeneration, 'from_pretrained', loader)
    with pytest.raises(LoaderReached):
        model.get_model('blip2-opt-2.7b')
    assert loader.call_args.kwargs['cache_dir'] == str(cache)
    assert loader.call_args.kwargs.get('local_files_only', False) is not opt_out


@pytest.mark.parametrize('identifier', ['vjepa1-vitl', 'vjepa1-wav2vec2',
                                      'random-vjepa1-random-wav2vec2'])
def test_vjepa_direct_download_is_guarded(cache, monkeypatch, identifier):
    import brainscore
    from brainscore.models.vjepa_v1 import model
    monkeypatch.setattr(model, '_default_cache_dir', lambda: cache)
    loader = Mock(side_effect=AssertionError('attempted URL download'))
    monkeypatch.setattr('urllib.request.urlretrieve', loader)
    with pytest.raises(downloads.ModelDownloadRequired, match=identifier):
        brainscore.load_model(identifier)
    loader.assert_not_called()


def test_existing_vjepa_checkpoint_needs_no_download(cache, monkeypatch):
    from brainscore.models.vjepa_v1.model import _download_checkpoint
    checkpoint = cache.parent / 'vitl16.pth.tar'
    checkpoint.write_bytes(b'cached-checkpoint-placeholder')
    loader = Mock(side_effect=AssertionError('attempted URL download'))
    monkeypatch.setattr('urllib.request.urlretrieve', loader)
    assert _download_checkpoint('https://example.org/weights', checkpoint) == checkpoint
    downloads.shutil.disk_usage.assert_not_called()
