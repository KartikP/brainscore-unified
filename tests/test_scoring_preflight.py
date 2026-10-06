"""All scoring entry points reject cheap failures before model construction."""
import importlib
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('module_name', ['brainscore', 'brainscore_vision', 'brainscore_language'])
@pytest.mark.parametrize('failure', ['unknown_benchmark', 'missing_data', 'cache'])
def test_preflight_does_not_load_model(module_name, failure, tmp_path, monkeypatch):
    module = importlib.import_module(module_name)
    run = module.score if module_name == 'brainscore' else module._run_score
    monkeypatch.setattr(module, 'load_model', lambda _: pytest.fail('model loaded too early'))
    error = KeyError if failure == 'unknown_benchmark' else FileNotFoundError

    def load(_):
        if failure != 'cache':
            raise error('benchmark prerequisite missing')
        return SimpleNamespace(identifier='benchmark')

    monkeypatch.setattr(module, 'load_benchmark', load)
    if failure == 'cache':
        path = tmp_path / 'cache'
        path.symlink_to(tmp_path / 'missing' / 'cache')
        monkeypatch.setenv('RESULTCACHING_HOME', str(path))
        monkeypatch.setenv('RESULTCACHING_DISABLE', '0')
        with pytest.raises(OSError, match='Result cache is not writable'):
            run('model', 'benchmark', check_mem=False)
    else:
        with pytest.raises(error, match='prerequisite'):
            run('model', 'benchmark', check_mem=False)


def test_declared_missing_asset_fails_before_either_factory(monkeypatch, tmp_path):
    import brainscore
    from brainscore.data import local
    asset = local.LocalAsset(
        name='test-asset', env_var='UMI_TEST_ASSET', default_path='unused',
        kind='data', why_local='test fixture', source='test', obtain='provide a file',
        used_by=['offline-test-benchmark'],
    )
    monkeypatch.setitem(local.REGISTRY, asset.name, asset)
    monkeypatch.setenv('UMI_TEST_ASSET', str(tmp_path / 'missing'))
    for name in ('load_model', 'load_benchmark'):
        monkeypatch.setattr(brainscore, name, lambda _: pytest.fail('factory called'))
    with pytest.raises(local.LocalDataMissing, match='test-asset'):
        brainscore.score('model', 'offline-test-benchmark')


def test_doctor_reports_invalid_cache(monkeypatch, tmp_path):
    from brainscore import doctor
    path = tmp_path / 'cache'
    path.write_text('not a directory')
    monkeypatch.setenv('RESULTCACHING_HOME', str(path))
    monkeypatch.setenv('RESULTCACHING_DISABLE', '0')
    monkeypatch.setattr(doctor, '_dependency_rows', lambda: [])
    monkeypatch.setattr(doctor, '_asset_rows', lambda: [])
    text, problems = doctor.report()
    assert problems == 1
    assert 'Result cache is not writable' in text
    assert doctor.main() == 1


@pytest.mark.parametrize('module_name', ['brainscore', 'brainscore_vision', 'brainscore_language'])
def test_bad_cache_precedes_benchmark_construction(module_name, tmp_path, monkeypatch):
    module = importlib.import_module(module_name)
    run = module.score if module_name == 'brainscore' else module._run_score
    path = tmp_path / 'cache'
    path.write_text('not a directory')
    monkeypatch.setenv('RESULTCACHING_HOME', str(path))
    monkeypatch.setenv('RESULTCACHING_DISABLE', '0')
    for name in ('load_model', 'load_benchmark'):
        monkeypatch.setattr(module, name, lambda _: pytest.fail('factory called'))
    with pytest.raises(OSError, match='Result cache is not writable'):
        run('model', 'benchmark', check_mem=False)


@pytest.mark.parametrize('module_name', ['brainscore', 'brainscore_vision', 'brainscore_language'])
def test_each_scoring_call_gets_a_fresh_weight_scope(module_name, monkeypatch):
    import torch
    from brainscore_core import extraction_cache as cache
    from brainscore_core.metrics import Score
    module = importlib.import_module(module_name)
    run = module.score if module_name == 'brainscore' else module._run_score
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    tensor = torch.ones(2)
    keys = []
    reads = []
    original = cache._hash_tensor
    def counted(value):
        reads.append(1)
        return original(value)
    monkeypatch.setattr(cache, '_hash_tensor', counted)
    class Benchmark:
        identifier = 'test-benchmark'
        required_modalities = {'text'}
        parent = 'behavior'
        def preallocate_memory(self, model):
            pass
        def __call__(self, model):
            keys.append(cache.fingerprint(tensor))
            assert cache.fingerprint(tensor) == keys[-1]
            return Score(0.5)
    model = SimpleNamespace(
        identifier='test-model', available_modalities={'text'}, required_modalities={'text'},
        in_channels={'text'}, required_channels={'text'}, out_channels={'behavior'}, region_layer_map={},
    )
    monkeypatch.setattr(module, 'load_model', lambda _: model)
    monkeypatch.setattr(module, 'load_benchmark', lambda _: Benchmark())
    run('model', 'benchmark', check_mem=False)
    tensor.data.mul_(2)  # Untracked mutation is safe between, not within, runs.
    run('model', 'benchmark', check_mem=False)
    assert len(reads) == 2
    assert keys[0] != keys[1]
    assert cache._weight_hashes.get() is None
