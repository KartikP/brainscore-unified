"""Exercise qualification failures without weights, datasets or cloud access."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from brainscore.validation import integration
from brainscore.validation.integration_checks import record_placement


def run_probe(tmp_path, source, *, expected=None, collect=False):
    test = tmp_path / 'test_probe.py'
    test.write_text(source)
    expected = expected if expected is not None else ['test_probe.py::test_one']
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'expected_collection': expected}))
    report = tmp_path / 'evidence.json'
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    env.pop('UMI_INTEGRATION_DEVICE', None)
    command = [sys.executable, '-m', 'pytest', str(test), '-q',
               '-p', 'pytest_check.plugin',
               '-p', 'brainscore.validation.integration_checks',
               '--umi-integration-profile', str(profile),
               '--umi-integration-report', str(report)]
    if collect:
        command.append('--collect-only')
    run = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=40)
    assert report.exists(), run.stdout + run.stderr
    return run, json.loads(report.read_text())


@pytest.mark.parametrize('collect', [True, False])
def test_locked_selection_passes_with_complete_evidence(tmp_path, collect):
    run, report = run_probe(tmp_path, 'def test_one():\n    assert True\n', collect=collect)
    assert run.returncode == 0, run.stdout + run.stderr
    assert report['complete']
    assert report['collect_only'] is collect
    assert bool(report['phases']) is not collect


@pytest.mark.parametrize('source', [
    'import pytest\ndef test_one():\n    pytest.skip("missing asset")\n',
    'import pytest\n@pytest.mark.xfail(reason="defect")\ndef test_one():\n    assert False\n',
    'import pytest\n@pytest.fixture(autouse=True)\ndef cleanup():\n    yield\n    raise ValueError("cleanup")\ndef test_one(): pass\n',
    'import pytest\npytest.skip("missing package", allow_module_level=True)\n',
    'def test_one():\n    assert False\n',
    'import pytest_check as check\ndef test_one():\n    check.equal(1, 2)\n',
    'import socket\ndef test_one():\n    socket.create_connection(("127.0.0.1", 443))\n',
])
def test_incomplete_or_failed_execution_cannot_pass(tmp_path, source):
    run, report = run_probe(tmp_path, source)
    assert run.returncode != 0
    assert report['returncode'] != 0


@pytest.mark.parametrize('expected', [[], ['test_probe.py::test_missing'],
                                    ['test_probe.py::test_one'] * 2])
def test_changed_selection_stops_before_any_fixture(tmp_path, expected):
    source = ('from pathlib import Path\nimport pytest\n'
              '@pytest.fixture(autouse=True)\ndef expensive():\n'
              '    Path("MODEL_LOADED").touch()\n'
              'def test_one(): pass\n')
    run, report = run_probe(tmp_path, source, expected=expected)
    assert run.returncode != 0
    assert not report['complete']
    assert not (tmp_path / 'MODEL_LOADED').exists()


def test_assets_are_copied_and_content_changes_are_rejected(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'weights').write_bytes(b'checkpoint')
    home = tmp_path / 'home'
    home.mkdir()
    profile = {'id': 'native-text', 'required_caches': ['huggingface']}
    record = {'profile': 'native-text', 'assets': {'huggingface': {
        'path': str(source), 'sha256': integration.fingerprint(source)[0], 'build': 'fixture-v1'}}}
    integration.prepare_assets(record, profile, home)
    (home / 'huggingface' / 'weights').write_bytes(b'changed only in private copy')
    assert (source / 'weights').read_bytes() == b'checkpoint'
    (source / 'weights').write_bytes(b'new checkpoint')
    with pytest.raises(ValueError, match='differs'):
        integration.prepare_assets(record, profile, home)


def test_environment_discards_credentials_cache_overrides_and_pytest_filters(tmp_path, monkeypatch):
    for key in ('AWS_ACCESS_KEY_ID', 'HF_TOKEN', 'PYTHONPATH', 'PYTEST_ADDOPTS',
                'TRANSFORMERS_CACHE', 'MT_IMAGENET_PATH', 'MT_HOME'):
        monkeypatch.setenv(key, 'inherited-value')
    env = integration.child_environment(tmp_path, 'cpu')
    assert all(key not in env for key in ('AWS_ACCESS_KEY_ID', 'HF_TOKEN', 'PYTHONPATH',
                                        'PYTEST_ADDOPTS', 'TRANSFORMERS_CACHE'))
    assert env['HOME'] == str(tmp_path)
    assert env['RESULTCACHING_HOME'].startswith(str(tmp_path))
    assert env['HF_HUB_OFFLINE'] == '1'
    assert env['MT_HOME'] == str(tmp_path / 'model_tools')
    assert 'MT_IMAGENET_PATH' not in env


@pytest.mark.parametrize('defect', ['missing', 'not-hdf5', 'missing-dataset', 'too-short', 'valid'])
def test_pca_input_is_validated_before_any_assets_are_copied(tmp_path, monkeypatch, defect):
    import h5py

    source = tmp_path / 'source'
    source.mkdir()
    path = source / 'imagenet2012.hdf5'
    if defect == 'not-hdf5':
        path.write_bytes(b'not a dataset')
    elif defect != 'missing':
        with h5py.File(path, 'w') as handle:
            if defect != 'missing-dataset':
                handle.create_dataset('val/images', shape=(50_000 if defect == 'valid' else 1, 1, 1, 3),
                                      dtype='uint8')
    # Keep the directory nonempty so a missing HDF5 gets a useful PCA error.
    (source / 'README').write_text('synthetic input fixture')
    record = {'profile': 'vision-layer-mapping', 'assets': {'model_tools': {
        'path': str(source), 'sha256': integration.fingerprint(source)[0], 'build': 'synthetic'}}}
    profile = {'id': 'vision-layer-mapping', 'required_caches': ['model_tools']}
    home = tmp_path / 'home'
    home.mkdir()
    if defect == 'valid':
        integration.prepare_assets(record, profile, home)
        assert (home / 'model_tools' / path.name).read_bytes() == path.read_bytes()
    else:
        monkeypatch.setattr(integration.shutil, 'copytree', lambda *args: pytest.fail('copied invalid inputs'))
        with pytest.raises(ValueError, match='PCA input'):
            integration.prepare_assets(record, profile, home)


def test_layer_mapping_manifest_requires_pca_inputs():
    manifest = Path(__file__).resolve().parents[1] / 'install' / 'integration-tests.json'
    profile = next(item for item in json.loads(manifest.read_text())['profiles']
                   if item['id'] == 'vision-layer-mapping')
    assert 'model_tools' in profile['required_caches']


@pytest.mark.parametrize('limit, reason', [(0.3, 'timeout'), (30, 'memory_limit')])
def test_process_group_stops_at_resource_limit(tmp_path, limit, reason):
    command = [sys.executable, '-c', 'import time; time.sleep(60)']
    result = integration.bounded_run(command, cwd=tmp_path, env=os.environ.copy(),
                                     log=tmp_path / 'log', seconds=limit,
                                     memory_bytes=1 if reason == 'memory_limit' else 1024**3)
    assert result['stop_reason'] == reason
    assert result['returncode'] != 0
    assert result['seconds'] < 10


def test_timeout_stops_descendants(tmp_path):
    import psutil
    command = [sys.executable, '-c',
               'import subprocess, sys, time; from pathlib import Path; '
               'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]); '
               'Path("child.pid").write_text(str(child.pid)); time.sleep(60)']
    result = integration.bounded_run(command, cwd=tmp_path, env=os.environ.copy(),
                                     log=tmp_path / 'log', seconds=1, memory_bytes=1024**3)
    assert result['stop_reason'] == 'timeout'
    pid = int((tmp_path / 'child.pid').read_text())
    try:
        child = psutil.Process(pid)
        child.wait(timeout=3)
    except psutil.NoSuchProcess:
        pass
    except psutil.TimeoutExpired:
        assert child.status() == psutil.STATUS_ZOMBIE


@pytest.mark.parametrize('evidence', [{}, {'complete': False, 'returncode': 0},
                                    {'complete': True, 'returncode': 1}])
def test_missing_or_incomplete_reports_fail(tmp_path, evidence):
    path = tmp_path / 'report.json'
    with pytest.raises(FileNotFoundError):
        integration.read_evidence(path)
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match='failed evidence'):
        integration.read_evidence(path)


def test_cuda_worker_cannot_count_cpu_model_as_gpu_evidence(monkeypatch):
    import torch
    monkeypatch.setenv('UMI_INTEGRATION_DEVICE', 'cuda')
    with pytest.raises(AssertionError, match='expected cuda'):
        record_placement(SimpleNamespace(), torch.nn.Linear(2, 2), identifier='small', dtype='torch.float32')


def test_cpu_placement_records_actual_dtype(monkeypatch):
    import torch
    monkeypatch.setenv('UMI_INTEGRATION_DEVICE', 'cpu')
    config = SimpleNamespace()
    record_placement(config, torch.nn.Linear(2, 2), identifier='small', dtype='torch.float32')
    assert config._umi_model_placements == [
        {'identifier': 'small', 'devices': ['cpu'], 'dtypes': ['torch.float32']}]


@pytest.mark.parametrize('kind', ['clip', 'gpt2', 'qwen'])
def test_native_profile_dispatch_assertions_with_small_assemblies(kind):
    # Check the new assertion paths offline; trained-model qualification is separate.
    import numpy as np
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    from tests.test_umi_review_regressions import _Extractor
    from tests import test_text_wrapper, test_vlm_vision_wrapper

    wrapper = _Extractor(np.arange(12).reshape(3, 4))
    stimuli = StimulusSet({'stimulus_id': ['a', 'b', 'c'],
                          'sentence': ['one', 'two', 'three']})
    stimuli.identifier = 'dispatch-assertions'
    if kind == 'clip':
        test_text_wrapper.TestTextWrapperIntegration().test_works_as_brainscore_model_preprocessor(wrapper, stimuli)
    elif kind == 'gpt2':
        test_text_wrapper.TestTextWrapperCausalMode().test_process_matches_wrapper(wrapper, stimuli)
    else:
        stimuli = StimulusSet({'stimulus_id': ['a', 'b', 'c'],
                              'image_file_name': ['a.png', 'b.png', 'c.png']})
        stimuli.identifier = 'dispatch-image-assertions'
        test_vlm_vision_wrapper.TestVLMVisionWrapperIntegration().test_works_as_brainscore_model_preprocessor(wrapper, stimuli)


def test_preflight_failure_keeps_a_report(tmp_path, monkeypatch):
    monkeypatch.setattr(integration.sys, 'platform', 'unsupported')
    output = tmp_path / 'evidence'
    assert integration.qualify(root=tmp_path, revisions=tmp_path, wheels=tmp_path,
                               manifest=tmp_path, assets=tmp_path, profile_id='native-text', output=output) == 1
    report = json.loads((output / 'report.json').read_text())
    assert not report['complete']
    assert 'Linux' in report['error']


@pytest.mark.parametrize('defect', [None, 'collection', 'identity', 'execution', 'junit'])
def test_runner_requires_both_phases_and_installed_evidence(tmp_path, monkeypatch, defect):
    monkeypatch.setattr(integration.sys, 'platform', 'linux')
    monkeypatch.setattr(integration, 'verify', lambda *args: None)
    monkeypatch.setattr(integration.subprocess, 'check_output', lambda *args, **kwargs: '')
    monkeypatch.setattr(integration, 'stage_tests', lambda *args: None)
    monkeypatch.setattr(integration, 'prepare_assets', lambda *args: {'copied_bytes': 0})
    monkeypatch.setattr(integration.psutil, 'virtual_memory', lambda: SimpleNamespace(available=100 * 1024**3))
    profile = {'id': 'fixture', 'repository': 'unified', 'device': 'cpu',
               'expected_collection': ['test_one'], 'expected_count': 1,
               'pytest_args': ['tests/test_one.py'],
               'proposed_limits': {'timeout_minutes': 1, 'memory_gib': 1}}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'profiles': [profile]}))
    record = tmp_path / 'record.json'
    record.write_text('{}')
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[1:4] == ['-I', '-m', 'brainscore.validation.installed']
        collecting = '--collect-only' in command
        report = {'complete': True, 'returncode': 0, 'collect_only': collecting,
                  'selected': ['test_one'], 'phases': {} if collecting else {
                      'test_one': {'setup': 'passed', 'call': 'passed', 'teardown': 'passed'}}}
        if defect == 'collection' and collecting:
            report['selected'] = []
        if defect == 'execution' and not collecting:
            report['phases'] = {}
        Path(command[command.index('--umi-integration-report') + 1]).write_text(json.dumps(report))
        identity = {'complete': defect != 'identity', 'returncode': 0, 'import_violations': []}
        Path(command[command.index('--identity') + 1]).write_text(json.dumps(identity))
        if not collecting and defect != 'junit':
            Path(command[command.index('--junitxml') + 1]).write_text('<testsuites/>')
        return {'returncode': 0}

    monkeypatch.setattr(integration, 'bounded_run', run)
    output = tmp_path / 'out'
    code = integration.qualify(root=tmp_path, revisions=record, wheels=tmp_path,
                               manifest=manifest, assets=record, profile_id='fixture', output=output)
    assert code == (0 if defect is None else 1)
    assert len(calls) == (1 if defect in ('collection', 'identity') else 2)
    assert json.loads((output / 'report.json').read_text())['complete'] == (defect is None)
