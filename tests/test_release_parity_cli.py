import pytest
import json
from contextlib import contextmanager
from brainscore.validation import benchmark_parity as parity
from brainscore.validation.run_parity import main

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('selection', ['not-a-benchmark', parity.PEREIRA_CASES[0].unified])
def test_invalid_or_unpaired_selection_fails_before_inference(selection):
    with pytest.raises(SystemExit) as error:
        main(['--resnet18', 'unused', '--gpt2', 'unused', '--only', selection])
    assert error.value.code == 2


@pytest.fixture
def simulated_runtime(monkeypatch):
    from brainscore.validation import parity_runtime
    monkeypatch.setattr(parity_runtime, 'checkpoint_manifest', lambda *args: {'synthetic': True})
    @contextmanager
    def execution(*args):
        yield {'simulated': True}
    monkeypatch.setattr(parity_runtime, 'execution', execution)


def test_paired_guard_failure_fails_cli(monkeypatch, tmp_path, simulated_runtime):
    monkeypatch.setattr(parity, 'validate_case', lambda case, factory, **kwargs: {'benchmark': case.unified})
    def fail(reports, **kwargs):
        raise AssertionError('drift detected')
    monkeypatch.setattr(parity, 'assert_pereira_score_drift', fail)
    code = main(['--resnet18', 'unused', '--gpt2', 'unused', '--out', str(tmp_path/'parity.json')])
    assert code == 1
    report = json.loads((tmp_path/'parity.json').read_text())
    assert 'drift detected' in report['failures']['paired_drift:linear']


def test_release_manifest_includes_ridge():
    assert len(parity.RELEASE_CASES) == 8
    assert len(parity.RIDGE_CASES) == 2
    assert all('ridge' in case.legacy for case in parity.RIDGE_CASES)


@pytest.mark.parametrize('options', [
    ['--device', 'cuda', '--language-precision', 'float64-eager'],
    ['--policy', parity.CPU_FP32_POLICY],
    ['--policy', parity.CPU_FP32_POLICY, '--device', 'cuda'],
    ['--policy', parity.CPU_FP32_POLICY, '--device', 'cpu', '--language-precision', 'float64-eager'],
    ['--policy', parity.CPU_FP32_POLICY, '--device', 'cpu', '--only', parity.RIDGE_CASES[0].unified],
    ['--threads', '0'],
])
def test_invalid_execution_profile_fails_before_inference(options):
    with pytest.raises(SystemExit) as error:
        main(['--resnet18', 'unused', '--gpt2', 'unused', *options])
    assert error.value.code == 2


def test_failed_observations_survive_cli(monkeypatch, tmp_path, simulated_runtime):
    def fail(case, factory, **kwargs):
        raise parity.ParityFailure('activation failure', {'benchmark': case.unified,
            'routes': {'native': {'score': .8, 'raw': .4}}})
    monkeypatch.setattr(parity, 'validate_case', fail)
    out = tmp_path/'failed.json'
    assert main(['--resnet18', 'unused', '--gpt2', 'unused', '--out', str(out)]) == 1
    report = json.loads(out.read_text())
    assert len(report['failed_observations']) == 8
    assert report['failed_observations'][parity.CASES[0].unified]['routes']['native']['score'] == .8
    assert not report['release_complete']


@pytest.mark.parametrize('reference', [False, True])
def test_reference_success_is_separate_from_fp32_release(monkeypatch, tmp_path, simulated_runtime, reference):
    monkeypatch.setattr(parity, 'validate_case', lambda case, factory, **kwargs: {'benchmark': case.unified})
    monkeypatch.setattr(parity, 'assert_pereira_score_drift', lambda *args, **kwargs: None)
    out = tmp_path/'success.json'
    options = ['--language-precision', 'float64-eager'] if reference else []
    assert main(['--resnet18', 'unused', '--gpt2', 'unused', '--device', 'cpu',
                 '--out', str(out), *options]) == 0
    report = json.loads(out.read_text())
    assert report['selection_complete']
    assert report['release_complete'] is not reference
    assert report['reference_only'] is reference


def test_cuda_request_cannot_fall_back_to_cpu(monkeypatch):
    import torch
    from brainscore.validation.parity_runtime import execution
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='no CPU fallback'):
        with execution('cuda', 1):
            pytest.fail('Unavailable CUDA entered execution')


def test_checkpoint_manifest_detects_changed_weights(tmp_path):
    from brainscore.validation.parity_runtime import checkpoint_manifest
    weights, gpt2 = tmp_path/'resnet.pt', tmp_path/'gpt2'
    weights.write_bytes(b'resnet')
    gpt2.mkdir()
    (gpt2/'model.safetensors').write_bytes(b'weights-v1')
    first = checkpoint_manifest(weights, gpt2)
    (gpt2/'model.safetensors').write_bytes(b'weights-v2')
    second = checkpoint_manifest(weights, gpt2)
    assert first['resnet18'] == second['resnet18']
    assert first['gpt2/model.safetensors'] != second['gpt2/model.safetensors']


def test_cpu_profile_rejects_different_or_extra_tokenizer_inputs():
    from brainscore.validation.parity_runtime import CPU_GPT2_SHA256, require_cpu_reference
    manifest = {f'gpt2/{name}': {'sha256': digest} for name, digest in CPU_GPT2_SHA256.items()}
    require_cpu_reference(manifest)
    manifest['gpt2/added_tokens.json'] = {'sha256': 'new-tokenizer-state'}
    with pytest.raises(ValueError, match='calibrated GPT-2'):
        require_cpu_reference(manifest)
    del manifest['gpt2/added_tokens.json']
    manifest['gpt2/model.safetensors']['sha256'] = 'different-weights'
    with pytest.raises(ValueError, match='calibrated GPT-2'):
        require_cpu_reference(manifest)


@pytest.mark.parametrize('options', [
    ['--device', 'cpu'],
    ['--device', 'cuda', '--language-precision', 'float64-eager'],
    ['--device', 'cuda', '--only', parity.RIDGE_CASES[0].unified],
])
def test_l4_profile_rejects_invalid_execution_or_unpaired_ridge(options):
    with pytest.raises(SystemExit) as error:
        main(['--resnet18', 'unused', '--gpt2', 'unused', '--policy', parity.L4_FP32_POLICY, *options])
    assert error.value.code == 2


def test_l4_profile_rejects_another_gpu_before_scoring(monkeypatch, tmp_path):
    from brainscore.validation import parity_runtime
    monkeypatch.setattr(parity_runtime, 'checkpoint_manifest', lambda *args: {
        f'gpt2/{name}': {'sha256': digest} for name, digest in parity_runtime.CPU_GPT2_SHA256.items()})
    @contextmanager
    def execution(*args):
        yield {'cuda_name': 'NVIDIA A10G'}
    monkeypatch.setattr(parity_runtime, 'execution', execution)
    monkeypatch.setattr(parity, 'validate_case', lambda *args, **kwargs: pytest.fail('Unexpected inference'))
    out = tmp_path/'wrong-gpu.json'
    assert main(['--resnet18', 'unused', '--gpt2', 'unused', '--device', 'cuda',
        '--policy', parity.L4_FP32_POLICY, '--out', str(out)]) == 1
    report = json.loads(out.read_text())
    assert 'requires NVIDIA L4' in report['failures']['preflight_or_runtime']
    assert not report['reports'] and not report['release_complete']
