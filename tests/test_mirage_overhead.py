"""Timing arithmetic and matched execution order, without loading MIRAGE."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

spec = importlib.util.spec_from_file_location(
    'measure_mirage_overhead',
    Path(__file__).parents[1] / 'examples/mirage_umi/measure_overhead.py',
)
measurement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measurement)


def test_execution_order_balances_every_position():
    orders = measurement.execution_orders(6)
    assert len(set(orders)) == 6
    for position in range(3):
        assert sorted(order[position] for order in orders) == sorted(measurement.MODES * 2)
    assert measurement.execution_orders(3) == orders[:3]


@pytest.mark.parametrize('repetitions', [0, 1, 2, 4, -3])
def test_execution_order_rejects_unbalanced_counts(repetitions):
    with pytest.raises(ValueError, match='multiple of three'):
        measurement.execution_orders(repetitions)


def test_summary_excludes_warmups_and_uses_paired_ratios():
    rows = [
        {'case': 'clip', 'block': block, 'mode': mode, 'wall_s': duration, 'warmup': block == 0}
        for block, values in enumerate([(999, 1, 1), (10, 12, 13), (20, 22, 24), (30, 33, 36)])
        for mode, duration in zip(measurement.MODES, values)
    ]
    result = measurement.summarize(rows)['clip']
    assert result['wall']['direct'] == {'n': 3, 'median_s': 20, 'min_s': 10, 'max_s': 30}
    assert result['relative_to_direct']['umi']['paired_overhead_percent'] == pytest.approx([20, 10, 10])
    assert result['relative_to_direct']['recorded']['median_paired_overhead_percent'] == pytest.approx(20)


@pytest.mark.parametrize('kind', ['cli', 'worker'])
def test_phase_timers_preserve_outputs_and_write_sidecars(tmp_path, kind):
    import json
    import os
    import subprocess
    import sys

    import numpy as np

    package = tmp_path / 'brain_enc'
    for directory in [package, package / 'eval', package / 'cli']:
        directory.mkdir(exist_ok=True)
        (directory / '__init__.py').write_text('')
    (tmp_path / 'torch.py').write_text('class cuda:\n    @staticmethod\n    def synchronize():\n        pass\n')
    (package / 'eval/infer_fmri.py').write_text(
        'import sys\nfrom types import SimpleNamespace\nimport numpy as np\n'
        'def run_inference(request):\n    return np.arange(6, dtype=np.float32).reshape(2, 3)\n'
        'def main():\n    np.save(sys.argv[1], run_inference(SimpleNamespace(device="cuda")))\n'
    )
    (package / 'cli/infer_fmri.py').write_text(
        'from brain_enc.eval.infer_fmri import main\nmain()\n'
    )
    worker = tmp_path / 'worker.py'
    worker.write_text(
        'import hashlib\nimport sys\nfrom pathlib import Path\n'
        'from types import SimpleNamespace\nimport numpy as np\n'
        'def sha256(path):\n    return hashlib.sha256(path.read_bytes()).hexdigest()\n'
        'def main():\n    from brain_enc.eval.infer_fmri import run_inference\n'
        '    assert sha256(Path(sys.argv[2])) == hashlib.sha256(b"fixture").hexdigest()\n'
        '    np.save(sys.argv[1], run_inference(SimpleNamespace(device="cuda")))\n'
    )
    checkpoint = tmp_path / 'model.safetensors'
    checkpoint.write_bytes(b'fixture')
    output = tmp_path / 'output.npy'
    timings = tmp_path / 'timings.json'
    target = ['-m', 'brain_enc.cli.infer_fmri', str(output)] if kind == 'cli' else [
        str(worker), str(output), str(checkpoint)]
    script = Path(__file__).parents[1] / 'examples/mirage_umi/timing_worker.py'
    environment = {**os.environ, 'PYTHONPATH': str(tmp_path)}
    subprocess.run([sys.executable, str(script), '--timings', str(timings), '--', *target],
                   env=environment, check=True, timeout=20)
    np.testing.assert_array_equal(np.load(output), np.arange(6, dtype=np.float32).reshape(2, 3))
    info = json.loads(timings.read_text())
    assert info['status'] == 'complete'
    assert info['worker_body_s'] >= info['native_inference_s'] >= 0
    assert info['imports_s'] >= 0
    assert info['initial_cuda_sync_s'] >= 0
    assert ('checkpoint_verification_s' in info) == (kind == 'worker')
