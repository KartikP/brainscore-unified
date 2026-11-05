"""Audit a completed run without running model inference again. Run remotely."""
import argparse
import json
from pathlib import Path
import platform

import numpy as np

import brainscore
from brainscore.data.sun2025.data import load_prepared
from brainscore.data.sun2025.data_packaging.prepare import checksum
from brainscore.experiments.runner import ExperimentResult
from brainscore.experiments.sun2025 import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, help='Optional separately versioned reanalysis directory')
    args = parser.parse_args()
    analysis_root = args.analysis or args.run
    root, manifest = load_prepared(args.data)
    stimuli = brainscore.load_stimulus_set('Sun2025-second-order-motion', root=root)
    reference = brainscore.load_dataset('Sun2025-second-order-motion', root=root)
    result = ExperimentResult(args.run / 'model', None)
    neural, behavior = result.neural('V1'), result.behavior(measure='text')
    assert neural.sizes['presentation'] == behavior.sizes['presentation'] == len(stimuli)
    assert set(neural['stimulus_id'].values) == set(stimuli.stimulus_id)
    np.testing.assert_array_equal(neural['stimulus_id'].values, behavior['stimulus_id'].values)
    assert neural.sizes['time_bin'] == 8, 'Expected eight temporal patches from sixteen frames'
    assert np.isfinite(neural.values).all()
    _, _, original = analyze(behavior, stimuli, reference)
    _, _, reordered = analyze(behavior.isel(presentation=slice(None, None, -1)),
                               stimuli, reference.isel(presentation=slice(None, None, -1)))
    assert original == reordered, 'Scoring must not depend on saved row order'
    saved = json.loads((analysis_root / 'summary.json').read_text())
    for key, value in original.items():
        assert saved[key] == value, f'Saved analysis changed: {key}'
    repo = Path(brainscore.__file__).resolve().parent.parent
    paths = [repo / 'brainscore/data/__init__.py']
    for folder in ('brainscore/data/sun2025', 'brainscore/experiments/sun2025', 'examples/sun2025'):
        paths.extend((repo / folder).rglob('*.py'))
    paths.append(repo / 'tests/test_sun2025.py')
    from huggingface_hub import try_to_load_from_cache
    from qwen_video import CHECKPOINT, REVISION
    config = try_to_load_from_cache(CHECKPOINT, 'config.json', revision=REVISION)
    if not isinstance(config, str):
        raise FileNotFoundError('The pinned model configuration is absent from the local cache')
    snapshot = Path(config).parent
    weights = {path.name: checksum(path) for path in sorted(snapshot.glob('*.safetensors'))}
    if not weights:
        raise FileNotFoundError('No pinned checkpoint weight files found')
    receipt = {
        'status': 'passed', 'scope': 'one-scene real-model pilot' if not manifest['complete'] else 'full study',
        'trial_count': len(stimuli), 'source_manifest_sha256': checksum(root / 'source-manifest.json'),
        'prepared_manifest_sha256': checksum(root / 'manifest.json'),
        'same_presentation_ids_for_neural_and_behavior': True,
        'analysis_invariant_to_row_order': True,
        'neural_shape': list(neural.shape), 'neural_dims': list(neural.dims),
        'neural_recording': neural.attrs['recording'],
        'python': platform.python_version(), 'weights_sha256': weights,
        'source_sha256': {str(path.relative_to(repo)): checksum(path) for path in paths},
        'summary_sha256': checksum(analysis_root / 'summary.json'),
        'report_parser': original['report_parser'],
        'control_summary_sha256': checksum(args.run / 'control-summary.json'),
    }
    (analysis_root / 'qualification.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
