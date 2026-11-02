"""Run inside the separately installed upstream Python 3.12 runtime."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


SOURCE_REVISION = 'c594e3124e2a9cb7b8a26a750b373e583a7fb297'
WEIGHTS_SHA256 = '8ee898b4d83466a72192577a492e89bececdd563d0b2fd82f1679dfd1dbce69d'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Upstream MIRAGE requires Python 3.12')
    import brain_enc
    import numpy as np
    import torch
    import yaml
    from brain_enc.eval.infer_fmri import (
        FMRI_TR_S, HRF_DELAY_S, InferenceRequest, SUBJECTS,
        WINDOW_DURATION_S, run_inference,
    )

    request = json.loads(Path(sys.argv[1]).read_text())
    run_dir = Path(request['run_dir'])
    checkpoint = run_dir / 'model.safetensors'
    source_dir = Path(brain_enc.__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', '-C', str(source_dir), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError('MIRAGE source revision differs from the reviewed revision')
    subprocess.run(['git', '-C', str(source_dir), 'diff', '--exit-code', 'HEAD', '--', 'src'],
                   check=True, capture_output=True)
    weights_sha256 = sha256(checkpoint)
    if weights_sha256 != WEIGHTS_SHA256:
        raise RuntimeError('MIRAGE weights differ from the pinned public checkpoint')
    config = yaml.safe_load((run_dir / 'config.yaml').read_text())
    backbone_revision = config['data']['text']['revision']
    if not isinstance(backbone_revision, str) or len(backbone_revision) != 40:
        raise RuntimeError('Pin the Qwen backbone revision in config.yaml before inference')
    native_request = InferenceRequest(
        video=Path(request['video']), transcript=Path(request['transcript']),
        run_dir=run_dir, checkpoint=checkpoint,
        subject_idx=SUBJECTS.index(request['subject']), output=Path(request['output']),
        device=request['device'], batch_size=request['batch_size'],
    )
    values = run_inference(native_request)
    if values.dtype != np.float32:
        raise RuntimeError(f'Unexpected native output dtype: {values.dtype}')
    np.save(request['output'], values, allow_pickle=False)
    provenance = {
        'implementation': 'native MIRAGE run_inference',
        'source_revision': revision, 'checkpoint_sha256': weights_sha256,
        'checkpoint_revision': 'b38a252c28e5416336423f4eda0c42d7e1df4e9c',
        'config_sha256': sha256(run_dir / 'config.yaml'), 'config': config,
        'backbone_revision': backbone_revision, 'subject': request['subject'],
        'device': request['device'], 'batch_size': request['batch_size'],
        'output_dtype': str(values.dtype), 'python': sys.version.split()[0],
        'packages': {name: importlib.metadata.version(name) for name in (
            'numpy', 'torch', 'transformers', 'x-transformers', 'safetensors', 'pydantic')},
        'cuda_version': torch.version.cuda,
        'cuda_device': torch.cuda.get_device_name() if request['device'].startswith('cuda') else None,
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'video_sha256': sha256(Path(request['video'])),
        'transcript_sha256': sha256(Path(request['transcript'])),
        'timing': {'tr_s': FMRI_TR_S, 'hrf_delay_s': HRF_DELAY_S,
                   'window_s': WINDOW_DURATION_S, 'causal_within_window': False},
        'umi_base_revision': 'abf74bd55a48bd6585d08ccd3320dd30a1e9ceca',
        'integration_sha256': {name: sha256(Path(__file__).with_name(name))
                               for name in ('model.py', 'runtime.py', 'worker.py')},
    }
    Path(request['provenance']).write_text(json.dumps(provenance, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
