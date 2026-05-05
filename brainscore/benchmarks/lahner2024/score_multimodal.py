"""Score the vjepa1-wav2vec2 multimodal A+V model on the Lahner2024
multimodal benchmark variant. Compares against the previously-shipped
V-JEPA v1 video-only score (0.5329 on the same visual-ROI mask).

Outputs:
    /tmp/lahner_multimodal_score.json — raw + ROI scores + breakdown
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

import brainscore  # noqa: E402


def main():
    log = lambda msg: print(f'[{time.time() - t0:6.1f}s] {msg}', flush=True)
    t0 = time.time()

    log('loading model...')
    model = brainscore.load_model('vjepa1-wav2vec2')

    out = {}
    for variant in ('multimodal-visualROI', 'multimodal'):
        full_id = f'Lahner2024-fMRI-naturalistic-{variant}'
        log(f'scoring {full_id} ...')
        b = brainscore.load_benchmark(full_id)
        score = b(model)
        out[variant] = {
            'raw_r': float(score.attrs['raw']),
            'mean_r': score.attrs['mean_r'],
            'n_voxels_scored': score.attrs['n_voxels_scored'],
            'n_videos': score.attrs['n_videos'],
            'n_features_video': score.attrs['n_features_video'],
            'n_features_audio': score.attrs['n_features_audio'],
            'pipeline': score.attrs['pipeline'],
        }
        log(f'  raw r = {out[variant]["raw_r"]:.4f}'
            f'  (n_voxels={out[variant]["n_voxels_scored"]})')

    out_path = Path('/tmp/lahner_multimodal_score.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    log(f'wrote {out_path}')


if __name__ == '__main__':
    main()
