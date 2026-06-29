"""Quick smoke test for the TR-resolved multimodal Lahner benchmark.

Runs all 5 modes (concat / per_modality / video_only / audio_only /
banded) on the auditory-ROI variant only — auditory has 715 voxels
(vs 4042 for visual-ROI), so per-voxel ridge is ~6x faster. Verifies
the pipeline runs end-to-end and produces sensible numbers.

Output: /tmp/lahner_tr_multimodal_smoke.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import brainscore  # noqa: E402
from brainscore.benchmarks.lahner2024.benchmark_timeresolved_multimodal \
    import (Lahner2024BOLDMoments_timeresolved_multimodal_auditoryROI)


def main():
    log = lambda msg: print(f'[{time.time() - t0:6.1f}s] {msg}', flush=True)
    t0 = time.time()

    log('loading model: vjepa1-wav2vec2 (signal both towers)...')
    model = brainscore.load_model('vjepa1-wav2vec2')
    log('model loaded')

    log('preloading TR-resolved assembly (~11 GB from S3 if not cached)...')
    b_pre = Lahner2024BOLDMoments_timeresolved_multimodal_auditoryROI()
    _ = b_pre.assembly  # force lazy load
    log(f'assembly: {dict(b_pre.assembly.sizes)}')
    log('events sidecar...')
    _ = b_pre.events
    log(f'events rows: {len(b_pre.events)}')
    log('voxel mask...')
    _ = b_pre.voxel_mask
    log(f'mask voxels kept: {int(b_pre.voxel_mask.sum())}')

    out = {}
    for mode in ('video_only', 'audio_only', 'concat', 'per_modality',
                 'banded'):
        log(f'scoring TR-resolved auditoryROI mode={mode} ...')
        b = Lahner2024BOLDMoments_timeresolved_multimodal_auditoryROI(
            mode=mode)
        score = b(model)
        out[mode] = {
            'raw_r': float(score.attrs['raw']),
            'mean_r': score.attrs['mean_r'],
            'n_voxels_scored': score.attrs['n_voxels_scored'],
            'pipeline': score.attrs['pipeline'],
        }
        if 'n_features_video' in score.attrs:
            out[mode]['n_features_video'] = score.attrs['n_features_video']
            out[mode]['n_features_audio'] = score.attrs['n_features_audio']
        log(f'  raw r = {out[mode]["raw_r"]:.4f}')

    out_path = Path('/tmp/lahner_tr_multimodal_smoke.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    log(f'wrote {out_path}')


if __name__ == '__main__':
    main()
