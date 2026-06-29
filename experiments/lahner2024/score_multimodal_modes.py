"""Score vjepa1-wav2vec2 in all 4 multimodal modes on Lahner-visualROI.

Decomposes the negative concat result by isolating each modality's
contribution and comparing per-modality ridge to ridge-on-concat.

Outputs:
    /tmp/lahner_multimodal_modes.json — raw r per mode
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import brainscore  # noqa: E402
from brainscore.benchmarks.lahner2024.benchmark_multimodal import (  # noqa: E402
    Lahner2024BOLDMoments_multimodal,
)


def main():
    log = lambda msg: print(f'[{time.time() - t0:6.1f}s] {msg}', flush=True)
    t0 = time.time()

    log('loading model (one-shot — wrappers cache across modes)...')
    model = brainscore.load_model('vjepa1-wav2vec2')

    out = {}
    for mode in ('concat', 'per_modality', 'video_only', 'audio_only',
                 'banded'):
        log(f'scoring mode={mode} on visualROI ...')
        b = Lahner2024BOLDMoments_multimodal(
            reliability_threshold=0.3,
            mode=mode,
            identifier_suffix=f'-multimodal-visualROI-{mode}',
        )
        score = b(model)
        entry = {
            'raw_r': float(score.attrs['raw']),
            'mean_r': score.attrs['mean_r'],
            'n_voxels_scored': score.attrs['n_voxels_scored'],
            'n_features_video': score.attrs['n_features_video'],
            'n_features_audio': score.attrs['n_features_audio'],
            'pipeline': score.attrs['pipeline'],
        }
        if 'banded_alpha_video_per_fold' in score.attrs:
            entry['banded_alpha_video_per_fold'] = (
                score.attrs['banded_alpha_video_per_fold'])
            entry['banded_alpha_audio_per_fold'] = (
                score.attrs['banded_alpha_audio_per_fold'])
        out[mode] = entry
        log(f'  raw r = {entry["raw_r"]:.4f}')

    out_path = Path('/tmp/lahner_multimodal_modes.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    log(f'wrote {out_path}')


if __name__ == '__main__':
    main()
