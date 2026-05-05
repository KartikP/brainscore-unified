"""α-ablation: score video_only at multiple Ridge α values on Lahner
visual-ROI to isolate banded ridge's pure multimodal contribution.

Banded ridge picked α_video=10 + α_audio=10000 across all 5 folds and
beat video_only(α=1) by +0.044 raw r. Some of that gain may come from
better-tuned video α alone, independent of audio. This script scores
video_only with α ∈ {1, 10, 100, 1000} so we can read off how much of
the +0.044 is α-tuning and how much is genuine multimodal lift.

Reuses the cached video activations (~32 MB pickle), so this is
ridge-only — fast.

Output: /tmp/lahner_alpha_ablation.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

import numpy as np  # noqa: E402

import brainscore  # noqa: E402
from brainscore.benchmarks.lahner2024.benchmark_multimodal import (  # noqa: E402
    Lahner2024BOLDMoments_multimodal,
)


ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def main():
    log = lambda msg: print(f'[{time.time() - t0:6.1f}s] {msg}', flush=True)
    t0 = time.time()

    log('loading model...')
    model = brainscore.load_model('vjepa1-wav2vec2')

    log('extracting features once via the multimodal benchmark...')
    b = Lahner2024BOLDMoments_multimodal(
        reliability_threshold=0.3,
        mode='video_only',
        identifier_suffix='-multimodal-visualROI-alpha-ablation',
    )
    features, clip_ids, modality_per_neuroid = b._multimodal_features(model)
    v_mask = modality_per_neuroid == 'video'
    X_video = features[:, v_mask]
    log(f'  video features: {X_video.shape}')

    # Align neural to features and apply ROI mask once
    neural = b._average_repetitions()
    neural_aligned = neural.sel(
        stimulus_id=list(clip_ids)
    ).transpose('stimulus_id', 'neuroid')
    Y = neural_aligned.values
    mask = b._get_voxel_mask()
    if mask is not None:
        Y = Y[:, mask]
    log(f'  neural matrix: {Y.shape}')

    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge

    n = X_video.shape[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    out = {}
    for alpha in ALPHAS:
        log(f'scoring video_only with alpha={alpha} ...')
        fold_preds = np.zeros_like(Y)
        for train_idx, test_idx in kf.split(np.arange(n)):
            reg = Ridge(alpha=alpha).fit(
                X_video[train_idx], Y[train_idx])
            fold_preds[test_idx] = reg.predict(X_video[test_idx])
        # per-voxel Pearson on held-out predictions
        n_voxels = Y.shape[1]
        per_voxel_r = np.zeros(n_voxels)
        for j in range(n_voxels):
            yt = Y[:, j]
            yp = fold_preds[:, j]
            if yt.std() > 0 and yp.std() > 0:
                per_voxel_r[j] = np.corrcoef(yt, yp)[0, 1]
            else:
                per_voxel_r[j] = np.nan
        per_voxel_r = per_voxel_r[~np.isnan(per_voxel_r)]
        median_r = float(np.median(per_voxel_r))
        out[f'video_only_alpha_{alpha:g}'] = {
            'raw_r': median_r,
            'mean_r': float(np.mean(per_voxel_r)),
            'alpha': alpha,
            'n_voxels_scored': int(len(per_voxel_r)),
        }
        log(f'  raw r = {median_r:.4f}')

    out_path = Path('/tmp/lahner_alpha_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    log(f'wrote {out_path}')


if __name__ == '__main__':
    main()
