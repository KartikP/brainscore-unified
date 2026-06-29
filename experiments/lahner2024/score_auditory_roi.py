"""Score vjepa1-wav2vec2 in all 5 modes + α-ablation on Lahner
auditory-ROI (Destrieux: HG, lateral STG, planum polare, planum tempo,
transverse temporal sulcus = ~715 fsaverage5 voxels).

Predictions:
- audio_only ≫ video_only on auditory voxels (asymmetry flips)
- concat / per_modality still fail for the predicted reasons
- banded ridge picks LOW α_audio (audio is the signal here) and HIGH α_video
- The fair multimodal baseline is audio_only at audio's optimal α; if
  banded beats THAT at matched α, we have genuine multimodal lift

Output: /tmp/lahner_auditory_modes.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import brainscore  # noqa: E402
from brainscore.benchmarks._scoring_utils import (  # noqa: E402
    kfold_ridge_predictions,
    pearson_summary,
)
from brainscore.benchmarks.lahner2024.benchmark_multimodal import (  # noqa: E402
    Lahner2024BOLDMoments_multimodal_auditoryROI,
)


ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0)


def main():
    log = lambda msg: print(f'[{time.time() - t0:6.1f}s] {msg}', flush=True)
    t0 = time.time()

    log('loading model...')
    model = brainscore.load_model('vjepa1-wav2vec2')

    out = {}

    # ── 5 modes ────────────────────────────────────────────────
    for mode in ('concat', 'per_modality', 'video_only', 'audio_only',
                 'banded'):
        log(f'scoring mode={mode} on auditoryROI ...')
        b = Lahner2024BOLDMoments_multimodal_auditoryROI(mode=mode)
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

    # ── α-ablation: audio_only across α grid ─────────────────────
    # Reuse the multimodal feature extraction once, then loop α
    log('extracting features once for α-ablation...')
    b = Lahner2024BOLDMoments_multimodal_auditoryROI(mode='audio_only')
    features, clip_ids, modality_per_neuroid = b._multimodal_features(model)
    a_mask = modality_per_neuroid == 'audio'
    v_mask = modality_per_neuroid == 'video'
    X_audio = features[:, a_mask]
    X_video = features[:, v_mask]

    neural = b._average_repetitions()
    neural_aligned = neural.sel(
        stimulus_id=list(clip_ids)).transpose('stimulus_id', 'neuroid')
    Y = neural_aligned.values
    voxel_mask = b._get_voxel_mask()
    if voxel_mask is not None:
        Y = Y[:, voxel_mask]

    def _score(X, alpha):
        fold_preds = kfold_ridge_predictions(
            X, Y, alpha=alpha, n_splits=5, random_state=0, dtype=None)
        _, median_r, _ = pearson_summary(Y, fold_preds)
        return median_r

    out['alpha_ablation'] = {}
    for alpha in ALPHA_GRID:
        log(f'α-ablation: audio_only @ α={alpha} ...')
        r = _score(X_audio, alpha)
        out['alpha_ablation'][f'audio_only_alpha_{alpha:g}'] = {
            'raw_r': r, 'alpha': alpha}
        log(f'  raw r = {r:.4f}')
        log(f'α-ablation: video_only @ α={alpha} ...')
        r = _score(X_video, alpha)
        out['alpha_ablation'][f'video_only_alpha_{alpha:g}'] = {
            'raw_r': r, 'alpha': alpha}
        log(f'  raw r = {r:.4f}')

    out_path = Path('/tmp/lahner_auditory_modes.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    log(f'wrote {out_path}')


if __name__ == '__main__':
    main()
