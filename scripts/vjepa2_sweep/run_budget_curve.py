"""Scientifically-valid layer-mapping comparison on V-JEPA2 / Lahner visual-ROI.

The earlier four-strategy bar chart was confounded: strategies used different
numbers of features, and at a fixed ridge penalty fewer features score higher,
so absolute heights couldn't rank the strategies. This run removes BOTH
confounds:

  * RidgeCV with per-voxel alpha tuning   (no fixed-penalty artifact)
  * noise-ceiling normalization            (score = fraction of explainable signal)
  * budget-matched curve                   (within-layer vs pooled-across-layers,
                                            scored at IDENTICAL feature counts K)
  * matched random-selection null at every K

Outputs (/tmp/vjepa2_sweep/budget_curve.json): the curve (raw + normalized),
the two full-budget reference points, and metadata.

Run on EC2 (GPU) — reuses the V-JEPA2 feature cache warmed by run_vjepa2_sweep.
"""
import json
import os
import sys
import time

sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
import numpy as np

t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

N_LAYERS = 24
TOP_N_LAYERS = 3
BUDGETS = [16, 32, 64, 100, 128, 256, 512, 1024]
ALPHA_GRID = (1., 10., 100., 1000., 10000., 100000.)
N_NULL_SEEDS = 5
OUT_DIR = '/tmp/vjepa2_sweep'


def main():
    import warnings; warnings.filterwarnings('ignore')
    os.makedirs(OUT_DIR, exist_ok=True)
    import brainscore
    from brainscore.tools import explore_layer_mapping, score_budget_curve
    from brainscore.tools.layer_mapping import extract_features_by_layer

    log('load benchmark + BOLD target (visual ROI) + noise ceiling...')
    b = brainscore.load_benchmark('Lahner2024-fMRI-naturalistic-visualROI')
    asm = b._average_repetitions()
    mask = b._get_voxel_mask()
    Y = np.asarray(asm.transpose('stimulus_id', 'neuroid').values, np.float64)
    reliability = np.asarray(b._split_half_reliability(), np.float64)  # full-set, per voxel
    if mask is not None:
        Y = Y[:, mask]
        reliability = reliability[mask]
    # noise ceiling for a predictivity correlation = sqrt(reliability): the
    # highest correlation a perfect model could reach given measurement noise.
    ceiling = np.sqrt(np.clip(reliability, 0.0, 1.0))
    stim_ids = [str(s) for s in asm['stimulus_id'].values]
    vids = b._videos_stimulus_set()
    log(f'  {Y.shape[1]} ROI voxels; ceiling median={np.nanmedian(ceiling):.3f}')

    log('extract V-JEPA2 features at all 24 layers (cache hit expected)...')
    m = brainscore.load_model('vjepa2-vitl')
    vw = m._preprocessors['video']
    layers = [f'encoder.layer.{i}' for i in range(N_LAYERS)]
    feats = extract_features_by_layer(vw, vids, layers)
    f_ids = list(feats.pop('_stimulus_id'))
    f_index = {s: i for i, s in enumerate(f_ids)}
    common = [s for s in stim_ids if s in f_index]
    yi = [stim_ids.index(s) for s in common]
    fi = [f_index[s] for s in common]
    Yc = Y[yi]
    feats = {l: feats[l][fi] for l in layers}
    log(f'  aligned on {len(common)} stimuli; Y={Yc.shape}')

    log('explore layer mapping (localizer/test split, fixed alpha for ranking)...')
    res = explore_layer_mapping(feats, Yc, localizer_frac=0.5, alpha=1.0, seed=0)
    log(f'  best layer = {res.best_layer}; top-{TOP_N_LAYERS} = {res.top_layers(TOP_N_LAYERS)}')

    log('budget-matched curve — RAW (RidgeCV per-voxel alpha, no ceiling)...')
    curve_raw = score_budget_curve(
        feats, Yc, res, budgets=BUDGETS, top_n_layers=TOP_N_LAYERS,
        alpha_grid=ALPHA_GRID, n_null_seeds=N_NULL_SEEDS, noise_ceiling=None)

    log('budget-matched curve — NORMALIZED (divide by per-voxel noise ceiling)...')
    curve_norm = score_budget_curve(
        feats, Yc, res, budgets=BUDGETS, top_n_layers=TOP_N_LAYERS,
        alpha_grid=ALPHA_GRID, n_null_seeds=N_NULL_SEEDS, noise_ceiling=ceiling)

    out = {
        'model': 'vjepa2-vitl',
        'benchmark': 'Lahner2024-fMRI-naturalistic-visualROI',
        'protocol': ('budget-matched curve; RidgeCV per-voxel alpha tuning; '
                     'functional-localization split; matched random nulls; '
                     'ceiling = sqrt(split-half reliability)'),
        'n_stimuli': len(common), 'n_voxels': int(Yc.shape[1]),
        'ceiling_median': round(float(np.nanmedian(ceiling)), 4),
        'best_layer': res.best_layer, 'top_layers': res.top_layers(TOP_N_LAYERS),
        'alpha_grid': list(ALPHA_GRID), 'budgets': BUDGETS, 'n_null_seeds': N_NULL_SEEDS,
        'curve_raw': curve_raw,
        'curve_normalized': curve_norm,
    }
    json.dump(out, open(f'{OUT_DIR}/budget_curve.json', 'w'), indent=2)
    log('DONE. summary (normalized):')
    log(f"  whole best layer ({res.best_layer}, all units): "
        f"{curve_norm['whole_layer_r']:.4f}")
    log(f"  all {TOP_N_LAYERS} top layers concat: {curve_norm['several_layers_r']:.4f}")
    for e in curve_norm['within_layer']:
        log(f"  within-layer  K={e['k']:5d}: {e['r']:.4f}  (random {e['random_null']:.4f})")
    for e in curve_norm['pooled_layers']:
        log(f"  pooled-layers K={e['k']:5d}: {e['r']:.4f}  (random {e['random_null']:.4f})")


if __name__ == '__main__':
    main()
