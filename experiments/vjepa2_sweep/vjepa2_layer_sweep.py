"""V-JEPA2 ViT-L per-layer (+ per-unit) brain-prediction sweep on Lahner2024.

V-JEPA2's region was mapped to encoder.layer.16 by analogy to V-JEPA v1, without
its own sweep — which is why it scored below CLIP. This finds V-JEPA2's actual
brain-optimal layer on the Lahner visual-ROI, and the per-unit predictivity for a
layer x unit selection heatmap.

Pipeline (reuses the registered Lahner benchmark for the BOLD target + ROI mask):
  1. video stimulus set + averaged-rep BOLD on the visual-ROI voxels  (benchmark)
  2. V-JEPA2 features at ALL 24 encoder layers in one VideoWrapper pass, mean over
     the 32 temporal tubelets -> (n_videos, 1024) per layer
  3. per layer: 5-fold ridge per voxel (no feature scaler, matching the v1 sweep
     protocol) -> median per-voxel Pearson r over the ROI
  4. per unit: max |Pearson r| of that unit's activation with any ROI voxel ->
     a (24, 1024) predictivity matrix; top-K per layer = the selected units

Outputs (default /tmp/vjepa2_sweep):
  sweep.json   — per_layer_r[24], best_layer, best_r, current_layer(16) r, top-K
  unit_predictivity.npy — (24, 1024) per-unit predictivity (heatmap source)

Run on EC2 (GPU). ~30-50 min (the 24-layer forward over ~1026 clips is the cost).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
import numpy as np

t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

N_LAYERS = 24
HIDDEN = 1024
TOP_K = 100   # units selected per layer for the selection heatmap


def per_voxel_cv_ridge(X, Y, alpha=1.0, n_splits=5, seed=0):
    """5-fold ridge per voxel (no scaler); return per-voxel Pearson r."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    n = X.shape[0]
    preds = np.full_like(Y, np.nan, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(np.arange(n)):
        reg = Ridge(alpha=alpha).fit(X[tr], Y[tr])
        preds[te] = reg.predict(X[te])
    Yc = Y - Y.mean(0, keepdims=True)
    Pc = preds - preds.mean(0, keepdims=True)
    num = (Yc * Pc).sum(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(den > 0, num / den, np.nan)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', default='Lahner2024-fMRI-naturalistic-visualROI')
    ap.add_argument('--model', default='vjepa2-vitl')
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--current_layer', type=int, default=16)
    ap.add_argument('--out', default='/tmp/vjepa2_sweep')
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    import brainscore
    log(f'load benchmark {args.benchmark} ...')
    b = brainscore.load_benchmark(args.benchmark)
    # averaged-over-reps BOLD, then restrict to the ROI voxels
    asm = b._average_repetitions()
    mask = b._get_voxel_mask()
    # assembly is (neuroid, stimulus_id) — transpose to (stimulus, voxel)
    Y_all = np.asarray(asm.transpose('stimulus_id', 'neuroid').values, dtype=np.float64)
    Y = Y_all[:, mask] if mask is not None else Y_all
    stim_ids = [str(s) for s in asm['stimulus_id'].values]
    log(f'  BOLD target Y={Y.shape}  ({len(stim_ids)} stimuli)')

    vids = b._videos_stimulus_set()
    log(f'  video stimulus set: {len(vids)} videos')

    log(f'load model {args.model} + extract all {N_LAYERS} layers (one pass)...')
    m = brainscore.load_model(args.model)
    vw = m._preprocessors['video']
    layers = [f'encoder.layer.{i}' for i in range(N_LAYERS)]
    feats = vw(vids, layers=layers)                            # (pres, time_bin, neuroid)
    # mean over the temporal tubelets -> (pres, neuroid)
    if 'time_bin' in feats.dims:
        feats = feats.mean('time_bin')
    feats = feats.transpose('presentation', 'neuroid')
    f_stim = [str(s) for s in feats['stimulus_id'].values]
    layer_coord = np.asarray(feats['layer'].values).ravel()
    vals = np.asarray(feats.values, dtype=np.float32)          # (pres, 24*1024)
    log(f'  features {vals.shape}; layers present: {len(set(layer_coord))}')

    # align feature rows to the BOLD stimulus order
    f_index = {s: i for i, s in enumerate(f_stim)}
    common = [s for s in stim_ids if s in f_index]
    yi = [stim_ids.index(s) for s in common]
    fi = [f_index[s] for s in common]
    Y = Y[yi]
    vals = vals[fi]
    log(f'  aligned on {len(common)} stimuli; Y={Y.shape}')

    per_layer_r = []
    unit_pred = np.zeros((N_LAYERS, HIDDEN), dtype=np.float32)
    # precompute z-scored Y for per-unit correlation
    Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    for li in range(N_LAYERS):
        cols = np.where(layer_coord == f'encoder.layer.{li}')[0]
        X = vals[:, cols].astype(np.float64)                   # (n, 1024)
        r = per_voxel_cv_ridge(X, Y, alpha=args.alpha)
        med = float(np.nanmedian(r))
        per_layer_r.append(med)
        # per-unit predictivity: max |corr| of each unit with any ROI voxel
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
        corr = (Xz.T @ Yz) / X.shape[0]                        # (1024, n_voxels)
        unit_pred[li] = np.abs(corr).max(1).astype(np.float32)
        log(f'  layer {li:2d}: median r = {med:.4f}')

    best_layer = int(np.argmax(per_layer_r))

    # ── compare four layer-mapping approaches, all scored the same way ──
    def layer_X(li):
        cols = np.where(layer_coord == f'encoder.layer.{li}')[0]
        return vals[:, cols].astype(np.float64)

    def score_X(Xmat):
        return float(np.nanmedian(per_voxel_cv_ridge(Xmat, Y, alpha=args.alpha)))

    TOP_N_LAYERS = 3
    top_layers = [int(li) for li in np.argsort(per_layer_r)[::-1][:TOP_N_LAYERS]]
    topk_units = {int(li): np.argsort(unit_pred[li])[::-1][:TOP_K].tolist()
                  for li in top_layers}

    # 1. standard: single best full layer (already have its r)
    standard_r = per_layer_r[best_layer]
    # 2. unit selection within the best layer: top-K units only
    unit_within_X = layer_X(best_layer)[:, topk_units[best_layer]]
    unit_within_r = score_X(unit_within_X)
    # 3. multiple full layers: concat the top-N full layers
    multi_full_X = np.concatenate([layer_X(li) for li in top_layers], axis=1)
    multi_full_r = score_X(multi_full_X)
    # 4. CompositeSelector: top-K units from each of the top-N layers
    composite_X = np.concatenate(
        [layer_X(li)[:, topk_units[li]] for li in top_layers], axis=1)
    composite_r = score_X(composite_X)

    approaches = [
        {'name': 'standard layer mapping', 'detail': f'single best full layer (encoder.layer.{best_layer})',
         'n_features': HIDDEN, 'r': round(standard_r, 4)},
        {'name': 'unit selection within a layer', 'detail': f'top-{TOP_K} units of encoder.layer.{best_layer}',
         'n_features': TOP_K, 'r': round(unit_within_r, 4)},
        {'name': 'multiple full layers', 'detail': f'concat of layers {top_layers}',
         'n_features': HIDDEN * TOP_N_LAYERS, 'r': round(multi_full_r, 4)},
        {'name': 'CompositeSelector', 'detail': f'top-{TOP_K} units from each of layers {top_layers}',
         'n_features': TOP_K * TOP_N_LAYERS, 'r': round(composite_r, 4)},
    ]
    log('  approach comparison:')
    for a in approaches:
        log(f"    {a['name']:30s} r={a['r']:.4f}  ({a['n_features']} feat)")

    result = {
        'model': args.model, 'benchmark': args.benchmark, 'alpha': args.alpha,
        'n_stimuli': len(common), 'n_voxels': int(Y.shape[1]),
        'per_layer_r': [round(x, 4) for x in per_layer_r],
        'best_layer': best_layer, 'best_r': round(per_layer_r[best_layer], 4),
        'current_layer': args.current_layer,
        'current_layer_r': round(per_layer_r[args.current_layer], 4),
        'top_k': TOP_K, 'top_n_layers': TOP_N_LAYERS, 'top_layers': top_layers,
        'approaches': approaches,
        # the CompositeSelector spec: per-layer selected unit indices
        'composite_selector': {f'encoder.layer.{li}': topk_units[li] for li in top_layers},
        'selected_units': {str(li): np.argsort(unit_pred[li])[::-1][:TOP_K].tolist()
                           for li in range(N_LAYERS)},
    }
    np.save(f'{args.out}/unit_predictivity.npy', unit_pred)
    json.dump(result, open(f'{args.out}/sweep.json', 'w'), indent=2)
    log(f'DONE best_layer={best_layer} (r={result["best_r"]}) vs '
        f'current layer {args.current_layer} (r={result["current_layer_r"]})')
    log(f'  saved -> {args.out}/sweep.json + unit_predictivity.npy')


if __name__ == '__main__':
    main()
