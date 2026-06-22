"""Run the productionized layer-mapping tool on V-JEPA2 / Lahner visual-ROI.

Uses brainscore.tools (functional-localization split: select units on the
localizer, score on held-out test). Reuses the V-JEPA2 feature cache warmed by
the first extraction, so this is fast on a second run.

Outputs (/tmp/vjepa2_sweep):
  sweep_honest.json     — per_layer_r, best_layer, four-approach scores, composite spec
  unit_predictivity.npy — (24, n_units) localizer per-unit predictivity (heatmap source)
"""
import json
import sys
import time

sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
import numpy as np

t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

N_LAYERS = 24
CURRENT = 16


def main():
    import warnings; warnings.filterwarnings('ignore')
    import brainscore
    from brainscore.tools import explore_layer_mapping, score_approaches
    from brainscore.tools.layer_mapping import extract_features_by_layer

    log('load benchmark + BOLD target (visual ROI)...')
    b = brainscore.load_benchmark('Lahner2024-fMRI-naturalistic-visualROI')
    asm = b._average_repetitions()
    mask = b._get_voxel_mask()
    Y = np.asarray(asm.transpose('stimulus_id', 'neuroid').values, np.float64)
    Y = Y[:, mask] if mask is not None else Y
    stim_ids = [str(s) for s in asm['stimulus_id'].values]
    vids = b._videos_stimulus_set()

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

    log('explore layer mapping (localizer/test split)...')
    res = explore_layer_mapping(feats, Yc, localizer_frac=0.5, alpha=1.0, seed=0)
    appr = score_approaches(feats, Yc, res, top_n_layers=3, top_k=100, alpha=1.0)
    comp = res.composite_selector(n_layers=3, k=100)
    best_idx = res.best_index

    # Benchmark-protocol score (5-fold CV over the full set, no localizer split),
    # so the best-layer number is directly comparable to the video_encoding curve
    # (which is benchmark 5-fold median-r, not localizer/test-split r).
    from brainscore.tools.layer_mapping import per_voxel_cv_ridge
    log('benchmark-protocol (5-fold CV) score at best layer + current layer 16...')
    best_layer_path = res.best_layer
    cv5_best = float(np.nanmedian(per_voxel_cv_ridge(feats[best_layer_path], Yc, alpha=1.0)))
    cv5_cur = float(np.nanmedian(per_voxel_cv_ridge(feats[f'encoder.layer.{CURRENT}'], Yc, alpha=1.0)))
    log(f'  cv5 best(layer {best_idx})={cv5_best:.4f}  cv5 current(16)={cv5_cur:.4f}')

    out = {
        'model': 'vjepa2-vitl', 'benchmark': 'Lahner2024-fMRI-naturalistic-visualROI',
        'protocol': 'functional-localization split (select on localizer, score on held-out test)',
        'n_stimuli': len(common), 'n_voxels': int(Yc.shape[1]),
        'per_layer_r': [round(x, 4) for x in res.per_layer_r],
        'best_layer': res.best_layer, 'best_r': round(res.best_r, 4),
        'current_layer': f'encoder.layer.{CURRENT}',
        'current_layer_r': round(res.per_layer_r[CURRENT], 4),
        'top_layers': res.top_layers(3),
        'approaches': appr,
        'composite_selector': {lp: list(idx) for lp, idx in comp.layers},
        'benchmark_protocol_5fold_cv': {
            'best_layer_r': round(cv5_best, 4),
            'current_layer_16_r': round(cv5_cur, 4),
            'note': 'full-set 5-fold KFold ridge, no scaler — comparable to the video_encoding curve',
        },
    }
    np.save('/tmp/vjepa2_sweep/unit_predictivity.npy', res.unit_predictivity)
    json.dump(out, open('/tmp/vjepa2_sweep/sweep_honest.json', 'w'), indent=2)
    log(f'DONE best={res.best_layer} (r={res.best_r:.4f}) vs current '
        f'encoder.layer.{CURRENT} (r={res.per_layer_r[CURRENT]:.4f})')
    for a in appr:
        log(f"  {a['name']:30s} r={a['r']:.4f}  ({a['n_features']} feat)")


if __name__ == '__main__':
    main()
