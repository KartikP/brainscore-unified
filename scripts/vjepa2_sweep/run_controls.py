"""Controls for the counterintuitive findings (low effective dimensionality;
depth/layer-combination barely helps; random subsets score high).

The worry: is "~13-64 brain-predictive dimensions" a real fact, or a pipeline
artifact (the ridge, the ceiling, too few stimuli)? These controls test it:

  CONTROL 1 — label-shuffle null. Permute which BOLD response goes with which
    clip, then re-score. If the pipeline is honest, the single-layer score and
    the whole PCA-dimensionality curve must COLLAPSE to ~0. (Guards against the
    metric manufacturing structure from noise.)

  CONTROL 2 — the BRAIN's own intrinsic dimensionality. PCA the BOLD itself.
    If the reliable brain signal in this ROI is itself only ~tens of dimensions
    (likely: ~1000 stimuli, ceiling 0.73, a fairly homogeneous ROI), then a model
    can't possibly need more — the low model-dimensionality is a property of the
    DATA, not a deficiency of V-JEPA2. Reframes the whole result.

  CONTROL 3 — per-brain-PC predictability. Decompose the BOLD into PCs; predict
    each PC from the best layer. The model should predict the first ~tens of PCs
    well and the rest near zero. The count of well-predicted PCs IS the brain-
    predictive dimensionality, derived independently of the score-curve — it must
    agree with the ~64 number if that number is real.

All on V-JEPA2 best layer / Lahner visual-ROI; honest localizer/test split.
"""
import json, sys, time, os
sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
import numpy as np
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)
ALPHA_GRID = (1., 10., 100., 1000., 10000., 100000.)
OUT_DIR = '/tmp/vjepa2_sweep'


def _pvp(Yt, Yp):
    Yc = Yt - Yt.mean(0, keepdims=True); Pc = Yp - Yp.mean(0, keepdims=True)
    num = (Yc * Pc).sum(0); den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, np.nan)


def main():
    import warnings; warnings.filterwarnings('ignore')
    import brainscore
    from brainscore.tools import explore_layer_mapping, effective_dimensionality
    from brainscore.tools.layer_mapping import (
        extract_features_by_layer, per_voxel_train_test, normalize_by_ceiling)
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV

    log('load benchmark + ceiling + best-layer features...')
    b = brainscore.load_benchmark('Lahner2024-fMRI-naturalistic-visualROI')
    asm = b._average_repetitions(); mask = b._get_voxel_mask()
    Y = np.asarray(asm.transpose('stimulus_id', 'neuroid').values, np.float64)
    rel = np.asarray(b._split_half_reliability(), np.float64)
    if mask is not None: Y = Y[:, mask]; rel = rel[mask]
    ceiling = np.sqrt(np.clip(rel, 0, 1))
    stim_ids = [str(s) for s in asm['stimulus_id'].values]
    m = brainscore.load_model('vjepa2-vitl'); vw = m._preprocessors['video']
    layers = [f'encoder.layer.{i}' for i in range(24)]
    feats = extract_features_by_layer(vw, b._videos_stimulus_set(), layers)
    f_ids = list(feats.pop('_stimulus_id')); f_index = {s: i for i, s in enumerate(f_ids)}
    common = [s for s in stim_ids if s in f_index]
    yi = [stim_ids.index(s) for s in common]; fi = [f_index[s] for s in common]
    Yc = Y[yi]; feats = {l: feats[l][fi] for l in layers}
    res = explore_layer_mapping(feats, Yc, localizer_frac=0.5, alpha=1.0, seed=0)
    L, T = res.localizer_idx, res.test_idx
    X = feats[res.best_layer]
    log(f'  best layer={res.best_layer}; Y={Yc.shape}')

    def med_norm(r): return float(np.nanmedian(normalize_by_ceiling(r, ceiling)))

    # CONTROL 1 — label-shuffle null --------------------------------------------
    rng = np.random.RandomState(0)
    permL = rng.permutation(len(L)); permT = rng.permutation(len(T))
    real = med_norm(per_voxel_train_test(X[L], Yc[L], X[T], Yc[T], alpha=ALPHA_GRID))
    shuf = med_norm(per_voxel_train_test(X[L], Yc[L][permL], X[T], Yc[T][permT], alpha=ALPHA_GRID))
    log(f'CONTROL 1 (shuffle): real={real:.4f}  shuffled-labels={shuf:.4f}  (shuffled must be ~0)')

    # CONTROL 2 — brain's own intrinsic dimensionality --------------------------
    brain_pr = effective_dimensionality(Yc)
    pcaY = PCA(n_components=min(len(L), Yc.shape[1]) - 1).fit(Yc[L])
    cvY = np.cumsum(pcaY.explained_variance_ratio_)
    dims_brain = {p: int(np.searchsorted(cvY, p) + 1) for p in (0.5, 0.9, 0.95)}
    log(f'CONTROL 2 (brain dim): participation-ratio={brain_pr:.1f}; '
        f'PCs for 50/90/95% var = {dims_brain}')

    # CONTROL 3 — per-brain-PC predictability -----------------------------------
    n_pc = 60
    pcaY60 = PCA(n_components=n_pc).fit(Yc[L])
    ZL = pcaY60.transform(Yc[L]); ZT = pcaY60.transform(Yc[T])
    reg = RidgeCV(alphas=np.asarray(ALPHA_GRID), alpha_per_target=True).fit(X[L], ZL)
    pc_r = _pvp(ZT, reg.predict(X[T]))            # test r per brain PC
    well = int(np.sum(pc_r > 0.2))
    log(f'CONTROL 3 (per-PC): PC1-10 r = {[round(float(v),3) for v in pc_r[:10]]}')
    log(f'CONTROL 3: PCs predicted at r>0.2 = {well} of {n_pc}  (≈ the brain-predictive dim)')

    out = {
        'best_layer': res.best_layer, 'n_voxels': int(Yc.shape[1]), 'n_stimuli': len(common),
        'control1_shuffle': {'real': round(real, 4), 'shuffled_labels': round(shuf, 4)},
        'control2_brain_dim': {'participation_ratio': round(brain_pr, 2),
                               'pcs_for_50pct_var': dims_brain[0.5],
                               'pcs_for_90pct_var': dims_brain[0.9],
                               'pcs_for_95pct_var': dims_brain[0.95]},
        'control3_per_pc': {'pc_test_r_first20': [round(float(v), 4) for v in pc_r[:20]],
                            'n_pcs_predicted_above_0.2': well, 'n_pcs_tested': n_pc},
    }
    json.dump(out, open(f'{OUT_DIR}/controls.json', 'w'), indent=2)
    log('DONE — controls:')
    log(f'  1 shuffle null:   real {real:.3f} vs shuffled {shuf:.3f}  (PASS if shuffled≈0)')
    log(f'  2 brain dim:      PR {brain_pr:.1f}; 90% var in {dims_brain[0.9]} PCs')
    log(f'  3 per-PC:         {well}/{n_pc} brain PCs predicted at r>0.2')


if __name__ == '__main__':
    main()
