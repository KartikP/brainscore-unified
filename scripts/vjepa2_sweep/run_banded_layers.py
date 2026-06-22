"""Does combining ALL layers (properly weighted) beat the single best layer?

Answers "banded ridge over all 24 layers — same or higher?" plus the
dimensionality questions behind it. On V-JEPA2 / Lahner visual-ROI, all scores
ceiling-normalized (÷ sqrt split-half reliability), honest localizer/test split:

  1. single best layer            — the 0.776 reference
  2. best-per-voxel layer         — each voxel uses its own best layer (chosen on
                                    the localizer): the "voxel-heterogeneity" ceiling
  3. banded ridge over 24 layers  — himalaya MultipleKernelRidgeCV, one kernel per
                                    layer, per-band weight tuned by CV random search
  4. shuffled-bands null          — banded over 24 RANDOM equal-size feature groups;
                                    if banded(layers) ≈ this, the layer structure
                                    adds nothing beyond "more bands = more knobs"
  5. PCA dimensionality curve      — how many dims of the best layer reach 95% of its
                                    score (brain-predictive dimensionality)
  6. effective dimensionality      — participation ratio of the best layer

Interpretation:
  banded ≈ single best layer  → depth is redundant; one layer holds all the signal.
  banded >  single best layer  → layers carry complementary signal (likely different
                                voxels prefer different depths across the ROI gradient).
  banded ≈ shuffled-bands null → the gain (if any) is extra parameters, not layers.

Run on EC2. Needs himalaya (pip install himalaya); reuses the V-JEPA2 feature cache.
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
ALPHA_GRID = (1., 10., 100., 1000., 10000., 100000.)
PCA_DIMS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
OUT_DIR = '/tmp/vjepa2_sweep'


def _per_voxel_pearson(Yt, Yp):
    Yc = Yt - Yt.mean(0, keepdims=True); Pc = Yp - Yp.mean(0, keepdims=True)
    num = (Yc * Pc).sum(0); den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, np.nan)


def main():
    import warnings; warnings.filterwarnings('ignore')
    os.makedirs(OUT_DIR, exist_ok=True)
    import brainscore
    from brainscore.tools import explore_layer_mapping, effective_dimensionality
    from brainscore.tools.layer_mapping import (
        extract_features_by_layer, per_voxel_train_test, normalize_by_ceiling)
    from sklearn.linear_model import RidgeCV
    from sklearn.decomposition import PCA

    log('load benchmark + BOLD + noise ceiling...')
    b = brainscore.load_benchmark('Lahner2024-fMRI-naturalistic-visualROI')
    asm = b._average_repetitions()
    mask = b._get_voxel_mask()
    Y = np.asarray(asm.transpose('stimulus_id', 'neuroid').values, np.float64)
    reliability = np.asarray(b._split_half_reliability(), np.float64)
    if mask is not None:
        Y = Y[:, mask]; reliability = reliability[mask]
    ceiling = np.sqrt(np.clip(reliability, 0.0, 1.0))
    stim_ids = [str(s) for s in asm['stimulus_id'].values]
    vids = b._videos_stimulus_set()

    log('extract V-JEPA2 features (cache hit expected)...')
    m = brainscore.load_model('vjepa2-vitl')
    vw = m._preprocessors['video']
    layers = [f'encoder.layer.{i}' for i in range(N_LAYERS)]
    feats = extract_features_by_layer(vw, vids, layers)
    f_ids = list(feats.pop('_stimulus_id'))
    f_index = {s: i for i, s in enumerate(f_ids)}
    common = [s for s in stim_ids if s in f_index]
    yi = [stim_ids.index(s) for s in common]; fi = [f_index[s] for s in common]
    Yc = Y[yi]
    feats = {l: feats[l][fi] for l in layers}
    log(f'  aligned on {len(common)} stimuli; Y={Yc.shape}')

    res = explore_layer_mapping(feats, Yc, localizer_frac=0.5, alpha=1.0, seed=0)
    L, T = res.localizer_idx, res.test_idx
    best = res.best_layer
    log(f'  best layer = {best}')

    def med_norm(r_test):
        return float(np.nanmedian(normalize_by_ceiling(r_test, ceiling)))

    # 1. single best layer (RidgeCV, ceiling-normalized) -----------------------
    single = med_norm(per_voxel_train_test(feats[best][L], Yc[L], feats[best][T],
                                           Yc[T], alpha=ALPHA_GRID))
    log(f'1. single best layer:          {single:.4f}')

    # 2. best-per-voxel layer (select on HELD-OUT localizer CV, score on test) --
    # Selection must use held-out data: training-fit r is optimistic and biases
    # the per-voxel pick toward high-capacity late layers that generalize worse.
    from sklearn.model_selection import KFold

    def cv_r_per_voxel(X, Yv, n_splits=5):
        preds = np.full_like(Yv, np.nan)
        for tr, va in KFold(n_splits, shuffle=True, random_state=0).split(X):
            reg = RidgeCV(alphas=np.asarray(ALPHA_GRID), alpha_per_target=True).fit(X[tr], Yv[tr])
            preds[va] = reg.predict(X[va])
        return _per_voxel_pearson(Yv, preds)

    n_vox = Yc.shape[1]
    r_sel = np.zeros((N_LAYERS, n_vox)); r_test = np.zeros((N_LAYERS, n_vox))
    for i, l in enumerate(layers):
        X = feats[l]
        r_sel[i] = cv_r_per_voxel(X[L], Yc[L])                    # unbiased selection
        reg = RidgeCV(alphas=np.asarray(ALPHA_GRID), alpha_per_target=True).fit(X[L], Yc[L])
        r_test[i] = _per_voxel_pearson(Yc[T], reg.predict(X[T]))  # honest score
    pick = np.nanargmax(r_sel, axis=0)
    bpv_test = r_test[pick, np.arange(n_vox)]
    best_per_voxel = float(np.nanmedian(normalize_by_ceiling(bpv_test, ceiling)))
    layer_hist = {int(k): int(v) for k, v in zip(*np.unique(pick, return_counts=True))}
    log(f'2. best-per-voxel layer:       {best_per_voxel:.4f}  (layer usage {layer_hist})')

    # 5+6. dimensionality of the best layer ------------------------------------
    eff_dim = effective_dimensionality(feats[best])
    n_comp = min(feats[best].shape[1], len(L) - 1)  # PCA full solver: ≤ n_samples
    pca = PCA(n_components=n_comp).fit(feats[best][L])
    Zl = pca.transform(feats[best][L]); Zt = pca.transform(feats[best][T])
    pca_curve = []
    for d in [d for d in PCA_DIMS if d <= n_comp]:
        s = med_norm(per_voxel_train_test(Zl[:, :d], Yc[L], Zt[:, :d], Yc[T], alpha=ALPHA_GRID))
        pca_curve.append({'d': d, 'r': round(s, 4)})
    target95 = 0.95 * single
    dim95 = next((e['d'] for e in pca_curve if e['r'] >= target95), PCA_DIMS[-1])
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    var90 = int(np.searchsorted(cumvar, 0.90) + 1)
    log(f'5. effective dim={eff_dim:.1f}; dims for 95% of score={dim95}; dims for 90% var={var90}')

    # 3+4. banded ridge over layers + shuffled-bands null ----------------------
    from himalaya.backend import set_backend
    from himalaya.kernel_ridge import MultipleKernelRidgeCV
    set_backend('numpy')

    def banded(bands_L, bands_T):
        # one linear kernel per band, each normalized to unit mean-diagonal so
        # bands are comparable; himalaya tunes the per-band weights by CV.
        Ks_tr, Ks_te = [], []
        for XbL, XbT in zip(bands_L, bands_T):
            Ktr = XbL @ XbL.T
            scale = np.trace(Ktr) / Ktr.shape[0]
            Ks_tr.append(Ktr / scale); Ks_te.append((XbT @ XbL.T) / scale)
        Ks_tr = np.stack(Ks_tr); Ks_te = np.stack(Ks_te)
        mkr = MultipleKernelRidgeCV(
            kernels='precomputed', solver='random_search',
            solver_params=dict(n_iter=50, alphas=np.logspace(0, 12, 13),
                               n_targets_batch=1000, progress_bar=False), cv=5)
        mkr.fit(Ks_tr, Yc[L])
        return _per_voxel_pearson(Yc[T], np.asarray(mkr.predict(Ks_te)))

    log('3. banded ridge over 24 real layers...')
    banded_real = med_norm(banded([feats[l][L] for l in layers],
                                   [feats[l][T] for l in layers]))
    log(f'   banded(24 layers):          {banded_real:.4f}')

    log('4. shuffled-bands null (24 random equal feature groups)...')
    allL = np.concatenate([feats[l][L] for l in layers], axis=1)
    allT = np.concatenate([feats[l][T] for l in layers], axis=1)
    rng = np.random.RandomState(0)
    perm = rng.permutation(allL.shape[1])
    groups = np.array_split(perm, N_LAYERS)
    banded_null = med_norm(banded([allL[:, g] for g in groups],
                                  [allT[:, g] for g in groups]))
    log(f'   banded(24 shuffled bands):  {banded_null:.4f}')

    out = {
        'model': 'vjepa2-vitl', 'benchmark': 'Lahner2024-fMRI-naturalistic-visualROI',
        'n_stimuli': len(common), 'n_voxels': int(n_vox),
        'best_layer': best, 'ceiling_median': round(float(np.nanmedian(ceiling)), 4),
        'single_best_layer': round(single, 4),
        'best_per_voxel_layer': round(best_per_voxel, 4),
        'best_per_voxel_layer_usage': layer_hist,
        'banded_24_layers': round(banded_real, 4),
        'banded_shuffled_bands_null': round(banded_null, 4),
        'effective_dimensionality': round(eff_dim, 2),
        'pca_score_curve': pca_curve,
        'dims_for_95pct_of_score': int(dim95),
        'dims_for_90pct_variance': int(var90),
        'alpha_grid': list(ALPHA_GRID),
    }
    json.dump(out, open(f'{OUT_DIR}/banded_layers.json', 'w'), indent=2)
    log('DONE — summary:')
    log(f"  single best layer      {single:.4f}")
    log(f"  best-per-voxel layer   {best_per_voxel:.4f}")
    log(f"  banded (24 layers)     {banded_real:.4f}")
    log(f"  banded (shuffled null) {banded_null:.4f}")
    log(f"  effective dim {eff_dim:.1f} | 95%-of-score at {dim95} dims | 90%-var at {var90} dims")


if __name__ == '__main__':
    main()
