"""Calibrate the banded-ridge estimate — is banded(24)=0.724 < single-layer=0.776
a real finding or an optimizer artifact?

Banded ridge is strictly more expressive than a single layer (it can place all
weight on the best band), so a correctly-optimized banded MUST be >= the single
best layer. Getting 0.724 means either (a) the himalaya kernel-ridge PROTOCOL
itself scores lower than RidgeCV (a calibration offset, not science), or (b) the
random search over the 24-band simplex with n_iter=50 under-explored.

This disentangles them:
  - single-via-himalaya (1 band = layer 16, same normalization) calibrates the
    protocol: if it's ~0.72 not ~0.776, the offset is the protocol, and the fair
    comparison is banded(24) vs THIS.
  - banded(24, n_iter=500) gives the search a real chance on the full simplex.
  - banded(top-6 layers, n_iter=300) uses a low-dim simplex where random search
    is reliable — the cleanest "do the good layers combine to beat the best one".

All ceiling-normalized, same localizer/test split. Reuses the feature cache.
"""
import json, os, sys, time
sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
import numpy as np
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)
N_LAYERS = 24
OUT_DIR = '/tmp/vjepa2_sweep'


def _pvp(Yt, Yp):
    Yc = Yt - Yt.mean(0, keepdims=True); Pc = Yp - Yp.mean(0, keepdims=True)
    num = (Yc * Pc).sum(0); den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, np.nan)


def main():
    import warnings; warnings.filterwarnings('ignore')
    import brainscore
    from brainscore.tools import explore_layer_mapping
    from brainscore.tools.layer_mapping import extract_features_by_layer, normalize_by_ceiling
    from himalaya.backend import set_backend
    from himalaya.kernel_ridge import MultipleKernelRidgeCV
    set_backend('numpy')

    log('load benchmark + ceiling + features...')
    b = brainscore.load_benchmark('Lahner2024-fMRI-naturalistic-visualROI')
    asm = b._average_repetitions(); mask = b._get_voxel_mask()
    Y = np.asarray(asm.transpose('stimulus_id', 'neuroid').values, np.float64)
    rel = np.asarray(b._split_half_reliability(), np.float64)
    if mask is not None: Y = Y[:, mask]; rel = rel[mask]
    ceiling = np.sqrt(np.clip(rel, 0, 1))
    stim_ids = [str(s) for s in asm['stimulus_id'].values]
    m = brainscore.load_model('vjepa2-vitl'); vw = m._preprocessors['video']
    layers = [f'encoder.layer.{i}' for i in range(N_LAYERS)]
    feats = extract_features_by_layer(vw, b._videos_stimulus_set(), layers)
    f_ids = list(feats.pop('_stimulus_id')); f_index = {s: i for i, s in enumerate(f_ids)}
    common = [s for s in stim_ids if s in f_index]
    yi = [stim_ids.index(s) for s in common]; fi = [f_index[s] for s in common]
    Yc = Y[yi]; feats = {l: feats[l][fi] for l in layers}
    res = explore_layer_mapping(feats, Yc, localizer_frac=0.5, alpha=1.0, seed=0)
    L, T = res.localizer_idx, res.test_idx
    best = res.best_layer; top6 = res.top_layers(6)
    log(f'  best={best}; top6={top6}; Y={Yc.shape}')

    def banded(band_layers, n_iter):
        Ks_tr, Ks_te = [], []
        for l in band_layers:
            XL, XT = feats[l][L], feats[l][T]
            K = XL @ XL.T; sc = np.trace(K) / K.shape[0]
            Ks_tr.append(K / sc); Ks_te.append((XT @ XL.T) / sc)
        Ks_tr = np.stack(Ks_tr); Ks_te = np.stack(Ks_te)
        mkr = MultipleKernelRidgeCV(
            kernels='precomputed', solver='random_search',
            solver_params=dict(n_iter=n_iter, alphas=np.logspace(-3, 12, 16),
                               n_targets_batch=1000, progress_bar=False), cv=5)
        mkr.fit(Ks_tr, Yc[L])
        r = _pvp(Yc[T], np.asarray(mkr.predict(Ks_te)))
        return float(np.nanmedian(normalize_by_ceiling(r, ceiling)))

    log('A. single layer via himalaya (1 band = best) — protocol calibration...')
    single_h = banded([best], n_iter=5)
    log(f'   single-via-himalaya: {single_h:.4f}  (RidgeCV single was 0.7765)')

    log('B. banded(24 layers), n_iter=500...')
    b24 = banded(layers, n_iter=500)
    log(f'   banded(24, 500): {b24:.4f}')

    log('C. banded(top-6 layers), n_iter=300...')
    b6 = banded(top6, n_iter=300)
    log(f'   banded(top-6, 300): {b6:.4f}')

    out = {'single_ridgecv': 0.7765, 'single_via_himalaya': round(single_h, 4),
           'banded_24_niter500': round(b24, 4), 'banded_top6_niter300': round(b6, 4),
           'best_layer': best, 'top6': top6}
    json.dump(out, open(f'{OUT_DIR}/banded_calibrate.json', 'w'), indent=2)
    log('DONE:')
    log(f'  single (RidgeCV)        0.7765')
    log(f'  single (himalaya 1-band){single_h:.4f}  <- protocol offset vs RidgeCV')
    log(f'  banded 24 layers        {b24:.4f}')
    log(f'  banded top-6 layers     {b6:.4f}')


if __name__ == '__main__':
    main()
