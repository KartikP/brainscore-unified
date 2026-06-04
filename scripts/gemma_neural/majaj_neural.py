"""MajajHong2015 V4/IT neural scoring for standalone-extracted features (bsu env).

Two modes:
  --dump   : load MajajHong2015.public (V4 + IT), write a stimulus manifest
             (stimulus_id,image_path) + save the neural response matrices per region.
  --score  : align features.npy (from gemma_neural_extract) with the neural data by
             stimulus_id and PLS-regress (25 comp, 5-fold CV), reporting median
             per-neuroid Pearson r per region — the standard MajajHong encoding metric.

Decouples Gemma feature extraction (gemma4 env) from neural scoring (bsu env), same
pattern as the 2-AFC choices->score split.
"""
import argparse, csv, json, os

import numpy as np


def dump(out):
    from brainscore_vision import load_dataset
    os.makedirs(out, exist_ok=True)
    asm = load_dataset('MajajHong2015.public')
    asm = asm.squeeze()
    ss = asm.stimulus_set
    # manifest
    sids = [str(s) for s in asm['stimulus_id'].values]
    uniq = list(dict.fromkeys(sids))
    with open(os.path.join(out, 'manifest.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['stimulus_id', 'image_path'])
        for sid in uniq:
            w.writerow([sid, str(ss.get_stimulus(sid))])
    # neural response matrix per region, averaged over repetitions -> (n_stim, n_neuroid)
    import xarray as xr
    region = asm['region'].values
    np.savez(os.path.join(out, 'neural.npz'),
             stimulus_id=np.array(sids), region=region, responses=np.asarray(asm.values))
    print(f'dumped {len(uniq)} stimuli, neural shape {asm.shape}, regions {sorted(set(region))}', flush=True)


def score(out, features_dir):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    from scipy.stats import pearsonr
    nd = np.load(os.path.join(out, 'neural.npz'), allow_pickle=True)
    feats = np.load(os.path.join(features_dir, 'features.npy'))
    fids = json.load(open(os.path.join(features_dir, 'image_ids.json')))
    fmap = {sid: feats[i] for i, sid in enumerate(fids)}
    # align: average neural over presentations per (stimulus, neuroid)
    sids = nd['stimulus_id']; resp = nd['responses']; region = nd['region']
    import pandas as pd
    df = pd.DataFrame(resp); df['sid'] = sids
    avg = df.groupby('sid').mean()                      # (n_unique_stim, n_neuroid)
    shared = [s for s in avg.index if s in fmap]
    X = np.stack([fmap[s] for s in shared])
    Y = avg.loc[shared].values
    out_scores = {}
    for reg in sorted(set(region)):
        cols = np.where(region == reg)[0]
        Yr = Y[:, cols]
        kf = KFold(5, shuffle=True, random_state=0)
        preds = np.zeros_like(Yr)
        for tr, te in kf.split(X):
            pls = PLSRegression(n_components=25)
            pls.fit(X[tr], Yr[tr]); preds[te] = pls.predict(X[te])
        rs = [pearsonr(Yr[:, j], preds[:, j])[0] for j in range(Yr.shape[1])]
        out_scores[reg] = round(float(np.nanmedian(rs)), 4)
    res = {'benchmark': 'MajajHong2015 V4/IT (Gemma-4 encoder-free features)',
           'n_stimuli': len(shared), 'feature_dim': int(X.shape[1]),
           'median_r_by_region': out_scores}
    json.dump(res, open(os.path.join(out, 'gemma_score.json'), 'w'), indent=2)
    print(json.dumps(res, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['dump', 'score'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--features_dir', default='')
    args = ap.parse_args()
    if args.mode == 'dump':
        dump(args.out)
    else:
        score(args.out, args.features_dir or args.out)


if __name__ == '__main__':
    main()
