"""Temporal-shift NULL on real Algonauts BOLD.

The most load-bearing validity check for any temporal/multimodal encoding
benchmark: if mis-timing the model features relative to the brain does NOT
degrade prediction, the "alignment" was never carrying stimulus-locked
information. Here we run it on real CNeuroMod BOLD (sub-01 Friends), sweeping
the HRF delay from the correct value out to clearly-wrong values. The score
must peak near the true delay (~3-5 TRs at TR=1.49 s) and collapse toward the
shuffle floor as |mis-timing| grows.

Uses the cached CLIP video features + the same run-aware stacking as the
production pipeline; a single ridge (alpha fixed) per shift keeps it fast. This
is a relative-shape test, not an absolute-score test.

Run on EC2 (CPU; cached features). Writes JSON to --out.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def stack(X, n_TRs, run_idx, W, D):
    """Run-aware stimulus-window stacking with HRF delay D (same as production)."""
    n_feat = X.shape[1]
    Xst = np.zeros((n_TRs, W * n_feat), dtype=np.float32)
    for offset in range(W):
        shift = D + (W - 1 - offset)
        for i in range(n_TRs):
            src = i - shift
            if src >= 0 and run_idx[src] == run_idx[i]:
                Xst[i, offset * n_feat:(offset + 1) * n_feat] = X[src]
    return Xst


def ridge_cv_median_r(X, Y, run_idx, alpha=1000.0, n_splits=5, seed=0):
    """5-fold ridge (closed form), per-parcel Pearson, return median over parcels."""
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    folds = rng.randint(0, n_splits, size=n)
    preds = np.zeros_like(Y)
    for f in range(n_splits):
        te = folds == f
        tr = ~te
        Xtr, Ytr, Xte = X[tr], Y[tr], X[te]
        xm = Xtr.mean(0, keepdims=True)
        ym = Ytr.mean(0, keepdims=True)
        Xtr_c = Xtr - xm
        W = np.linalg.solve(Xtr_c.T @ Xtr_c + alpha * np.eye(Xtr.shape[1]),
                            Xtr_c.T @ (Ytr - ym))
        preds[te] = (Xte - xm) @ W + ym
    # per-parcel Pearson r (vectorized)
    P = preds - preds.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = (P * Yc).sum(0)
    den = np.sqrt((P ** 2).sum(0) * (Yc ** 2).sum(0)) + 1e-12
    r = num / den
    return float(np.median(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assembly', default='Algonauts2025-friends-sub01')
    ap.add_argument('--video_cache',
                    default='/home/ubuntu/.brainio/algonauts2025/video_features_clip.npz')
    ap.add_argument('--stimulus_window', type=int, default=5)
    ap.add_argument('--true_delay', type=int, default=3)
    ap.add_argument('--shifts', default='-12,-6,-3,0,3,6,9,12,18,24',
                    help='HRF-delay values to test (TRs). True delay should peak.')
    ap.add_argument('--alpha', type=float, default=1000.0)
    ap.add_argument('--out', default='/tmp/algonauts_temporal_shift_null.json')
    args = ap.parse_args()

    import brainscore
    print('loading benchmark + assembly...', flush=True)
    b = brainscore.load_benchmark(args.assembly)
    a_run = list(b.assembly['run'].values)
    a_stim = list(b.assembly['stimulus_id'].values)
    Y = np.asarray(b.assembly.values, dtype=np.float32)        # (n_TRs, 1000)
    n_TRs = Y.shape[0]
    X = np.load(args.video_cache)['X'].astype(np.float32)      # (n_TRs, 768)
    assert X.shape[0] == n_TRs, f'feature/BOLD mismatch {X.shape[0]} vs {n_TRs}'
    print(f'  n_TRs={n_TRs}  X={X.shape}  Y={Y.shape}', flush=True)

    seen, run_idx = {}, np.empty(n_TRs, dtype=np.int64)
    for i, (s, r) in enumerate(zip(a_stim, a_run)):
        key = (str(s), str(r))
        seen.setdefault(key, len(seen))
        run_idx[i] = seen[key]

    # shuffle floor: pair shuffled features with BOLD (no real timing)
    perm = np.random.RandomState(0).permutation(n_TRs)
    Xsh = stack(X[perm], n_TRs, run_idx, args.stimulus_window, args.true_delay)
    floor = ridge_cv_median_r(Xsh, Y, run_idx, alpha=args.alpha)
    print(f'shuffle floor: {floor:.4f}', flush=True)

    shifts = [int(s) for s in args.shifts.split(',')]
    curve = {}
    for D in shifts:
        Xst = stack(X, n_TRs, run_idx, args.stimulus_window, D)
        r = ridge_cv_median_r(Xst, Y, run_idx, alpha=args.alpha)
        curve[D] = r
        flag = '  <-- true delay' if D == args.true_delay else ''
        print(f'  HRF delay {D:+3d} TRs : median r = {r:.4f}{flag}', flush=True)

    best_D = max(curve, key=curve.get)
    result = {
        'assembly': args.assembly, 'true_delay': args.true_delay,
        'shuffle_floor': floor, 'curve': curve, 'best_delay': best_D,
        'peak_r': curve[best_D],
        'peak_near_true': abs(best_D - args.true_delay) <= 3,
        'mistimed_collapses': curve[max(shifts)] < 0.5 * curve[best_D],
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nbest delay {best_D} (r={curve[best_D]:.4f}); floor {floor:.4f}; '
          f'peak_near_true={result["peak_near_true"]}; '
          f'mistimed_collapses={result["mistimed_collapses"]}', flush=True)
    print(f'written {args.out}', flush=True)


if __name__ == '__main__':
    main()
