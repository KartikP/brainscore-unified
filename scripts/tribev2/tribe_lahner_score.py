"""Direct-comparison score: TRIBEv2 20484-vertex preds vs Lahner fsaverage5 GLM betas.
TRIBEv2 outputs brain space directly (trained encoder) -> per-voxel Pearson across clips,
NOT a ridge readout. In-distribution (TRIBEv2 trained on Lahner) -> sanity score, not a leaderboard claim."""
import json
import numpy as np
import xarray as xr

NC = "/home/ubuntu/.brainio/2c7f1d2e5724b8cc3c5cf47986e956c4f13001e4/assy_Lahner2024-fMRI.nc"
da = xr.open_dataarray(NC).isel(time_bin=0)                       # (neuroid, presentation)
sid = np.asarray(da["stimulus_id"].values)
data = np.asarray(da.transpose("presentation", "neuroid").values, dtype="float64")  # (10260, 20484)

preds = np.load("/tmp/tribe_lahner_preds.npz")
beta = {}
for s in preds.files:                                            # mean over the clip s reps
    m = sid == s
    if m.sum() > 0:
        beta[s] = data[m].mean(0)
sids = list(beta)
print(f"matched {len(sids)}/{len(preds.files)} pred clips to assembly", flush=True)
X = np.stack([preds[s] for s in sids]).astype("float64")          # TRIBEv2 preds (n, 20484)
Y = np.stack([beta[s] for s in sids])                             # BOLD betas  (n, 20484)

def per_voxel_r(A, B):
    Ac = A - A.mean(0); Bc = B - B.mean(0)
    den = np.sqrt((Ac**2).sum(0) * (Bc**2).sum(0))
    return (Ac*Bc).sum(0) / np.where(den == 0, np.nan, den)

r = per_voxel_r(X, Y)
rnull = per_voxel_r(X[np.random.RandomState(0).permutation(len(sids))], Y)  # clip-shuffle null
res = {"n_clips": len(sids), "n_voxels": int(X.shape[1]),
       "median_r": float(np.nanmedian(r)), "mean_r": float(np.nanmean(r)),
       "top10pct_median_r": float(np.nanmedian(np.sort(r[np.isfinite(r)])[-int(np.isfinite(r).sum()*0.1):])),
       "shuffle_null_median_r": float(np.nanmedian(rnull)),
       "frac_voxels_r_gt_0.1": float(np.nanmean(r > 0.1))}
print("SCORE", json.dumps(res, indent=2), flush=True)
json.dump(res, open("/tmp/tribe_lahner_score.json", "w"), indent=2)
