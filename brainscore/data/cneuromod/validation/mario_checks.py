"""Mario checks from the packaged time series (Paugam et al. 2025, bioRxiv 10.1101/2025.11.28.691119).

1. ``glm``, a check of the data and of the per-TR button columns (not a figure of the paper): per
   participant, a linear model of each parcel on the presses of left-thumb buttons (LEFT, RIGHT, UP, DOWN)
   and of right-thumb buttons (A, B) in each TR and the seconds of play, each convolved with the SPM HRF per
   run, one intercept per run, each parcel z-scored within each run. The contrast left - right thumb must be
   positive in the right somatomotor parcels and negative in the left ones (contralateral motor cortex).
2. ``map``, the check of the paper's Fig. 2c (encoding maps of their untrained network, highest in "the
   visual cortex, the dorsal attentional path and in the motor cortex"): the per-parcel r of a score of
   ``Paugam2025-fMRI-mario-cortex`` averaged over the participants by parcel name, and its mean per network
   of the atlas names (Vis, SomMot, DorsAttn, ...). The paper releases no map, so the comparison is the
   ranking of the networks the paper names.

    python -m brainscore.data.cneuromod.validation.mario_checks glm --packaged <folder> --out <folder>
    python -m brainscore.data.cneuromod.validation.mario_checks map --score <score.json> --out <folder>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TR_SECONDS = 1.49
LEFT_THUMB = ('LEFT', 'RIGHT', 'UP', 'DOWN')
RIGHT_THUMB = ('A', 'B')


def network(parcel_name: str) -> str:
    """The network of a cneuromod2026 parcel name: ``7Networks_LH_SomMot_12`` -> ``SomMot``; a subcortical or
    cerebellar parcel (no hemisphere prefix) -> ``subcortex/cerebellum``."""
    parts = str(parcel_name).removeprefix('7Networks_').split('_')
    return parts[1] if len(parts) > 2 and parts[0] in ('LH', 'RH') else 'subcortex/cerebellum'


def design(series) -> tuple:
    """``(regressors TR x 3, run label per TR)``: left thumb, right thumb, play; HRF per run."""
    from brainscore.temporal import hrf_convolve
    # presses, not seconds held: a participant who holds a button most of the time (sub-01 holds B in 79 % of
    # its play) has a nearly constant held-seconds regressor
    left = sum(series[f'presses_{b}'].values for b in LEFT_THUMB)
    right = sum(series[f'presses_{b}'].values for b in RIGHT_THUMB)
    play = np.clip(series['frame_stop'].values - series['frame_start'].values, 0, None) / 60.0
    X = np.stack([left, right, play], axis=1).astype(float)
    runs = np.array([f'{s}-{r}' for s, r in zip(series['session'].values, series['run'].values)])
    for run in np.unique(runs):
        X[runs == run] = hrf_convolve(X[runs == run], 1 / TR_SECONDS)
    return X, runs


def contrast(Y: np.ndarray, X: np.ndarray, runs: np.ndarray) -> np.ndarray:
    """The t value of left - right thumb per parcel (OLS, one intercept per run)."""
    Y = Y.astype(float).copy()
    for run in np.unique(runs):
        inside = runs == run
        Y[inside] = (Y[inside] - Y[inside].mean(0)) / (Y[inside].std(0) + 1e-8)
    intercepts = (runs[:, None] == np.unique(runs)[None]).astype(float)
    full = np.concatenate([X, intercepts], axis=1)
    beta, *_ = np.linalg.lstsq(full, Y, rcond=None)
    residual = Y - full @ beta
    dof = len(Y) - np.linalg.matrix_rank(full)
    sigma2 = (residual ** 2).sum(0) / dof
    c = np.zeros(full.shape[1])
    c[:2] = (1, -1)
    variance = c @ np.linalg.pinv(full.T @ full) @ c
    return (c @ beta) / np.sqrt(sigma2 * variance)


def glm(packaged: Path, out: Path) -> pd.DataFrame:
    from brainscore.data.cneuromod.data import load_timeseries
    rows = []
    for folder in sorted(packaged.glob('sub-*')):
        series = load_timeseries(folder.name, packaged)
        X, runs = design(series)
        t = contrast(np.asarray(series.values), X, runs)
        names = series['parcel_name'].values.astype(str)
        for hemisphere in ('LH', 'RH'):
            motor = np.array([n.removeprefix('7Networks_').startswith(f'{hemisphere}_SomMot') for n in names])
            rows.append({'participant': folder.name, 'hemisphere': hemisphere, 'parcels': int(motor.sum()),
                         'mean_t': float(t[motor].mean()), 'median_t': float(np.median(t[motor])),
                         'n_trs': int(len(runs)), 'n_runs': int(len(np.unique(runs))),
                         'top_positive': names[np.argsort(-t)[:3]].tolist(),
                         'top_negative': names[np.argsort(t)[:3]].tolist()})
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'mario_glm_hands.csv', index=False)
    return table


def parcel_map(score_json: Path, out: Path) -> tuple:
    """``(per-parcel mean r over participants, per-network table)`` of a saved score."""
    score = json.loads(Path(score_json).read_text())
    frames = [pd.DataFrame({'parcel_name': part['parcel_names'], 'r': part['per_parcel_r'], 'participant': name})
              for name, part in score['per_participant'].items()]
    table = pd.concat(frames, ignore_index=True)
    parcels = table.groupby('parcel_name')['r'].mean().sort_values(ascending=False).reset_index()
    parcels['network'] = [network(n) for n in parcels['parcel_name']]
    networks = (parcels.groupby('network')['r'].agg(['mean', 'median', 'count'])
                .sort_values('mean', ascending=False).reset_index())
    out.mkdir(parents=True, exist_ok=True)
    parcels.to_csv(out / 'mario_map_parcels.csv', index=False)
    networks.to_csv(out / 'mario_map_networks.csv', index=False)
    return parcels, networks


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('check', choices=('glm', 'map'))
    parser.add_argument('--packaged')
    parser.add_argument('--score')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.check == 'glm':
        print(glm(Path(args.packaged), Path(args.out)).to_string())
    else:
        print(parcel_map(Path(args.score), Path(args.out))[1].to_string())


if __name__ == '__main__':
    main()
