"""Tests for the faithful Rajalingham2018 2-AFC scoring core.

Uses small synthetic human data (no S3 download) to verify the choice->assembly
->d-prime->correlate pipeline: random model choices score ~0, while a model that
copies the human majority scores clearly higher. The real-data calibration
(i2n raw -0.02 for random, 0.33 for human-majority) lives in the build notes.
"""
import numpy as np
import pandas as pd
import pytest

from . import benchmark as B


def _synthetic(seed=0, n_obj=6, imgs_per_obj=8, reps=10):
    """Synthetic 2-AFC data with a BROAD spread of per-condition difficulty, so
    the d-prime response matrix has real variance (degenerate near-constant
    matrices give NaN correlations)."""
    rng = np.random.RandomState(seed)
    objs = [f'o{i}' for i in range(n_obj)]
    rows = []          # human per-trial rows
    conds = []         # one row per (image, dist) condition
    for obj in objs:
        for k in range(imgs_per_obj):
            iid = f'{obj}_img{k}'
            for dist in objs:
                if dist == obj:
                    continue
                conds.append({'image_id': iid, 'sample_obj': obj, 'dist_obj': dist})
                # difficulty spans chance->easy so conditions differ (signal to correlate on)
                p = rng.uniform(0.5, 0.98)
                for _ in range(reps):
                    choice = obj if rng.rand() < p else dist
                    rows.append({'image_id': iid, 'sample_obj': obj, 'dist_obj': dist, 'choice': choice})
    return pd.DataFrame(rows), pd.DataFrame(conds), objs


def test_choices_to_assembly_coords():
    _, conds, _ = _synthetic()
    conds['choice'] = conds['sample_obj']
    asm = B.choices_to_assembly(conds)
    assert asm.dims == ('presentation',)
    # in xarray 2022.3 these are MultiIndex levels on 'presentation', not top-level coords
    names = set(asm.indexes['presentation'].names)
    for c in ('stimulus_id', 'sample_obj', 'dist_obj', 'truth'):
        assert c in names
    assert (asm['truth'].values == asm['sample_obj'].values).all()


def test_random_scores_near_zero_and_copy_scores_higher():
    human_df, conds, _ = _synthetic(seed=1)
    rng = np.random.RandomState(7)

    rand = conds.copy()
    flip = rng.rand(len(rand)) < 0.5
    rand['choice'] = np.where(flip, rand['sample_obj'], rand['dist_obj'])

    # model copies the human-majority choice per condition
    maj = (human_df.assign(cs=(human_df['choice'] == human_df['sample_obj']).astype(float))
           .groupby(['image_id', 'sample_obj', 'dist_obj'])['cs'].mean().reset_index())
    maj['choice'] = np.where(maj['cs'] >= 0.5, maj['sample_obj'], maj['dist_obj'])
    copy = maj[['image_id', 'sample_obj', 'dist_obj', 'choice']]

    r_rand = B.score_choices(rand, human_df=human_df, metric_name='i2n', repetitions=5)
    r_copy = B.score_choices(copy, human_df=human_df, metric_name='i2n', repetitions=5)

    assert np.isfinite(r_rand['raw']) and np.isfinite(r_copy['raw'])
    assert abs(r_rand['raw']) < 0.2          # random ~ 0
    assert r_copy['raw'] > r_rand['raw'] + 0.1   # copying humans clearly beats random
