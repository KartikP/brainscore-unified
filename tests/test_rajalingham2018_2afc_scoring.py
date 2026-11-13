"""Tests for the faithful Rajalingham2018 2-AFC scoring core.

Uses small synthetic human data (no S3 download) to verify the choice->assembly
->d-prime->correlate pipeline: random model choices score ~0, while a model that
copies the human majority scores clearly higher. The real-data calibration
(i2n raw -0.02 for random, 0.33 for human-majority) lives in the build notes.
"""
import numpy as np
import pandas as pd
import pytest

from brainscore.benchmarks.rajalingham2018_2afc import benchmark as B


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


def test_two_afc_model_runs_through_participate():
    """A subject with an answer part answers through participate; trials that share an
    image (both sides of one condition) stay separate because the driver joins
    answers to its own trial table by the trial stimulus_id."""
    from brainscore import BrainScoreModel
    from brainscore.benchmarks.rajalingham2018_2afc.score_2afc import INSTRUCTION_DIRECT, run_trials
    stim = pd.DataFrame([
        {'stimulus_id': 'trial_00000L', 'image_path': 'm0', 'image_id': 'img0',
         'sample_obj': 'wrench', 'dist_obj': 'rhino', 'left_obj': 'wrench', 'right_obj': 'rhino'},
        {'stimulus_id': 'trial_00000R', 'image_path': 'm1', 'image_id': 'img0',
         'sample_obj': 'wrench', 'dist_obj': 'rhino', 'left_obj': 'rhino', 'right_obj': 'wrench'},
    ])
    model = BrainScoreModel('left', in_channels={'vision:image'},
                            answer=lambda stimulus_row, instruction, label_set: 'LEFT')
    assert set(model.in_channels) == {'vision:image', 'instruction', 'control'}
    choices = run_trials(model, stim, INSTRUCTION_DIRECT)
    assert list(choices['choice']) == ['wrench', 'rhino']
    assert list(choices['image_id']) == ['img0', 'img0']
    assert list(choices['left_obj']) == ['wrench', 'rhino']
