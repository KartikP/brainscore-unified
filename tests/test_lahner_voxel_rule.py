"""Lahner2024 version 2: the voxels that count, and the region selected before the fit.

One rule in every variant: a voxel counts when its held-out r is finite and, with a
measured ceiling, its ceiling is finite and above zero. The other voxels stay in
``per_voxel_r`` (NaN where r is undefined), are left out of every median and mean,
and are counted in ``n_voxels_excluded``. The time-resolved benchmark keeps the
region's voxels before the fit, so voxels outside the region never enter it.

Each expected value is computed here from the data with scikit-learn, not read
from the benchmark. Every test runs the benchmark call on synthetic data behind
the registered loaders (``tests/lahner_synthetic.py``).
"""
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from brainscore.benchmarks.lahner2024 import (
    VISUAL_ROI_CEILING, Lahner2024BOLDMoments, Lahner2024BOLDMoments_timeresolved, RunBlocks)
from brainscore.benchmarks.lahner2024.benchmark import _split_half_reliability
from brainscore.benchmarks.lahner2024.benchmark_timeresolved import TR_SEC
from brainscore.temporal import double_gamma_hrf, hrf_convolve
from tests.lahner_synthetic import (
    CLIPS, GLM_REPS, N_TR, N_VOXELS, Recorder, glm_assembly, glm_values, install, tr_events,
    tr_runs, unit_values)

WIDTH = 4   # features per clip of the default Recorder


def pearson(Y, P):
    """Per-column Pearson r in float64; NaN for a constant column."""
    Y, P = np.asarray(Y, dtype=float), np.asarray(P, dtype=float)
    Yc, Pc = Y - Y.mean(0), P - P.mean(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (Yc * Pc).sum(0) / den, np.nan)


def held_out_r(X, Y, folds):
    held = np.empty(Y.shape, dtype=float)
    for train, test in folds:
        held[test] = Ridge(alpha=1.0).fit(X[train], Y[train]).predict(X[test])
    return pearson(Y, held)


def mask_of(voxels):
    mask = np.zeros(N_VOXELS, dtype=bool)
    mask[list(voxels)] = True
    return mask


# ── GLM betas ─────────────────────────────────────────────────────


def glm_expected_r(values, mask):
    """5 shuffled folds over the clips (seed 0), ridge at penalty 1, on the clip means."""
    features = np.stack([unit_values(clip, 'neural:IT', WIDTH)[0] for clip in CLIPS])
    neural = values.reshape(GLM_REPS, len(CLIPS), N_VOXELS).mean(0)[:, mask]
    folds = list(KFold(n_splits=5, shuffle=True, random_state=0).split(features))
    return held_out_r(features, neural, folds)


def test_a_constant_voxel_in_the_region_is_left_out_and_counted(monkeypatch, tmp_path):
    values = glm_values()
    constant = 10005
    values[:, constant] = 3.0
    mask = mask_of(range(10000, 10040))
    install(monkeypatch, tmp_path, glm=values)
    score = Lahner2024BOLDMoments(voxel_mask_fn=lambda: mask)(Recorder({'vision:video'}))

    expected = glm_expected_r(values, mask)
    position = int(np.flatnonzero(np.flatnonzero(mask) == constant)[0])
    r = score.attrs['per_voxel_r']
    assert len(r) == mask.sum() == score.attrs['voxel_mask_n_kept']
    assert np.isnan(r[position]) and np.isnan(expected[position])
    counts = ~np.isnan(expected)
    np.testing.assert_allclose(r[counts], expected[counts], rtol=1e-4, atol=1e-6)
    assert (score.attrs['n_voxels_scored'], score.attrs['n_voxels_excluded']) == (39, 1)
    assert float(score) == pytest.approx(np.median(expected[counts]), rel=1e-4)
    assert score.attrs['mean_r'] == pytest.approx(np.mean(expected[counts]), rel=1e-4)
    assert score.attrs['ceiling_aggregation'] == 'scalar'


def test_with_a_measured_ceiling_a_voxel_without_r_or_ceiling_is_left_out(monkeypatch, tmp_path):
    """Version 1 divided the shorter vector of finite r by every ceiling: a shape error
    here. Version 2 leaves out the constant voxel (no r, no ceiling) and the voxel
    whose reliability is at or below zero (an r, no ceiling)."""
    values = glm_values()
    constant = 20000
    values[:, constant] = 3.0
    reliability = _split_half_reliability(glm_assembly(values))
    reliable = [int(v) for v in np.flatnonzero(reliability > 0.3)[-30:]]
    unreliable = int(np.flatnonzero((reliability <= 0) & (np.arange(N_VOXELS) < 1000))[0])
    mask = mask_of(reliable + [constant, unreliable])
    install(monkeypatch, tmp_path, glm=values)
    score = Lahner2024BOLDMoments(ceiling=VISUAL_ROI_CEILING, voxel_mask_fn=lambda: mask)(
        Recorder({'vision:video'}))

    expected = glm_expected_r(values, mask)
    ceiling = np.sqrt(np.clip(reliability[mask], 0.0, 1.0))
    counts = ~np.isnan(expected) & (ceiling > 0)
    r = score.attrs['per_voxel_r']
    indices = list(np.flatnonzero(mask))
    assert np.isnan(r[indices.index(constant)])
    assert np.isfinite(r[indices.index(unreliable)])   # kept in the vector, not in the median
    assert (score.attrs['n_voxels_scored'], score.attrs['n_voxels_excluded']) == (30, 2)
    assert score.attrs['ceiling_aggregation'] == 'per_voxel' and score.attrs['normalized'] is True
    assert float(score) == pytest.approx(np.median(expected[counts] / ceiling[counts]), rel=1e-4)
    assert float(score.attrs['raw']) == pytest.approx(np.median(expected[counts]), rel=1e-4)
    assert float(score.attrs['ceiling']) == pytest.approx(np.median(ceiling[counts]), rel=1e-6)


# ── TR time series ────────────────────────────────────────────────


def tr_design(runs, events):
    """Each run's design: the clip's features at floor(onset / TR), HRF-convolved."""
    kernel = double_gamma_hrf(duration_sec=32.0, sampling_rate_hz=1.0 / TR_SEC)
    designs = []
    for run in runs:
        impulses = np.zeros((N_TR, WIDTH), dtype=np.float32)
        own = events[(events['subject'] == run['subject']) & (events['task'] == run['task'])
                     & (events['session'] == run['session']) & (events['run'] == run['run'])]
        for _, event in own.iterrows():
            impulses[int(np.floor(event['onset_sec'] / TR_SEC))] += unit_values(
                event['stimulus_id'], 'neural:IT', WIDTH)[0]
        designs.append(hrf_convolve(impulses, sampling_rate_hz=1.0 / TR_SEC, hrf=kernel))
    return designs


def tr_bold(runs, events, outside, inside_constant):
    """BOLD driven by the design inside the region; very large and unrelated outside."""
    rng = np.random.default_rng(7)
    designs = tr_design(runs, events)
    bold = np.stack([design[:, np.arange(N_VOXELS) % WIDTH] for design in designs])
    bold = bold + 0.5 * rng.standard_normal(bold.shape)
    bold[:, :, outside] = 1e7 + 1e6 * rng.standard_normal((len(runs), N_TR, int(outside.sum())))
    bold[:, :, inside_constant] = 7.0
    return bold


def test_the_region_is_selected_before_the_fit(monkeypatch, tmp_path):
    runs = tr_runs()
    events = tr_events(runs)
    region = mask_of(range(0, N_VOXELS, 9))
    constant = 9 * 100
    bold = tr_bold(runs, events, ~region, constant)
    install(monkeypatch, tmp_path, runs=runs, bold=bold)
    benchmark = Lahner2024BOLDMoments_timeresolved(folds=RunBlocks(2), voxel_mask_fn=lambda: region)
    score = benchmark(Recorder({'vision:video'}))

    # Expected: ridge on the region's voxels only, BOLD z-scored per run and voxel,
    # the two blocks of two runs held out in turn.
    X = np.concatenate(tr_design(runs, events))
    Y = []
    for run_bold in bold[:, :, region]:
        sd = run_bold.std(0)
        Y.append(np.where(sd > 0, (run_bold - run_bold.mean(0)) / np.where(sd > 0, sd, 1), 0.0))
    Y = np.concatenate(Y)
    rows = np.repeat(np.arange(len(runs)), N_TR)
    folds = [(rows >= 2, rows < 2), (rows < 2, rows >= 2)]
    expected = held_out_r(X, Y, folds)

    r = score.attrs['per_voxel_r']
    position = list(np.flatnonzero(region)).index(constant)
    assert len(r) == region.sum() and np.isnan(r[position]) and np.isnan(expected[position])
    counts = ~np.isnan(expected)
    np.testing.assert_allclose(r[counts], expected[counts], rtol=1e-4, atol=1e-5)
    assert score.attrs['n_voxels_excluded'] == 1
    assert float(score) == pytest.approx(np.median(expected[counts]), rel=1e-4)

    # The voxels outside the region never enter the fit: replacing them changes no bit.
    (tmp_path / 'other').mkdir()
    calm = bold.copy()
    calm[:, :, ~region] = 0.0
    install(monkeypatch, tmp_path / 'other', runs=runs, bold=calm)
    again = Lahner2024BOLDMoments_timeresolved(folds=RunBlocks(2), voxel_mask_fn=lambda: region)(
        Recorder({'vision:video'}))
    np.testing.assert_array_equal(again.attrs['per_voxel_r'], r)
    assert repr(float(again)) == repr(float(score))


@pytest.mark.parametrize('confounds', [(), ('trans_x', 'rot_y', 'csf')])
def test_a_voxel_is_preprocessed_alike_in_any_selection(monkeypatch, tmp_path, confounds):
    """Confound regression and z-score of a kept voxel give the same bits whether the
    region or every voxel is preprocessed: no step mixes voxels, even in rounding.
    A constant voxel stays zero, with or without confounds (the rounding of the
    confound regression is not scaled up); a nearly constant voxel is standardized."""
    runs = tr_runs()
    rng = np.random.default_rng(12)
    bold = 10000.0 + 50.0 * rng.standard_normal((len(runs), N_TR, N_VOXELS))
    bold[:, :, 5] = 10000.0
    bold[:, :, 6] = 10000.0 + 0.001 * (np.arange(N_TR) % 2)
    install(monkeypatch, tmp_path, runs=runs, bold=bold)
    benchmark = Lahner2024BOLDMoments_timeresolved(confounds=confounds)
    design = np.zeros((len(runs), N_TR, 2), dtype=np.float32)
    every = benchmark._concatenate_with_mask(design, benchmark.assembly)[1]
    assert every.dtype == np.float32
    assert not every[:, 5].any()
    assert every[:, 6].all() and every[:, 6].std() == pytest.approx(1, abs=1e-5)
    for voxels in ([5], [5, 6], [0, 5, 6, 9], range(0, N_VOXELS, 7)):
        region = mask_of(voxels)
        np.testing.assert_array_equal(
            benchmark._concatenate_with_mask(design, benchmark.assembly, region)[1], every[:, region])


@pytest.mark.parametrize('r,ceiling,kept', [
    ([0.2, np.inf, 0.8], None, [0.2, 0.8]),
    ([0.2, -np.inf, np.nan, 0.8], None, [0.2, 0.8]),
    ([0.2, 0.4, 0.6, 0.8], [1.0, np.inf, 0.0, -0.5], [0.2]),
    ([0.2, 0.4, 0.6], [0.5, np.nan, 0.5], [0.2, 0.6]),
])
def test_only_finite_r_and_valid_ceilings_count(r, ceiling, kept):
    from types import SimpleNamespace
    from brainscore_core.metrics import Score
    from brainscore.benchmarks.lahner2024.benchmark import voxel_score
    benchmark = SimpleNamespace(ceiling=Score(1.0), version=2, voxel_mask=None)
    score = voxel_score(benchmark, np.array(r), None if ceiling is None else np.array(ceiling))
    assert score.attrs['n_voxels_scored'] == len(kept)
    assert score.attrs['n_voxels_excluded'] == len(r) - len(kept)
    assert score.attrs['mean_r'] == pytest.approx(np.mean(kept))
    assert float(score.attrs['raw']) == pytest.approx(np.median(kept))
