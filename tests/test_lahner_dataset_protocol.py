"""Lahner2024 GLM betas follow the protocol of the data set: one fit on the training-set
videos, the correlation on the test-set videos (id above 1000), the voxel mask from the
training-set videos and the ceilings from the scored test-set videos.

Synthetic data behind the registered loaders (``tests/lahner_synthetic.py``): six
training clips and five test clips, four participants each. Expected values are
computed here with scikit-learn, in the same process.
"""
import numpy as np
import pytest
from sklearn.linear_model import Ridge

import brainscore
import brainscore.benchmarks.lahner2024.benchmark as lahner
from brainscore.benchmarks.lahner2024 import VISUAL_ROI_CEILING, Lahner2024BOLDMoments
from brainscore.benchmarks.lahner2024.benchmark import (
    _split_half_reliability, boldmoments_test_set, reliability_on_test_set)
from tests.lahner_synthetic import (
    CLIPS, GLM_CLIPS, GLM_REPS, N_VOXELS, TEST_CLIPS, Recorder, glm_assembly, glm_values,
    install, unit_values)

VISUAL_ROI = 'Lahner2024-fMRI-naturalistic-visualROI'
MULTIMODAL_ROI = 'Lahner2024-fMRI-naturalistic-multimodal-visualROI'
MASK_VARIANTS = [(VISUAL_ROI, {'vision:video'}, ('IT',)),
                 (MULTIMODAL_ROI, {'vision:video', 'audio'}, ('IT', 'A1'))]


def _replace(values, clips, seed):
    """``values`` with the betas of ``clips`` replaced by unrelated random values."""
    values = values.copy()
    rows = np.array([clip in clips for _ in range(GLM_REPS) for clip in GLM_CLIPS])
    values[rows] = np.random.default_rng(seed).standard_normal((rows.sum(), N_VOXELS)) * 5
    return values


class _Spy:
    """Wraps the benchmark's ``cross_validated_r``: keeps the folds, counts the fits and
    keeps the predictions of each fit."""

    def __init__(self, monkeypatch):
        self.folds, self.fits = [], []
        original = lahner.cross_validated_r

        def spy(features, Y, folds, fit, **kwargs):
            folds = list(folds)
            self.folds.append(folds)

            def counted(train_bands, Y_train, test_bands, groups=None):
                predictions, chosen = fit(train_bands, Y_train, test_bands, groups)
                self.fits.append((len(Y_train), np.array(predictions)))
                return predictions, chosen
            return original(features, Y, folds, counted, **kwargs)
        monkeypatch.setattr(lahner, 'cross_validated_r', spy)


def _score(monkeypatch, folder, identifier, channels, regions, values):
    folder.mkdir()
    install(monkeypatch, folder, glm=values)
    spy = _Spy(monkeypatch)
    subject = Recorder(channels, regions=regions)
    benchmark = brainscore.load_benchmark(identifier)
    return benchmark(subject), benchmark, subject, spy


def pearson(Y, P):
    Y, P = np.asarray(Y, dtype=float), np.asarray(P, dtype=float)
    Yc, Pc = Y - Y.mean(0), P - P.mean(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (Yc * Pc).sum(0) / den, np.nan)


def test_the_test_set_is_the_videos_above_1000():
    assert boldmoments_test_set(glm_assembly()['stimulus_id'].values) == sorted(TEST_CLIPS)


@pytest.mark.parametrize('kept, message', [(CLIPS, 'no test-set video'),
                                           (TEST_CLIPS, 'no training-set video')])
def test_a_benchmark_refuses_an_assembly_without_one_of_the_sets(monkeypatch, tmp_path, kept,
                                                                 message):
    values = glm_values()
    install(monkeypatch, tmp_path, glm=values)
    rows = np.array([clip in kept for _ in range(GLM_REPS) for clip in GLM_CLIPS])
    part = glm_assembly(values).isel(presentation=rows)
    monkeypatch.delitem(brainscore.data_registry, 'Lahner2024-fMRI')
    monkeypatch.setitem(brainscore.data_registry, 'Lahner2024-fMRI', lambda *a, **k: part)
    monkeypatch.setattr(lahner.Lahner2024BOLDMoments, 'stimulus_set', property(
        lambda self: (lambda s: s[s['stimulus_id'].isin(kept)])(brainscore.load_stimulus_set('BOLDMoments'))))
    subject = Recorder({'vision:video'})
    with pytest.raises(ValueError, match=message):
        brainscore.load_benchmark('Lahner2024-fMRI-naturalistic')(subject)
    assert subject.seen == []                     # it fails before the subject runs


def test_one_rule_gives_the_test_set_reliability():
    """``reliability_on_test_set`` (the GLM ceilings; the time-resolved mask until its own
    change) is ``_split_half_reliability`` on the test-set videos, and a call needs a
    subset: no caller can compute on every video by default."""
    assembly = glm_assembly()
    np.testing.assert_array_equal(reliability_on_test_set(assembly),
                                  _split_half_reliability(assembly, stimuli=TEST_CLIPS))
    with pytest.raises(TypeError):
        _split_half_reliability(assembly)


@pytest.mark.parametrize('identifier, channels, regions', MASK_VARIANTS)
def test_the_mask_follows_the_training_videos(monkeypatch, tmp_path, identifier, channels,
                                              regions):
    values = glm_values()
    _, benchmark, _, _ = _score(monkeypatch, tmp_path / 'a', identifier, channels, regions,
                                values)
    _, other, _, _ = _score(monkeypatch, tmp_path / 'b', identifier, channels, regions,
                            _replace(values, CLIPS, seed=12))
    assert not np.array_equal(other.voxel_mask, benchmark.voxel_mask)


def test_one_fit_on_the_training_rows_and_r_on_the_test_rows(monkeypatch, tmp_path):
    values = glm_values()
    score, _, subject, spy = _score(monkeypatch, tmp_path / 'a', 'Lahner2024-fMRI-naturalistic',
                                    {'vision:video'}, ('IT',), values)
    [(_, _, ids, _)] = subject.seen
    [[(train, test)]] = spy.folds                 # one call, one fixed split
    assert sorted(np.asarray(ids)[train]) == sorted(CLIPS)
    assert sorted(np.asarray(ids)[test]) == sorted(TEST_CLIPS)
    assert len(spy.fits) == 1 and spy.fits[0][0] == len(CLIPS)
    assert (score.attrs['n_videos_fitted'], score.attrs['n_videos_scored']) == (6, 5)

    features = np.stack([unit_values(clip, 'neural:IT')[0] for clip in ids])
    neural = values.reshape(GLM_REPS, len(GLM_CLIPS), N_VOXELS).mean(0)
    neural = neural[[GLM_CLIPS.index(clip) for clip in ids]]
    predictions = Ridge(alpha=1.0).fit(features[train], neural[train]).predict(features[test])
    expected = pearson(neural[test], predictions)
    counts = np.isfinite(expected)
    np.testing.assert_allclose(score.attrs['per_voxel_r'][counts], expected[counts],
                               rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('identifier, channels, regions', MASK_VARIANTS)
def test_no_test_response_reaches_the_fit_or_the_mask(monkeypatch, tmp_path, identifier,
                                                      channels, regions):
    """Random test-set betas leave the mask and the predictions equal; the score moves."""
    values = glm_values()
    score, benchmark, _, spy = _score(monkeypatch, tmp_path / 'a', identifier, channels,
                                      regions, values)
    other, other_benchmark, _, other_spy = _score(monkeypatch, tmp_path / 'b', identifier,
                                                  channels, regions,
                                                  _replace(values, TEST_CLIPS, seed=11))
    np.testing.assert_array_equal(other_benchmark.voxel_mask, benchmark.voxel_mask)
    np.testing.assert_array_equal(other_spy.fits[0][1], spy.fits[0][1])
    assert not np.array_equal(other.attrs['per_voxel_r'], score.attrs['per_voxel_r'])


def _fixed_mask_score(monkeypatch, folder, values):
    """A measured ceiling over a fixed set of voxels, so that the mask cannot move."""
    folder.mkdir()
    install(monkeypatch, folder, glm=values)
    mask = np.zeros(N_VOXELS, dtype=bool)
    mask[N_VOXELS - 200:] = True
    return Lahner2024BOLDMoments(ceiling=VISUAL_ROI_CEILING, voxel_mask_fn=lambda: mask,
                                 identifier='Lahner2024-fMRI-naturalistic')(
        Recorder({'vision:video'}))


def test_ceilings_do_not_see_the_training_videos(monkeypatch, tmp_path):
    values = glm_values()
    score = _fixed_mask_score(monkeypatch, tmp_path / 'a', values)
    other = _fixed_mask_score(monkeypatch, tmp_path / 'b', _replace(values, CLIPS, seed=13))
    assert float(other.attrs['ceiling']) == float(score.attrs['ceiling'])


def test_ceilings_follow_the_test_videos(monkeypatch, tmp_path):
    values = glm_values()
    score = _fixed_mask_score(monkeypatch, tmp_path / 'a', values)
    other = _fixed_mask_score(monkeypatch, tmp_path / 'b', _replace(values, TEST_CLIPS, seed=14))
    assert float(other.attrs['ceiling']) != float(score.attrs['ceiling'])


@pytest.mark.parametrize('identifier, channels, regions', MASK_VARIANTS)
def test_mask_from_the_training_videos_ceilings_from_the_test_videos(monkeypatch, tmp_path,
                                                                     identifier, channels,
                                                                     regions):
    """Both visual-ROI variants: the mask is reliability >= 0.3 on the training-set
    videos, the ceiling of each voxel is sqrt(reliability) on the test-set videos, and a
    voxel whose ceiling is not positive is left out and counted in ``n_voxels_excluded``."""
    values = glm_values()
    score, benchmark, _, _ = _score(monkeypatch, tmp_path / 'a', identifier, channels,
                                    regions, values)
    assembly = glm_assembly(values)
    mask = _split_half_reliability(assembly, stimuli=CLIPS) >= 0.3
    np.testing.assert_array_equal(benchmark.voxel_mask, mask)
    ceiling = np.sqrt(np.clip(reliability_on_test_set(assembly)[mask], 0.0, 1.0))
    r = score.attrs['per_voxel_r']
    counts = np.isfinite(r) & (ceiling > 0)
    assert score.attrs['ceiling_aggregation'] == 'per_voxel' and score.attrs['normalized'] is True
    assert score.attrs['n_voxels_excluded'] == int((~counts).sum())
    assert float(score.attrs['ceiling']) == pytest.approx(np.median(ceiling[counts]), rel=1e-12)
    assert float(score) == pytest.approx(np.median(r[counts] / ceiling[counts]), rel=1e-12)
    assert float(benchmark.ceiling) == VISUAL_ROI_CEILING


GLM_VARIANTS = [
    ('Lahner2024-fMRI-naturalistic', {'vision:video'}, ('IT',)),
    ('Lahner2024-fMRI-naturalistic', {'vision:image'}, ('IT',)),          # picture route
    (VISUAL_ROI, {'vision:image'}, ('IT',)),
    ('Lahner2024-fMRI-naturalistic-multimodal', {'vision:video', 'audio'}, ('IT', 'A1')),
    ('Lahner2024-fMRI-naturalistic-multimodal', {'audio'}, ('IT', 'A1')),  # sound route alone
    (MULTIMODAL_ROI, {'vision:image', 'audio'}, ('IT', 'A1')),
    ('Lahner2024-fMRI-naturalistic-multimodal-auditoryROI', {'vision:video', 'audio'},
     ('IT', 'A1')),
]


@pytest.mark.parametrize('identifier, channels, regions', GLM_VARIANTS)
def test_every_glm_variant_through_participate(monkeypatch, tmp_path, identifier, channels,
                                               regions):
    """Each variant presents every video of the table under its own identifier, fits on
    the six training clips and scores the five test clips."""
    install(monkeypatch, tmp_path)
    subject = Recorder(channels, regions=regions)
    score = brainscore.score(subject, brainscore.load_benchmark(identifier), check_mem=False)
    assert np.isfinite(float(score))
    assert (score.attrs['n_videos_fitted'], score.attrs['n_videos_scored']) == (6, 5)
    assert 'n_videos' not in score.attrs
    for _, _, ids, table in subject.seen:
        assert {i.split('_t')[0] for i in ids} == set(GLM_CLIPS)
        assert table.startswith('BOLDMoments') and '-train' not in table
