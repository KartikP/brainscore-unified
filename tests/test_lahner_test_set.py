"""Lahner2024 GLM betas: the voxel mask and the ceilings come from the test-set videos
(id above 1000), and the variants that use them score only the other videos.

Synthetic data behind the registered loaders (``tests/lahner_synthetic.py``): six
training clips and five test clips, four participants each.
"""
import numpy as np
import pytest

import brainscore
from brainscore.benchmarks.lahner2024.benchmark import (
    _split_half_reliability, boldmoments_test_set, reliability_on_test_set)
from tests.lahner_synthetic import (
    CLIPS, GLM_CLIPS, GLM_REPS, N_VOXELS, TEST_CLIPS, Recorder, glm_assembly, glm_values,
    install)

VISUAL_ROI = 'Lahner2024-fMRI-naturalistic-visualROI'
MASK_VARIANTS = [(VISUAL_ROI, {'vision:video'}, ('IT',)),
                 ('Lahner2024-fMRI-naturalistic-multimodal-visualROI',
                  {'vision:video', 'audio'}, ('IT', 'A1'))]


def _replace(values, clips, seed):
    """``values`` with the betas of ``clips`` replaced by unrelated random values."""
    values = values.copy()
    rows = np.array([clip in clips for _ in range(GLM_REPS) for clip in GLM_CLIPS])
    values[rows] = np.random.default_rng(seed).standard_normal((rows.sum(), N_VOXELS)) * 5
    return values


def _score(monkeypatch, folder, identifier, channels, regions, values):
    folder.mkdir()
    install(monkeypatch, folder, glm=values)
    subject = Recorder(channels, regions=regions)
    benchmark = brainscore.load_benchmark(identifier)
    return benchmark(subject), benchmark, subject


def test_the_test_set_is_the_videos_above_1000():
    assert boldmoments_test_set(glm_assembly()['stimulus_id'].values) == sorted(TEST_CLIPS)


def test_a_mask_variant_refuses_an_assembly_without_test_videos(monkeypatch, tmp_path):
    values = glm_values()
    install(monkeypatch, tmp_path, glm=values)
    training = glm_assembly(values).isel(presentation=np.array(
        [clip in CLIPS for _ in range(GLM_REPS) for clip in GLM_CLIPS]))
    monkeypatch.delitem(brainscore.data_registry, 'Lahner2024-fMRI')
    monkeypatch.setitem(brainscore.data_registry, 'Lahner2024-fMRI', lambda *a, **k: training)
    with pytest.raises(ValueError, match='no test-set video'):
        brainscore.load_benchmark(VISUAL_ROI)(Recorder({'vision:video'}))


def test_one_rule_gives_the_reliability_for_both_classes():
    """``reliability_on_test_set`` is ``_split_half_reliability`` on the test-set videos, and
    a call needs a subset: no caller can compute on every video by default."""
    assembly = glm_assembly()
    np.testing.assert_array_equal(reliability_on_test_set(assembly),
                                  _split_half_reliability(assembly, stimuli=TEST_CLIPS))
    assert not np.array_equal(reliability_on_test_set(assembly),
                              _split_half_reliability(assembly, stimuli=GLM_CLIPS))
    with pytest.raises(TypeError):
        _split_half_reliability(assembly)


@pytest.mark.parametrize('identifier, channels, regions', MASK_VARIANTS)
def test_mask_and_ceilings_do_not_see_the_training_videos(monkeypatch, tmp_path, identifier,
                                                          channels, regions):
    """Random training betas leave the mask and the ceilings equal; no test video is
    presented or scored."""
    values = glm_values()
    score, benchmark, subject = _score(monkeypatch, tmp_path / 'a', identifier, channels,
                                       regions, values)
    other, other_benchmark, _ = _score(monkeypatch, tmp_path / 'b', identifier, channels,
                                       regions, _replace(values, CLIPS, seed=11))
    np.testing.assert_array_equal(other.attrs['voxel_mask_indices'],
                                  score.attrs['voxel_mask_indices'])
    np.testing.assert_array_equal(other_benchmark.voxel_mask, benchmark.voxel_mask)
    assert score.attrs['n_videos'] == len(CLIPS)
    for _, _, ids, _ in subject.seen:
        assert sorted(ids) == sorted(CLIPS)
    if 'ceiling' in score.attrs and score.attrs['ceiling_aggregation'] == 'per_voxel':
        assert float(other.attrs['ceiling']) == float(score.attrs['ceiling'])


@pytest.mark.parametrize('identifier, channels, regions', MASK_VARIANTS)
def test_mask_and_ceilings_follow_the_test_videos(monkeypatch, tmp_path, identifier,
                                                  channels, regions):
    values = glm_values()
    score, _, _ = _score(monkeypatch, tmp_path / 'a', identifier, channels, regions, values)
    other, _, _ = _score(monkeypatch, tmp_path / 'b', identifier, channels, regions,
                         _replace(values, TEST_CLIPS, seed=12))
    assert not np.array_equal(other.attrs['voxel_mask_indices'], score.attrs['voxel_mask_indices'])


def test_visual_roi_ceilings_are_sqrt_reliability_of_the_test_videos(monkeypatch, tmp_path):
    values = glm_values()
    score, benchmark, _ = _score(monkeypatch, tmp_path / 'a', VISUAL_ROI, {'vision:video'},
                                 ('IT',), values)
    reliability = _split_half_reliability(glm_assembly(values), stimuli=TEST_CLIPS)
    mask = reliability >= 0.3
    np.testing.assert_array_equal(benchmark.voxel_mask, mask)
    ceiling = np.sqrt(np.clip(reliability[mask], 0.0, 1.0))
    counts = np.isfinite(score.attrs['per_voxel_r']) & (ceiling > 0)
    assert float(score.attrs['ceiling']) == pytest.approx(np.median(ceiling[counts]), rel=1e-12)


# Measured at the base commit 6ee39c0 on this synthetic data: the variants with no
# reliability mask and no ceiling score every video, and their value must not move.
# The comparison allows the last digits of another platform's linear algebra (measured:
# at most 8e-8 between macOS and Linux); a changed set of videos moves a value far more.
UNCHANGED = [
    ('Lahner2024-fMRI-naturalistic', {'vision:video'}, ('IT',), -0.15729528665542603),
    ('Lahner2024-fMRI-naturalistic-multimodal', {'vision:video', 'audio'}, ('IT', 'A1'),
     -0.05648249760270119),
    ('Lahner2024-fMRI-naturalistic-multimodal-auditoryROI', {'vision:video', 'audio'},
     ('IT', 'A1'), -0.07669255137443542),
]


@pytest.mark.parametrize('identifier, channels, regions, base', UNCHANGED)
def test_variants_without_mask_or_ceiling_keep_their_value(monkeypatch, tmp_path, identifier,
                                                           channels, regions, base):
    score, _, subject = _score(monkeypatch, tmp_path / 'a', identifier, channels, regions,
                               None)
    assert float(score) == pytest.approx(base, abs=1e-6)
    assert score.attrs['n_videos'] == len(GLM_CLIPS)
    for _, _, ids, _ in subject.seen:
        assert sorted(ids) == sorted(GLM_CLIPS)


@pytest.mark.parametrize('in_channels', [{'vision:video'}, {'vision:image'}])
def test_visual_roi_through_participate_presents_only_training_videos(monkeypatch, tmp_path,
                                                                     in_channels):
    install(monkeypatch, tmp_path)
    subject = Recorder(in_channels)
    score = brainscore.score(subject, brainscore.load_benchmark(VISUAL_ROI), check_mem=False)
    assert np.isfinite(float(score)) and score.attrs['n_videos'] == len(CLIPS)
    [(_, outputs, ids, _)] = subject.seen
    clips = {i.split('_t')[0] for i in ids}
    assert outputs == ['neural:IT'] and clips == set(CLIPS)
