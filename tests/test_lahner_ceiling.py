"""The visual-ROI variant must divide by the measured noise ceiling, not by 1.0.

Before 2026-07-31 this benchmark reported a raw, undivided median correlation while
MajajHong/Pereira reported ceiling-normalized scores, so the two could not be shown on
one scale. The ceiling was measured on EC2 (task7_ceiling.json) and pinned; these tests
guard that it stays applied and stays sane.

No model weights, no data download.
"""
import numpy as np
import pytest

import brainscore
from brainscore.benchmarks.lahner2024 import VISUAL_ROI_CEILING, Lahner2024BOLDMoments
from tests.lahner_synthetic import N_VOXELS, Recorder, install


@pytest.mark.unit
def test_visual_roi_ceiling_is_applied_not_one():
    bench = brainscore.benchmark_registry['Lahner2024-fMRI-naturalistic-visualROI']()
    assert float(bench.ceiling) == pytest.approx(VISUAL_ROI_CEILING)
    assert float(bench.ceiling) != 1.0, (
        "visual-ROI reverted to an undivided score; its numbers are then not "
        "comparable to MajajHong/Pereira and must not be shown beside them")


@pytest.mark.unit
def test_ceiling_is_a_plausible_correlation_ceiling():
    # sqrt(reliability) for a predictivity correlation: must be a real correlation
    # bound. A value >1 would inflate every score; <=0 would divide by ~nothing.
    assert 0.0 < VISUAL_ROI_CEILING <= 1.0


@pytest.mark.unit
def test_normalized_beats_raw_but_stays_below_one():
    # V-JEPA v1 raw on this benchmark, measured 2026-07-31 (task6_result.json).
    vjepa1_raw = 0.5328519444188079
    normalized = vjepa1_raw / VISUAL_ROI_CEILING
    assert normalized > vjepa1_raw, "a ceiling <1 must raise the score"
    assert normalized < 1.0, (
        "normalized score at/above 1.0 means the ceiling is underestimated -- "
        "the model would be claimed to beat the data's own reliability")


@pytest.mark.unit
def test_whole_cortex_variant_does_not_borrow_the_roi_ceiling():
    # The ceiling was measured on the 4042-voxel ROI mask. Whole cortex has a very
    # different reliability profile, so applying this number there would be wrong.
    assert float(Lahner2024BOLDMoments().ceiling) == 1.0


def test_whole_cortex_takes_the_scalar_path(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path)
    score = brainscore.load_benchmark('Lahner2024-fMRI-naturalistic')(Recorder({'vision:video'}))
    assert score.attrs['ceiling_aggregation'] == 'scalar'
    assert score.attrs['normalized'] is False
    assert float(score) == float(score.attrs['raw'])


def test_roi_normalizes_per_voxel_and_names_the_scored_voxels(monkeypatch, tmp_path):
    """The repo convention is median(r_i / c_i), not median(r_i) / median(c_i); the
    per-voxel scores are ordered over the kept voxels, and the indices place them back
    on the surface."""
    install(monkeypatch, tmp_path)
    score = brainscore.load_benchmark('Lahner2024-fMRI-naturalistic-visualROI')(
        Recorder({'vision:video'}))
    assert score.attrs['ceiling_aggregation'] == 'per_voxel'
    assert score.attrs['normalized'] is True
    indices = score.attrs['voxel_mask_indices']
    assert len(indices) == score.attrs['voxel_mask_n_kept'] == len(score.attrs['per_voxel_r'])
    assert np.all(np.diff(indices) > 0) and indices[-1] < N_VOXELS
