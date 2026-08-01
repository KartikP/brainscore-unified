"""The visual-ROI variant must divide by the measured noise ceiling, not by 1.0.

Before 2026-07-31 this benchmark reported a raw, undivided median correlation while
MajajHong/Pereira reported ceiling-normalized scores, so the two could not be shown on
one scale. The ceiling was measured on EC2 (task7_ceiling.json) and pinned; these tests
guard that it stays applied and stays sane.

No model weights, no data download -- these assert on benchmark construction only.
"""
import pytest

from brainscore.benchmarks.lahner2024.benchmark import (
    VISUAL_ROI_CEILING,
    Lahner2024BOLDMoments_visualROI,
    Lahner2024BOLDMoments,
)


@pytest.mark.unit
def test_visual_roi_ceiling_is_applied_not_one():
    bench = Lahner2024BOLDMoments_visualROI()
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
def test_per_voxel_ceiling_returns_none_when_not_normalized():
    # Whole cortex keeps ceiling=1.0, so it must take the scalar path unchanged --
    # the per-voxel branch must not silently alter variants that were never normalized.
    whole = Lahner2024BOLDMoments()
    assert whole._per_voxel_ceiling(mask=None) is None


@pytest.mark.unit
def test_roi_variant_uses_the_per_voxel_branch():
    # The repo convention is median(r_i / c_i), not median(r_i) / median(c_i).
    # Guard that the ROI variant is wired to the per-voxel helper at all: it must not
    # report ceiling == 1.0, which is what sends it down the scalar fallback.
    roi = Lahner2024BOLDMoments_visualROI()
    assert float(roi.ceiling) != 1.0


@pytest.mark.unit
def test_whole_cortex_variant_does_not_borrow_the_roi_ceiling():
    # The ceiling was measured on the 4042-voxel ROI mask. Whole cortex has a very
    # different reliability profile, so applying this number there would be wrong.
    whole = Lahner2024BOLDMoments()
    assert float(whole.ceiling) == 1.0


@pytest.mark.unit
def test_scored_voxels_can_be_placed_back_on_a_surface():
    """The per-voxel scores are ordered over KEPT voxels; without the index mapping
    they cannot be rendered on a cortex at all (and a wrong guess would draw a
    plausible-looking but anatomically meaningless map)."""
    import inspect
    from brainscore.benchmarks.lahner2024 import benchmark as mod
    src = inspect.getsource(mod)
    assert "score.attrs['voxel_mask_indices']" in src, (
        "voxel_mask_indices is what lets per-voxel scores be placed back onto "
        "fsaverage5; dropping it silently breaks every cortical figure")
    assert "np.flatnonzero(mask)" in src
