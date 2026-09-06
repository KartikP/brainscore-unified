"""Tests for the LanA language-network mask on fsaverage5.

The mask decides which vertices a reported mean is taken over, so an error here
moves a headline number without failing anything. Two of its assumptions are
silent if wrong — that fsaverage5 is the leading block of the fsaverage7 mesh,
and that both the atlas and this benchmark order the hemispheres left-first —
so both are checked against the data rather than asserted in a comment.
"""

import numpy as np
import pytest

from brainscore.benchmarks.lebel2023.language_mask import (
    FSAVERAGE5_PER_HEMISPHERE, lana_mask, lana_probabilities, mask_summary)

pytestmark = pytest.mark.unit

N_VERTICES = 2 * FSAVERAGE5_PER_HEMISPHERE


def _atlas_available():
    try:
        lana_probabilities()
        return True
    except (FileNotFoundError, ImportError):
        return False


needs_atlas = pytest.mark.skipif(
    not _atlas_available(),
    reason='LanA atlas not present; set BRAINSCORE_LANA_ATLAS (osf.io/kzwbh)')


class TestMaskShape:
    def test_fsaverage5_hemisphere_size(self):
        assert FSAVERAGE5_PER_HEMISPHERE == 10242      # ico5: 10 * 4**5 + 2

    def test_top_fraction_is_validated(self):
        for bad in (0, -0.1, 1.5):
            with pytest.raises(ValueError, match='top_fraction'):
                lana_mask(top_fraction=bad)


class TestMaskSummary:
    def test_reports_both_conventions_for_one_fit(self):
        """A mean within the mask and a median over all cortex, same scores."""
        r = np.zeros(N_VERTICES)
        mask = np.zeros(N_VERTICES, dtype=bool)
        mask[:100] = True
        r[:100] = 0.5
        summary = mask_summary(r, mask)
        assert summary['language_mask_mean'] == pytest.approx(0.5)
        assert summary['whole_cortex_median'] == pytest.approx(0.0)
        assert summary['n_mask_vertices'] == 100

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match='against'):
            mask_summary(np.zeros(10), np.ones(11, dtype=bool))


@needs_atlas
class TestAgainstTheAtlas:
    def test_mask_holds_the_requested_share_of_cortex(self):
        mask = lana_mask(top_fraction=0.10)
        assert len(mask) == N_VERTICES
        assert mask.mean() == pytest.approx(0.10, abs=0.005)

    def test_mask_is_left_lateralised(self):
        """LanA is a left-lateralised network.

        This is the check that the atlas files are being read and concatenated
        left-first: a right-first read would invert the ratio.
        """
        mask = lana_mask()
        left = mask[:FSAVERAGE5_PER_HEMISPHERE].sum()
        assert left / mask.sum() > 0.6, 'mask is not left-lateralised'

    def test_probabilities_are_a_probability(self):
        p = lana_probabilities()
        assert p.shape == (N_VERTICES,)
        assert 0.0 <= p.min() and p.max() <= 1.0

    def test_downsampling_takes_the_leading_ico5_block(self):
        """fsaverage5 is the fsaverage7 prefix, not an interpolation.

        FreeSurfer's icosahedra are hierarchical, so the first 10242 vertices of
        the 163842-vertex mesh *are* the coarser mesh. Pinned by re-reading the
        source and confirming the kept block is its head.
        """
        import nibabel as nib
        from brainscore.benchmarks.lebel2023 import language_mask
        full = np.asarray(nib.load(
            language_mask._atlas_dir() / 'LH_LanA_n804.nii.gz').dataobj).squeeze()
        assert len(full) == 163842                      # ico7: 10 * 4**7 + 2
        kept = lana_probabilities()[:FSAVERAGE5_PER_HEMISPHERE]
        assert np.allclose(kept, full[:FSAVERAGE5_PER_HEMISPHERE])
