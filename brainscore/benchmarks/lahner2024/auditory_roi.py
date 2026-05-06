"""Auditory-ROI mask for Lahner2024 fsaverage5 voxels.

Built from the Destrieux 2009 surface atlas (10242 vertices per hemi,
matching the Lahner assembly). Selects early auditory + planum cortex:

    33: G_temp_sup-G_T_transv     (Heschl's gyrus — primary auditory)
    34: G_temp_sup-Lateral        (lateral STG — auditory association)
    35: G_temp_sup-Plan_polar     (planum polare)
    36: G_temp_sup-Plan_tempo     (planum temporale — Wernicke)
    75: S_temporal_transverse     (transverse temporal sulcus)

Excludes label 74 (S_temporal_sup, full superior temporal sulcus) —
that's a broader region that includes multimodal social-perception
voxels and would dilute a clean audio-cortex mask.

Total: ~526 voxels across both hemispheres on fsaverage5.

The Lahner assembly is concatenated as (left || right) → 20484 voxels.
We build a boolean mask of that length and the existing voxel-mask
plumbing in ``Lahner2024BOLDMoments_multimodal`` consumes it as-is.
"""
from typing import Iterable

import numpy as np


# Destrieux label indices for early auditory + planum cortex
DEFAULT_AUDITORY_LABELS = (33, 34, 35, 36, 75)


def build_auditory_mask(label_indices: Iterable[int] = None) -> np.ndarray:
    """Return a boolean mask over fsaverage5 cortical voxels (20484-length)
    for the given Destrieux label indices. Concatenation order matches
    the Lahner assembly: left hemisphere (0..10241) then right (10242..20483).
    """
    from nilearn.datasets import fetch_atlas_surf_destrieux

    label_indices = tuple(label_indices) if label_indices is not None \
        else DEFAULT_AUDITORY_LABELS

    atlas = fetch_atlas_surf_destrieux()
    left = atlas['map_left']
    right = atlas['map_right']
    if left.shape[0] != 10242 or right.shape[0] != 10242:
        raise ValueError(
            f"Expected fsaverage5 (10242 vertices per hemi); got "
            f"left={left.shape}, right={right.shape}.")

    mask = np.zeros(20484, dtype=bool)
    label_set = set(label_indices)
    mask[:10242] = np.isin(left, list(label_set))
    mask[10242:] = np.isin(right, list(label_set))
    return mask
