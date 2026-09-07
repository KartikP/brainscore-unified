"""LanA language-network mask on the fsaverage5 surface.

The reference pipeline reports its headline as a *mean within the LanA language
network*, where this benchmark reports a *median over all cortex*. Those are
different quantities, and comparing them was the original source of an apparent
deficit. This module builds the mask so both can be reported.

The atlas is an external Fedorenko-lab resource (`osf.io/kzwbh`,
doi:10.17605/OSF.IO/KZWBH), not redistributed here. Download its **FS Atlas**
archive — the FreeSurfer-surface arm, which avoids projecting from an MNI volume
— and point ``BRAINSCORE_LANA_ATLAS`` at the directory holding
``LH_LanA_n804.nii.gz`` and ``RH_LanA_n804.nii.gz``.

Those files are fsaverage7 (163842 vertices per hemisphere) and this benchmark
is fsaverage5 (10242). FreeSurfer's icosahedra are hierarchical, so fsaverage5
is exactly the first 10242 vertices of fsaverage7 rather than something needing
interpolation. That is checked rather than assumed: see
``tests/test_lebel2023_language_mask.py``.
"""

import pathlib

import numpy as np

from ...data import local

FSAVERAGE5_PER_HEMISPHERE = 10242
DEFAULT_TOP_FRACTION = 0.10          # the fraction the reference pipeline masks at
_HEMISPHERES = ('LH', 'RH')


def _atlas_dir():
    return local.LANA_ATLAS.resolved()


def lana_probabilities(atlas_dir=None):
    """Per-vertex LanA probability on fsaverage5, left hemisphere first.

    :return: ``(20484,)`` float array in ``[0, 1]``
    """
    import nibabel as nib

    directory = pathlib.Path(atlas_dir) if atlas_dir else _atlas_dir()
    hemispheres = []
    for hemisphere in _HEMISPHERES:
        path = directory / f'{hemisphere}_LanA_n804.nii.gz'
        if not path.exists():
            raise local.LocalDataMissing(local.LANA_ATLAS.instructions())
        values = np.asarray(nib.load(path).dataobj).squeeze()
        # fsaverage5 is the leading ico5 block of the fsaverage7 mesh.
        hemispheres.append(values[:FSAVERAGE5_PER_HEMISPHERE])
    return np.concatenate(hemispheres).astype(np.float64)


def lana_mask(top_fraction=DEFAULT_TOP_FRACTION, atlas_dir=None):
    """Boolean fsaverage5 mask of the most language-selective vertices.

    Thresholds on rank rather than on probability so the mask holds a fixed
    share of cortex regardless of how the atlas is scaled.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError(f'top_fraction must be in (0, 1]; got {top_fraction}')
    probabilities = lana_probabilities(atlas_dir)
    n_selected = int(round(len(probabilities) * top_fraction))
    cutoff = np.partition(probabilities, -n_selected)[-n_selected]
    mask = probabilities >= cutoff
    return mask


def mask_summary(per_vertex_r, mask):
    """Both conventions for one fit, so neither has to be re-run to compare.

    The mask is applied to an existing whole-cortex fit rather than refitting on
    the masked vertices. Keeping one fit means the mask is purely a reporting
    choice: the benchmark selects a single shared ridge penalty across its
    targets, so refitting on a subset would also move that penalty and the
    masked and unmasked numbers would no longer describe the same model.
    """
    per_vertex_r = np.asarray(per_vertex_r, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if per_vertex_r.shape != mask.shape:
        raise ValueError(f'{per_vertex_r.shape} scores against {mask.shape} mask')
    inside = per_vertex_r[mask]
    half = len(mask) // 2
    return {
        'whole_cortex_median': float(np.median(per_vertex_r)),
        'whole_cortex_mean': float(per_vertex_r.mean()),
        'language_mask_median': float(np.median(inside)),
        'language_mask_mean': float(inside.mean()),
        'n_mask_vertices': int(mask.sum()),
        'mask_left_fraction': float(mask[:half].mean()),
        'mask_right_fraction': float(mask[half:].mean()),
        'mask_mean_left': float(per_vertex_r[:half][mask[:half]].mean()),
        'mask_mean_right': float(per_vertex_r[half:][mask[half:]].mean()),
    }
