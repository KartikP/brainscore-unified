"""Tests for the NSD-surface staging in brainscore.topographic_support.

Uses a synthetic surface assembly + a mock vertex->xyz function, so the
selection / reliability-filter / coord-attach logic is exercised without
nilearn or the real NSD assembly.
"""
import numpy as np
import pytest

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore.topographic_support import stage_nsd_surface_target, has_tissue_coords


def _mock_xyz(vertex_index, hemisphere, surf='infl', rh_offset=200.0):
    v = np.asarray(vertex_index, dtype=float)
    return np.stack([v, v * 2.0, v * 3.0], axis=1)   # x == vertex_index


def _synth_surface_assembly():
    # 12 neuroids: 0-5 subj01/IT, 6-7 subj01/V1, 8-11 subj02/IT; all lh.
    n_pres, n = 10, 12
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_pres, n))
    subject = np.array(['subj01'] * 8 + ['subj02'] * 4)
    region = np.array(['IT'] * 6 + ['V1'] * 2 + ['IT'] * 4)
    hemisphere = np.array(['lh'] * n)
    vertex_index = np.arange(n)
    # subj01/IT verts (idx 0-5) get nc = [2,8,12,3,20,1]; >5 keeps idx 1,2,4.
    nc = np.array([2, 8, 12, 3, 20, 1, 99, 99, 99, 99, 99, 99], dtype=float)
    return NeuroidAssembly(
        data, dims=['presentation', 'neuroid'],
        coords={'stimulus_id': ('presentation', [f's{i}' for i in range(n_pres)]),
                'subject': ('neuroid', subject),
                'region': ('neuroid', region),
                'hemisphere': ('neuroid', hemisphere),
                'vertex_index': ('neuroid', vertex_index),
                'nc_testset': ('neuroid', nc)})


def test_stage_selects_subject_region_hemi_and_filters_reliability():
    asm = _synth_surface_assembly()
    brain = stage_nsd_surface_target(
        asm, subject='subj01', region='IT', hemisphere='lh',
        nc_threshold=5.0, vertex_xyz_fn=_mock_xyz)
    assert brain.sizes['neuroid'] == 3          # idx 1,2,4 survive nc>5
    assert has_tissue_coords(brain)
    # mock xyz sets x == vertex_index → the surviving vertices are 1,2,4
    assert sorted(brain['tissue_x'].values.tolist()) == [1.0, 2.0, 4.0]


def test_stage_empty_selection_raises():
    asm = _synth_surface_assembly()
    with pytest.raises(ValueError, match="no vertices"):
        stage_nsd_surface_target(asm, subject='nobody', region='IT',
                                 vertex_xyz_fn=_mock_xyz)


def test_stage_threshold_too_high_raises():
    asm = _synth_surface_assembly()
    with pytest.raises(ValueError, match="survive nc_testset"):
        stage_nsd_surface_target(asm, subject='subj01', region='IT',
                                 hemisphere='lh', nc_threshold=1000.0,
                                 vertex_xyz_fn=_mock_xyz)
