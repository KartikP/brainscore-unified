"""Lahner2024 with the sound track: which subjects it takes, and which voxels it scores.

Synthetic data behind the registered loaders (``tests/lahner_synthetic.py``);
real forward passes run on EC2.
"""
import numpy as np
import pytest
import xarray as xr

import brainscore
from brainscore.benchmarks.lahner2024._util import read_stimulus_ids
from tests.lahner_synthetic import N_VOXELS, Recorder, install


def test_benchmark_rejects_unimodal_subject():
    """An audio-less subject must raise immediately, not silently
    produce video-only output under a multimodal label."""
    from brainscore import BrainScoreModel

    video_only = BrainScoreModel(
        identifier='video-only',
        model=None,
        region_layer_map={'IT': 'layer.10'},
        extractor={'video': lambda *a, **k: None},
    )
    b = brainscore.load_benchmark('Lahner2024-fMRI-naturalistic-multimodal-visualROI')
    with pytest.raises(ValueError, match="both"):
        b(video_only)


@pytest.mark.parametrize('identifier, source', [
    ('Lahner2024-fMRI-naturalistic-multimodal-visualROI', None),
    ('Lahner2024-fMRI-naturalistic-multimodal-auditoryROI', 'build_auditory_mask'),
])
def test_roi_registry_lines_score_their_voxels(monkeypatch, tmp_path, identifier, source):
    install(monkeypatch, tmp_path)
    score = brainscore.load_benchmark(identifier)(
        Recorder({'vision:video', 'audio'}, regions=('IT', 'A1')))
    assert score.attrs['voxel_mask_n_total'] == N_VOXELS
    assert 0 < score.attrs['voxel_mask_n_kept'] < N_VOXELS
    assert score.attrs.get('voxel_mask_source') == source
    assert ('reliability_threshold' in score.attrs) is (source is None)


def test_one_voxel_mask_part_at_a_time():
    from brainscore.benchmarks.lahner2024 import Lahner2024BOLDMoments, build_auditory_mask
    with pytest.raises(ValueError, match='not both'):
        Lahner2024BOLDMoments(reliability_threshold=0.3, voxel_mask_fn=build_auditory_mask)


def test_read_stimulus_ids_from_multiindex():
    """Works whether the presentation dim is a MultiIndex or a plain coord."""
    arr = xr.DataArray(
        np.zeros((2, 3)), dims=('presentation', 'neuroid'),
        coords={'stimulus_id': ('presentation', ['s0', 's1']),
                'extra': ('presentation', ['a', 'b'])},
    )
    assert list(read_stimulus_ids(arr.set_index(presentation=['stimulus_id', 'extra']))) == ['s0', 's1']
    assert list(read_stimulus_ids(arr)) == ['s0', 's1']
