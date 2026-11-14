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


def _tr_multimodal():
    from brainscore.benchmarks.lahner2024 import Lahner2024BOLDMoments_timeresolved, RunBlocks
    return Lahner2024BOLDMoments_timeresolved(
        channels=('vision:image', 'vision:video', 'audio'), folds=RunBlocks(2))


@pytest.mark.parametrize('build', [
    lambda: brainscore.load_benchmark('Lahner2024-fMRI-naturalistic-multimodal-visualROI'),
    _tr_multimodal,
], ids=['glm', 'tr'])
def test_both_classes_give_one_answer_for_a_subject_that_takes_one_stream(
        monkeypatch, tmp_path, build):
    """Pre-flight alone decides: a subject without ``neural:A1`` is N/A, and a
    subject that takes the clips but no sound runs as a partial match; its one
    session requests both regions. Its score names the inputs scored and carries
    no two-stream label."""
    from brainscore_core.compatibility import CompatibilityError, check_compatibility
    install(monkeypatch, tmp_path)
    benchmark = build()
    with pytest.raises(CompatibilityError, match='cannot produce output channels'):
        check_compatibility(Recorder({'vision:video'}), benchmark)
    subject = Recorder({'vision:video'}, regions=('IT', 'A1'))
    score = brainscore.score(subject, benchmark, check_mem=False)
    assert score.attrs['compatibility']['match'] == 'partial'
    assert score.attrs['compatibility']['dropped_inputs'] == ['audio', 'vision:image']
    assert score.attrs['scored_inputs'] == ['vision:video']
    assert 'mode' not in score.attrs and 'n_features_audio' not in score.attrs
    assert 'multimodal' not in score.attrs['pipeline']
    assert [outputs for _, outputs, _, _ in subject.seen] == [['neural:IT', 'neural:A1']]


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
