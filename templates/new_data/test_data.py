"""Tests are the spec. Data should: register under both ids, load, and be shaped so a
benchmark can actually use it.

The last part is what these check hardest. A stimulus set and an assembly can each look
fine alone and still be unusable together — mismatched ids, a missing modality column, a
single coord that never becomes a MultiIndex. Every assertion below corresponds to a real
failure mode, named in its message.
"""
import os

os.environ.setdefault('RESULTCACHING_DISABLE', '1')

import numpy as np
import pytest


def test_registered_under_both_ids():
    import brainscore
    assert 'your-stimuli' in brainscore.stimulus_set_registry
    assert 'your-measurements' in brainscore.data_registry


def test_stimulus_set_is_loadable_and_dispatchable():
    import brainscore
    stimuli = brainscore.load_stimulus_set('your-stimuli')

    assert len(stimuli) > 0
    assert 'stimulus_id' in stimuli.columns
    # Dispatch picks a modality by RECOGNIZED column name; without one, process()
    # raises "No recognized modality columns".
    recognized = {'image_file_name', 'image_path', 'filename',
                  'sentence', 'text', 'video_path',
                  'audio_path', 'audio_file_name', 'audio_file'}
    assert recognized & set(stimuli.columns), (
        f'no recognized modality column in {list(stimuli.columns)}; a model cannot '
        f'tell what kind of data this is')

    # Every id must resolve to a file that exists — a real model opens these.
    assert hasattr(stimuli, 'stimulus_paths')
    for stimulus_id in stimuli['stimulus_id'].values:
        path = stimuli.stimulus_paths[stimulus_id]
        assert os.path.exists(path), f'stimulus_paths[{stimulus_id!r}] -> missing {path}'


def test_assembly_is_loadable_and_well_formed():
    import brainscore
    assembly = brainscore.load_dataset('your-measurements')

    assert 'presentation' in assembly.dims and 'neuroid' in assembly.dims
    assert np.isfinite(np.asarray(assembly.values)).all(), 'measurements contain NaN/inf'

    # `assembly['name']` reads both plain coords and MultiIndex levels; `.coords` alone
    # under-reports, which is how missing metadata goes unnoticed.
    for coord in ('stimulus_id', 'neuroid_id'):
        assert assembly[coord] is not None

    # Two coords per axis, or brainio never builds the MultiIndex and metrics later fail
    # with "no stimulus_id on the presentation axis".
    levels = set(assembly.indexes['presentation'].names or [])
    assert len(levels) >= 2, (
        f'presentation axis carries {levels}; give it at least two coords so the '
        f'MultiIndex is built')


def test_assembly_and_stimulus_set_agree():
    """The pairing benchmarks depend on, and the easiest thing to get subtly wrong."""
    import brainscore
    stimuli = brainscore.load_stimulus_set('your-stimuli')
    assembly = brainscore.load_dataset('your-measurements')

    in_stimuli = set(np.asarray(stimuli['stimulus_id'].values).astype(str))
    in_assembly = set(np.asarray(assembly['stimulus_id'].values).astype(str))
    assert in_assembly <= in_stimuli, (
        f'measured stimuli missing from the stimulus set: '
        f'{sorted(in_assembly - in_stimuli)[:5]}')


def test_carries_the_coord_metrics_stratify_on():
    """Cross-validated metrics split by a presentation coord and fail loudly without it."""
    import brainscore
    assembly = brainscore.load_dataset('your-measurements')
    try:
        values = np.asarray(assembly['object_name'].values)
    except KeyError:                                          # pragma: no cover
        pytest.fail("no 'object_name' on the presentation axis; cross-validated metrics "
                    "stratify on it and raise 'Expected stratification coordinate'. "
                    "Rename to match your metric if you use a different coord.")
    assert len(set(values)) >= 2, 'stratification needs at least two categories'
