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
    FILE_BACKED = {'image_file_name', 'image_path', 'filename', 'video_path',
                   'audio_path', 'audio_file_name', 'audio_file'}
    INLINE = {'sentence', 'text'}          # TextWrapper reads the value itself
    present = (FILE_BACKED | INLINE) & set(stimuli.columns)
    assert present, (
        f'no recognized modality column in {list(stimuli.columns)}; a model cannot '
        f'tell what kind of data this is')

    if FILE_BACKED & present:
        # Only file-backed modalities need paths on disk — a real model opens them.
        # Text stimuli carry their content inline and have no files, so requiring
        # stimulus_paths there would reject a perfectly valid language plugin.
        assert hasattr(stimuli, 'stimulus_paths'), (
            'file-backed stimuli need a stimulus_paths mapping')
        for stimulus_id in stimuli['stimulus_id'].values:
            path = stimuli.stimulus_paths[stimulus_id]
            assert os.path.exists(path), (
                f'stimulus_paths[{stimulus_id!r}] -> missing {path}')
    else:
        column = sorted(INLINE & present)[0]
        values = [str(v).strip() for v in stimuli[column].values]
        assert all(values), f'empty {column!r} values; inline stimuli carry their content'


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
    """The pairing benchmarks depend on, and the easiest thing to get subtly wrong.

    A subset check is NOT enough. A model predicts one row per stimulus in the set, so
    if the assembly is missing even one measurement the prediction and target have
    different lengths and the metric fails — while a subset assertion happily passes.
    That is the shape of a half-finished data conversion, so check for it exactly.
    """
    import brainscore
    stimuli = brainscore.load_stimulus_set('your-stimuli')
    assembly = brainscore.load_dataset('your-measurements')

    in_stimuli = list(np.asarray(stimuli['stimulus_id'].values).astype(str))
    in_assembly = list(np.asarray(assembly['stimulus_id'].values).astype(str))

    assert len(set(in_stimuli)) == len(in_stimuli), 'duplicate stimulus_id in the stimulus set'
    assert len(set(in_assembly)) == len(in_assembly), 'duplicate stimulus_id in the assembly'
    assert set(in_assembly) == set(in_stimuli), (
        f'stimulus_id sets differ — measured but not shown: '
        f'{sorted(set(in_assembly) - set(in_stimuli))[:5]}; '
        f'shown but not measured: {sorted(set(in_stimuli) - set(in_assembly))[:5]}. '
        f'If your dataset genuinely measures only a subset, filter the stimulus set to '
        f'the measured ids so a model is never asked to predict rows you cannot score.')


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
