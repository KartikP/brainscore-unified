"""The activations cache must miss when the stimulus set's columns change.

@store_xarray keys on (model, stimuli_identifier, layers) -- NOT on the stimulus set's
columns. With a fixed identifier, a cached assembly built from one column set gets
merged with a fresh extraction built from another, and pandas raises an opaque
`AssertionError: Length of new_levels (N) must be <= self.nlevels (M)` from deep inside
a MultiIndex recode. That is what blocked scoring CLIP on Algonauts entirely.

Folding a column signature into the identifier turns that silent collision into a cache
miss, which is correct: different metadata columns are different stimuli.
"""
import hashlib
import inspect

import pytest

from brainscore.benchmarks.algonauts2025 import benchmark as algo


@pytest.mark.unit
def test_frame_stimulus_identifier_includes_a_column_signature():
    src = inspect.getsource(algo)
    assert 'column_signature' in src, (
        "the frame stimulus-set identifier no longer varies with its columns; a stale "
        "cache entry with different columns will collide on the presentation MultiIndex")
    assert "'|'.join(sorted(map(str, df.columns)))" in src, (
        "the signature must be computed from the SORTED column names, so column order "
        "alone does not change the cache key")


@pytest.mark.unit
def test_signature_is_stable_under_column_reordering_and_changes_with_content():
    def sig(columns):
        return hashlib.md5('|'.join(sorted(map(str, columns))).encode()).hexdigest()[:8]

    assert sig(['a', 'b', 'c']) == sig(['c', 'a', 'b']), (
        "reordering columns must NOT change the cache key -- it is the same metadata")
    assert sig(['a', 'b', 'c']) != sig(['a', 'b', 'c', 'd']), (
        "adding a column MUST change the cache key -- that is the collision being fixed")


@pytest.mark.unit
def test_frame_table_identifier_is_unchanged_and_the_video_table_has_its_own(tmp_path):
    """Frame identifier: the md5 of the sorted frame-table columns, as before."""
    import pandas as pd
    from unittest.mock import patch
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends

    frame_signature = hashlib.md5(
        '|'.join(sorted(['stimulus_id', 'frame_id', 'clip_id', 't_within_run',
                         'image_file_name'])).encode()).hexdigest()[:8]
    assert frame_signature == hashlib.md5(
        b'clip_id|frame_id|image_file_name|stimulus_id|t_within_run').hexdigest()[:8]

    movies = pd.DataFrame({'stimulus_id': ['m1'], 'video_path': [str(tmp_path / 'm1.mkv')]})
    (tmp_path / 'm1.mkv').write_bytes(b'')
    # The cutter is patched out below, so put the two clip files in place of its work.
    (tmp_path / 'clips' / 'm1').mkdir(parents=True)
    for index in range(2):
        (tmp_path / 'clips' / 'm1' / f'{index:04d}.mp4').write_bytes(b'')
    for form, tag in (('image', 'frames'), ('video', 'clips')):
        benchmark = Algonauts2025Friends(subject=1, form=form, assembly_root=tmp_path)
        benchmark._stimulus_set = movies
        with patch.object(Algonauts2025Friends, '_stim_id_to_n_TRs', return_value={'m1': 2}), \
                patch.object(Algonauts2025Friends, '_run_pool'), \
                patch.object(Algonauts2025Friends, '_fill_trailing_files', return_value=0):
            table = (benchmark._expand_to_per_TR_stimuli())
        if form == 'image':
            expected = f'algonauts2025-friends-sub01-frames-{frame_signature}'
            image_identifier = table.identifier
            assert table.identifier == expected
        else:
            column_signature = hashlib.md5(
                b'clip_id|frame_id|stimulus_id|t_within_run|video_path').hexdigest()[:8]
            assert table.identifier == f'algonauts2025-friends-sub01-clips-{column_signature}'
            assert table.identifier != image_identifier
            assert list(table['video_path']) == [
                str(tmp_path / 'clips' / 'm1' / '0000.mp4'),
                str(tmp_path / 'clips' / 'm1' / '0001.mp4')]
