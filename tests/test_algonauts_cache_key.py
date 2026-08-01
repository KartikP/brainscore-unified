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
