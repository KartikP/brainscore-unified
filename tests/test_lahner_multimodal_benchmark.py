"""Structural tests for Lahner2024-fMRI-naturalistic-multimodal.

Skips anything that needs a real forward pass (model weights, audio files
on disk) — those run out-of-band via score_multimodal.py. This file just
checks that the benchmark registers, validates candidates, and assembles
the per-modality stim sets correctly.
"""
import pytest

import brainscore


def test_benchmark_registered():
    assert ('Lahner2024-fMRI-naturalistic-multimodal'
            in brainscore.benchmark_registry)
    assert ('Lahner2024-fMRI-naturalistic-multimodal-visualROI'
            in brainscore.benchmark_registry)


def test_multimodal_model_registered():
    assert 'vjepa1-wav2vec2' in brainscore.model_registry


def test_benchmark_rejects_unimodal_candidate():
    """An audio-less candidate must raise immediately, not silently
    produce video-only output under a multimodal label."""
    from brainscore_core.model_interface import BrainScoreModel

    audio_only = BrainScoreModel(
        identifier='video-only',
        model=None,
        region_layer_map={'IT': 'layer.10'},
        preprocessors={'video': lambda *a, **k: None},
    )
    b = brainscore.load_benchmark(
        'Lahner2024-fMRI-naturalistic-multimodal-visualROI')
    with pytest.raises(ValueError, match="both"):
        b(audio_only)


def test_to_2d_collapses_time_bin_axis():
    """The benchmark mean-pools the time_bin axis before concat to
    sidestep cross-modality time-grid misalignment."""
    import numpy as np
    import xarray as xr
    from brainscore.benchmarks.lahner2024.benchmark_multimodal import (
        Lahner2024BOLDMoments_multimodal,
    )
    arr = xr.DataArray(
        np.array([[[1.0, 2.0], [3.0, 4.0]]]),  # (1 pres, 2 time, 2 neuroid)
        dims=('presentation', 'time_bin', 'neuroid'),
    )
    out = Lahner2024BOLDMoments_multimodal._to_2d(arr)
    assert 'time_bin' not in out.dims
    np.testing.assert_allclose(out.values, [[2.0, 3.0]])


def test_to_2d_passthrough_when_no_time_bin():
    import numpy as np
    import xarray as xr
    from brainscore.benchmarks.lahner2024.benchmark_multimodal import (
        Lahner2024BOLDMoments_multimodal,
    )
    arr = xr.DataArray(
        np.array([[1.0, 2.0]]), dims=('presentation', 'neuroid'))
    out = Lahner2024BOLDMoments_multimodal._to_2d(arr)
    assert out.dims == ('presentation', 'neuroid')


def test_read_stimulus_ids_from_multiindex():
    """Should work whether the presentation dim is a MultiIndex or
    a plain coord — the benchmark needs to handle both."""
    import numpy as np
    import pandas as pd
    import xarray as xr
    from brainscore.benchmarks.lahner2024.benchmark_multimodal import (
        Lahner2024BOLDMoments_multimodal,
    )
    # MultiIndex form: emulate set_index output
    arr = xr.DataArray(
        np.zeros((2, 3)), dims=('presentation', 'neuroid'),
        coords={'stimulus_id': ('presentation', ['s0', 's1']),
                'extra': ('presentation', ['a', 'b'])},
    )
    arr_mi = arr.set_index(presentation=['stimulus_id', 'extra'])
    ids = Lahner2024BOLDMoments_multimodal._read_stimulus_ids(arr_mi)
    assert list(ids) == ['s0', 's1']
    # Plain coord form
    ids2 = Lahner2024BOLDMoments_multimodal._read_stimulus_ids(arr)
    assert list(ids2) == ['s0', 's1']
