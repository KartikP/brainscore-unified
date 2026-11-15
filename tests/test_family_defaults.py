"""Family plumbing lives once: the defaults and shared helpers give the values of the copies they replace.

The guard references (``tests/test_wrapper_cache_guard.py``) pin what each
wrapper extracts. The tests here pin the rest: a recording through the Qwen-VL
defaults of ``VLMVisionWrapper``, and each shared model helper against a frozen
copy of the code it replaced.
"""
import numpy as np
import pytest
import torch
import xarray
from PIL import Image

from tests.test_wrapper_cache_guard import _PatchProcessor, _image_files, _stimuli, vlm_vision_wrapper

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    monkeypatch.chdir(tmp_path)


# -- VLMVisionWrapper: the Qwen-VL layout is the default -------------------------

def test_a_qwen_pass_is_recorded_as_the_passive_call_gives_it():
    wrapper = vlm_vision_wrapper(batch_size=2)
    stimuli = _stimuli('family-vlm', image_file_name=_image_files()[1:2])      # a 4x6 image: 6 patches
    passive = wrapper(stimuli, layers=['blocks.0', 'blocks.1'])
    processed = _PatchProcessor()(images=[Image.open(stimuli['image_file_name'][0]).convert('RGB')])
    with wrapper.record(['blocks.0', 'blocks.1'], stimuli) as recording, torch.no_grad():
        wrapper._model(processed['pixel_values'], grid_thw=processed['image_grid_thw'])
        taken = recording.take(stimuli.iloc[0])
    for layer, assembly in taken.items():
        expected = passive.isel(neuroid=passive['layer'].values == layer)
        np.testing.assert_array_equal(assembly.values, expected.values)
        assert list(assembly['neuroid_id'].values) == list(expected['neuroid_id'].values)


def test_without_a_grid_each_output_row_is_one_image():
    wrapper = vlm_vision_wrapper()
    assert wrapper._patch_counts({'pixel_values': torch.zeros(2, 12)}) is None
    assert wrapper._patch_counts({'image_grid_thw': torch.tensor([[1, 2, 3], [2, 2, 2]])}) == [6, 8]
    with pytest.raises(TypeError):
        type(wrapper)(wrapper._model, None, patch_count_fn=lambda out: None)


# -- tubelet video models -------------------------------------------------------

def _old_tubelet_mean(arr, temporal, spatial):
    """Frozen copy of the V-JEPA 1, V-JEPA 2 and VideoMAE post-hooks before they were shared."""
    B, seq_len, hidden = arr.shape
    if seq_len == temporal * spatial:
        t_steps, s_steps = temporal, spatial
    elif seq_len % spatial == 0:
        t_steps, s_steps = seq_len // spatial, spatial
    elif seq_len % temporal == 0:
        t_steps, s_steps = temporal, seq_len // temporal
    else:
        raise ValueError('unable to infer')
    return arr.reshape(B, t_steps, s_steps, hidden).mean(axis=2)


def _old_fit(frames, count):
    """Frozen copy of the V-JEPA 1 and 2 frame pad and subsample before it was shared."""
    frames = list(frames)
    if len(frames) < count:
        frames = frames + [frames[-1]] * (count - len(frames))
    elif len(frames) > count:
        idxs = [int(round(i * (len(frames) - 1) / (count - 1))) for i in range(count)]
        frames = [frames[i] for i in idxs]
    return frames


@pytest.mark.parametrize('module, hook, temporal, spatial', [
    ('vjepa', '_vjepa_post_hook', 32, 256), ('vjepa_v1', '_vjepa_v1_post_hook', 8, 196),
    ('videomae_base', '_videomae_post_hook', 8, 196)])
def test_each_tubelet_registration_reduces_as_its_old_copy(module, hook, temporal, spatial):
    import importlib
    post_hook = getattr(importlib.import_module(f'brainscore.models.{module}.model'), hook)
    rng = np.random.default_rng(0)
    for seq in (temporal * spatial, 3 * spatial, temporal * 5):
        arr = rng.standard_normal((2, seq, 4)).astype(np.float32)
        result, expected = post_hook(arr), _old_tubelet_mean(arr, temporal, spatial)
        assert result.dtype == expected.dtype and np.array_equal(result, expected)


@pytest.mark.parametrize('count', [16, 64])
def test_fit_frames_gives_the_old_frames(count):
    from brainscore.models._shared import fit_frames
    frames = [np.full((2, 2, 3), i, np.uint8) for i in range(70)]
    for n in (1, 10, count, 70):
        assert [f is g for f, g in zip(fit_frames(frames[:n], count), _old_fit(frames[:n], count))] == [True] * count
    with pytest.raises(ValueError, match='empty frame list'):
        fit_frames([], count)
