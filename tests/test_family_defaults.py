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
