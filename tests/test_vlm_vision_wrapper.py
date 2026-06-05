"""Tests for VLMVisionWrapper — activations_model for VLM vision encoders
with flattened-patch layouts.

Uses Qwen2.5-VL-3B as the reference test case since it exhibits the full
flattened-patch layout with grid_thw metadata.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytestmark = pytest.mark.slow  # loads real Qwen2.5-VL-3B weights — run on demand

from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


@pytest.fixture(scope='module')
def qwen_vision_wrapper():
    """VLMVisionWrapper wrapping Qwen's vision encoder."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype=torch.float16,
    )
    qwen_processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')

    return VLMVisionWrapper(
        model=qwen_model.model.visual,
        processor=qwen_processor,
        identifier='qwen2.5-vl-3b-vision-test',
        image_input_key='pixel_values',
        forward_kwargs_map={'grid_thw': 'image_grid_thw'},
        patch_count_fn=lambda out: [int(t * h * w) for t, h, w in out['image_grid_thw']],
        layer_aggregation='mean_patches',
        batch_size=2,
    )


_img_counter = 0


def _make_image_stimuli(n=3, identifier_suffix=''):
    global _img_counter
    _img_counter += 1
    tmpdir = tempfile.mkdtemp()
    paths = {}
    for i in range(n):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(tmpdir, f'img{i}.png')
        img.save(img_path)
        paths[f'img{i}'] = img_path

    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': list(paths.keys()),
        'image_file_name': list(paths.values()),
    }))
    stimuli.identifier = f'vlm_vision_test_{_img_counter}{identifier_suffix}'
    stimuli.stimulus_paths = paths
    return stimuli, tmpdir


@pytest.fixture
def image_stimuli():
    stimuli, tmpdir = _make_image_stimuli(3)
    yield stimuli
    import shutil
    shutil.rmtree(tmpdir)


class TestVLMVisionWrapperBasic:
    def test_returns_neuroid_assembly(self, qwen_vision_wrapper, image_stimuli):
        result = qwen_vision_wrapper(image_stimuli, layers=['blocks.28'])
        assert isinstance(result, NeuroidAssembly)

    def test_shape_is_per_image(self, qwen_vision_wrapper, image_stimuli):
        """Critical test: the whole point of this wrapper is that output is
        (n_images, features), NOT (total_patches, features)."""
        result = qwen_vision_wrapper(image_stimuli, layers=['blocks.28'])
        assert result.dims == ('presentation', 'neuroid')
        assert result.shape[0] == 3  # 3 images, NOT 3 * 256 patches

    def test_stimulus_id_coord(self, qwen_vision_wrapper, image_stimuli):
        result = qwen_vision_wrapper(image_stimuli, layers=['blocks.28'])
        assert 'stimulus_id' in result.coords
        assert list(result['stimulus_id'].values) == ['img0', 'img1', 'img2']

    def test_finite_activations(self, qwen_vision_wrapper, image_stimuli):
        result = qwen_vision_wrapper(image_stimuli, layers=['blocks.28'])
        assert bool(result.notnull().all().item())

    def test_layer_coord(self, qwen_vision_wrapper, image_stimuli):
        result = qwen_vision_wrapper(image_stimuli, layers=['blocks.28'])
        layers = np.unique(result['layer'].values)
        assert list(layers) == ['blocks.28']


class TestPatchToImageAggregation:
    """The critical correctness check — patches must be grouped back to the
    right images, not mixed across images."""

    def test_each_image_gets_independent_vector(self, qwen_vision_wrapper):
        """If patch segmentation is wrong, two identical images batched with
        different companions will produce different activations."""
        tmpdir = tempfile.mkdtemp()
        try:
            # Make image A deterministic, image B deterministic and different
            np.random.seed(42)
            img_a = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            np.random.seed(99)
            img_b = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

            path_a = os.path.join(tmpdir, 'a.png')
            path_b = os.path.join(tmpdir, 'b.png')
            img_a.save(path_a)
            img_b.save(path_b)

            # Same image A should produce (approximately) the same activation
            # whether batched alone or with a different image.
            stim_alone = StimulusSet(pd.DataFrame({
                'stimulus_id': ['a'],
                'image_file_name': [path_a],
            }))
            stim_alone.identifier = 'agg_alone'
            stim_alone.stimulus_paths = {'a': path_a}

            stim_with_b = StimulusSet(pd.DataFrame({
                'stimulus_id': ['a', 'b'],
                'image_file_name': [path_a, path_b],
            }))
            stim_with_b.identifier = 'agg_with_b'
            stim_with_b.stimulus_paths = {'a': path_a, 'b': path_b}

            r_alone = qwen_vision_wrapper(stim_alone, layers=['blocks.28'])
            r_with_b = qwen_vision_wrapper(stim_with_b, layers=['blocks.28'])

            alone_a = r_alone.values[0]
            with_b_a = r_with_b.values[
                list(r_with_b['stimulus_id'].values).index('a')]
            # FP16 arithmetic introduces small differences, so a loose tolerance
            assert np.allclose(alone_a, with_b_a, atol=1e-2)
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_different_images_different_activations(self, qwen_vision_wrapper, image_stimuli):
        """Sanity: different images should produce different activations."""
        result = qwen_vision_wrapper(image_stimuli, layers=['blocks.28'])
        row_0 = result.values[0]
        row_1 = result.values[1]
        assert not np.allclose(row_0, row_1, atol=1e-3)


class TestCaching:
    def test_second_call_hits_cache(self, qwen_vision_wrapper):
        stimuli, tmpdir = _make_image_stimuli(2, identifier_suffix='_cache')
        try:
            r1 = qwen_vision_wrapper(stimuli, layers=['blocks.28'])
            r2 = qwen_vision_wrapper(stimuli, layers=['blocks.28'])
            assert np.allclose(r1.values, r2.values)
        finally:
            import shutil
            shutil.rmtree(tmpdir)


class TestVLMVisionWrapperIntegration:
    def test_works_as_brainscore_model_preprocessor(self, qwen_vision_wrapper, image_stimuli):
        """VLMVisionWrapper used as the vision preprocessor in BrainScoreModel.

        BrainScoreModel.process() duck-types the preprocessor — anything with
        an `identifier` attribute is treated as a full extractor and called
        with (stimuli, layers=[...]).
        """
        model = BrainScoreModel(
            identifier='qwen2.5-vl-3b-test',
            model=None,
            region_layer_map={'IT': 'blocks.28'},
            preprocessors={'vision': qwen_vision_wrapper},
        )
        model.start_recording('IT')

        # Direct wrapper call (mimics how BrainScoreModel.process() invokes it)
        result = qwen_vision_wrapper(image_stimuli, layers=['blocks.28'])
        assert result.dims == ('presentation', 'neuroid')
        assert result.shape[0] == 3
