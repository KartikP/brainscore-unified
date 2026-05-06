"""Qwen2.5-VL-3B (vision tower) + Wav2Vec2-base (audio tower).

Qwen-VL uses VLMVisionWrapper for its flattened-patch input layout
(rather than PytorchWrapper). The multimodal benchmark routes its
'vision' modality through frame-aggregation: extract N frames per
clip, run Qwen-VL on each, mean-pool over frames.

Single registration only. See blip2-wav2vec2 docstring for why we
skip null variants on these two heavy VLMs (CLIP and V-JEPA combos
already cover the four signal/null permutations).
"""
import torch

from brainscore_core.model_interface import BrainScoreModel


REGION_LAYER_MAP = {
    'V1': 'blocks.2',
    'V2': 'blocks.6',
    'V4': 'blocks.14',
    'IT': 'blocks.28',
    'A1': 'encoder.layers.6',
}

REGION_MODALITY_MAP = {
    'V1': 'vision',
    'V2': 'vision',
    'V4': 'vision',
    'IT': 'vision',
    'A1': 'audio',
}


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'qwen2.5-vl-wav2vec2'

    from transformers import (
        AutoProcessor, Qwen2_5_VLForConditionalGeneration)
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    from brainscore.models._multimodal_av_shared import build_audio_wrapper

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen2.5-VL-3B-Instruct',
        torch_dtype=torch.float16,
    )
    qwen_processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2.5-VL-3B-Instruct')

    vision_wrapper = VLMVisionWrapper(
        model=qwen_model.model.visual,
        processor=qwen_processor,
        identifier=f'{identifier}-vision',
        image_input_key='pixel_values',
        forward_kwargs_map={'grid_thw': 'image_grid_thw'},
        patch_count_fn=lambda out: [
            int(t * h * w) for t, h, w in out['image_grid_thw']],
        layer_aggregation='mean_patches',
        batch_size=4,
    )
    # Share the standalone Qwen vision-cache key — same backbone, same
    # forward pass for the visual side.
    vision_wrapper.backbone_id = 'qwen2.5-vl-3b-vision'

    audio_wrapper, _ = build_audio_wrapper(identifier, random_init=False)

    return BrainScoreModel(
        identifier=identifier,
        model=qwen_model,
        region_layer_map=REGION_LAYER_MAP,
        region_modality_map=REGION_MODALITY_MAP,
        preprocessors={
            'vision': vision_wrapper,
            'audio': audio_wrapper,
        },
        required_modalities={'vision', 'audio'},
        visual_degrees=8,
    )
