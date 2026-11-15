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

from brainscore import BrainScoreModel
from brainscore_core.hf_compat import pin_image_processor


REGION_LAYER_MAP = {
    'V1': 'blocks.2',
    'V2': 'blocks.6',
    'V4': 'blocks.14',
    'IT': 'blocks.28',
    'A1': 'encoder.layers.6',
}


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'qwen2.5-vl-wav2vec2'
    from brainscore.models._downloads import hf_preflight
    download = hf_preflight(identifier, 'Qwen/Qwen2.5-VL-3B-Instruct', 7.6)

    from transformers import (
        AutoProcessor, Qwen2_5_VLForConditionalGeneration)
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    from brainscore.models._shared import build_audio_wrapper

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen2.5-VL-3B-Instruct',
        torch_dtype=torch.float16,
        **download,
    )
    # Pin the image-processor implementation (see brainscore_core.hf_compat).
    qwen_processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2.5-VL-3B-Instruct', **download)
    qwen_processor = pin_image_processor(
        qwen_processor, 'Qwen/Qwen2.5-VL-3B-Instruct', **download)

    vision_wrapper = VLMVisionWrapper(
        model=qwen_model.model.visual,
        processor=qwen_processor,
        identifier=f'{identifier}-vision',
    )
    audio_wrapper, _ = build_audio_wrapper(identifier, random_init=False)

    return BrainScoreModel(
        identifier=identifier,
        model=qwen_model,
        region_layer_map=REGION_LAYER_MAP,
        extractor={
            'vision': vision_wrapper,
            'audio': audio_wrapper,
        },
    )
