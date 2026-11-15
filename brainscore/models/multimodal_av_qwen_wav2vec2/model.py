"""Qwen2.5-VL-3B (vision tower) + Wav2Vec2-base (audio tower).

Qwen-VL uses VLMVisionWrapper for its flattened-patch input layout
(rather than PytorchWrapper). The multimodal benchmark routes its
'vision' modality through frame-aggregation: extract N frames per
clip, run Qwen-VL on each, mean-pool over frames.

Single registration only. See blip2-wav2vec2 docstring for why we
skip null variants on these two heavy VLMs (CLIP and V-JEPA combos
already cover the four signal/null permutations).
"""
from brainscore import BrainScoreModel


REGION_LAYER_MAP = {
    'V1': 'blocks.2',
    'V2': 'blocks.6',
    'V4': 'blocks.14',
    'IT': 'blocks.28',
    'A1': 'encoder.layers.6',
}


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'qwen2.5-vl-wav2vec2'
    from brainscore.models._shared import build_audio_wrapper
    from brainscore.models.qwen25_vl_3b.model import load_qwen, picture_tower

    # The picture tower of qwen2.5-vl-3b: the same load and the same extractor.
    qwen_model, qwen_processor, _ = load_qwen(identifier, 'qwen2.5-vl-3b')
    vision_wrapper = picture_tower(qwen_model, qwen_processor, identifier)
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
