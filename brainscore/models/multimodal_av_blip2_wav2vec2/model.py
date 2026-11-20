"""BLIP-2 OPT-2.7B (vision tower) + Wav2Vec2-base (audio tower).

BLIP-2's vision_model is a frozen ViT-G/14 (~1B params). The
multimodal benchmark routes its 'vision' modality through frame-
aggregation: extract N frames per clip, run BLIP-2's vision encoder
on each, mean-pool over frames before regression.

ViT-G hook outputs (batch, 257, 1408) flatten to 361K features per
image — far too many for sklearn ridge. The standalone BLIP-2 model
applies LayerPCA(1000) at registration time to keep features
manageable. The picture extractor is the one of ``blip2-opt-2.7b``: the same
load (float16) and the same picture extractor (``load_blip2``,
``picture_extractor``).

Single registration only (no random-init null variants — BLIP-2 is
2.7B params and re-randomizing would be costly compute we'd repeat
on every cold load). The CLIP and V-JEPA combos already cover all
4 signal/null permutations between them.
"""
from brainscore import BrainScoreModel, LayerSelector
from brainscore.models._shared import build_audio_wrapper


REGION_LAYER_MAP = {
    'V1': 'encoder.layers.4',
    'V2': 'encoder.layers.10',
    'V4': 'encoder.layers.20',
    'IT': 'encoder.layers.34',
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
    assert identifier == 'blip2-wav2vec2'
    from brainscore.models.blip2_opt_2_7b.model import load_blip2, picture_extractor
    from brainscore.model_helpers.device import select_device

    blip_model, blip_processor = load_blip2(identifier)
    # The activations carry the reduced backbone's name; the cache key is the combo's.
    vision_wrapper = picture_extractor(blip_model, blip_processor, 'blip2-vision-pca1000',
                                   backbone_id=f'{identifier}-vision')

    audio_wrapper, _ = build_audio_wrapper(identifier, random_init=False)

    blip_model = blip_model.to(select_device())

    return BrainScoreModel(
        identifier=identifier,
        model=blip_model,
        region_layer_map={
            region: LayerSelector(name=layer, modality=REGION_MODALITY_MAP.get(region))
            for region, layer in REGION_LAYER_MAP.items()
        },
        extractor={
            'vision': vision_wrapper,
            'audio': audio_wrapper,
        },
    )
