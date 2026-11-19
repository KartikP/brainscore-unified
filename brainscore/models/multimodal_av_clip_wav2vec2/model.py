"""CLIP ViT-B/32 (vision tower) + Wav2Vec2-base (audio tower).

CLIP is a still-image VLM, so the multimodal benchmark routes through
the frame-aggregation path: extract N frames per clip, run CLIP on
each, mean-pool over frames before regression.

This is the "capable VLM" complement to the V-JEPA-class native-video
combos — it lets us check whether a contrastive image-text encoder
trained on web data carries comparable brain-prediction signal to
representation-reconstruction video models, when both are paired with
the same audio extractor.

Four signal/null permutations registered:
- clip-wav2vec2                  signal vision + signal audio
- random-clip-wav2vec2           null vision   + signal audio
- clip-random-wav2vec2           signal vision + null audio
- random-clip-random-wav2vec2    null vision   + null audio
"""

from brainscore import BrainScoreModel, LayerSelector
from brainscore.models._shared import build_audio_wrapper


REGION_LAYER_MAP = {
    # Vision tower (CLIP ViT-B/32 vision_model — paths relative to
    # the wrapped sub-module). Per the standalone clip-vit-b-32 entry.
    'V1': 'encoder.layers.1',
    'V2': 'encoder.layers.3',
    'V4': 'encoder.layers.6',
    'IT': 'encoder.layers.10',
    # Audio tower
    'A1': 'encoder.layers.6',
}

REGION_MODALITY_MAP = {
    'V1': 'vision',
    'V2': 'vision',
    'V4': 'vision',
    'IT': 'vision',
    'A1': 'audio',
}


def _build_vision_wrapper(combo_identifier: str, random_init: bool = False):
    """The picture extractor of ``clip-vit-b-32`` (or of ``random-vit-b-32`` with ``random_init``)."""
    from brainscore.models.clip_vit_b_32.model import load_clip, picture_extractor
    clip_model, processor = load_clip(random_init=random_init)
    # The activations carry the backbone's name; the cache key is the combo's.
    vision_wrapper = picture_extractor(clip_model, processor,
                                   'random-clip-vit-b-32' if random_init else 'clip-vit-b-32',
                                   backbone_id=f'{combo_identifier}-vision')
    return vision_wrapper


def get_model(identifier: str) -> BrainScoreModel:
    random_vision = identifier.startswith('random-clip')
    random_audio = 'random-wav2vec2' in identifier

    vision_wrapper = _build_vision_wrapper(
        identifier, random_init=random_vision)
    audio_wrapper, _ = build_audio_wrapper(
        identifier, random_init=random_audio)

    return BrainScoreModel(
        identifier=identifier,
        region_layer_map={
            region: LayerSelector(name=layer, modality=REGION_MODALITY_MAP.get(region))
            for region, layer in REGION_LAYER_MAP.items()
        },
        extractor={
            'vision': vision_wrapper,
            'audio': audio_wrapper,
        },
    )
