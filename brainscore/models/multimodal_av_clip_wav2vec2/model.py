"""CLIP ViT-B/32 (vision tower) + Wav2Vec2-base (audio tower).

CLIP is a still-image VLM, so the multimodal benchmark routes through
the frame-aggregation path: extract N frames per clip, run CLIP on
each, mean-pool over frames before regression.

This is the "capable VLM" complement to the V-JEPA-class native-video
combos — it lets us check whether a contrastive image-text encoder
trained on web data carries comparable brain-prediction signal to
representation-reconstruction video models, when both are paired with
the same audio tower.

Four signal/null permutations registered:
- clip-wav2vec2                  signal vision + signal audio
- random-clip-wav2vec2           null vision   + signal audio
- clip-random-wav2vec2           signal vision + null audio
- random-clip-random-wav2vec2    null vision   + null audio
"""

import functools

import numpy as np
import torch
from PIL import Image

from brainscore_core.model_interface import BrainScoreModel


SUPPORTED_IDENTIFIERS = (
    'clip-wav2vec2',
    'random-clip-wav2vec2',
    'clip-random-wav2vec2',
    'random-clip-random-wav2vec2',
)

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


def _load_preprocess_images(image_filepaths, processor, image_size=224):
    images = []
    for path in image_filepaths:
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img.copy())
    processed = processor(images=images, return_tensors='pt')
    return processed['pixel_values'].numpy()


def _build_vision_wrapper(combo_identifier: str, random_init: bool = False):
    """CLIP vision_model wrapped via PytorchWrapper."""
    from transformers import CLIPModel, CLIPProcessor
    from brainscore_vision.model_helpers.activations.pytorch import (
        PytorchWrapper)

    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    if random_init:
        torch.manual_seed(0)
        for p in clip_model.vision_model.parameters():
            p.data.normal_(mean=0.0, std=0.02)

    preprocessing = functools.partial(
        _load_preprocess_images, processor=processor, image_size=224)
    backbone_id = ('random-clip-vit-b-32' if random_init
                   else 'clip-vit-b-32')
    activations_model = PytorchWrapper(
        identifier=f'{combo_identifier}-vision',
        model=clip_model.vision_model,
        preprocessing=preprocessing,
    )
    activations_model.image_size = 224
    activations_model._extractor.identifier = backbone_id
    return preprocessing, activations_model


def get_model(identifier: str) -> BrainScoreModel:
    if identifier not in SUPPORTED_IDENTIFIERS:
        raise AssertionError(
            f"unknown clip+wav2vec2 identifier {identifier!r}; "
            f"expected one of {SUPPORTED_IDENTIFIERS}")
    random_vision = identifier.startswith('random-clip')
    random_audio = 'random-wav2vec2' in identifier

    from brainscore.models._multimodal_av_shared import build_audio_wrapper

    preprocessing, activations_model = _build_vision_wrapper(
        identifier, random_init=random_vision)
    audio_wrapper, _ = build_audio_wrapper(
        identifier, random_init=random_audio)

    return BrainScoreModel(
        identifier=identifier,
        model=None,
        region_layer_map=REGION_LAYER_MAP,
        region_modality_map=REGION_MODALITY_MAP,
        preprocessors={
            'vision': preprocessing,
            'audio': audio_wrapper,
        },
        activations_model=activations_model,
        required_modalities={'vision', 'audio'},
        visual_degrees=8,
    )
