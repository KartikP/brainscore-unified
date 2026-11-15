"""
CLIP ViT-B/32 registered as a BrainScoreModel -- the first VLM in Brain-Score.

Uses PytorchWrapper wrapping clip_model.vision_model for vision extraction,
and TextWrapper wrapping clip_model.text_model for text extraction. Both
share the same underlying CLIPModel.

PytorchWrapper wraps vision_model (not full CLIPModel) because CLIPModel.forward()
unconditionally invokes the text encoder, which fails without input_ids.
Similarly, TextWrapper wraps text_model directly. All region_layer_map paths
are therefore relative to the respective sub-module root.
"""

import functools

import torch
from transformers import CLIPConfig, CLIPModel, CLIPProcessor

from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore import BrainScoreModel, LayerSelector
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_core.hf_compat import pin_image_processor
from brainscore.models._shared import _load_preprocess_images


REGION_LAYER_MAP = {
    # Vision regions -- relative to vision_model (PytorchWrapper root)
    'V1': 'encoder.layers.1',
    'V2': 'encoder.layers.3',
    'V4': 'encoder.layers.6',
    'IT': 'encoder.layers.10',
    # Language region -- relative to text_model (TextWrapper root)
    'language_system': 'encoder.layers.10',
}


def load_clip(random_init: bool = False):
    """CLIP ViT-B/32 and its processor (image part pinned): the one load of the family.

    ``random_init`` gives the untrained null: the same architecture from the
    checkpoint's config, weights initialised by the model with torch seed 0.
    """
    # Pin the image-processor implementation: transformers 5 rebinds the class
    # names, moving the default from PIL to torchvision and shifting pixels.
    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_processor = pin_image_processor(clip_processor, 'openai/clip-vit-base-patch32')
    if random_init:
        config = CLIPConfig.from_pretrained('openai/clip-vit-base-patch32')
        torch.manual_seed(0)
        return CLIPModel(config), clip_processor
    return CLIPModel.from_pretrained('openai/clip-vit-base-patch32'), clip_processor


def picture_tower(clip_model, clip_processor, identifier: str, backbone_id=None) -> PytorchWrapper:
    """The picture extractor of every CLIP registration: the vision model with the image loader.

    ``identifier`` labels the activations; ``backbone_id`` names the stored
    path of the legacy wrapper (its ``identifier`` when None).
    """
    return PytorchWrapper(
        identifier=identifier,
        backbone_id=backbone_id,
        model=clip_model.vision_model,
        preprocessing=functools.partial(_load_preprocess_images, processor=clip_processor),
    )


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'clip-vit-b-32'
    clip_model, clip_processor = load_clip()
    vision_wrapper = picture_tower(clip_model, clip_processor, identifier)

    # Text: TextWrapper wraps text_model sub-module. CLIP's text encoder is
    # bidirectional (BERT-like), so mean aggregation is appropriate.
    text_wrapper = TextWrapper(
        model=clip_model.text_model,
        tokenizer=clip_processor.tokenizer,
        identifier=f'{identifier}-text',
        layer_aggregation='mean',
        max_length=77,
    )

    return BrainScoreModel(
        identifier=identifier,
        model=clip_model,
        region_layer_map={
            region: LayerSelector(
                name=layer,
                modality='text' if region == 'language_system' else 'vision',
            )
            for region, layer in REGION_LAYER_MAP.items()
        },
        extractor={
            'vision': vision_wrapper,
            'text': text_wrapper,
        },
        # Behavioral readout uses the final vision encoder block (matches
        # the 'IT' layer convention). Behavioral benchmarks like Ferguson2024
        # will fit a logistic readout on top of this layer's features.
        behavioral_readout_layer='encoder.layers.10',
        # The readout derives the choice from its probabilities by argmax.
        capability_config={'behavior-sessions': {'behavior:choice': ('readout', 'argmax')}},
    )
