"""
Null-control model: randomly-initialized CLIP ViT-B/32.

Same architecture as the pretrained CLIP we register in clip_vit_b_32/,
but with freshly random weights. Never trained on anything. Used as a
sanity-check baseline: a logistic readout trained on 400 stimuli should
only do as well as random-feature regression allows, which for a balanced
binary task approaches 0.5 (chance). Any benchmark that gives this model
a non-chance score is either leaky (label leak into stimulus metadata)
or the logistic readout alone is over-expressive on the training set.

Deterministic seed so re-runs are reproducible.
"""

import functools

import torch
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPProcessor

from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore_core.model_interface import BrainScoreModel
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_core.hf_compat import pin_image_processor


REGION_LAYER_MAP = {
    'V1': 'encoder.layers.1',
    'V2': 'encoder.layers.3',
    'V4': 'encoder.layers.6',
    'IT': 'encoder.layers.10',
    'language_system': 'encoder.layers.10',
}


def _load_preprocess_images(image_filepaths, processor, image_size=224):
    images = []
    for path in image_filepaths:
        with Image.open(path) as img:
            if img.mode not in ('RGB',):
                img = img.convert('RGB')
            images.append(img.copy())
    processed = processor(images=images, return_tensors='pt')
    return processed['pixel_values'].numpy()


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'random-vit-b-32'

    # Same architecture as CLIP ViT-B/32 but untrained. Config is loaded
    # from the pretrained checkpoint (so architecture is identical) but
    # weights are reinitialized via from_config.
    config = CLIPConfig.from_pretrained('openai/clip-vit-base-patch32')
    # Deterministic random init for reproducibility
    torch.manual_seed(0)
    clip_model = CLIPModel(config)

    # We still need the processor for its preprocessing pipeline
    # (pixel normalization, resizing). These are NOT learned — just
    # standard image preprocessing — so it's fine to reuse.
# Pin the image-processor implementation: transformers 5 rebinds the class
    # names, moving the default from PIL to torchvision and shifting pixels.
    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_processor = pin_image_processor(clip_processor, 'openai/clip-vit-base-patch32')

    preprocessing = functools.partial(
        _load_preprocess_images,
        processor=clip_processor,
        image_size=224,
    )

    activations_model = PytorchWrapper(
        identifier=identifier,
        model=clip_model.vision_model,
        preprocessing=preprocessing,
    )
    activations_model.image_size = 224

    text_wrapper = TextWrapper(
        model=clip_model.text_model,
        tokenizer=clip_processor.tokenizer,
        identifier=f'{identifier}-text',
        layer_aggregation='mean_tokens',
        max_length=77,
    )

    return BrainScoreModel(
        identifier=identifier,
        model=clip_model,
        region_layer_map=REGION_LAYER_MAP,
        preprocessors={
            'vision': preprocessing,
            'text': text_wrapper,
        },
        activations_model=activations_model,
        visual_degrees=8,
        behavioral_readout_layer='encoder.layers.10',
    )
