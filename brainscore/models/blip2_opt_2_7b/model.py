"""
BLIP-2 OPT-2.7B registered as a BrainScoreModel — second VLM with a different
architecture from CLIP and Qwen.

Architecture: ViT-G/14 vision encoder + Q-Former + OPT-2.7B causal LM. This
is a third VLM topology (encoder + adapter + decoder), distinct from CLIP's
dual-encoder pair and Qwen's flattened-patch native VLM.

Vision input is standard (batch, C, H, W) — wraps cleanly with PytorchWrapper,
unlike Qwen which needed VLMVisionWrapper. This validates that the unified
interface scales across VLM families with very different vision input layouts:
- CLIP: dual encoder, vision_model.encoder.layers.{N}
- Qwen: flattened patches, VLMVisionWrapper
- BLIP-2: standard ViT, PytorchWrapper

Layer counts:
  vision_model.encoder.layers.{0-38}             — 39 ViT-G blocks
  language_model.model.decoder.layers.{0-31}     — 32 OPT decoder layers
"""

import functools

import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore_core.model_interface import BrainScoreModel
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper


REGION_LAYER_MAP = {
    # Vision regions — relative to vision_model (PytorchWrapper root)
    'V1': 'encoder.layers.4',
    'V2': 'encoder.layers.10',
    'V4': 'encoder.layers.20',
    'IT': 'encoder.layers.34',
    # Language region — relative to language_model.model.decoder (TextWrapper root)
    'language_system': 'layers.28',
}


def _load_preprocess_images(image_filepaths, processor, image_size=224):
    """Load and preprocess images using BLIP-2's processor."""
    images = []
    for path in image_filepaths:
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img.copy())

    # BLIP-2 processor with images-only input
    processed = processor(images=images, return_tensors='pt')
    return processed['pixel_values'].numpy()


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'blip2-opt-2.7b'

    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        'Salesforce/blip2-opt-2.7b',
        torch_dtype=torch.float16,
    )
    blip_processor = AutoProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')

    preprocessing = functools.partial(
        _load_preprocess_images,
        processor=blip_processor,
        image_size=224,
    )

    # Vision: PytorchWrapper wraps vision_model. BLIP-2's vision_model can be
    # called standalone (forward signature: pixel_values -> last_hidden_state)
    # so the full model wrapper isn't needed.
    activations_model = PytorchWrapper(
        identifier=identifier,
        model=blip_model.vision_model,
        preprocessing=preprocessing,
    )
    activations_model.image_size = 224

    # Text: TextWrapper wraps the OPT decoder. Causal LM → last_token aggregation.
    # Wrap language_model.model.decoder so layer paths are 'layers.{N}'.
    text_wrapper = TextWrapper(
        model=blip_model.language_model.model.decoder,
        tokenizer=blip_processor.tokenizer,
        identifier=f'{identifier}-text',
        layer_aggregation='last_token',
        max_length=512,
    )

    return BrainScoreModel(
        identifier=identifier,
        model=blip_model,
        region_layer_map=REGION_LAYER_MAP,
        preprocessors={
            'vision': preprocessing,
            'text': text_wrapper,
        },
        activations_model=activations_model,
        visual_degrees=8,
    )
