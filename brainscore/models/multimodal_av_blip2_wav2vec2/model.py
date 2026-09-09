"""BLIP-2 OPT-2.7B (vision tower) + Wav2Vec2-base (audio tower).

BLIP-2's vision_model is a frozen ViT-G/14 (~1B params). The
multimodal benchmark routes its 'vision' modality through frame-
aggregation: extract N frames per clip, run BLIP-2's vision encoder
on each, mean-pool over frames before regression.

ViT-G hook outputs (batch, 257, 1408) flatten to 361K features per
image — far too many for sklearn ridge. The standalone BLIP-2 model
applies LayerPCA(1000) at registration time to keep features
manageable; we reuse the same pattern here.

Single registration only (no random-init null variants — BLIP-2 is
2.7B params and re-randomizing would be costly compute we'd repeat
on every cold load). The CLIP and V-JEPA combos already cover all
4 signal/null permutations between them.
"""
import functools

import torch
from PIL import Image

from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.hf_compat import pin_image_processor


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


def _load_preprocess_images(image_filepaths, processor, image_size=224):
    images = []
    for path in image_filepaths:
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img.copy())
    processed = processor(images=images, return_tensors='pt')
    return processed['pixel_values'].numpy()


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'blip2-wav2vec2'
    from brainscore.models._downloads import hf_preflight
    download = hf_preflight(identifier, 'Salesforce/blip2-opt-2.7b', 15.5)

    from transformers import AutoProcessor, Blip2ForConditionalGeneration
    from brainscore_vision.model_helpers.activations.pca import LayerPCA
    from brainscore_vision.model_helpers.activations.pytorch import (
        PytorchWrapper)
    from brainscore.models._multimodal_av_shared import build_audio_wrapper

    # Pin the image-processor implementation (see brainscore_core.hf_compat).
    blip_processor = AutoProcessor.from_pretrained(
        'Salesforce/blip2-opt-2.7b', **download)
    blip_processor = pin_image_processor(
        blip_processor, 'Salesforce/blip2-opt-2.7b', **download)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        'Salesforce/blip2-opt-2.7b', torch_dtype=torch.float32, **download)

    preprocessing = functools.partial(
        _load_preprocess_images, processor=blip_processor, image_size=224)
    activations_model = PytorchWrapper(
        identifier=f'{identifier}-vision',
        model=blip_model.vision_model,
        preprocessing=preprocessing,
    )
    activations_model.image_size = 224
    LayerPCA.hook(activations_model, n_components=1000)
    activations_model._extractor.identifier = 'blip2-vision-pca1000'

    audio_wrapper, _ = build_audio_wrapper(identifier, random_init=False)

    if torch.cuda.is_available():
        target_device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        target_device = torch.device('mps')
    else:
        target_device = torch.device('cpu')
    blip_model = blip_model.to(target_device)

    return BrainScoreModel(
        identifier=identifier,
        model=blip_model,
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
