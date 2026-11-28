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
from typing import List

import torch
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPProcessor

from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore import BrainScoreModel, CompositeSelector, LayerSelector
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
# The region cortex: every k-th vision block, at most 12 (the rule of the study models); of 12 blocks, all.
CORTEX_LAYERS = tuple(f'encoder.layers.{i}' for i in range(12))


def region_layers() -> dict:
    """The region map of the CLIP registrations: ``REGION_LAYER_MAP`` from its towers, and ``cortex``."""
    layers = {region: LayerSelector(name=layer, modality='text' if region == 'language_system' else 'vision')
              for region, layer in REGION_LAYER_MAP.items()}
    layers['cortex'] = CompositeSelector(tuple((layer, None) for layer in CORTEX_LAYERS), modality='vision')
    return layers

# The zero-shot registration matches each image against the text of each label of the task.
ZERO_SHOT_IDENTIFIER = 'clip-vit-b-32-zero-shot'
ZERO_SHOT_PROMPT = 'a photo of a {label}.'


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


def picture_extractor(clip_model, clip_processor, identifier: str, backbone_id=None) -> PytorchWrapper:
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


def zero_shot_answer(clip_model, clip_processor):
    """The ``answer`` part of the zero-shot registration: CLIP's own zero-shot method.

    The image and the prompt of each label (``ZERO_SHOT_PROMPT``) go through the
    network; its ``logits_per_image`` (the cosine similarity of the projections
    scaled by ``logit_scale``) gives one logit per label, and the label with the
    largest softmax probability is the answer. The labels are the task's own; a label that is not text is refused.
    The registration states the method ``similarity``: the part declares no instruction
    input, so a session never gives it one (``instruction`` is always None here).
    """
    from brainscore.model_helpers.api_behavioral import row_parts
    from brainscore.model_helpers.device import select_device

    # The complete network (projection heads included) and its inputs go to the
    # device the wrappers select.
    device = select_device()
    clip_model.to(device)
    # A prompt longer than the checkpoint's text positions is cut at that limit.
    limit = clip_model.config.text_config.max_position_embeddings

    def answer(stimulus_row, instruction, label_set: List[str]) -> str:
        if not label_set or not all(isinstance(label, str) for label in label_set):
            raise ValueError(f'{ZERO_SHOT_IDENTIFIER}: a Choice needs its labels as text; '
                             f'got {label_set!r}.')
        image_column, _ = row_parts(stimulus_row, f'{ZERO_SHOT_IDENTIFIER} answer')
        if image_column is None:
            raise ValueError(f'{ZERO_SHOT_IDENTIFIER}: the stimulus row has no image.')
        with Image.open(stimulus_row[image_column]) as image:
            image = image.convert('RGB')
        pixels = clip_processor(images=[image], return_tensors='pt')['pixel_values']
        tokens = clip_processor.tokenizer([ZERO_SHOT_PROMPT.format(label=label) for label in label_set],
                                          padding=True, truncation=True, max_length=limit,
                                          return_tensors='pt')
        with torch.no_grad():
            # CLIP's own image-text logits (cosine times logit_scale) in transformers 4 and 5.
            probabilities = clip_model(
                pixel_values=pixels.to(device), input_ids=tokens['input_ids'].to(device),
                attention_mask=tokens['attention_mask'].to(device)).logits_per_image.softmax(dim=-1)
        return label_set[int(probabilities[0].argmax())]

    return answer


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier in ('clip-vit-b-32', ZERO_SHOT_IDENTIFIER)
    clip_model, clip_processor = load_clip()
    zero_shot = identifier == ZERO_SHOT_IDENTIFIER
    # Both registrations name their extractors 'clip-vit-b-32': one activation cache.
    identifier_of_network = 'clip-vit-b-32'
    vision_wrapper = picture_extractor(clip_model, clip_processor, identifier_of_network)

    # Text: TextWrapper wraps text_model sub-module. CLIP's text encoder is
    # bidirectional (BERT-like), so mean aggregation is appropriate.
    text_wrapper = TextWrapper(
        model=clip_model.text_model,
        tokenizer=clip_processor.tokenizer,
        identifier=f'{identifier_of_network}-text',
        layer_aggregation='mean',
        max_length=77,
    )

    region_layer_map = region_layers()
    extractor = {
        'vision': vision_wrapper,
        'text': text_wrapper,
    }
    # Behavioral readout uses the final vision encoder block (matches
    # the 'IT' layer convention). Behavioral benchmarks like Ferguson2024
    # will fit a logistic readout on top of this layer's features.
    if zero_shot:
        # The answer part owns the choice; the readout keeps the probabilities. The
        # choice needs a picture, so a choice task with no picture is N/A at pre-flight.
        return BrainScoreModel(
            identifier=identifier,
            model=clip_model,
            region_layer_map=region_layer_map,
            extractor=extractor,
            behavioral_readout_layer='encoder.layers.10',
            answer=zero_shot_answer(clip_model, clip_processor),
            answer_owners={'behavior:choice': 'similarity'},
            output_inputs={'behavior:choice': {'vision:image'}},
        )
    # The readout derives the choice from its probabilities by argmax.
    return BrainScoreModel(
        identifier=identifier,
        model=clip_model,
        region_layer_map=region_layer_map,
        extractor=extractor,
        behavioral_readout_layer='encoder.layers.10',
        answer_owners={'behavior:choice': ('readout', 'argmax')},
    )
