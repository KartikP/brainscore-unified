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

from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore import BrainScoreModel, LayerSelector
from brainscore.models.clip_vit_b_32.model import REGION_LAYER_MAP, load_clip, picture_tower


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'random-vit-b-32'

    # Same architecture as CLIP ViT-B/32 but untrained (``load_clip(random_init=True)``);
    # the processor is reused: its preprocessing is not learned.
    clip_model, clip_processor = load_clip(random_init=True)
    vision_wrapper = picture_tower(clip_model, clip_processor, identifier)

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
        # language_system is recorded from the text tower, every other region from
        # the vision tower: the towers each layer path is relative to.
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
        behavioral_readout_layer='encoder.layers.10',
        # The readout derives the choice from its probabilities by argmax.
        capability_config={'behavior-sessions': {'behavior:choice': ('readout', 'argmax')}},
    )
