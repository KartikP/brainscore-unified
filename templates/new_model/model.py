"""TODO: one-line description of the model and what it can be scored on.

A model is ONE BrainScoreModel construction. The optional slots you fill decide which
dispatch branches process() can take (see
https://brain-score.github.io/public/UMI/architecture.html for the full router):
  - extractor     -> neural recording + behavioral readout
  - answer        -> behavioral answers in words (instruction-following)
  - policy        -> closed loop (EnvironmentStep)
  - intervention  -> lesion/perturbation (StateChange)
Leave a slot as None if the model doesn't support that capability.

**This file runs as-is.** It registers a small randomly-initialised CNN, so the whole
path — load, record, process, get an assembly back — works offline the moment you copy
the folder. Swap in your real backbone a piece at a time and keep the tests green.
The random weights are scaffolding, not a scientific claim: a real registration loads
pretrained weights here.
"""
import torch.nn as nn

from brainscore import BrainScoreModel
from brainscore_vision.model_helpers.activations.pytorch import (
    load_preprocess_images)
from brainscore.model_helpers.vision_wrapper import VisionWrapper

IDENTIFIER = 'your-model'


def get_model() -> BrainScoreModel:
    # TODO: load your backbone (nn.Module). Use None for an output-only / API model.
    # Layer names matter: whatever you put in region_layer_map below has to resolve
    # against this module tree, so give the layers you intend to record real names.
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=7, stride=4, padding=3),   # '0'
        nn.ReLU(),                                             # '1'
        nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # '2'  <- recorded below
        nn.ReLU(),                                             # '3'
        nn.AdaptiveAvgPool2d((4, 4)),                          # '4'
    ).eval()

    # TODO: pick the wrapper that matches your model. One of:
    #   VisionWrapper (any vision: image / VLM / video — dispatches internally; pass kind= to force)
    #   TextWrapper (LM) · AudioWrapper (HF audio encoder)
    # Configure input preparation on the wrapper; the subject routes each input
    # kind (vision, text, audio) to the extractor registered for it.
    vision_extractor = VisionWrapper(
        identifier=IDENTIFIER,
        model=backbone,
        preprocessing=lambda paths: load_preprocess_images(paths, image_size=64),
    )

    return BrainScoreModel(
        identifier=IDENTIFIER,
        model=backbone,
        region_layer_map={'IT': '2'},   # brain region -> layer to record. TODO: yours.
        extractor={'vision': vision_extractor},
        # --- optional capability slots (see templates/new_capability/) ---
        # answer=build_answer(...),
        # policy=build_policy(...),
        # intervention=build_intervention(...),
        # behavioral_readout_layer='TODO.layer',
        # backbone_id='shared-backbone-id',                     # share a feature cache
    )
