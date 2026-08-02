"""TODO: one-line description of the model and what it can be scored on.

A model is ONE BrainScoreModel construction. The optional slots you fill decide which
dispatch branches process() can take (see
https://brain-score.github.io/public/UMI/architecture.html for the full router):
  - activations_model  -> neural encoding + behavioral readout
  - generation_fn      -> behavioral generation (instruction-following)
  - action_fn          -> embodied closed-loop (EnvironmentStep)
  - state_change_fn    -> lesion/perturbation (StateChange)
Leave a slot as None if the model doesn't support that capability.

**This file runs as-is.** It registers a small randomly-initialised CNN, so the whole
path — load, record, process, get an assembly back — works offline the moment you copy
the folder. Swap in your real backbone a piece at a time and keep the tests green.
The random weights are scaffolding, not a scientific claim: a real registration loads
pretrained weights here.
"""
import torch.nn as nn

from brainscore_core.model_interface import BrainScoreModel
from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper, load_preprocess_images)

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
    # VisionWrapper fronts PytorchWrapper / VLMVisionWrapper / VideoWrapper, so a vision
    # model usually needs only that one surface. PytorchWrapper is used directly here to
    # keep the example's moving parts visible.
    activations_model = PytorchWrapper(
        identifier=IDENTIFIER,
        model=backbone,
        preprocessing=lambda paths: load_preprocess_images(paths, image_size=64))

    # TODO: simple per-modality callables (resize/normalize, tokenize, resample).
    #   preprocessors.keys() IS the model's supported_modalities — there is no separate
    #   declaration to keep in sync, and no hasattr check anywhere in the codebase.
    # Identity is right when the wrapper already does the preprocessing, as here.
    preprocessors = {'vision': lambda stimuli: stimuli}

    return BrainScoreModel(
        identifier=IDENTIFIER,
        model=backbone,
        region_layer_map={'IT': '2'},   # brain region -> layer to record. TODO: yours.
        preprocessors=preprocessors,
        activations_model=activations_model,
        # --- optional capability slots (see templates/new_capability/) ---
        # generation_fn=...,
        # action_fn=...,
        # state_change_fn=...,
        # behavioral_readout_layer='TODO.layer',
        # region_modality_map={'IT': 'vision', 'A1': 'audio'},  # multimodal routing
        # backbone_id='shared-backbone-id',                     # share a feature cache
    )
