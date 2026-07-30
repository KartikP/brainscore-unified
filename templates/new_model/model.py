"""TODO: one-line description of the model and what it can be scored on.

A model is ONE BrainScoreModel construction. The optional slots you fill decide which
dispatch branches process() can take (see https://brain-score.github.io/public/UMI/architecture.html for the full router):
  - activations_model  -> neural encoding + behavioral readout
  - generation_fn      -> behavioral generation (instruction-following)
  - action_fn          -> embodied closed-loop (EnvironmentStep)
  - state_change_fn    -> lesion/perturbation (StateChange)
Leave a slot as None if the model doesn't support that capability.
"""
from brainscore_core.model_interface import BrainScoreModel


def get_model() -> BrainScoreModel:
    # TODO: load your backbone (nn.Module). Use None for an output-only / API model.
    backbone = ...

    # TODO: pick the wrapper that matches your model and build it. One of:
    #   VisionWrapper (any vision: image / VLM / video — dispatches internally; pass kind= to force)
    #   TextWrapper (LM) · AudioWrapper (HF audio encoder)
    # VisionWrapper fronts PytorchWrapper / VLMVisionWrapper / VideoWrapper, so a vision
    # model only needs this one surface:  VisionWrapper(backbone, preprocessing, identifier=...)
    activations_model = ...

    # TODO: simple per-modality callables (resize/normalize, tokenize, resample).
    #   preprocessors.keys() IS the model's supported_modalities.
    preprocessors = {'vision': ...}

    return BrainScoreModel(
        identifier='your-model',
        model=backbone,
        region_layer_map={'IT': 'TODO.layer.path'},   # brain region -> layer to record
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
