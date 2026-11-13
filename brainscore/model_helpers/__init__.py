"""Model harnesses (activations-model wrappers).

Each wrapper is named after what it wraps and is imported from its own module:
``PytorchWrapper`` (still images, from
``brainscore_vision.model_helpers.activations.pytorch``), ``VLMVisionWrapper``
(images in a vision-language model with flattened patches), ``VideoWrapper``
(video clips), ``TextWrapper`` and ``AudioWrapper``.
"""
