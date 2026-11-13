"""Extractors: the wrappers that record a network's layers for a modality.

Each wrapper is named after what it wraps and is imported from its own module:
``PytorchWrapper`` (still images, from
``brainscore_vision.model_helpers.activations.pytorch``), ``VLMVisionWrapper``
(images in a vision-language model with flattened patches), ``VLMVideoWrapper``
(clips through a vision-language model's own video processor), ``VideoWrapper``
(video clips), ``TextWrapper`` and ``AudioWrapper``. The plumbing they share (layer
lookup, hooks, batching, the cache key) is in ``_extractors``; in each wrapper of
this folder a layer name that ends in ``':input'`` records the module's first input.
"""
