"""Model harnesses (activations-model wrappers).

``VisionWrapper`` is the single vision entry point -- it dispatches to the
frame / temporal / VLM-patch strategy internally. The other wrappers
(``TextWrapper``, ``AudioWrapper``, ``VideoWrapper``, ``VLMVisionWrapper``) are
imported from their own modules.
"""
from .vision_wrapper import VisionWrapper

__all__ = ["VisionWrapper"]
