"""VisionWrapper -- one vision activations surface that dispatches internally.

Interface-design decision (Schrimpf review, 2026): a user wraps a vision model
with a single ``VisionWrapper`` and never has to categorize it as still-image,
native-temporal, or flattened-patch VLM. The wrapper inspects the model and
routes to the focused internal extraction strategy that actually fits it:

  - ``PytorchWrapper``    -- frame-based CNN / ViT vision (the common case)
  - ``VideoWrapper``      -- native-temporal video models, (B, T, C, H, W)
  - ``VLMVisionWrapper``  -- flattened-patch VLM vision encoders

The three strategies genuinely differ (tensor layout, hook post-processing), so
they stay separate, focused, and independently testable -- they are just kept
*behind* this one surface rather than chosen by the user.

Auto-detection is a heuristic. When it cannot decide, or guesses wrong, pass
``kind='frame' | 'temporal' | 'vlm'`` explicitly -- that is the escape hatch.
All three feed the single ``vision`` channel; a benchmark only ever sees ``vision``.
"""
from typing import Any, Callable, Optional

# class-name substrings that mark a native-temporal video model
_TEMPORAL_NAME_HINTS = ('video', 'vjepa', 'v_jepa', 'videomae', 'timesformer', 'temporal')
# constructor kwargs that only a native-temporal model would pass
_TEMPORAL_KWARGS = ('num_frames', 'target_fps', 'frame_sampler', 'hook_time_axis',
                    'post_hook_fn', 'context_window_ms', 'context_stride_ms', 'max_clip_ms')


def _infer_kind(model: Any, processor: Optional[Any], kwargs: dict) -> str:
    """Best-effort dispatch -> 'vlm' | 'temporal' | 'frame'.

    Order: (1) a HuggingFace ``processor`` implies a patch-layout VLM encoder;
    (2) a temporal class-name or a temporal-only kwarg implies a native-temporal
    model; (3) otherwise the common still-image case. Deliberately conservative:
    when unsure it returns ``'frame'``. Override with ``kind=``.
    """
    if processor is not None:
        return 'vlm'
    name = type(model).__name__.lower()
    if any(h in name for h in _TEMPORAL_NAME_HINTS) or any(k in kwargs for k in _TEMPORAL_KWARGS):
        return 'temporal'
    return 'frame'


class VisionWrapper:
    """Single vision activations-model surface; dispatches to a focused strategy.

    ``VisionWrapper(model, preprocessing)`` covers still-image models. Pass a
    HuggingFace ``processor=`` for a flattened-patch VLM vision encoder, or
    ``kind='temporal'`` for a native-temporal video model. The resolved strategy
    name is on ``.kind`` and the underlying wrapper on ``.strategy``.
    """

    _VALID = ('auto', 'frame', 'temporal', 'vlm')

    def __init__(self, model: Any, preprocessing: Optional[Callable] = None, *,
                 identifier: Optional[str] = None, backbone_id: Optional[str] = None,
                 kind: str = 'auto', processor: Optional[Any] = None, **kwargs):
        if kind not in self._VALID:
            raise ValueError(f"kind must be one of {self._VALID}, got {kind!r}")
        chosen = _infer_kind(model, processor, kwargs) if kind == 'auto' else kind
        self.kind = chosen

        if chosen == 'vlm':
            if processor is None:
                raise ValueError("VisionWrapper(kind='vlm') needs `processor=` "
                                 "(the model's HuggingFace image processor).")
            from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
            self.strategy = VLMVisionWrapper(model, processor, identifier=identifier,
                                             backbone_id=backbone_id, **kwargs)
        elif chosen == 'temporal':
            if preprocessing is None:
                raise ValueError("VisionWrapper(kind='temporal') needs `preprocessing=`.")
            from brainscore.model_helpers.video_wrapper import VideoWrapper
            self.strategy = VideoWrapper(model, preprocessing, identifier=identifier,
                                         backbone_id=backbone_id, **kwargs)
        else:  # frame
            if preprocessing is None:
                raise ValueError("VisionWrapper(kind='frame') needs `preprocessing=`.")
            from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
            self.strategy = PytorchWrapper(model, preprocessing, identifier=identifier,
                                           backbone_id=backbone_id, **kwargs)

    @property
    def identifier(self):
        return self.strategy.identifier

    @identifier.setter
    def identifier(self, value):
        self.strategy.identifier = value

    def __call__(self, *args, **kwargs):
        return self.strategy(*args, **kwargs)

    def __getattr__(self, name):
        # everything else (layer hooks, caching internals, ...) delegates to the strategy
        strategy = self.__dict__.get('strategy')
        if strategy is None:
            raise AttributeError(name)
        return getattr(strategy, name)
