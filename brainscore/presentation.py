"""Optional stimulus presentation, before model-specific input preparation.

Components transform decoded payloads. They do not schedule sessions, step an
environment, or claim physical display/audio calibration.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from brainscore_core.streaming import StreamEvent, parse_channel


@dataclass(frozen=True)
class Screen:
    """Place a floating-point image on a canvas, with contrast about background.

    Image values must be in [0, 1], shaped (height, width) or (height, width,
    channels). Coordinates and canvas dimensions are pixels. No resizing,
    visual-angle conversion, or timing is implied.
    """
    height: int
    width: int
    top: int = 0
    left: int = 0
    background: float = 0.5
    contrast: float = 1.0

    def __post_init__(self) -> None:
        for name in ('height', 'width', 'top', 'left'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{name} must be an integer pixel count')
        if min(self.height, self.width) <= 0 or min(self.top, self.left) < 0:
            raise ValueError('Canvas dimensions must be positive and offsets nonnegative')
        if not np.isfinite(self.background) or not 0 <= self.background <= 1:
            raise ValueError('background must be in [0, 1]')
        if not np.isfinite(self.contrast) or not 0 <= self.contrast <= 1:
            raise ValueError('contrast must be in [0, 1]')

    def describe(self) -> dict[str, Any]:
        """Return the presentation settings saved with each delivered event."""
        return {'component': 'Screen', 'version': 1, 'height': self.height,
                'width': self.width, 'top': self.top, 'left': self.left,
                'background': self.background, 'contrast': self.contrast,
                'coordinate_units': 'pixels'}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return a new canvas; never modify the supplied image."""
        image = np.asarray(image)
        if image.ndim not in (2, 3) or not np.issubdtype(image.dtype, np.floating):
            raise ValueError('Screen expects a floating-point HxW or HxWxC image')
        if not image.size or not np.isfinite(image).all() or np.any((image < 0) | (image > 1)):
            raise ValueError('Image values must be finite and in [0, 1]')
        h, w = image.shape[:2]
        if self.top + h > self.height or self.left + w > self.width:
            raise ValueError('Image does not fit on the screen; clipping is not implicit')
        canvas = np.full((self.height, self.width) + image.shape[2:], self.background, dtype=image.dtype)
        canvas[self.top:self.top + h, self.left:self.left + w] = (
            self.background + self.contrast * (image - self.background)
        )
        return canvas


@dataclass(frozen=True)
class Speaker:
    """Attenuate normalized waveform amplitude; this is not calibrated loudness.

    Payload: {'samples': floating array, 'sample_rate': positive integer}.
    Samples are mono (time,) or multi-channel (time, channels), in [-1, 1].
    Sample rate, sample count, and any other payload fields remain unchanged.
    """
    gain: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.gain) or not 0 <= self.gain <= 1:
            raise ValueError('gain must be in [0, 1]; amplification is not supported')

    def describe(self) -> dict[str, Any]:
        """Return amplitude settings, without claiming physical sound levels."""
        return {'component': 'Speaker', 'version': 1, 'gain': self.gain,
                'gain_units': 'linear_amplitude'}

    def __call__(self, audio: Mapping[str, Any]) -> dict[str, Any]:
        """Return an independent audio payload with attenuated samples."""
        if not isinstance(audio, Mapping) or 'samples' not in audio or 'sample_rate' not in audio:
            raise ValueError('Speaker expects samples and sample_rate')
        rate = audio['sample_rate']
        if isinstance(rate, bool) or not isinstance(rate, (int, np.integer)) or rate <= 0:
            raise ValueError('sample_rate must be a positive integer in Hz')
        samples = np.asarray(audio['samples'])
        if samples.ndim not in (1, 2) or not np.issubdtype(samples.dtype, np.floating):
            raise ValueError('Samples must be a floating-point time or time-by-channel array')
        if not samples.size or not np.isfinite(samples).all() or np.any(np.abs(samples) > 1):
            raise ValueError('Samples must be finite and in [-1, 1]')
        result = deepcopy(dict(audio))
        result['samples'] = samples * self.gain
        return result


class Presentation:
    """Apply an ordered list of payload components to each exact input channel.

    Components are callables with describe() returning JSON-serializable settings.
    No registration or subclass is needed. Unconfigured channels pass through as
    independent copies. Input events, timestamps, and channel names are preserved.
    """

    def __init__(self, *, components: Mapping[str, Sequence[Callable]]) -> None:
        self._components = {}
        for channel, sequence in components.items():
            parse_channel(channel)
            prepared = tuple(sequence)
            for component in prepared:
                if not callable(component) or not callable(getattr(component, 'describe', None)):
                    raise TypeError('A presentation component must be callable and implement describe()')
                json.dumps(component.describe(), allow_nan=False)
            self._components[channel] = prepared

    def __call__(self, event: StreamEvent) -> StreamEvent:
        """Prepare one input and attach the settings actually used to produce it."""
        if not isinstance(event, StreamEvent):
            raise TypeError('Presentation expects a StreamEvent')
        result = deepcopy(event)
        history = result.meta.get('presentation', [])
        if not isinstance(history, list):
            raise ValueError('event.meta[presentation] must be a list')
        for component in self._components.get(event.channel, ()):
            settings = deepcopy(component.describe())
            json.dumps(settings, allow_nan=False)
            result.payload = component(result.payload)
            history.append(settings)
        if history:
            result.meta['presentation'] = history
        return result
