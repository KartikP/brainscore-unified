"""Record layers while something else runs the model, one presentation at a time.

An extractor's ``record(layers)`` returns one of these. ``BrainScoreModel``
uses it to record regions during the forward pass that produces an answer, so
a stimulus is shown once for both measurements. ``take()`` returns the first
forward pass since the previous ``take()``: for a generating model that is the
prompt pass; the passes for generated tokens are not kept.
"""
from contextlib import contextmanager
from typing import Callable, Dict, List

import numpy as np


class LiveRecording:
    def __init__(self, layers: List[str], aggregate: Callable, package: Callable):
        self.layers, self._aggregate, self._package = list(layers), aggregate, package
        self._first: Dict[str, np.ndarray] = {}

    def hook(self, layer: str):
        def capture(_module, _inputs, output):
            if layer in self._first:
                return          # a later pass (a generated token): keep the prompt pass
            if isinstance(output, (tuple, list)):
                output = output[0]
            activity = output.detach().cpu()
            if activity.dtype.is_floating_point:
                activity = activity.float()      # numpy has no bfloat16
            self._first[layer] = activity.numpy()
        return capture

    def take(self):
        """The first forward pass since the last take, as a one-presentation assembly."""
        missing = [layer for layer in self.layers if layer not in self._first]
        if missing:
            raise ValueError(f'No forward pass reached {missing} since the last presentation.')
        activity = {layer: self._aggregate(self._first[layer]) for layer in self.layers}
        self._first.clear()
        return self._package(activity)


@contextmanager
def live_recording(get_layer: Callable, layers: List[str], aggregate: Callable,
                   package: Callable):
    """Hook ``layers`` (resolved with ``get_layer``) until the block exits."""
    recording = LiveRecording(layers, aggregate, package)
    handles = [get_layer(layer).register_forward_hook(recording.hook(layer))
               for layer in recording.layers]
    try:
        yield recording
    finally:
        for handle in handles:
            handle.remove()
